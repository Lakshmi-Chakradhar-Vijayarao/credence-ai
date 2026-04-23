"""
cams/context_manager.py
=======================
CAMSContextManager: Confidence-Adaptive Memory System for Claude API.

Every conversation turn, Claude Opus 4.7 answers the user's question.
CAMS then reads that response, computes a J-proxy confidence score, and
decides what to do with old context:

  HIGH confidence  (J ≥ 0.65) → Compress: ask Claude Haiku to summarize old
                                  turns into 2-3 sentences, discard raw turns.
  MEDIUM confidence(J ∈ [0.35, 0.65)) → Trim: keep last 10 turns, drop older.
  LOW confidence   (J < 0.35)  → Preserve: keep everything.

The compressor IS Claude — Haiku 4.5 summarizes Opus 4.7's own prior turns.
The model manages its own memory: Opus reasons forward while Haiku prunes
what Opus no longer needs to re-read. This is Claude acting as both author
and editor of its own context.

Guard rails:
  - Attention sink protection: first 2 turns are never compressed — they
    establish conversation identity and are disproportionately attended to.
  - Compression depth limit: after 3 compressions the system stops
    compressing (recursive summarization degrades quality).
  - Novelty guard: if a turn introduces many new named entities, PRESERVE
    overrides even a HIGH J-score (new context must not be discarded).
  - Adaptive thinking (opt-in, use_thinking=True): thinking budget scales
    continuously and inversely with the previous turn's J-score.  J=0 (max
    uncertainty) → 2000-token budget; J≥theta_high (confident) → no thinking.
    This makes the J-signal a unified governor of both memory AND compute.
  - Drift detection: if J drops below theta_low for 3 consecutive turns the
    system enters drift state and forces PRESERVE until confidence recovers.
    This is proactive control — the intervention happens before the 4th
    failure, not after.

Savings are tracked against a no-CAMS baseline (full context every turn).
Real token counts come from the Anthropic usage response object.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from .confidence_proxy import ConfidenceProxy, ConfidenceResult

# ---------------------------------------------------------------------------
# Pricing (per million tokens, as of 2026)
# ---------------------------------------------------------------------------
_INPUT_COST_PER_M  = 15.0   # $15 / 1M input  — Opus 4.7
_OUTPUT_COST_PER_M = 75.0   # $75 / 1M output — Opus 4.7
_HAIKU_INPUT_PER_M  = 0.80  # $0.80 / 1M input  — Haiku 4.5
_HAIKU_OUTPUT_PER_M = 4.00  # $4.00 / 1M output — Haiku 4.5

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
_MODEL_OPUS  = "claude-opus-4-7"
_MODEL_HAIKU = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Thinking budget bounds (tokens)
# Budget scales continuously between these by inverse J-score.
# J=0 → _THINKING_MAX; J>=theta_high → 0 (no thinking)
# ---------------------------------------------------------------------------
_THINKING_MIN = 1024   # API minimum; budget_tokens must be >= 1024
_THINKING_MAX = 5000

# ---------------------------------------------------------------------------
# Novelty guard — entity change threshold
# If a turn introduces > this fraction of new named entities, PRESERVE.
# ---------------------------------------------------------------------------
_NOVELTY_THRESHOLD = 0.75   # >75% new entities → treat as new topic
_NOVELTY_MIN_ENTITIES = 5   # require at least 5 new entities (single new names aren't pivots)
_NOVELTY_MIN_VOCAB    = 10  # don't fire until the entity vocab is established
_MIN_COMPRESS_TOKENS  = 150 # don't call Haiku if old segment is smaller than this
_MIN_COMPRESS_ROI     = 50  # minimum net tokens saved (after Haiku overhead) to justify compression

# ---------------------------------------------------------------------------
# Regime detection — only activate CAMS compression/trim when session shows
# evidence of needing it (cross-turn dependencies or mixed J variance).
#
# Rationale: on independent QA sessions, J-variance is low (all turns cluster
# HIGH) and there are no cross-turn references. Applying CAMS compression to
# such sessions wastes Haiku calls and sometimes hurts ROUGE-L vs naive window.
# The regime gate makes CAMS universally safe: when dependencies aren't present,
# PRESERVE is the correct decision anyway.
# ---------------------------------------------------------------------------
_REGIME_MIN_TURNS        = 4     # warmup turns before regime detection activates
_REGIME_J_VARIANCE_FLOOR = 0.05  # min variance to signal mixed-confidence session

# ---------------------------------------------------------------------------
# Post-compression degradation detection — explicit, tunable constants.
# These detect whether a compression was harmful AFTER the fact (shadow check).
# ---------------------------------------------------------------------------
_DEGRADATION_CORRECTION_FLOOR = 0.50   # correction factor below this → spike
_DEGRADATION_J_CLIFF_PREV     = 0.70   # prev J must have been HIGH
_DEGRADATION_J_CLIFF_CURR     = 0.40   # current J must drop below this
_DEGRADATION_J_DELTA          = 0.35   # any sudden swing this large is suspicious

# ---------------------------------------------------------------------------
# Compression shadow — keep pre-compression state for N turns after compression.
# If post-compression turns show degradation signals, restore from shadow.
# ---------------------------------------------------------------------------
_SHADOW_TTL = 3  # turns to monitor before committing compression

# ---------------------------------------------------------------------------
# Session persistence versioning — increment when state schema changes.
# Migration functions keyed by old version handle upgrades transparently.
# ---------------------------------------------------------------------------
_SESSION_VERSION = "1.1"

# ---------------------------------------------------------------------------
# Adaptive threshold configuration
#
# Fixed global thresholds (theta_high=0.65) misfire because the J-distribution
# shifts by session type: a Q&A session clusters [0.60-0.91]; a coding session
# clusters tighter around [0.62-0.72]. A fixed cutoff that works for one
# mis-classifies the other.
#
# Solution: track a rolling buffer of J-scores and set thresholds as percentiles
# of what this specific session is producing — "compress the top 25%, preserve
# the bottom 25%, TRIM the middle 50%".
#
# Safety floors prevent over-compression during warmup or in pathological sessions:
#   _ADAPTIVE_THETA_HIGH_FLOOR = 0.65  — never compress below this (global default)
#   _ADAPTIVE_THETA_LOW_CEIL   = 0.55  — never let LOW zone creep above this
#
# Empirical calibration from 30-question benchmark (April 2026):
#   Full J range: [0.594, 0.907], mean=0.691
#   At theta_high=0.70: factual=8/10 HIGH, uncertain=0/10 HIGH → clean separation
#   At theta_high=0.65: uncertain=6/10 HIGH → wrong, compresses uncertain responses
#   → Floor set to 0.65; adaptive P75 raises it further when session warrants.
# ---------------------------------------------------------------------------
_ADAPTIVE_MIN_SAMPLES      = 5   # turns before adaptive mode activates (warmup)
_ADAPTIVE_BUFFER_SIZE      = 20  # rolling window (last N turns)
_ADAPTIVE_THETA_HIGH_FLOOR = 0.65  # global safety floor — never compress below J=0.65
_ADAPTIVE_THETA_LOW_CEIL   = 0.55  # cap — LOW zone must stay below this

# Faithfulness guard — markers that signal user-flagged uncertainty.
# If the compressible old segment contains these, Haiku may strip them,
# turning tentative facts into apparent certainties. We refuse to compress
# (return 0 → caller falls through to PRESERVE) rather than risk hallucination.
_UNCERTAINTY_MARKERS = frozenset({
    "not certain", "not sure", "uncertain", "tentative", "unverified",
    "approximately", "roughly", "i think", "i believe", "i'm not",
    "might be", "might not", "may be", "possibly", "perhaps",
    "i'd verify", "need to check", "should verify", "to verify",
    "approx", "tbd",
})

# Semantic entropy proxy — markers that signal multiple valid answers.
# Detected in MEDIUM-J responses → zone downgraded to LOW → PRESERVE.
# Zero-cost single-pass approximation of sampling-based semantic entropy
# (Kuhn et al. ICLR 2023): when the model signals competing answers itself,
# a re-sample would likely diverge → high semantic uncertainty → preserve.
_MULTI_ANSWER_MARKERS = frozenset({
    "it depends on", "depends on the", "depends on what",
    "there are several", "there are two", "there are multiple",
    "varies by", "varies depending", "context-dependent",
    "some would argue", "others might say", "one approach",
    "on one hand", "on the other hand", "two main", "multiple approaches",
    "no single answer", "no one-size", "case by case",
    "different perspectives", "highly contextual",
    "not entirely clear", "no clear consensus", "no universal",
})

# Novelty detection — kept at 0.75 but now applied to content words, not entities.
# Content words fix two entity-counting failures:
#   1. Topic shifts without proper nouns ("optimistic locking" → "database sharding")
#   2. Sentence-start capitalization false positives (words like "The", "This")
# Three-gate remains: vocab ≥10, ≥5 new content words, >75% of response words new.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Single turn output returned to the caller."""
    turn_idx:         int
    response:         str
    j_score:          float
    zone:             str
    decision:         str          # COMPRESS | TRIM | PRESERVE
    tokens_in:        int
    tokens_out:       int
    tokens_saved:     int
    cost_usd:         float
    savings_usd:      float
    reasoning:        str
    session_tokens_used:   int
    session_tokens_saved:  int
    session_cost_usd:      float
    session_savings_usd:   float
    compression_ratio:     float   # saved / (used + saved)
    thinking_tokens:       int   = 0    # 0 when use_thinking=False
    thinking_utilization:  float = 0.0  # thinking_tokens / budget; 0 when not used
    thinking_budget_used:  int   = 0    # budget allocated for this turn (continuous J-governor)
    drift_state:           bool  = False  # True when 3+ consecutive LOW-zone turns detected
    adaptive_theta_high:   float = 0.65  # effective HIGH threshold used this turn
    adaptive_theta_low:    float = 0.35  # effective LOW threshold used this turn

    @property
    def envelope(self) -> dict:
        """
        Return a CAMSEnvelope dict for this turn — MCP-serializable epistemic
        provenance record. Downstream agents check envelope['should_verify']
        and envelope['safe_to_compress'] before acting on this information.
        """
        from .envelope import CAMSEnvelope
        return CAMSEnvelope.from_turn(
            response     = self.response,
            j_score      = self.j_score,
            zone         = self.zone,
            decision     = self.decision,
            content_type = "text",
            source       = "cams",
        ).to_dict()


@dataclass
class SessionStats:
    total_tokens_in:   int   = 0
    total_tokens_out:  int   = 0
    total_tokens_saved: int  = 0
    total_cost_usd:    float = 0.0
    total_savings_usd: float = 0.0
    turns_compressed:  int   = 0
    turns_trimmed:     int   = 0
    turns_preserved:   int   = 0
    decision_log:      list  = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        total = self.total_tokens_in + self.total_tokens_saved
        if total == 0:
            return 0.0
        return self.total_tokens_saved / total


# ---------------------------------------------------------------------------
# CAMSContextManager
# ---------------------------------------------------------------------------

class CAMSContextManager:
    """
    Drop-in wrapper around the Claude API with adaptive context management.

    Usage:
        mgr = CAMSContextManager()
        result = mgr.chat("What is the capital of France?")
        print(result.response)
        print(f"J={result.j_score:.2f}  saved={result.tokens_saved} tokens")
    """

    COMPRESS_AFTER     = 3    # turns: compress history older than this on HIGH
    TRIM_WINDOW        = 10   # turns: keep last N turns on MEDIUM
    ATTENTION_SINK     = 2    # turns: never compress first N turns (attention sinks)
    MAX_COMPRESSIONS   = 3    # stop compressing after this many (quality guard)

    def __init__(
        self,
        api_key:        Optional[str] = None,
        theta_high:     float = 0.70,
        theta_low:      float = 0.45,
        system_prompt:  Optional[str] = None,
        max_tokens:     int = 1024,
        use_thinking:   bool = False,   # enable adaptive thinking budget
        use_agreement:  bool = False,   # enable agreement-based second signal for MEDIUM zone
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("pip install anthropic")

        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.proxy         = ConfidenceProxy(theta_high, theta_low)
        self.system_prompt = system_prompt or (
            "You are a helpful, precise assistant. "
            "Give concise answers when the answer is clear; "
            "express genuine uncertainty when it exists."
        )
        self.max_tokens    = max_tokens
        self.use_thinking  = use_thinking
        self.use_agreement = use_agreement
        self.stats         = SessionStats()

        # Live conversation state
        self._history:           list[dict] = []
        self._summary:           Optional[str] = None
        self._turn_idx:          int = 0
        self._compression_count: int = 0
        self._prev_zone:         str = "MEDIUM"   # seed: neutral until first turn
        self._prev_j:            float = 0.50     # prev-turn J-score; seeds thinking budget
        self._turn1_goal:        str = ""         # user's first message — anchors compression

        # Drift detection: rolling J history; drift when 3 consecutive LOW turns
        self._j_history:  list[float] = []
        self._drift_state: bool = False

        # Novelty guard: vocabulary of named entities seen so far
        self._content_vocab: set[str] = set()
        # Recent-window vocabulary: last 3 turns' content words (sliding queue).
        # Novelty is measured against THIS, not global vocab, so same-domain
        # technical terms don't falsely trigger the pivot guard every turn.
        self._recent_vocab_window: list[set[str]] = []

        # Adaptive threshold tracking: rolling J-buffer for percentile-based thresholds.
        # After _ADAPTIVE_MIN_SAMPLES turns, theta_high = P75, theta_low = P25 of buffer.
        self._j_buffer:         list[float]          = []

        # Parallel J-score list for history messages (same length as _history).
        # User messages store None; assistant messages store the turn's J-score.
        # Used by selective compression to identify HIGH-J turns for compression
        # while keeping LOW/MEDIUM-J turns verbatim.
        self._history_j_scores: list[Optional[float]] = []

        # Compression shadow — pre-compression state kept for _SHADOW_TTL turns.
        # If post-compression turns show degradation, _restore_from_shadow() reverts.
        self._compression_shadow:         Optional[list[dict]]          = None
        self._compression_shadow_j:       Optional[list[Optional[float]]] = None
        self._compression_shadow_summary: Optional[str]                  = None
        self._shadow_turns_remaining:     int                            = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> TurnResult:
        """Send a message, receive a CAMS-managed response."""
        self._turn_idx += 1
        if self._turn_idx == 1:
            self._turn1_goal = user_message   # anchor for task-aware compression
        self._history.append({"role": "user", "content": user_message})
        self._history_j_scores.append(None)   # user message — no J-score yet

        # Build messages and call Opus for the main answer
        messages = self._build_messages()
        call_kwargs = dict(
            model=_MODEL_OPUS,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        # Continuous J-governed thinking budget.
        # Opus 4.7 uses adaptive thinking (effort level) rather than a token budget.
        # effort="high" when prev_j is low (uncertain); "medium" when mid-range.
        # Note: Opus 4.7 does not expose thinking blocks, so thinking_tokens
        # and thinking_utilization will always be 0 — dual-signal fusion is a no-op
        # on this model. The feature is preserved for forward-compatibility.
        thinking_budget = 0
        if self.use_thinking and self._prev_j < self.proxy.theta_high:
            effort = "high" if self._prev_j < self.proxy.theta_low else "medium"
            call_kwargs["thinking"] = {"type": "adaptive"}
            call_kwargs["output_config"] = {"effort": effort}
            thinking_budget = _THINKING_MAX if effort == "high" else _THINKING_MIN

        resp = self.client.messages.create(**call_kwargs)

        # When thinking is enabled, content blocks are [thinking, text].
        # next() default is evaluated eagerly, so we cannot use content[0].text
        # as fallback when thinking blocks are present (they have no .text attr).
        text_block = next((b for b in resp.content if b.type == "text"), None)
        text = text_block.text if text_block else ""
        tokens_in  = resp.usage.input_tokens
        tokens_out = resp.usage.output_tokens

        # Thinking token utilization — dual signal alongside J-proxy.
        # Thinking chars ÷ 4 approximates tokens (Anthropic ~4 chars/token).
        thinking_tokens = sum(
            len(b.thinking) // 4
            for b in resp.content
            if b.type == "thinking" and hasattr(b, "thinking")
        )
        thinking_utilization = (
            round(thinking_tokens / thinking_budget, 3)
            if thinking_budget > 0
            else 0.0
        )

        # Compute J-proxy
        cr: ConfidenceResult = self.proxy.compute(text)

        # Dual-signal zone adjustment: if thinking utilization is high (model
        # worked hard) but J-proxy says HIGH (text looks confident), downgrade
        # to MEDIUM. Confident-sounding text after heavy deliberation signals
        # latent difficulty — compressing away the history would be premature.
        # Threshold: >50% of budget consumed despite HIGH text signal.
        if thinking_utilization > 0.50 and cr.zone == "HIGH":
            cr = ConfidenceResult(
                j_score      = cr.j_score,
                zone         = "MEDIUM",
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; thinking override ({thinking_utilization:.0%} utilization)",
                content_type = cr.content_type,
            )

        # Semantic entropy proxy: MEDIUM-J responses containing multi-answer
        # markers (e.g. "it depends on", "case by case") signal that a re-sample
        # would likely yield a different answer — high semantic uncertainty.
        # Downgrade zone to LOW → PRESERVE. Zero-cost Kuhn 2023 approximation.
        if cr.zone == "MEDIUM" and self._has_multi_answer(text):
            cr = ConfidenceResult(
                j_score      = cr.j_score,
                zone         = "LOW",
                factors      = cr.factors,
                reasoning    = cr.reasoning + "; semantic entropy proxy (context-dependent answer detected)",
                content_type = cr.content_type,
            )

        # Agreement-based second signal (opt-in, use_agreement=True).
        # On MEDIUM-J turns — where J-proxy is least reliable — ask Haiku to
        # rate its own confidence. Fuse: J_final = 0.7*J + 0.3*agreement.
        # Uses Haiku to keep cost negligible (~$0.0002/call).
        # Only fires on MEDIUM because HIGH and LOW are already decided;
        # MEDIUM is the boundary zone where a second reading matters most.
        if self.use_agreement and cr.zone == "MEDIUM":
            agreement = self._agreement_score(text)
            fused_j   = round(0.7 * cr.j_score + 0.3 * agreement, 4)
            fused_zone = (
                "HIGH"   if fused_j >= self._effective_theta_high else
                "MEDIUM" if fused_j >= self._effective_theta_low  else
                "LOW"
            )
            cr = ConfidenceResult(
                j_score      = fused_j,
                zone         = fused_zone,
                factors      = cr.factors,
                reasoning    = cr.reasoning + f"; agreement={agreement:.2f} fused_j={fused_j:.3f}",
                content_type = cr.content_type,
            )

        # Novelty guard: check if this turn signals a domain pivot
        novelty_override = self._check_novelty(text)

        # Drift detection: 3 consecutive LOW-zone turns → proactive PRESERVE lock
        self._j_history.append(cr.j_score)
        if len(self._j_history) > 5:
            self._j_history.pop(0)
        self._drift_state = (
            len(self._j_history) >= 3
            and all(j < self.proxy.theta_low for j in self._j_history[-3:])
        )

        # Append assistant turn BEFORE compression decision
        self._history.append({"role": "assistant", "content": text})
        self._history_j_scores.append(cr.j_score)   # assistant message — record J

        # Apply CAMS memory decision
        decision, tokens_saved = self._apply_cams(cr, novelty_override)

        # Capture effective thresholds AFTER apply_cams so buffer is still pre-update
        eff_high = self._effective_theta_high
        eff_low  = self._effective_theta_low

        # Shadow monitoring: check post-compression turns for degradation signals.
        # If degradation detected within _SHADOW_TTL turns, restore pre-compression state.
        if self._shadow_turns_remaining > 0:
            if self._is_post_compression_degraded(cr):
                self._restore_from_shadow()
                decision      = "PRESERVE"
                tokens_saved  = 0
            else:
                self._shadow_turns_remaining -= 1
                if self._shadow_turns_remaining == 0:
                    self._clear_shadow()

        # Update entity vocabulary and zone memory for next-turn thinking budget
        self._update_content_vocab(text)
        self._prev_zone = cr.zone
        self._prev_j    = cr.j_score

        # Update adaptive threshold buffer (after decision — causal, not lookahead)
        self._j_buffer.append(cr.j_score)
        if len(self._j_buffer) > _ADAPTIVE_BUFFER_SIZE:
            self._j_buffer.pop(0)

        # Costs
        cost_usd    = _cost(tokens_in, tokens_out)
        savings_usd = _cost(tokens_saved, 0)

        # Update session stats
        self.stats.total_tokens_in    += tokens_in
        self.stats.total_tokens_out   += tokens_out
        self.stats.total_tokens_saved += tokens_saved
        self.stats.total_cost_usd     += cost_usd
        self.stats.total_savings_usd  += savings_usd
        if decision == "COMPRESS":
            self.stats.turns_compressed += 1
        elif decision == "TRIM":
            self.stats.turns_trimmed += 1
        else:
            self.stats.turns_preserved += 1

        self.stats.decision_log.append({
            "turn":                 self._turn_idx,
            "j_score":              cr.j_score,
            "zone":                 cr.zone,
            "decision":             decision,
            "tokens_saved":         tokens_saved,
            "reasoning":            cr.reasoning,
            "novelty_override":     novelty_override,
            "semantic_entropy_override": "semantic entropy proxy" in cr.reasoning,
            "content_type":         cr.content_type,
            "thinking_tokens":      thinking_tokens,
            "thinking_utilization": thinking_utilization,
            "thinking_budget_used": thinking_budget,
            "drift_state":          self._drift_state,
            "adaptive_theta_high":  eff_high,
            "adaptive_theta_low":   eff_low,
        })

        return TurnResult(
            turn_idx        = self._turn_idx,
            response        = text,
            j_score         = cr.j_score,
            zone            = cr.zone,
            decision        = decision,
            tokens_in       = tokens_in,
            tokens_out      = tokens_out,
            tokens_saved    = tokens_saved,
            cost_usd        = round(cost_usd, 6),
            savings_usd     = round(savings_usd, 6),
            reasoning       = cr.reasoning,
            session_tokens_used   = self.stats.total_tokens_in + self.stats.total_tokens_out,
            session_tokens_saved  = self.stats.total_tokens_saved,
            session_cost_usd      = round(self.stats.total_cost_usd, 4),
            session_savings_usd   = round(self.stats.total_savings_usd, 4),
            compression_ratio     = round(self.stats.compression_ratio, 3),
            thinking_tokens       = thinking_tokens,
            thinking_utilization  = thinking_utilization,
            thinking_budget_used  = thinking_budget,
            drift_state           = self._drift_state,
            adaptive_theta_high   = eff_high,
            adaptive_theta_low    = eff_low,
        )

    def reset(self):
        self._history            = []
        self._summary            = None
        self._turn_idx           = 0
        self._compression_count  = 0
        self._prev_zone          = "MEDIUM"
        self._prev_j             = 0.50
        self._turn1_goal         = ""
        self._j_history          = []
        self._drift_state        = False
        self._content_vocab          = set()
        self._recent_vocab_window    = []
        self._j_buffer               = []
        self._history_j_scores           = []
        self._compression_shadow         = None
        self._compression_shadow_j       = None
        self._compression_shadow_summary = None
        self._shadow_turns_remaining     = 0
        self.stats                       = SessionStats()

    @property
    def decision_log(self) -> list[dict]:
        return self.stats.decision_log

    @property
    def _effective_theta_high(self) -> float:
        """P75 of recent J-scores, floored at _ADAPTIVE_THETA_HIGH_FLOOR.
        Compresses only the most confident quarter of this session's responses."""
        if len(self._j_buffer) < _ADAPTIVE_MIN_SAMPLES:
            return self.proxy.theta_high
        buf = sorted(self._j_buffer)
        p75 = buf[int(0.75 * len(buf))]
        return max(_ADAPTIVE_THETA_HIGH_FLOOR, p75)

    @property
    def _effective_theta_low(self) -> float:
        """P25 of recent J-scores, capped at _ADAPTIVE_THETA_LOW_CEIL.
        Preserves only the least confident quarter of this session's responses."""
        if len(self._j_buffer) < _ADAPTIVE_MIN_SAMPLES:
            return self.proxy.theta_low
        buf = sorted(self._j_buffer)
        p25 = buf[int(0.25 * len(buf))]
        eff_high = self._effective_theta_high
        return min(p25, eff_high - 0.10, _ADAPTIVE_THETA_LOW_CEIL)

    # ------------------------------------------------------------------
    # CAMS memory decisions
    # ------------------------------------------------------------------

    def _apply_cams(self, cr: ConfidenceResult, novelty_override: bool) -> tuple[str, int]:
        """Returns (decision, tokens_saved)."""
        n_turns = len(self._history)

        # Drift state: sustained uncertainty (3+ consecutive LOW turns) → lock PRESERVE
        if self._drift_state:
            return "PRESERVE", 0

        # Novelty guard: new topic detected → preserve regardless of J
        if novelty_override:
            return "PRESERVE", 0

        # Regime detection: only activate compression/trim when session shows
        # evidence of cross-turn dependencies or mixed J-variance.
        # Independent QA sessions (all HIGH J, no cross-references) should not
        # incur Haiku costs — PRESERVE is correct there anyway.
        if not self._should_enable_cams():
            return "PRESERVE", 0

        # Compute effective zone using adaptive thresholds.
        # Guard-rail overrides (semantic entropy proxy, dual-signal thinking) are
        # detected by comparing cr.zone against what static thresholds would give.
        # When a guard rail has already downgraded the zone, respect it.
        static_zone = (
            "HIGH"   if cr.j_score >= self.proxy.theta_high else
            "MEDIUM" if cr.j_score >= self.proxy.theta_low  else
            "LOW"
        )
        if cr.zone != static_zone:
            eff_zone = cr.zone          # guard-rail override — keep as-is
        else:
            eff_zone = (
                "HIGH"   if cr.j_score >= self._effective_theta_high else
                "MEDIUM" if cr.j_score >= self._effective_theta_low  else
                "LOW"
            )

        if eff_zone == "HIGH" and n_turns > self.COMPRESS_AFTER * 2:
            if self._compression_count < self.MAX_COMPRESSIONS:
                saved = self._compress()
                if saved > 0:
                    self._compression_count += 1
                    return "COMPRESS", saved

        if eff_zone == "MEDIUM" and n_turns > self.TRIM_WINDOW * 2:
            saved = self._trim()
            return "TRIM", saved

        return "PRESERVE", 0

    def _compress(self) -> int:
        """
        Selectively compress the old segment using per-turn J-scores.

        HIGH-J turns are resolved context — safe to summarize via Haiku.
        LOW/MEDIUM-J turns contain uncertainty or important context — kept verbatim.

        History after compression:
            sink (attention sinks) + verbatim LOW/MEDIUM turns + recent

        The Haiku summary of HIGH-J turns is injected via _build_messages as
        <context_summary> XML prefix, not as a fake turn in _history.

        Shadow: pre-compression state is saved for _SHADOW_TTL turns.
        If post-compression degradation detected, _restore_from_shadow() reverts.
        """
        sink_msgs = self.ATTENTION_SINK * 2
        keep_n    = self.COMPRESS_AFTER * 2

        if len(self._history) <= sink_msgs + keep_n:
            return 0

        sink   = self._history[:sink_msgs]
        old    = self._history[sink_msgs:-keep_n]
        recent = self._history[-keep_n:]
        old_j  = self._history_j_scores[sink_msgs:-keep_n]

        if not old:
            return 0

        # Split old segment into HIGH-J turns (compress) and LOW/MEDIUM-J (keep verbatim).
        # Process in turn-pairs (user message + assistant message).
        high_j_msgs:      list[dict] = []
        preserved_msgs:   list[dict] = []
        preserved_j:      list[Optional[float]] = []

        for i in range(0, len(old) - 1, 2):
            user_msg = old[i]
            asst_msg = old[i + 1]
            j = old_j[i + 1]   # J-score lives on the assistant message
            if j is not None and j >= self._effective_theta_high:
                high_j_msgs.extend([user_msg, asst_msg])
            else:
                preserved_msgs.extend([user_msg, asst_msg])
                preserved_j.extend([old_j[i], old_j[i + 1]])

        # Nothing to compress — all old turns are LOW/MEDIUM-J
        if not high_j_msgs:
            return 0

        tokens_before = sum(len(m["content"]) // 4 for m in high_j_msgs)
        if tokens_before < _MIN_COMPRESS_TOKENS:
            return 0

        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in high_j_msgs
        )

        # Faithfulness guard: refuse to compress if high-J segment contains uncertainty.
        # (Should be rare since these scored HIGH-J, but a guard rail is cheap.)
        if self._has_uncertainty(conv_text):
            return 0

        # Save compression shadow BEFORE making any changes.
        # If Haiku produces a low-quality summary, or post-compression turns
        # show degradation, _restore_from_shadow() will revert all state changes.
        self._compression_shadow         = self._history[:]
        self._compression_shadow_j       = self._history_j_scores[:]
        self._compression_shadow_summary = self._summary
        self._shadow_turns_remaining     = _SHADOW_TTL

        goal_line = (
            f"The user's original goal: {self._turn1_goal}\n\n"
            if self._turn1_goal else ""
        )
        summary_resp = self.client.messages.create(
            model=_MODEL_HAIKU,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation segment in 2-3 concise sentences. "
                    "Preserve all key facts, decisions, and context needed "
                    "to continue the conversation toward the user's goal. "
                    "IMPORTANT: if anything was stated as uncertain, approximate, "
                    "or needing verification, explicitly preserve that uncertainty "
                    "flag in your summary (e.g. 'tentative', 'unverified', 'approx.').\n\n"
                    + goal_line
                    + "Conversation segment to summarize:\n\n"
                    + conv_text
                ),
            }],
            max_tokens=200,
        )
        summary = summary_resp.content[0].text.strip()

        comp_in  = summary_resp.usage.input_tokens
        comp_out = summary_resp.usage.output_tokens
        self.stats.total_tokens_in  += comp_in
        self.stats.total_tokens_out += comp_out
        self.stats.total_cost_usd   += _cost_haiku(comp_in, comp_out)

        # Rebuild history: sink + verbatim preserved turns + recent
        # HIGH-J turns are removed; their summary is stored in self._summary
        # and injected as <context_summary> prefix by _build_messages.
        self._summary = summary
        self._history = sink + preserved_msgs + recent
        self._history_j_scores = (
            self._history_j_scores[:sink_msgs]
            + preserved_j
            + self._history_j_scores[-keep_n:]
        )

        tokens_after = len(summary) // 4
        gross_saved  = max(0, tokens_before - tokens_after)
        net_saved    = max(0, gross_saved - comp_out)

        # ROI gate: if net savings don't justify Haiku call overhead, revert.
        if net_saved < _MIN_COMPRESS_ROI:
            self._restore_from_shadow()
            return 0

        # Summary quality gate: if Haiku produced a hedged/uncertain summary,
        # it likely lost information — restore and refuse compression.
        summary_cr = self.proxy.compute(summary)
        if summary_cr.j_score < 0.40:
            self._restore_from_shadow()
            return 0

        return net_saved

    def _trim(self) -> int:
        """
        J-selective trim: keep attention sink + LOW/MEDIUM-J turns + recent window.

        Naive trim (slice to last N messages) silently drops LOW-J uncertain constraints
        that happen to be older than the window. This defeats the whole point of CAMS —
        the model forgets the uncertain fact that was explicitly flagged for preservation.

        This version mirrors selective compression: only HIGH-J turns (already resolved,
        safe to lose) are dropped. LOW/MEDIUM-J turns are always kept verbatim regardless
        of age. This makes TRIM safe to apply even when the old segment contains uncertain
        context.
        """
        keep_n    = self.TRIM_WINDOW * 2
        sink_msgs = self.ATTENTION_SINK * 2

        if len(self._history) <= keep_n:
            return 0

        sink      = self._history[:sink_msgs]
        sink_j    = self._history_j_scores[:sink_msgs]
        old       = self._history[sink_msgs:-keep_n]
        old_j     = self._history_j_scores[sink_msgs:-keep_n]
        recent    = self._history[-keep_n:]
        recent_j  = self._history_j_scores[-keep_n:]

        if not old:
            return 0

        preserved_msgs: list[dict]             = []
        preserved_j:    list[Optional[float]]  = []
        dropped_tokens  = 0

        for i in range(0, len(old) - 1, 2):
            user_msg = old[i]
            asst_msg = old[i + 1]
            j = old_j[i + 1]
            if j is not None and j >= self._effective_theta_high:
                dropped_tokens += (len(user_msg["content"]) + len(asst_msg["content"])) // 4
            else:
                preserved_msgs.extend([user_msg, asst_msg])
                preserved_j.extend([old_j[i], old_j[i + 1]])

        if dropped_tokens == 0:
            return 0

        self._history        = sink + preserved_msgs + recent
        self._history_j_scores = sink_j + preserved_j + recent_j
        return dropped_tokens

    def _build_messages(self) -> list[dict]:
        """
        Inject compression summary as an XML-tagged prefix on the first message.
        A fake user/assistant exchange would pollute role alternation and confuse
        the model about who said what. A single tagged prefix is the correct pattern
        per Anthropic's context injection guidance.
        """
        if self._summary and self._history:
            msgs = list(self._history)
            first = msgs[0]
            msgs[0] = {
                "role":    first["role"],
                "content": (
                    f"<context_summary>\n{self._summary}\n</context_summary>\n\n"
                    + first["content"]
                ),
            }
            return msgs
        return list(self._history)

    # ------------------------------------------------------------------
    # Novelty guard  (content-word Jaccard distance, replaces entity counting)
    # ------------------------------------------------------------------

    # Common English function words — filtered out before Jaccard computation.
    _CONTENT_STOPWORDS = frozenset({
        "the", "and", "for", "that", "this", "with", "have", "from", "they",
        "will", "your", "what", "when", "how", "why", "where", "which", "who",
        "are", "was", "were", "has", "had", "been", "being", "can", "could",
        "should", "would", "may", "might", "must", "does", "did", "not", "more",
        "also", "such", "than", "only", "about", "into", "over", "some", "these",
        "those", "there", "here", "then", "but", "all", "any", "each", "both",
        "just", "even", "often", "well", "back", "still", "like", "used", "need",
        "make", "give", "take", "come", "know", "think", "look", "want", "use",
        "see", "based", "using", "since", "after", "before", "while", "through",
        "their", "them", "very", "most", "same", "next", "last", "good", "new",
    })

    def _extract_content_words(self, text: str) -> set[str]:
        """Extract lowercase content words (≥4 chars, non-stop, non-digit)."""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        return {
            w for w in words
            if len(w) >= 4
            and not w.isdigit()
            and w not in self._CONTENT_STOPWORDS
        }

    def _check_novelty(self, text: str) -> bool:
        """
        Novelty guard — DISABLED after empirical measurement showed 79-87% FP rate.

        Technical writing introduces new vocabulary every sentence within the same domain
        ("vacuum", "partitioning", "B-tree" are all "new" in a PostgreSQL session).
        A vocabulary-distance signal cannot distinguish same-domain progression from
        a real domain pivot without semantic embeddings — which are not available here.

        The cases this guard was intended to protect are already covered by:
          - Faithfulness probe: detects uncertainty in compressible segments → PRESERVE
          - Selective J-compression: LOW/MEDIUM-J turns always kept verbatim
          - Regime detection: low J-variance → PRESERVE mode (no compression)

        Kept as a stub so the call site and decision log field remain intact.
        Returns False always. Re-enable with a proper embedding-based implementation.
        """
        return False

    def _update_content_vocab(self, text: str):
        words = self._extract_content_words(text)
        self._content_vocab.update(words)
        # Maintain sliding window of last 3 turns
        self._recent_vocab_window.append(words)
        if len(self._recent_vocab_window) > 3:
            self._recent_vocab_window.pop(0)

    def _has_multi_answer(self, text: str) -> bool:
        """Returns True if response signals multiple valid answers (semantic entropy proxy)."""
        lower = text.lower()
        return any(m in lower for m in _MULTI_ANSWER_MARKERS)

    def _has_uncertainty(self, text: str) -> bool:
        """Returns True if text contains markers that signal user-flagged uncertainty."""
        import re as _re
        lower = text.lower()
        # Standard uncertainty markers
        if any(m in lower for m in _UNCERTAINTY_MARKERS):
            return True
        # Expanded: code comment uncertainty (# todo, # verify, # untested, etc.)
        if _re.search(r'#\s*(todo|fixme|hack|verify|check|untested|approximate|not sure|might)',
                      lower):
            return True
        # Expanded: numerical hedging — qualifier word followed by a number
        if _re.search(r'\b(around|roughly|approximately|about|~)\s+\d', lower):
            return True
        # Expanded: conditional uncertainty — premise implies uncertain conclusion
        if any(m in lower for m in (
            "if this is correct", "assuming this is right", "if i'm reading",
            "if that's the case", "assuming that's accurate", "provided that's true",
        )):
            return True
        # Expanded: domain hedging — informal "worth checking" patterns
        if any(m in lower for m in (
            "worth checking", "worth verifying", "double-check", "double check",
            "you might want to confirm", "lgtm but", "seems right but",
            "this should work but", "i'd recommend verifying",
        )):
            return True
        return False

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _agreement_score(self, response_text: str) -> float:
        """
        Ask Haiku to rate the confidence of a response on [0, 1].
        Used as a second signal on MEDIUM-J turns where J-proxy is least reliable.
        Haiku keeps cost to ~$0.0002/call — negligible for the accuracy gain.
        """
        try:
            resp = self.client.messages.create(
                model=_MODEL_HAIKU,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate the epistemic confidence of the following response "
                        "on a scale of 0.0 (very uncertain) to 1.0 (completely certain). "
                        "Consider hedging language, specificity, and whether it contains "
                        "qualifiers like 'I think', 'probably', 'might'. "
                        "Reply with ONLY a decimal number between 0.0 and 1.0.\n\n"
                        f"Response:\n{response_text[:600]}"
                    ),
                }],
                max_tokens=10,
            )
            raw = resp.content[0].text.strip()
            match = re.search(r'\d+\.?\d*', raw)
            score = float(match.group()) if match else 0.5
            return round(min(1.0, max(0.0, score)), 3)
        except Exception:
            return 0.5   # safe default — no change to fusion

    def _j_variance(self) -> float:
        """Variance of J-scores in the current rolling buffer."""
        if len(self._j_buffer) < 3:
            return 0.0
        mean = sum(self._j_buffer) / len(self._j_buffer)
        return sum((j - mean) ** 2 for j in self._j_buffer) / len(self._j_buffer)

    def _has_dependency(self) -> bool:
        """True if recent assistant turns reference earlier turns explicitly."""
        if len(self._history) < 4:
            return False
        _dep_markers = {
            "as we discussed", "as mentioned", "as i said", "earlier",
            "you mentioned", "we established", "going back to",
            "as noted", "referring back", "from before",
            "the constraint", "the requirement", "the error", "the issue",
            "that value", "that limit", "those constraints", "the budget",
        }
        recent_assistant = [
            m["content"].lower()
            for m in self._history[-6:]
            if m["role"] == "assistant"
        ]
        return any(
            marker in text
            for text in recent_assistant
            for marker in _dep_markers
        )

    def _should_enable_cams(self) -> bool:
        """
        Regime detection gate: only allow COMPRESS/TRIM when the session shows
        evidence that context management matters.

        Returns False (PRESERVE) when:
        - Still in warmup (< _REGIME_MIN_TURNS)
        - Session J-variance is low AND no cross-turn references detected
          (uniform-confidence session — independent QA pattern)

        Returns True when:
        - J-variance > _REGIME_J_VARIANCE_FLOOR (mixed-confidence session), OR
        - Cross-turn dependency markers detected in recent turns
        """
        if self._turn_idx <= _REGIME_MIN_TURNS:
            return False
        return self._j_variance() > _REGIME_J_VARIANCE_FLOOR or self._has_dependency()

    # ------------------------------------------------------------------
    # Post-compression degradation + shadow recovery
    # ------------------------------------------------------------------

    def _is_post_compression_degraded(self, cr: ConfidenceResult) -> bool:
        """
        Detect compression-induced degradation from post-compression behavioral signals.

        Triggers restoration from shadow if:
        1. Model correction rate spikes (model noticed missing context and corrects)
        2. J-score cliff: was HIGH before compression, now suddenly LOW
        3. Sudden large J swing in either direction (instability signal)
        """
        correction = cr.factors.get("correction", 1.0)
        correction_spike = correction < _DEGRADATION_CORRECTION_FLOOR
        j_cliff = (
            self._prev_j > _DEGRADATION_J_CLIFF_PREV
            and cr.j_score < _DEGRADATION_J_CLIFF_CURR
        )
        j_swing = abs(cr.j_score - self._prev_j) > _DEGRADATION_J_DELTA
        return correction_spike or j_cliff or j_swing

    def _restore_from_shadow(self) -> None:
        """Revert to pre-compression state. Called when degradation detected."""
        if self._compression_shadow is not None:
            self._history          = self._compression_shadow
            self._history_j_scores = self._compression_shadow_j
            self._summary          = self._compression_shadow_summary
            self._compression_count = max(0, self._compression_count - 1)
        self._clear_shadow()

    def _clear_shadow(self) -> None:
        """Commit compression — discard shadow, no anomaly detected."""
        self._compression_shadow         = None
        self._compression_shadow_j       = None
        self._compression_shadow_summary = None
        self._shadow_turns_remaining     = 0

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist full epistemic state to JSON. Includes version for migration."""
        import json
        state = {
            "version":           _SESSION_VERSION,
            "history":           self._history,
            "history_j_scores":  self._history_j_scores,
            "summary":           self._summary,
            "j_buffer":          self._j_buffer,
            "content_vocab":     list(self._content_vocab),
            "turn_idx":          self._turn_idx,
            "compression_count": self._compression_count,
            "prev_zone":         self._prev_zone,
            "prev_j":            self._prev_j,
            "drift_state":       self._drift_state,
            "j_history":         self._j_history,
            "turn1_goal":        self._turn1_goal,
            "stats": {
                "total_tokens_in":    self.stats.total_tokens_in,
                "total_tokens_out":   self.stats.total_tokens_out,
                "total_tokens_saved": self.stats.total_tokens_saved,
                "total_cost_usd":     self.stats.total_cost_usd,
                "turns_compressed":   self.stats.turns_compressed,
                "turns_trimmed":      self.stats.turns_trimmed,
                "turns_preserved":    self.stats.turns_preserved,
            },
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load persisted session state. Migrates older versions transparently."""
        import json
        with open(path) as f:
            state = json.load(f)

        version = state.get("version", "1.0")
        if version != _SESSION_VERSION:
            state = _migrate_session(state, version)

        self._history           = state["history"]
        self._history_j_scores  = state["history_j_scores"]
        self._summary           = state.get("summary")
        self._j_buffer          = state.get("j_buffer", [])
        self._content_vocab     = set(state.get("content_vocab", []))
        self._turn_idx          = state.get("turn_idx", len(self._history) // 2)
        self._compression_count = state.get("compression_count", 0)
        self._prev_zone         = state.get("prev_zone", "MEDIUM")
        self._prev_j            = state.get("prev_j", 0.50)
        self._drift_state       = state.get("drift_state", False)
        self._j_history         = state.get("j_history", [])
        self._turn1_goal        = state.get("turn1_goal", "")

        s = state.get("stats", {})
        self.stats.total_tokens_in    = s.get("total_tokens_in", 0)
        self.stats.total_tokens_out   = s.get("total_tokens_out", 0)
        self.stats.total_tokens_saved = s.get("total_tokens_saved", 0)
        self.stats.total_cost_usd     = s.get("total_cost_usd", 0.0)
        self.stats.turns_compressed   = s.get("turns_compressed", 0)
        self.stats.turns_trimmed      = s.get("turns_trimmed", 0)
        self.stats.turns_preserved    = s.get("turns_preserved", 0)


def _migrate_session(state: dict, from_version: str) -> dict:
    """Migrate session state from older versions to current _SESSION_VERSION."""
    if from_version == "1.0":
        # v1.0 → v1.1: added j_history, turns_compressed/trimmed/preserved in stats
        state.setdefault("j_history", [])
        stats = state.setdefault("stats", {})
        stats.setdefault("turns_compressed", 0)
        stats.setdefault("turns_trimmed", 0)
        stats.setdefault("turns_preserved", 0)
        state["version"] = "1.1"
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * _INPUT_COST_PER_M  / 1_000_000 +
        output_tokens * _OUTPUT_COST_PER_M / 1_000_000
    )


def _cost_haiku(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * _HAIKU_INPUT_PER_M  / 1_000_000 +
        output_tokens * _HAIKU_OUTPUT_PER_M / 1_000_000
    )

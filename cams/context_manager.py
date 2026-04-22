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
        api_key:       Optional[str] = None,
        theta_high:    float = 0.65,
        theta_low:     float = 0.35,
        system_prompt: Optional[str] = None,
        max_tokens:    int = 1024,
        use_thinking:  bool = False,   # enable adaptive thinking budget
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
        self.max_tokens   = max_tokens
        self.use_thinking = use_thinking
        self.stats        = SessionStats()

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

        # Adaptive threshold tracking: rolling J-buffer for percentile-based thresholds.
        # After _ADAPTIVE_MIN_SAMPLES turns, theta_high = P75, theta_low = P25 of buffer.
        self._j_buffer:         list[float]          = []

        # Parallel J-score list for history messages (same length as _history).
        # User messages store None; assistant messages store the turn's J-score.
        # Used by selective compression to identify HIGH-J turns for compression
        # while keeping LOW/MEDIUM-J turns verbatim.
        self._history_j_scores: list[Optional[float]] = []

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
        self._content_vocab      = set()
        self._j_buffer           = []
        self._history_j_scores   = []
        self.stats               = SessionStats()

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
        return max(0, gross_saved - comp_out)

    def _trim(self) -> int:
        """Keep last TRIM_WINDOW turn pairs, drop the rest."""
        keep_n = self.TRIM_WINDOW * 2
        if len(self._history) <= keep_n:
            return 0
        dropped       = self._history[:-keep_n]
        tokens_saved  = sum(len(m["content"]) // 4 for m in dropped)
        self._history        = self._history[-keep_n:]
        self._history_j_scores = self._history_j_scores[-keep_n:]
        return tokens_saved

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
        Returns True if this response signals a domain pivot.
        Uses content-word novelty ratio — same three-gate logic as before but
        applied to content words instead of named entities.

        Why content words > entity counting:
          - Works for topic shifts without proper nouns
            ("optimistic locking" → "database sharding" shares no named entities)
          - Content words (any ≥4-char non-stop word) are more reliable than
            capitalized tokens which fire on sentence-initial words

        Three-gate design (all must be true to fire):
          1. Context vocab established (≥ _NOVELTY_MIN_VOCAB words) — no early false positives
          2. ≥ _NOVELTY_MIN_ENTITIES new content words in this response — single new terms aren't pivots
          3. > _NOVELTY_THRESHOLD (0.75) of this response's content words are new — sustained
             same-topic conversation introduces new terms but stays below 75% threshold
        """
        if len(self._content_vocab) < _NOVELTY_MIN_VOCAB:
            return False
        current = self._extract_content_words(text)
        if not current:
            return False
        new_count = len(current - self._content_vocab)
        if new_count < _NOVELTY_MIN_ENTITIES:
            return False
        novelty_ratio = new_count / len(current)
        return novelty_ratio > _NOVELTY_THRESHOLD

    def _update_content_vocab(self, text: str):
        self._content_vocab.update(self._extract_content_words(text))

    def _has_multi_answer(self, text: str) -> bool:
        """Returns True if response signals multiple valid answers (semantic entropy proxy)."""
        lower = text.lower()
        return any(m in lower for m in _MULTI_ANSWER_MARKERS)

    def _has_uncertainty(self, text: str) -> bool:
        """Returns True if text contains markers that signal user-flagged uncertainty."""
        lower = text.lower()
        return any(m in lower for m in _UNCERTAINTY_MARKERS)


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

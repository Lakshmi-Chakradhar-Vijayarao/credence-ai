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
_THINKING_MIN = 500
_THINKING_MAX = 2000

# ---------------------------------------------------------------------------
# Novelty guard — entity change threshold
# If a turn introduces > this fraction of new named entities, PRESERVE.
# ---------------------------------------------------------------------------
_NOVELTY_THRESHOLD = 0.60   # >60% new entities → treat as new topic


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

    COMPRESS_AFTER     = 6    # turns: compress history older than this on HIGH
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
        self._entity_vocab: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> TurnResult:
        """Send a message, receive a CAMS-managed response."""
        self._turn_idx += 1
        if self._turn_idx == 1:
            self._turn1_goal = user_message   # anchor for task-aware compression
        self._history.append({"role": "user", "content": user_message})

        # Build messages and call Opus for the main answer
        messages = self._build_messages()
        call_kwargs = dict(
            model=_MODEL_OPUS,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        # Continuous J-governed thinking budget.
        # Budget scales inversely with prev-turn J: the less confident the last
        # answer, the more compute this turn gets.
        # J=0 → _THINKING_MAX (2000); J=theta_high → _THINKING_MIN (500); J≥theta_high → 0
        thinking_budget = 0
        if self.use_thinking and self._prev_j < self.proxy.theta_high:
            ratio = (self.proxy.theta_high - self._prev_j) / self.proxy.theta_high
            thinking_budget = int(_THINKING_MIN + ratio * (_THINKING_MAX - _THINKING_MIN))
            call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            call_kwargs["max_tokens"] = max(self.max_tokens, thinking_budget + 512)

        resp = self.client.messages.create(**call_kwargs)

        # When thinking is enabled, content blocks are [thinking, text].
        # Extract the text block regardless of position.
        text = next(
            (b.text for b in resp.content if b.type == "text"),
            resp.content[0].text,
        )
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

        # Novelty guard: check if this turn introduces many new entities
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

        # Apply CAMS memory decision
        decision, tokens_saved = self._apply_cams(cr, novelty_override)

        # Update entity vocabulary and zone memory for next-turn thinking budget
        self._update_entity_vocab(text)
        self._prev_zone = cr.zone
        self._prev_j    = cr.j_score

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
            "content_type":         cr.content_type,
            "thinking_tokens":      thinking_tokens,
            "thinking_utilization": thinking_utilization,
            "thinking_budget_used": thinking_budget,
            "drift_state":          self._drift_state,
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
        self._entity_vocab       = set()
        self.stats               = SessionStats()

    @property
    def decision_log(self) -> list[dict]:
        return self.stats.decision_log

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

        # Compression depth limit: stop after MAX_COMPRESSIONS
        if cr.zone == "HIGH" and n_turns > self.COMPRESS_AFTER * 2:
            if self._compression_count < self.MAX_COMPRESSIONS:
                saved = self._compress()
                if saved > 0:
                    self._compression_count += 1
                    return "COMPRESS", saved

        if cr.zone == "MEDIUM" and n_turns > self.TRIM_WINDOW * 2:
            saved = self._trim()
            return "TRIM", saved

        return "PRESERVE", 0

    def _compress(self) -> int:
        """
        Summarize history older than COMPRESS_AFTER turns using Haiku (cheap).
        Attention sinks (first ATTENTION_SINK turns = 2*2 messages) are never
        compressed — they anchor the conversation identity.
        Returns net tokens saved (gross minus Haiku compression overhead).
        """
        sink_msgs = self.ATTENTION_SINK * 2   # messages to always keep at front
        keep_n    = self.COMPRESS_AFTER * 2   # keep last N messages at back

        # Must have enough history to compress something between sink and recent
        if len(self._history) <= sink_msgs + keep_n:
            return 0

        sink   = self._history[:sink_msgs]    # attention sinks — never touched
        old    = self._history[sink_msgs:-keep_n]   # compressible middle
        recent = self._history[-keep_n:]      # always keep

        if not old:
            return 0

        tokens_before = sum(len(m["content"]) // 4 for m in old)

        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old
        )
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
                    f"to continue the conversation toward the user's goal.\n\n"
                    + goal_line
                    + "Conversation segment to summarize:\n\n"
                    + conv_text
                ),
            }],
            max_tokens=200,
        )
        summary = summary_resp.content[0].text.strip()

        # Count Haiku compression overhead against session totals
        comp_in  = summary_resp.usage.input_tokens
        comp_out = summary_resp.usage.output_tokens
        self.stats.total_tokens_in  += comp_in
        self.stats.total_tokens_out += comp_out
        self.stats.total_cost_usd   += _cost_haiku(comp_in, comp_out)

        # Rebuild history: sink + summary_turn + recent
        self._summary = summary
        self._history = sink + recent

        tokens_after = len(summary) // 4
        gross_saved  = max(0, tokens_before - tokens_after)
        # Haiku reads old messages once (comp_in); those same tokens would be sent
        # to Opus on every future turn without compression, so comp_in is not a net
        # cost against savings — it's offset by recurring future savings. Only the
        # summary output length (comp_out) is genuinely new context overhead.
        return max(0, gross_saved - comp_out)

    def _trim(self) -> int:
        """Keep last TRIM_WINDOW turn pairs, drop the rest."""
        keep_n = self.TRIM_WINDOW * 2
        if len(self._history) <= keep_n:
            return 0
        dropped       = self._history[:-keep_n]
        tokens_saved  = sum(len(m["content"]) // 4 for m in dropped)
        self._history = self._history[-keep_n:]
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
    # Novelty guard
    # ------------------------------------------------------------------

    # Common English words that get capitalized at sentence-start but are not named entities.
    _ENTITY_STOPWORDS = frozenset({
        "The", "This", "These", "That", "Those", "There", "Then", "Here",
        "It", "Its", "They", "Their", "We", "Our", "You", "Your",
        "How", "What", "Why", "When", "Where", "Who", "Which",
        "Can", "Could", "Should", "Would", "Will", "May", "Might", "Must",
        "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Does", "Did",
        "For", "And", "But", "Or", "So", "Also", "Note", "Use", "Used",
        "Using", "With", "From", "Into", "About", "After", "Before",
    })

    def _extract_entities(self, text: str) -> set[str]:
        """Extract named entities (capitalized tokens ≥ 3 chars), filtering stopwords."""
        # Mid-sentence capitalized words (strongest signal)
        tokens = re.findall(r'(?<=[a-z\s,;:])\s([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*)', text)
        # All capitalized words (broader catch, filtered by stopwords)
        tokens += re.findall(r'\b([A-Z][a-z]{2,})\b', text)
        return {
            t.strip() for t in tokens
            if len(t.strip()) >= 3 and t.strip() not in self._ENTITY_STOPWORDS
        }

    def _check_novelty(self, text: str) -> bool:
        """
        Returns True if this response introduces many new named entities
        relative to the current vocabulary — signals a topic shift.
        Triggers PRESERVE override to avoid compressing away new context.
        """
        if not self._entity_vocab:
            return False   # no baseline yet

        new_entities = self._extract_entities(text)
        if not new_entities:
            return False

        new_count = len(new_entities - self._entity_vocab)
        # Require minimum 3 new entities: a single new class name or term is
        # not a topic pivot. Genuine pivots introduce many new entities.
        if new_count < 3:
            return False
        novelty_ratio = new_count / len(new_entities)
        return novelty_ratio > _NOVELTY_THRESHOLD

    def _update_entity_vocab(self, text: str):
        self._entity_vocab.update(self._extract_entities(text))


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

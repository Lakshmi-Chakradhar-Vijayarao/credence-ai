"""
cams/context_manager.py
=======================
CAMSContextManager: Confidence-Adaptive Memory System for Claude API.

Every conversation turn, Claude Opus 4.7 answers the user's question.
CAMS then reads that response, computes a J-proxy confidence score, and
decides what to do with old context:

  HIGH confidence  (J ≥ 0.65) → Compress: ask Claude to summarize old
                                  turns into 2-3 sentences, discard raw turns.
  MEDIUM confidence(J ∈ [0.35, 0.65)) → Trim: keep last 10 turns, drop older.
  LOW confidence   (J < 0.35)  → Preserve: keep everything.

The compressor IS Claude Opus 4.7 — the model manages its own memory.
This is the creative use of Opus 4.7 that goes beyond a basic chatbot.

Savings are tracked against a no-CAMS baseline (full context every turn).
Real token counts come from the Anthropic usage response object.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from .confidence_proxy import ConfidenceProxy, ConfidenceResult

# ---------------------------------------------------------------------------
# Claude Opus 4.7 pricing (per million tokens, as of 2026)
# ---------------------------------------------------------------------------
_INPUT_COST_PER_M  = 15.0   # $15 / 1M input tokens
_OUTPUT_COST_PER_M = 75.0   # $75 / 1M output tokens


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

    MODEL          = "claude-opus-4-7"
    COMPRESS_AFTER = 6    # turns: compress history older than this on HIGH
    TRIM_WINDOW    = 10   # turns: keep last N turns on MEDIUM

    def __init__(
        self,
        api_key:       Optional[str] = None,
        theta_high:    float = 0.65,
        theta_low:     float = 0.35,
        system_prompt: Optional[str] = None,
        max_tokens:    int = 1024,
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
        self.max_tokens = max_tokens
        self.stats      = SessionStats()

        # Live conversation state
        self._history:     list[dict] = []   # raw turns kept in context
        self._summary:     Optional[str] = None  # compression summary
        self._turn_idx:    int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> TurnResult:
        """Send a message, receive a CAMS-managed response."""
        self._turn_idx += 1
        self._history.append({"role": "user", "content": user_message})

        # Call Claude with current (potentially compressed) history
        messages = self._build_messages()
        resp = self.client.messages.create(
            model=self.MODEL,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        text       = resp.content[0].text
        tokens_in  = resp.usage.input_tokens
        tokens_out = resp.usage.output_tokens

        # Compute J-proxy
        cr: ConfidenceResult = self.proxy.compute(text)

        # Append assistant turn BEFORE compression decision
        self._history.append({"role": "assistant", "content": text})

        # Apply CAMS memory decision
        decision, tokens_saved = self._apply_cams(cr)

        # Costs
        cost_usd     = _cost(tokens_in, tokens_out)
        savings_usd  = _cost(tokens_saved, 0)

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
            "turn":         self._turn_idx,
            "j_score":      cr.j_score,
            "zone":         cr.zone,
            "decision":     decision,
            "tokens_saved": tokens_saved,
            "reasoning":    cr.reasoning,
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
        )

    def reset(self):
        self._history  = []
        self._summary  = None
        self._turn_idx = 0
        self.stats     = SessionStats()

    @property
    def decision_log(self) -> list[dict]:
        return self.stats.decision_log

    # ------------------------------------------------------------------
    # CAMS memory decisions
    # ------------------------------------------------------------------

    def _apply_cams(self, cr: ConfidenceResult) -> tuple[str, int]:
        """Returns (decision, tokens_saved)."""
        n_turns = len(self._history)

        if cr.zone == "HIGH" and n_turns > self.COMPRESS_AFTER * 2:
            saved = self._compress()
            return "COMPRESS", saved

        if cr.zone == "MEDIUM" and n_turns > self.TRIM_WINDOW * 2:
            saved = self._trim()
            return "TRIM", saved

        return "PRESERVE", 0

    def _compress(self) -> int:
        """
        Summarize all but the last COMPRESS_AFTER turns using Claude.
        Replace raw history with the summary + recent turns.
        Returns approximate tokens saved.
        """
        keep_n  = self.COMPRESS_AFTER * 2   # keep last N messages
        old     = self._history[:-keep_n]
        recent  = self._history[-keep_n:]

        if not old:
            return 0

        tokens_before = sum(len(m["content"]) // 4 for m in old)

        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old
        )
        summary_resp = self.client.messages.create(
            model=self.MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation in 2-3 concise sentences. "
                    "Preserve all key facts, decisions, and context needed "
                    "to continue the conversation naturally:\n\n" + conv_text
                ),
            }],
            max_tokens=200,
        )
        summary = summary_resp.content[0].text.strip()

        # Store summary, rebuild history
        self._summary = summary
        self._history = recent
        tokens_after  = len(summary) // 4
        return max(0, tokens_before - tokens_after)

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
        """Inject compression summary at front if one exists."""
        if self._summary:
            prefix = [{
                "role":    "user",
                "content": f"[Earlier conversation summary: {self._summary}]",
            }, {
                "role":    "assistant",
                "content": "Understood, I have the context from our earlier conversation.",
            }]
            return prefix + self._history
        return list(self._history)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  * _INPUT_COST_PER_M  / 1_000_000 +
        output_tokens * _OUTPUT_COST_PER_M / 1_000_000
    )

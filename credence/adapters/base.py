"""
credence/adapters/base.py
=========================
Abstract base class for all model adapters.

All adapters produce AdapterTurnResult — a model-agnostic version of
TurnResult that works regardless of which provider generated the response.
The J-score computation, faithfulness probe, and epistemic ledger are
identical across all adapters; only the API call differs.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod

from ..confidence_proxy import CredenceProxy, CredenceResult
from ..context_manager  import (
    _UNCERTAINTY_MARKERS, _MULTI_ANSWER_MARKERS,
    _MIN_COMPRESS_TOKENS, _SUMMARY_FAITHFUL_THRESHOLD,
    _cost,
)


@dataclass
class AdapterTurnResult:
    """
    Model-agnostic turn result — same fields as TurnResult but works with
    any provider. Returned by all adapter.chat() calls.
    """
    response:              str
    j_score:               float
    zone:                  str
    decision:              str          # COMPRESS | TRIM | PRESERVE
    tokens_in:             int
    tokens_out:            int
    tokens_saved:          int
    model:                 str
    provider:              str
    uncertainty_preserved: bool  = False
    alignment_warnings:    list  = field(default_factory=list)
    caveat_injected:       bool  = False


class BaseAdapter(ABC):
    """
    Base class for all Credence model adapters.

    Subclasses implement:
        _generate(messages, system) -> (text, tokens_in, tokens_out)
        _compress_with_model(text)  -> summary_str

    Everything else — J-score computation, faithfulness probe, selective
    compression, drift detection, alignment check — is identical across
    all adapters.
    """

    COMPRESS_AFTER  = 3
    TRIM_WINDOW     = 10
    ATTENTION_SINK  = 2
    MAX_COMPRESSIONS = 3

    def __init__(
        self,
        theta_high:    float = 0.70,
        theta_low:     float = 0.45,
        system_prompt: Optional[str] = None,
        max_tokens:    int   = 1024,
        registry=None,
        session_id:    Optional[str] = None,
    ):
        self.proxy         = CredenceProxy(theta_high, theta_low)
        self.system_prompt = system_prompt or (
            "You are a helpful, precise assistant. "
            "Express genuine uncertainty when it exists."
        )
        self.max_tokens        = max_tokens
        self._registry         = registry
        self._session_id       = session_id

        self._history:            list         = []
        self._history_j_scores:   list         = []
        self._summary:            Optional[str] = None
        self._turn_idx:           int  = 0
        self._compression_count:  int  = 0
        self._j_history:          list = []
        self._drift_state:        bool = False
        self._last_faith_block:   bool = False
        self._pending_caveat:     Optional[str] = None

        self._tokens_saved_total: int = 0

    @property
    def provider(self) -> str:
        return "base"

    @property
    def model_name(self) -> str:
        return "unknown"

    @abstractmethod
    def _generate(self, messages: list, system: str) -> tuple[str, int, int]:
        """Call the provider API. Returns (text, tokens_in, tokens_out)."""
        ...

    @abstractmethod
    def _compress_with_model(self, text: str) -> str:
        """Call the compression model (Haiku equivalent). Returns summary."""
        ...

    # ------------------------------------------------------------------
    # Public API — same as ContextManager
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> AdapterTurnResult:
        self._turn_idx += 1
        self._history.append({"role": "user", "content": user_message})
        self._history_j_scores.append(None)

        # Truth Buffer + pending caveat injection
        augmented_system = self._build_system()

        # Clear pending caveat (consumed this turn)
        caveat_injected = self._pending_caveat is not None
        self._pending_caveat = None

        messages = self._build_messages()
        text, tokens_in, tokens_out = self._generate(messages, augmented_system)

        cr: CredenceResult = self.proxy.compute(text)

        # Semantic entropy proxy
        lower = text.lower()
        if cr.zone == "MEDIUM" and any(m in lower for m in _MULTI_ANSWER_MARKERS):
            from dataclasses import replace as _replace
            cr = _replace(cr, zone="LOW")

        # Drift detection
        self._j_history.append(cr.j_score)
        if len(self._j_history) > 5:
            self._j_history.pop(0)
        self._drift_state = (
            len(self._j_history) >= 3
            and all(j < self.proxy.theta_low for j in self._j_history[-3:])
        )

        # Alignment check (Governor)
        alignment_warnings = self._align_output(text)
        if alignment_warnings:
            self._pending_caveat = "\n".join(w["suggested_caveat"] for w in alignment_warnings)

        self._history.append({"role": "assistant", "content": text})
        self._history_j_scores.append(cr.j_score)

        # Memory decision
        decision, tokens_saved = self._apply_memory(cr)
        self._tokens_saved_total += tokens_saved

        uncertainty_preserved = (
            self._last_faith_block
            or (decision == "PRESERVE" and cr.zone in ("LOW", "MEDIUM"))
        )
        self._last_faith_block = False

        return AdapterTurnResult(
            response              = text,
            j_score               = cr.j_score,
            zone                  = cr.zone,
            decision              = decision,
            tokens_in             = tokens_in,
            tokens_out            = tokens_out,
            tokens_saved          = tokens_saved,
            model                 = self.model_name,
            provider              = self.provider,
            uncertainty_preserved = uncertainty_preserved,
            alignment_warnings    = alignment_warnings,
            caveat_injected       = caveat_injected,
        )

    def reset(self):
        self._history.clear()
        self._history_j_scores.clear()
        self._summary         = None
        self._turn_idx        = 0
        self._compression_count = 0
        self._j_history.clear()
        self._drift_state     = False
        self._pending_caveat  = None
        self._tokens_saved_total = 0

    # ------------------------------------------------------------------
    # Internal helpers (same logic as ContextManager)
    # ------------------------------------------------------------------

    def _build_system(self) -> str:
        base = self.system_prompt
        if self._registry is not None and self._session_id is not None:
            uncertain = self._registry.list_uncertain(self._session_id)
            if uncertain:
                lines = "\n".join(f"• [{c['zone']}] {c['content'][:120]}" for c in uncertain[:10])
                base  = (
                    f"{base}\n\nEPISTEMIC CONTEXT — UNVERIFIED CONSTRAINTS:\n{lines}\n"
                    "When discussing these, always acknowledge their uncertain status."
                )
        if self._pending_caveat:
            base = f"{base}\n\nEPISTEMIC GOVERNOR ALERT:\n{self._pending_caveat}"
        return base

    def _build_messages(self) -> list:
        if self._summary and self._history:
            msgs = list(self._history)
            msgs[0] = {
                "role":    msgs[0]["role"],
                "content": f"<context_summary>\n{self._summary}\n</context_summary>\n\n" + msgs[0]["content"],
            }
            return msgs
        return list(self._history)

    def _has_uncertainty(self, text: str) -> bool:
        lower = text.lower()
        return any(m in lower for m in _UNCERTAINTY_MARKERS)

    def _align_output(self, response_text: str) -> list:
        if self._registry is None or self._session_id is None:
            return []
        uncertain = self._registry.list_uncertain(self._session_id)
        if not uncertain:
            return []
        cr = self.proxy.compute(response_text)
        if cr.zone != "HIGH":
            return []
        response_words = set(response_text.lower().split())
        warnings = []
        for constraint in uncertain:
            c_words = set(constraint["content"].lower().split())
            overlap = response_words & c_words - {"the","a","an","is","are","was","to","of","in","for","and"}
            if len(overlap) >= 3 and constraint["zone"] in ("LOW", "MEDIUM"):
                warnings.append({
                    "constraint_id":      constraint["constraint_id"],
                    "constraint_content": constraint["content"],
                    "ledger_zone":        constraint["zone"],
                    "response_zone":      cr.zone,
                    "overlap_words":      list(overlap)[:6],
                    "suggested_caveat":   (
                        f"Previous response discussed '{constraint['content'][:80]}' "
                        f"which is unverified (zone={constraint['zone']}). "
                        "Acknowledge uncertainty if mentioned again."
                    ),
                })
        return warnings

    def _apply_memory(self, cr: CredenceResult) -> tuple[str, int]:
        n_turns = len(self._history)
        if self._drift_state:
            return "PRESERVE", 0
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
        sink_msgs = self.ATTENTION_SINK * 2
        keep_n    = self.COMPRESS_AFTER * 2
        if len(self._history) <= sink_msgs + keep_n:
            return 0
        old    = self._history[sink_msgs:-keep_n]
        old_j  = self._history_j_scores[sink_msgs:-keep_n]
        recent = self._history[-keep_n:]
        if not old:
            return 0
        high_j_msgs:    list = []
        preserved_msgs: list = []
        preserved_j:    list = []
        for i in range(0, len(old) - 1, 2):
            j = old_j[i + 1]
            if j is not None and j >= self.proxy.theta_high:
                high_j_msgs.extend([old[i], old[i + 1]])
            else:
                preserved_msgs.extend([old[i], old[i + 1]])
                preserved_j.extend([old_j[i], old_j[i + 1]])
        if not high_j_msgs:
            return 0
        tokens_before = sum(len(m["content"]) // 4 for m in high_j_msgs)
        if tokens_before < _MIN_COMPRESS_TOKENS:
            return 0
        conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in high_j_msgs)
        if self._has_uncertainty(conv_text):
            self._last_faith_block = True
            return 0
        summary = self._compress_with_model(conv_text)
        # Simple content-word overlap faithfulness check (fallback)
        orig_words = set(w for w in conv_text.lower().split() if len(w) >= 4)
        summ_words = set(w for w in summary.lower().split()  if len(w) >= 4)
        if orig_words and len(orig_words & summ_words) / len(orig_words) < _SUMMARY_FAITHFUL_THRESHOLD:
            return 0
        self._summary = summary
        sink = self._history[:sink_msgs]
        sink_j = self._history_j_scores[:sink_msgs]
        self._history          = sink + preserved_msgs + recent
        self._history_j_scores = (
            sink_j + preserved_j + self._history_j_scores[-keep_n:]
        )
        tokens_after = len(summary) // 4
        return max(0, tokens_before - tokens_after)

    def _trim(self) -> int:
        keep_n    = self.TRIM_WINDOW * 2
        sink_msgs = self.ATTENTION_SINK * 2
        if len(self._history) <= keep_n:
            return 0
        old       = self._history[sink_msgs:-keep_n]
        old_j     = self._history_j_scores[sink_msgs:-keep_n]
        recent    = self._history[-keep_n:]
        recent_j  = self._history_j_scores[-keep_n:]
        preserved_msgs:  list = []
        preserved_j:     list = []
        dropped = 0
        for i in range(0, len(old) - 1, 2):
            j = old_j[i + 1]
            if j is not None and j >= self.proxy.theta_high:
                dropped += (len(old[i]["content"]) + len(old[i + 1]["content"])) // 4
            else:
                preserved_msgs.extend([old[i], old[i + 1]])
                preserved_j.extend([old_j[i], old_j[i + 1]])
        if dropped == 0:
            return 0
        sink   = self._history[:sink_msgs]
        sink_j = self._history_j_scores[:sink_msgs]
        self._history          = sink + preserved_msgs + recent
        self._history_j_scores = sink_j + preserved_j + recent_j
        return dropped

"""
credence/pipeline_monitor.py
=============================
Epistemic middleware for multi-agent pipelines.

The problem this solves
-----------------------
When Agent A hands off to Agent B, epistemic state is lost.
Agent A said "I think the rate limit is ~50 req/min — unconfirmed."
Agent B receives only the text, not the uncertainty.
Agent B writes: RATE_LIMIT = 50  — no caveat, ships to production.

The fix
-------
PipelineMonitor sits between agents. It intercepts Agent A's output,
scans it for implicit uncertain claims (using the faithfulness probe
or Ghost Detector), registers them in the shared registry, then
generates an epistemic handoff block that gets prepended to Agent B's
system prompt. Agent B's Truth Buffer sees the constraints and enforces
qualifier propagation without any agent cooperation required.

This is the only part of Credence that's *active* at handoff time —
not advisory, not informational, actually injecting epistemic context.

Usage
-----
    monitor = PipelineMonitor(registry=reg, api_key=os.environ["ANTHROPIC_API_KEY"])

    # Agent A produces output
    agent_a_output = "The cache TTL should be 300s — from staging estimates only."

    # Monitor intercepts before passing to Agent B
    handoff = monitor.intercept(
        agent_output=agent_a_output,
        from_session="agent_a",
        to_session="agent_b",
    )

    # Prepend handoff block to Agent B's system prompt
    agent_b_system = handoff.system_block + "\n\n" + base_system
    # → Agent B's Truth Buffer now contains the uncertain constraint
    # → Consistency Enforcer fires if Agent B discusses cache TTL

Architecture note
-----------------
The monitor uses two extraction strategies, in order:
  1. Probe-based (deterministic, free): checks _UNCERTAINTY_MARKERS in Agent A's text.
     If canonical markers are found, registers them directly.
  2. Ghost Detector (Claude Haiku call, ~$0.0005): extracts *implicit* uncertain
     claims — things stated confidently but sourced from unconfirmed sources.
     Requires api_key. Falls back to probe-only if not available.

Both strategies write to the same registry. Agent B's ContextManager
reads from it via _augment_with_truth_buffer() on every turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .context_manager import _UNCERTAINTY_MARKERS
from .registry import CredenceRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MONITOR_MODEL = "claude-haiku-4-5-20251001"
_MONITOR_J_SCORE = 0.25          # conservative — intercept-registered claims start LOW
_MONITOR_ZONE = "LOW"
_MONITOR_GHOST_CONFIDENCE = 0.65  # minimum Ghost Detector confidence to register
_MIN_CLAIM_LEN = 12               # skip trivially short claims
_MAX_CLAIMS_PER_HANDOFF = 8       # cap to avoid flooding Agent B's context

_HANDOFF_HEADER = "EPISTEMIC HANDOFF — UNVERIFIED CLAIMS FROM UPSTREAM AGENT:"
_HANDOFF_FOOTER = (
    "These claims were stated by the upstream agent but are NOT verified. "
    "When referencing or implementing these values, state them as uncertain."
)

# Minimal stopwords for keyword overlap (reused from CE logic)
_STOP = frozenset({"the", "a", "an", "is", "are", "was", "were", "be", "been",
                   "in", "of", "to", "for", "and", "or", "with", "from", "at",
                   "by", "it", "as", "on", "that", "this", "we", "i", "you"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedClaim:
    """A single uncertain claim extracted from an agent's output."""
    content:    str
    confidence: float   # 0–1, from Ghost Detector or 0.3 default for probe
    source:     str     # "probe" or "ghost_detector"
    raw_quote:  str     # the sentence or fragment it came from
    cid:        str = ""  # filled after registry.register()


@dataclass
class EpistemicHandoff:
    """Result of a pipeline monitor intercept call.

    system_block:   str to prepend to Agent B's system prompt.
                    If n_injected == 0, this is an empty string (no overhead).
    from_session:   session_id of the upstream agent
    to_session:     session_id of the downstream agent
    n_extracted:    number of uncertain claims found in Agent A's output
    n_injected:     number actually registered and injected (≤ n_extracted)
    claims:         list of ExtractedClaim objects
    strategy:       "probe" | "ghost_detector" | "none"
    """
    system_block:  str
    from_session:  str
    to_session:    str
    n_extracted:   int
    n_injected:    int
    claims:        list[ExtractedClaim] = field(default_factory=list)
    strategy:      str = "none"

    @property
    def has_uncertain(self) -> bool:
        return self.n_injected > 0


# ---------------------------------------------------------------------------
# PipelineMonitor
# ---------------------------------------------------------------------------

class PipelineMonitor:
    """
    Epistemic middleware. Sits between agents in a pipeline.
    Extracts uncertain claims from Agent A's output and injects them
    into Agent B's epistemic context before Agent B is invoked.
    """

    def __init__(
        self,
        registry:          CredenceRegistry,
        api_key:           Optional[str] = None,
        use_ghost_detector: bool = True,
    ) -> None:
        self._registry    = registry
        self._api_key     = api_key
        self._use_ghost   = use_ghost_detector and (api_key is not None)

        self._client = None
        if self._use_ghost:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=api_key)
            except ImportError:
                self._use_ghost = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intercept(
        self,
        agent_output: str,
        from_session:  str,
        to_session:    str,
    ) -> EpistemicHandoff:
        """
        Intercept Agent A's output before it reaches Agent B.

        Scans agent_output for uncertain claims, registers them in the
        shared registry under from_session, and returns an EpistemicHandoff
        containing the system_block to prepend to Agent B's system prompt.

        No API call if probe finds markers; one cheap Haiku call otherwise.
        """
        claims: list[ExtractedClaim] = []
        strategy = "none"

        # Strategy 1 — Probe-based (free, deterministic)
        probe_claims = self._probe_extract(agent_output)
        if probe_claims:
            claims = probe_claims
            strategy = "probe"
        elif self._use_ghost:
            # Strategy 2 — Ghost Detector (one Haiku call)
            ghost_claims = self._ghost_extract(agent_output)
            if ghost_claims:
                claims = ghost_claims
                strategy = "ghost_detector"

        # Cap and register
        claims = claims[:_MAX_CLAIMS_PER_HANDOFF]
        injected = []
        for c in claims:
            if len(c.content) < _MIN_CLAIM_LEN:
                continue
            try:
                cid = self._registry.register(
                    c.content,
                    session_id=from_session,
                    j_score=c.confidence,
                    zone=_MONITOR_ZONE,
                    source="pipeline_monitor",
                )
                c.cid = cid
                injected.append(c)
            except Exception:
                pass

        # Build system block for Agent B
        system_block = self._build_system_block(injected, from_session)

        return EpistemicHandoff(
            system_block=system_block,
            from_session=from_session,
            to_session=to_session,
            n_extracted=len(claims),
            n_injected=len(injected),
            claims=injected,
            strategy=strategy,
        )

    def build_agent_b_system(
        self,
        handoff:       EpistemicHandoff,
        base_system:   str = "",
        include_gate:  bool = True,
    ) -> str:
        """
        Compose the full system prompt for Agent B by prepending the
        epistemic handoff block to the base system prompt.

        include_gate: if True, adds a final imperative line prohibiting
                      confident code embedding of unverified values.
        """
        parts = []
        if handoff.system_block:
            parts.append(handoff.system_block)
        if include_gate and handoff.n_injected > 0:
            parts.append(
                "GATE: Do NOT embed these values in code, config, or commands "
                "without explicitly marking them as unverified estimates."
            )
        if base_system:
            parts.append(base_system)
        return "\n\n".join(parts)

    def handoff_report(self, handoff: EpistemicHandoff) -> str:
        """Human-readable summary of what the monitor intercepted."""
        if not handoff.has_uncertain:
            return f"[PipelineMonitor] No uncertain claims found in {handoff.from_session} → {handoff.to_session}"
        lines = [
            f"[PipelineMonitor] {handoff.from_session} → {handoff.to_session}",
            f"  Strategy: {handoff.strategy}  |  Extracted: {handoff.n_extracted}  |  Registered: {handoff.n_injected}",
        ]
        for c in handoff.claims:
            lines.append(f"  ⚠ [conf={c.confidence:.2f}] {c.content[:80]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extraction strategies
    # ------------------------------------------------------------------

    def _probe_extract(self, text: str) -> list[ExtractedClaim]:
        """
        Deterministic probe: find sentences containing uncertainty markers.
        Returns one ExtractedClaim per marked sentence, up to _MAX_CLAIMS_PER_HANDOFF.
        """
        lower = text.lower()
        # Fast gate: if no markers at all, skip sentence splitting
        if not any(m in lower for m in _UNCERTAINTY_MARKERS):
            return []

        claims = []
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
        for sent in sentences:
            if len(sent) < _MIN_CLAIM_LEN:
                continue
            sl = sent.lower()
            matched = [m for m in _UNCERTAINTY_MARKERS if m in sl]
            if matched:
                # Use the sentence as both content and raw_quote
                claims.append(ExtractedClaim(
                    content=sent.strip(),
                    confidence=_MONITOR_J_SCORE,
                    source="probe",
                    raw_quote=sent.strip(),
                ))
        return claims

    def _ghost_extract(self, text: str) -> list[ExtractedClaim]:
        """
        Ghost Detector: one Haiku call to extract *implicit* uncertain claims
        — facts stated confidently but sourced from unconfirmed sources.
        Returns [] if the API call fails or no claims are found.
        """
        if self._client is None:
            return []

        prompt = (
            "You are an epistemic auditor. Extract implicit uncertain claims from "
            "this agent output — claims stated as facts but that are actually "
            "estimates, vendor-stated values, unconfirmed assumptions, or "
            "second-hand information.\n\n"
            "Return JSON array: [{\"claim\": str, \"confidence\": 0-1, \"quote\": str}]\n"
            "confidence = 1.0 means definitely uncertain. 0 = definitely verified.\n"
            "Only return claims with confidence >= 0.60.\n"
            "Max 6 items. If no uncertain claims found, return [].\n\n"
            f"Agent output:\n{text[:2000]}"
        )
        try:
            resp = self._client.messages.create(
                model=_MONITOR_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON from response
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group(0))
            claims = []
            for item in items:
                conf = float(item.get("confidence", 0))
                if conf < _MONITOR_GHOST_CONFIDENCE:
                    continue
                claim_text = str(item.get("claim", "")).strip()
                if len(claim_text) < _MIN_CLAIM_LEN:
                    continue
                claims.append(ExtractedClaim(
                    content=claim_text,
                    confidence=round(conf * 0.35, 4),  # scale to J-score range
                    source="ghost_detector",
                    raw_quote=str(item.get("quote", claim_text))[:200],
                ))
            return claims
        except Exception:
            return []

    # ------------------------------------------------------------------
    # System block builder
    # ------------------------------------------------------------------

    def _build_system_block(
        self,
        claims: list[ExtractedClaim],
        from_session: str,
    ) -> str:
        if not claims:
            return ""
        lines = [_HANDOFF_HEADER, ""]
        for c in claims:
            tag = "⚠⚠" if c.confidence < 0.20 else "⚠"
            lines.append(f"  {tag} [conf={c.confidence:.2f}] {c.content}")
        lines.append("")
        lines.append(_HANDOFF_FOOTER)
        return "\n".join(lines)

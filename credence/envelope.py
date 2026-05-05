"""
credence/envelope.py
================
CredenceEnvelope: epistemic provenance wrapper for every Credence output.

The envelope is the architectural pivot of Credence v2. It turns a local
context manager into a systemic epistemic integrity layer by making
the J-score travel with information across agent boundaries.

Without envelope: J-score is used, then discarded. Downstream agents
  have no visibility into the epistemic quality of information they receive.

With envelope: every response carries its J-score, source, verification
  status, chain depth, and uncertainty flags. Downstream agents can inspect
  provenance before compressing, summarizing, or acting on information.

Grounded in UProp (ACL 2025): uncertainty propagation in multi-step LLM
pipelines is a distinct problem from single-step uncertainty estimation.
Single-step methods fail to capture how uncertainty compounds across agent
handoffs. The envelope is the practical implementation of UProp's insight:
attach a trust score to each piece of information and propagate it.

Design constraints:
  - Fully JSON-serializable (no Python-only types) — MCP-transportable
  - Immutable after creation (dataclass with no setters) — safe across hops
  - trust_score degrades with chain_depth (empirically calibrated in Phase 5)
  - should_verify is the primary flag downstream agents should check
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

# Known trusted sources — internal Credence agents.
# External sources (unknown IDs) receive a trust penalty.
_TRUSTED_SOURCES = frozenset({"credence", "user", "system"})

# Trust penalty per agent hop beyond the source.
# Calibrated starting value — will be updated after Phase 5 multi-agent experiment.
_CHAIN_DEPTH_PENALTY = 0.05

# Below this trust_score, the envelope flags should_verify = True
_VERIFY_THRESHOLD = 0.40

# Trust penalty for unknown / unverified source agents
_UNKNOWN_SOURCE_PENALTY = 0.10


@dataclass(frozen=True)
class CredenceEnvelope:
    """
    Epistemic provenance wrapper for a single Credence response or agent output.

    Fields
    ------
    content             : The actual response text
    j_score             : Raw J-proxy score at time of generation (0–1)
    zone                : "HIGH" | "MEDIUM" | "LOW"
    source              : Agent ID or "user" / "credence" / "system"
    verified            : True if this has been independently cross-checked
    chain_depth         : Number of agent hops from original source (0 = fresh)
    uncertainty_preserved: True if faithfulness probe fired — uncertainty kept verbatim
    content_type        : "text" | "code" | "error" | "math"
    session_id          : Optional session identifier for tracing
    """

    content:               str
    j_score:               float
    zone:                  str
    source:                str
    verified:              bool
    chain_depth:           int
    uncertainty_preserved: bool
    content_type:          str
    session_id:            Optional[str] = None

    # ------------------------------------------------------------------
    # Derived properties (computed, not stored — frozen dataclass)
    # ------------------------------------------------------------------

    @property
    def source_trust_penalty(self) -> float:
        """Unknown sources receive a penalty on top of chain depth decay."""
        return 0.0 if self.source in _TRUSTED_SOURCES else _UNKNOWN_SOURCE_PENALTY

    @property
    def trust_score(self) -> float:
        """
        Composite trust score: raw J degraded by chain depth and source trust.

        trust_score = max(0, j_score - depth_penalty - source_penalty)

        At chain_depth=0, trust_score == j_score (for trusted sources).
        Each hop degrades trust by _CHAIN_DEPTH_PENALTY (default 0.05).
        Unknown sources take an additional _UNKNOWN_SOURCE_PENALTY (0.10).

        A 4-hop-old LOW-J result from an unknown source:
          trust_score = max(0, 0.30 - 4*0.05 - 0.10) = 0.0 → always verify.
        """
        raw = self.j_score - (self.chain_depth * _CHAIN_DEPTH_PENALTY) - self.source_trust_penalty
        return round(max(0.0, raw), 4)

    @property
    def should_verify(self) -> bool:
        """True when trust_score is too low to act on without verification."""
        return self.trust_score < _VERIFY_THRESHOLD and not self.verified

    @property
    def safe_to_compress(self) -> bool:
        """
        True only when trust is HIGH and uncertainty was NOT explicitly preserved.

        This is the primary check a downstream agent should perform before
        compressing or summarizing information received in an envelope.
        """
        return (
            self.trust_score >= _VERIFY_THRESHOLD
            and not self.uncertainty_preserved
            and not self.should_verify
            and self.zone == "HIGH"
        )

    # ------------------------------------------------------------------
    # Serialization (MCP-compatible)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain dict — JSON-serializable, MCP-transportable."""
        d = asdict(self)
        # Add computed properties for downstream consumers
        d["trust_score"]       = self.trust_score
        d["should_verify"]     = self.should_verify
        d["safe_to_compress"]  = self.safe_to_compress
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "CredenceEnvelope":
        """Reconstruct from a plain dict (e.g., received over MCP)."""
        return cls(
            content               = d["content"],
            j_score               = d["j_score"],
            zone                  = d["zone"],
            source                = d["source"],
            verified              = d["verified"],
            chain_depth           = d["chain_depth"],
            uncertainty_preserved = d["uncertainty_preserved"],
            content_type          = d["content_type"],
            session_id            = d.get("session_id"),
        )

    def propagate(self, new_source: Optional[str] = None) -> "CredenceEnvelope":
        """
        Create a new envelope for the next hop in a pipeline.

        Increments chain_depth (degrades trust), optionally updates source.
        The verified flag is reset — the new agent hasn't verified this.
        """
        return CredenceEnvelope(
            content               = self.content,
            j_score               = self.j_score,
            zone                  = self.zone,
            source                = new_source or self.source,
            verified              = False,
            chain_depth           = self.chain_depth + 1,
            uncertainty_preserved = self.uncertainty_preserved,
            content_type          = self.content_type,
            session_id            = self.session_id,
        )

    def verify(self) -> "CredenceEnvelope":
        """Return a copy with verified=True after independent confirmation."""
        return CredenceEnvelope(
            content               = self.content,
            j_score               = self.j_score,
            zone                  = self.zone,
            source                = self.source,
            verified              = True,
            chain_depth           = self.chain_depth,
            uncertainty_preserved = self.uncertainty_preserved,
            content_type          = self.content_type,
            session_id            = self.session_id,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_turn(
        cls,
        response:              str,
        j_score:               float,
        zone:                  str,
        decision:              str,
        content_type:          str = "text",
        source:                str = "credence",
        session_id:            Optional[str] = None,
        uncertainty_preserved: bool = False,
    ) -> "CredenceEnvelope":
        """
        Build an envelope from a ContextManager TurnResult fields.
        uncertainty_preserved must be passed explicitly by the caller —
        it is True when the faithfulness probe blocked compression, OR when
        the turn is LOW/MEDIUM zone and was preserved verbatim.
        The decision parameter is retained for backward compatibility.
        """
        return cls(
            content               = response,
            j_score               = j_score,
            zone                  = zone,
            source                = source,
            verified              = False,
            chain_depth           = 0,
            uncertainty_preserved = uncertainty_preserved,
            content_type          = content_type,
            session_id            = session_id,
        )

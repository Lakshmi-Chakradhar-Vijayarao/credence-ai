"""
credence/claim_extractor.py
===========================
LLM-based structured claim extraction — catches ghost constraints.

The faithfulness probe (CP1) relies on surface hedging markers:
"probably", "I think", "unconfirmed". It misses ghost constraints —
claims stated as confident facts that are actually unverified:

    "The Stripe rate limit is 50 req/min."   ← vendor call, no docs
    "Token expiry is 3600s."                 ← assumed, never checked
    "We use RS256 for JWT signing."          ← legacy assumption

ClaimExtractor sends the user's message to Haiku and asks it to
classify every factual claim by source type and confidence — not by
surface language. A claim can score LOW confidence even with no hedging
words if Haiku infers it's a vendor claim, an assumption, or hearsay.

This is the distribution-aware replacement for keyword scanning:
uncertainty is inferred from the *claim's epistemic origin*, not its
surface markers. Ghost constraints surface as has_surface_markers=False
with source_type in ("vendor_claim", "hearsay", "assumption").

Usage:
    from credence import ClaimExtractor, CredenceRegistry

    extractor = ClaimExtractor()
    claims = extractor.extract(user_message, client)
    for c in claims:
        if not c.has_surface_markers:
            print(f"Ghost constraint: {c.claim_text} (conf={c.confidence:.2f})")

    # Auto-register into registry:
    registered = extractor.extract_and_register(
        user_message, client, registry, session_id
    )

Cost: ~one Haiku call per user turn (~$0.0002). Off by default in
ContextManager (use_claim_extractor=True to enable).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import CredenceRegistry

# ---------------------------------------------------------------------------
# Source types — epistemic origin taxonomy
# ---------------------------------------------------------------------------
SOURCE_TYPES = frozenset({
    "verified_fact",   # confirmed in docs, code, official spec
    "user_estimate",   # user's own rough guess ("around 50")
    "vendor_claim",    # stated by vendor/sales/partner — not in contract
    "inference",       # derived from other facts ("so it must be...")
    "hearsay",         # told by someone, not verified ("teammate said...")
    "assumption",      # treated as true without checking
})

# ---------------------------------------------------------------------------
# Uncertainty types — epistemic gap taxonomy (multi-dimensional, not scalar)
# ---------------------------------------------------------------------------
UNCERTAINTY_TYPES = frozenset({
    "knowledge_gap",   # user simply doesn't have the verified information
    "ambiguity",       # the claim could mean multiple things
    "hearsay",         # origin is indirect / third-party
    "extrapolation",   # inferred from incomplete data (e.g. a trend)
    "none",            # verified, no meaningful uncertainty
})

# Confidence thresholds (matches registry decay zones)
_GHOST_CONFIDENCE_FLOOR = 0.60   # below this AND no surface markers → ghost constraint
_UNVERIFIED_THRESHOLD   = 0.75   # above this only verified_fact should sit

# Haiku JSON extraction prompt — the key design decision here is asking for
# source_type rather than asking "is this uncertain?". Haiku is better at
# reasoning about epistemic origin than about hedging language.
_EXTRACTION_PROMPT = """\
Analyze this text and extract every factual claim that could affect a technical decision.
For each claim, provide:
  - claim_text: a clean one-sentence restatement of the claim
  - value: the specific value/number/string if present, else ""
  - confidence: float 0.0–1.0 (how confident is the speaker, based on ORIGIN not hedging words)
  - source_type: one of: verified_fact, user_estimate, vendor_claim, inference, hearsay, assumption
  - uncertainty_type: one of: knowledge_gap, ambiguity, hearsay, extrapolation, none
  - has_surface_markers: true if text contains explicit hedging words like "probably/maybe/I think/unconfirmed/roughly/seems/not sure"

CRITICAL: Classify by epistemic origin, not surface language.
"The rate limit is 50 req/min" stated after a vendor call → vendor_claim, confidence=0.35
"The rate limit is 50 req/min" read from official API docs → verified_fact, confidence=0.95
A claim can be a ghost constraint if it sounds confident but comes from an unverified source.

Return ONLY a JSON array. No explanation. No markdown. If no claims found, return [].

Example output:
[
  {
    "claim_text": "rate limit is 50 req/min",
    "value": "50",
    "confidence": 0.35,
    "source_type": "vendor_claim",
    "uncertainty_type": "knowledge_gap",
    "has_surface_markers": false
  }
]

Text to analyze:
"""

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MAX_TOKENS  = 400


# ---------------------------------------------------------------------------
# StructuredClaim dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructuredClaim:
    """
    A factual claim with its full epistemic provenance.

    has_surface_markers=False + source_type in (vendor_claim, hearsay, assumption)
    + confidence < 0.60 → ghost constraint. The faithfulness probe misses these.
    """
    claim_text:          str
    value:               str   = ""
    confidence:          float = 0.5
    source_type:         str   = "assumption"
    uncertainty_type:    str   = "knowledge_gap"
    has_surface_markers: bool  = False
    raw_quote:           str   = ""
    constraint_id:       str   = ""   # populated after registry registration

    @property
    def is_ghost(self) -> bool:
        """
        True if this is a ghost constraint: implicitly uncertain (no hedging
        markers) but low-confidence by origin. The faithfulness probe misses these.
        """
        return (
            not self.has_surface_markers
            and self.confidence < _GHOST_CONFIDENCE_FLOOR
            and self.source_type in ("vendor_claim", "hearsay", "assumption", "inference")
        )

    @property
    def should_register(self) -> bool:
        """True if this claim should be added to the epistemic registry."""
        return self.confidence < _UNVERIFIED_THRESHOLD and self.source_type != "verified_fact"

    def to_dict(self) -> dict:
        return {
            "claim_text":          self.claim_text,
            "value":               self.value,
            "confidence":          round(self.confidence, 3),
            "source_type":         self.source_type,
            "uncertainty_type":    self.uncertainty_type,
            "has_surface_markers": self.has_surface_markers,
            "is_ghost":            self.is_ghost,
            "constraint_id":       self.constraint_id,
        }


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """
    Extracts structured epistemic claims from user messages using Haiku.

    Unlike the faithfulness probe (which scans for surface hedging markers),
    ClaimExtractor asks Haiku to reason about the *origin* of each claim.
    This catches ghost constraints that the probe misses.

    Design principle:
        Uncertainty is a property of epistemic origin, not surface language.
        A claim stated confidently from a sales call is more uncertain than
        a hedged claim verified in official documentation.
    """

    def __init__(
        self,
        model:      str   = _MODEL_HAIKU,
        max_tokens: int   = _MAX_TOKENS,
    ):
        self.model      = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, client) -> list[StructuredClaim]:
        """
        Extract structured claims from text. Returns empty list if Haiku
        fails or text contains no actionable claims.
        """
        if not text or not text.strip():
            return []

        try:
            raw = self._call_haiku(text, client)
            return self._parse_response(raw, text)
        except Exception:
            return []

    def extract_and_register(
        self,
        text:       str,
        client,
        registry:   "CredenceRegistry",
        session_id: str,
        current_turn: int = 0,
    ) -> list[StructuredClaim]:
        """
        Extract claims and auto-register unverified ones in the registry.
        Returns all extracted claims (including verified ones).
        ghost constraints are registered at their extracted confidence score.
        """
        claims = self.extract(text, client)
        for claim in claims:
            if claim.should_register:
                try:
                    cid = registry.register(
                        content    = claim.claim_text,
                        session_id = session_id,
                        j_score    = claim.confidence,
                        source     = f"extractor:{claim.source_type}",
                        turn       = current_turn,
                    )
                    claim.constraint_id = cid
                    registry.log_event(
                        constraint_id = cid,
                        event_type    = "claim_extract",
                        j_score       = claim.confidence,
                        zone          = _confidence_to_zone(claim.confidence),
                        notes         = (
                            f"source={claim.source_type} "
                            f"uncertainty={claim.uncertainty_type} "
                            f"ghost={claim.is_ghost}"
                        ),
                    )
                except Exception:
                    pass
        return claims

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_haiku(self, text: str, client) -> str:
        resp = client.messages.create(
            model    = self.model,
            messages = [{
                "role":    "user",
                "content": _EXTRACTION_PROMPT + text[:2000],
            }],
            max_tokens = self.max_tokens,
        )
        return resp.content[0].text.strip() if resp.content else "[]"

    @staticmethod
    def _parse_response(raw: str, original_text: str) -> list[StructuredClaim]:
        """Parse Haiku's JSON output into StructuredClaim objects."""
        # Strip markdown fences if Haiku wrapped in ```json```
        raw = re.sub(r"^```[^\n]*\n", "", raw.strip())
        raw = re.sub(r"\n```$", "", raw.strip())

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting a JSON array from the middle of text
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return []
            try:
                items = json.loads(m.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(items, list):
            return []

        claims = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                source = item.get("source_type", "assumption")
                if source not in SOURCE_TYPES:
                    source = "assumption"
                utype = item.get("uncertainty_type", "knowledge_gap")
                if utype not in UNCERTAINTY_TYPES:
                    utype = "knowledge_gap"
                conf = float(item.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                claims.append(StructuredClaim(
                    claim_text          = str(item.get("claim_text", "")).strip(),
                    value               = str(item.get("value", "")).strip(),
                    confidence          = conf,
                    source_type         = source,
                    uncertainty_type    = utype,
                    has_surface_markers = bool(item.get("has_surface_markers", False)),
                    raw_quote           = original_text[:200],
                ))
            except (TypeError, ValueError):
                continue

        return [c for c in claims if c.claim_text]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_to_zone(confidence: float) -> str:
    if confidence >= 0.70:
        return "HIGH"
    elif confidence >= 0.45:
        return "MEDIUM"
    return "LOW"

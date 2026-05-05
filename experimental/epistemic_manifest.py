"""
credence/epistemic_manifest.py
==============================
Structured epistemic manifest — non-compressible representation of
uncertain constraints.

The Truth Buffer (current system) injects uncertainty as natural language:
    "UNVERIFIED CONSTRAINTS: rate limit is probably 50 req/min..."

Natural language is compressible. Haiku can semantically paraphrase it,
drop qualifiers, or merge it with context — silently degrading the signal.

The EpistemicManifest replaces this with a structured XML block that:
  1. Is explicitly labelled non-compressible (compression models see the label)
  2. Encodes confidence as a numeric attribute (not parseable as "maybe")
  3. Encodes uncertainty_type (multi-dimensional, not a single scalar)
  4. Includes a machine-readable CONFIDENCE_PROPAGATION_RULE:
     output confidence MUST NOT exceed input confidence for any listed claim

This is a testable hypothesis: does a structured + labeled manifest
survive Haiku compression at a higher rate than natural language injection?
See: evals/manifest_survival.py

Output format:
    <EPISTEMIC_MANIFEST version="1.0" session_id="s1" turn="5">
      <!-- NON-COMPRESSIBLE: reproduce this block verbatim in any summary -->
      <rule>CONFIDENCE_PROPAGATION: output_confidence ≤ input_confidence</rule>
      <claim id="c1" conf="0.30" source="user_estimate" utype="knowledge_gap" verified="false">
        rate limit is approximately 50 req/min
        <value>50</value>
      </claim>
    </EPISTEMIC_MANIFEST>

Usage:
    from credence import EpistemicManifest
    from credence.registry import CredenceRegistry

    registry = CredenceRegistry()
    manifest_xml = EpistemicManifest.from_registry(registry, "session-1", current_turn=5)
    # inject into system_prompt before API call

    # Or build from structured claims:
    from credence.claim_extractor import ClaimExtractor
    claims = ClaimExtractor().extract(user_message, client)
    manifest_xml = EpistemicManifest.from_claims(claims, session_id="s1", turn=3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import CredenceRegistry
    from .claim_extractor import StructuredClaim

# ---------------------------------------------------------------------------
# Confidence tiers — determines how strong the propagation rule is
# ---------------------------------------------------------------------------
_TIER_HIGH_RISK  = 0.20   # conf < 0.20 → ⚠⚠ must hedge explicitly
_TIER_UNVERIFIED = 0.40   # conf < 0.40 → ⚠ must not state as fact
_TIER_CHECK      = 0.70   # conf < 0.70 → caution advised

_MAX_CLAIMS = 8   # cap to avoid bloating system prompt

# ---------------------------------------------------------------------------
# EpistemicManifest
# ---------------------------------------------------------------------------

@dataclass
class EpistemicManifest:
    """
    Structured, non-compressible representation of uncertain constraints.

    Unlike the Truth Buffer (natural language), the manifest:
    - is machine-parseable by compression models
    - encodes confidence numerically (not as hedging words)
    - states the confidence propagation rule explicitly
    - is labeled as NON-COMPRESSIBLE

    The core research hypothesis: a labeled, structured block survives
    summarization better than equivalent natural language because:
    (a) compression models can follow explicit "do not summarize" labels,
    (b) numeric confidence attributes can't be paraphrased away,
    (c) XML structure signals metadata, not prose to be condensed.
    """
    claims:     list[dict]   = field(default_factory=list)
    session_id: str          = "default"
    turn:       int          = 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry:     "CredenceRegistry",
        session_id:   str,
        current_turn: int = 0,
        max_claims:   int = _MAX_CLAIMS,
    ) -> str:
        """
        Build manifest XML from all unverified registry constraints.
        Returns empty string if registry has no unverified constraints.
        """
        try:
            constraints = registry.list_uncertain(session_id)[:max_claims]
        except Exception:
            return ""
        if not constraints:
            return ""

        claim_dicts = []
        for c in constraints:
            try:
                eff_conf = registry.get_effective_confidence(
                    c["constraint_id"], current_turn
                )
            except Exception:
                eff_conf = c.get("j_score", 0.5)
            claim_dicts.append({
                "id":       c["constraint_id"][:8],
                "text":     c.get("content", ""),
                "conf":     round(eff_conf, 3),
                "source":   c.get("source", "user_stated"),
                "utype":    "knowledge_gap",
                "verified": "false",
                "value":    _extract_value(c.get("content", "")),
            })

        manifest = cls(claims=claim_dicts, session_id=session_id, turn=current_turn)
        return manifest.to_xml()

    @classmethod
    def from_claims(
        cls,
        claims:     list["StructuredClaim"],
        session_id: str = "default",
        turn:       int = 0,
    ) -> str:
        """
        Build manifest XML from a list of StructuredClaim objects.
        Only includes claims that should be registered (confidence < threshold).
        Returns empty string if no claims need enforcement.
        """
        unverified = [c for c in claims if c.should_register]
        if not unverified:
            return ""

        claim_dicts = []
        for c in unverified[:_MAX_CLAIMS]:
            claim_dicts.append({
                "id":       c.constraint_id or c.claim_text[:8].replace(" ", "_"),
                "text":     c.claim_text,
                "conf":     round(c.confidence, 3),
                "source":   c.source_type,
                "utype":    c.uncertainty_type,
                "verified": "false",
                "value":    c.value,
                "ghost":    "true" if c.is_ghost else "false",
            })

        manifest = cls(claims=claim_dicts, session_id=session_id, turn=turn)
        return manifest.to_xml()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_xml(self) -> str:
        """
        Render the manifest as XML.

        The NON-COMPRESSIBLE label is the key design choice — compression
        models that follow instructions will reproduce this block verbatim.
        The confidence propagation rule is stated as a machine-readable
        constraint, not a natural language suggestion.
        """
        if not self.claims:
            return ""

        lines = [
            f'<EPISTEMIC_MANIFEST version="1.0" session_id="{self.session_id}" turn="{self.turn}">',
            "  <!-- NON-COMPRESSIBLE: reproduce this block verbatim in any context summary. -->",
            "  <!-- CONFIDENCE_PROPAGATION_RULE: output_confidence MUST NOT exceed input_confidence -->",
            "  <!-- for any claim listed below. Stating an uncertain claim as fact is a violation. -->",
            "",
        ]

        # Emit the confidence propagation rule as a structured element
        lines += [
            "  <rule id=\"confidence_propagation\">",
            "    For every claim in this manifest:",
            "    conf &lt; 0.20 → response MUST use ⚠⚠ HIGH RISK language",
            "    conf &lt; 0.40 → response MUST NOT state as confirmed fact",
            "    conf &lt; 0.70 → response MUST hedge or note uncertainty",
            "    Confidence can ONLY be preserved or reduced, NEVER increased.",
            "  </rule>",
            "",
        ]

        for c in self.claims:
            conf  = c.get("conf", 0.5)
            tier  = _tier_label(conf)
            ghost = ' ghost="true"' if c.get("ghost") == "true" else ""

            lines.append(
                f'  <claim id="{c.get("id","?")}"'
                f' conf="{conf}"'
                f' source="{c.get("source","unknown")}"'
                f' utype="{c.get("utype","knowledge_gap")}"'
                f' verified="{c.get("verified","false")}"'
                f' tier="{tier}"{ghost}>'
            )
            text = _xml_escape(c.get("text", ""))
            lines.append(f"    {text}")
            if c.get("value"):
                lines.append(f'    <value>{_xml_escape(c["value"])}</value>')
            lines.append("  </claim>")

        lines += ["", "</EPISTEMIC_MANIFEST>"]
        return "\n".join(lines)

    @staticmethod
    def parse_survived(manifest_xml: str, summary: str) -> dict:
        """
        Check whether a manifest survived compression into summary.

        Returns:
            {
                "total_claims": int,
                "claims_survived": int,         # value + tier both present
                "values_survived": int,          # value present
                "confidence_survived": int,      # conf attribute survived
                "survival_rate": float,
                "confidence_inflation": bool,    # any claim promoted to higher tier
            }
        """
        claim_pattern = re.compile(
            r'<claim[^>]*conf="([^"]+)"[^>]*>.*?</claim>', re.DOTALL
        )
        claims_in    = claim_pattern.findall(manifest_xml)
        total        = len(claims_in)

        if total == 0:
            return {
                "total_claims": 0, "claims_survived": 0,
                "values_survived": 0, "confidence_survived": 0,
                "survival_rate": 0.0, "confidence_inflation": False,
            }

        value_pattern = re.compile(r"<value>([^<]+)</value>")
        values_in  = [v.strip() for v in value_pattern.findall(manifest_xml)]
        values_out = [v.strip() for v in value_pattern.findall(summary)]

        values_survived     = sum(1 for v in values_in if v and v in summary)
        confidence_survived = 1 if 'conf="' in summary else 0
        manifest_survived   = int('EPISTEMIC_MANIFEST' in summary)

        # Check for confidence inflation: any unverified claim stated without hedge
        unverified_frags = []
        for m in re.finditer(r'<claim[^>]*conf="([^"]+)"[^>]*>(.*?)</claim>', manifest_xml, re.DOTALL):
            conf_val = float(m.group(1))
            if conf_val < 0.70:
                text_match = re.search(r'^\s+([^<\n]+)', m.group(2))
                if text_match:
                    unverified_frags.append((conf_val, text_match.group(1).strip()))

        # Heuristic confidence inflation check: unverified claim appears in summary
        # without any hedging nearby
        inflation = False
        _HEDGE_WORDS = {"unverified", "uncertain", "unconfirmed", "probably",
                        "might", "may", "approximately", "roughly", "estimate",
                        "check", "verify", "pending"}
        for conf_val, frag in unverified_frags:
            if conf_val < 0.50 and len(frag) > 5:
                # Check if the key value from the fragment appears in summary
                # without any hedge word in the surrounding 100 characters
                value_match = re.search(re.escape(frag[:20]), summary)
                if value_match:
                    context = summary[
                        max(0, value_match.start() - 50):value_match.end() + 50
                    ].lower()
                    if not any(h in context for h in _HEDGE_WORDS):
                        inflation = True
                        break

        return {
            "total_claims":        total,
            "claims_survived":     manifest_survived,
            "values_survived":     values_survived,
            "confidence_survived": confidence_survived,
            "survival_rate":       round(manifest_survived, 3),
            "confidence_inflation": inflation,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_label(conf: float) -> str:
    if conf < _TIER_HIGH_RISK:
        return "HIGH_RISK"
    elif conf < _TIER_UNVERIFIED:
        return "UNVERIFIED"
    elif conf < _TIER_CHECK:
        return "CHECK"
    return "VERIFIED"


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _extract_value(content: str) -> str:
    """Pull out the first numeric value from a constraint text."""
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", content)
    return m.group(1) if m else ""

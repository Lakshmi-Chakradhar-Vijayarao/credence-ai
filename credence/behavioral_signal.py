"""
credence/behavioral_signal.py
==============================
Tier 2 signal: Behavioral Consistency Score.

Tier 1 (J-proxy) measures linguistic assertiveness of a single response —
hedging rate, anchors, self-corrections. It is fast (no API calls) but
has a documented ceiling: confidently-stated wrong answers score HIGH-J,
indistinguishable from confidently-stated correct ones.

Tier 2 addresses this via behavioral consistency: draw N=5 Haiku
re-completions of the same question and measure pairwise ROUGE-L variance.
High variance → model is uncertain despite confident-sounding text.
Low variance → model consistently produces the same answer → signal for
HIGH-J routing is valid.

Fused score: J_final = w_j × J + (1 − w_j) × consistency
Default w_j = 0.70 (J-proxy is still the primary signal at API surfaces
where we can't inspect attention weights; consistency supplements it).

This is NOT the same as the SemanticEntropyProbe (which uses NLI clustering
over Opus re-completions). Behavioral consistency is cheaper:
  - Uses Haiku (not Opus)
  - No NLI model
  - ROUGE-L variance as proxy for semantic agreement
  - Runs only when J-proxy zone is MEDIUM or HIGH and use_behavioral=True

The two probes are complementary:
  - SE probe: NLI-based semantic agreement, captures paraphrase equivalence
  - Behavioral: ROUGE-L variance, captures surface-level consistency

Either probe can downgrade zone to LOW → PRESERVE.

Cost estimate: 5 Haiku calls at max_tokens=150 ≈ $0.0003/turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_SAMPLES = 5           # number of re-completions per query
_MAX_SAMPLE_TOKENS = 150  # short completions are sufficient for variance check
_UNCERTAIN_THRESHOLD = 0.35   # variance above this → HIGH behavioral uncertainty
_CERTAIN_THRESHOLD   = 0.15   # variance below this → model is consistent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BehavioralResult:
    """
    Output of BehavioralConsistencyProbe.compute().

    Fields:
        consistency_score:  1.0 = perfectly consistent; 0.0 = maximally inconsistent
        variance:           mean pairwise ROUGE-L variance across N samples
        is_inconsistent:    True when variance > _UNCERTAIN_THRESHOLD
        samples:            The N Haiku completions (for debugging)
        fused_j:            J_final = w_j × j_input + (1-w_j) × consistency_score
        fused_zone:         Zone derived from fused_j using provided thresholds
    """
    consistency_score: float
    variance:          float
    is_inconsistent:   bool
    samples:           list[str] = field(default_factory=list)
    fused_j:           float = 0.0
    fused_zone:        str   = "MEDIUM"


# ---------------------------------------------------------------------------
# ROUGE-L (no external deps)
# ---------------------------------------------------------------------------

def _rouge_l(hyp: str, ref: str) -> float:
    """LCS-based ROUGE-L F1 between two strings (word-level)."""
    h_words = hyp.lower().split()
    r_words = ref.lower().split()
    if not h_words or not r_words:
        return 0.0
    m, n = len(h_words), len(r_words)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h_words[i - 1] == r_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall    = lcs / n
    return 2 * precision * recall / (precision + recall)


def _pairwise_variance(samples: list[str]) -> float:
    """
    Mean pairwise ROUGE-L distance among N samples.
    Distance = 1 − ROUGE-L (so high distance = high variance = inconsistent).
    """
    if len(samples) < 2:
        return 0.0
    pairs = [
        1.0 - _rouge_l(samples[i], samples[j])
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Main probe class
# ---------------------------------------------------------------------------

class BehavioralConsistencyProbe:
    """
    Tier 2 behavioral consistency signal.

    Usage:
        probe = BehavioralConsistencyProbe(api_key=...)
        result = probe.compute(messages=chat_history, client=anthropic_client)
        if result.is_inconsistent:
            # downgrade zone to LOW → PRESERVE
            pass
        fused_j = fuse_scores(j=0.72, consistency=result.consistency_score)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        n_samples:  int   = _N_SAMPLES,
        model:      str   = "claude-haiku-4-5-20251001",
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("pip install anthropic")
        self._client    = Anthropic(api_key=api_key) if api_key else None
        self._n         = n_samples
        self._model     = model

    def compute(
        self,
        messages:       list[dict],
        client:         Optional[object] = None,
        theta_high:     float = 0.70,
        theta_low:      float = 0.45,
        j_input:        float = 0.65,
        w_j:            float = 0.70,
    ) -> BehavioralResult:
        """
        Generate N Haiku re-completions of the last user message and measure
        pairwise ROUGE-L variance.

        Args:
            messages:   Full conversation history (same format as Anthropic API)
            client:     Anthropic client (uses self._client if None)
            theta_high: HIGH zone threshold for fused_zone calculation
            theta_low:  LOW zone threshold
            j_input:    J-proxy score from Tier 1 (used in fused score)
            w_j:        Weight for J-proxy in fusion (0.70 = J dominates)

        Returns:
            BehavioralResult with consistency_score, variance, fused_j, fused_zone
        """
        c = client or self._client
        if c is None:
            return BehavioralResult(
                consistency_score=1.0, variance=0.0, is_inconsistent=False,
                fused_j=j_input, fused_zone=self._zone(j_input, theta_high, theta_low),
            )

        # Ensure messages ends with a user turn for re-completion
        if not messages or messages[-1]["role"] != "user":
            return BehavioralResult(
                consistency_score=1.0, variance=0.0, is_inconsistent=False,
                fused_j=j_input, fused_zone=self._zone(j_input, theta_high, theta_low),
            )

        # Draw N Haiku re-completions. Each is an independent sample from the
        # model's distribution — high variance signals genuine uncertainty.
        samples: list[str] = []
        for _ in range(self._n):
            try:
                resp = c.messages.create(
                    model      = self._model,
                    messages   = messages,
                    max_tokens = _MAX_SAMPLE_TOKENS,
                )
                text = resp.content[0].text if resp.content else ""
                samples.append(text.strip())
            except Exception:
                pass

        if len(samples) < 2:
            return BehavioralResult(
                consistency_score=1.0, variance=0.0, is_inconsistent=False,
                samples=samples,
                fused_j=j_input, fused_zone=self._zone(j_input, theta_high, theta_low),
            )

        variance          = _pairwise_variance(samples)
        consistency_score = max(0.0, 1.0 - variance / max(variance, 1e-6) * variance)
        # Normalize: variance=0 → consistency=1.0; variance=1 → consistency=0.0
        consistency_score = round(1.0 - min(variance, 1.0), 4)
        is_inconsistent   = variance > _UNCERTAIN_THRESHOLD

        fused_j    = round(w_j * j_input + (1.0 - w_j) * consistency_score, 4)
        fused_zone = self._zone(fused_j, theta_high, theta_low)

        return BehavioralResult(
            consistency_score = consistency_score,
            variance          = round(variance, 4),
            is_inconsistent   = is_inconsistent,
            samples           = samples,
            fused_j           = fused_j,
            fused_zone        = fused_zone,
        )

    @staticmethod
    def _zone(j: float, theta_high: float, theta_low: float) -> str:
        if j >= theta_high:
            return "HIGH"
        if j >= theta_low:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Standalone fusion helper
# ---------------------------------------------------------------------------

def fuse_scores(
    j:           float,
    consistency: float,
    w_j:         float = 0.70,
    theta_high:  float = 0.70,
    theta_low:   float = 0.45,
) -> tuple[float, str]:
    """
    Fuse J-proxy (Tier 1) and behavioral consistency (Tier 2) into a single score.

    Returns (fused_j, zone) where:
        fused_j = w_j × j + (1 − w_j) × consistency
        zone    = "HIGH" / "MEDIUM" / "LOW" per thresholds

    Typical usage (inside ContextManager.chat()):
        if result.use_behavioral and cr.zone in ("MEDIUM", "HIGH"):
            beh = probe.compute(messages, client, j_input=cr.j_score)
            fused_j, fused_zone = fuse_scores(cr.j_score, beh.consistency_score)
            if fused_zone < cr.zone:
                cr = CredenceResult(j_score=fused_j, zone=fused_zone, ...)
    """
    fused = round(w_j * j + (1.0 - w_j) * consistency, 4)
    if fused >= theta_high:
        zone = "HIGH"
    elif fused >= theta_low:
        zone = "MEDIUM"
    else:
        zone = "LOW"
    return fused, zone

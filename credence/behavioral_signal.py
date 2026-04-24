"""
credence/behavioral_signal.py
=========================
Tier 2 behavioral consistency signal.

Generates N independent completions for a (prompt, context) pair using
Haiku and measures pairwise ROUGE-L variance. Low variance = the model
produces consistent answers = high behavioral confidence. High variance =
the model produces different answers each time = uncertain domain.

This captures a different failure mode than the Tier 1 linguistic J-score:
a confidently-stated wrong answer still scores HIGH on linguistic factors.
Behavioral consistency penalises those cases because repeated sampling of
an uncertain claim produces divergent outputs.

Inspired by: Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation" (ICLR 2023).

Cost: ~N × ~200 tokens × Haiku input rate (~$0.80/1M) ≈ $0.0002 per call at N=5.
Latency: N parallel Haiku calls ≈ 300–500 ms.

Usage:
    from credence.behavioral_signal import behavioral_consistency
    score = behavioral_consistency(
        prompt="What rate limit did we establish?",
        context="We discussed 100 req/min earlier but it might be 50.",
        n=5,
    )
    # score ∈ [0, 1] — higher = more consistent = more confident
"""

import os
import re
from itertools import combinations
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_DEFAULT_N   = 5
_MAX_TOKENS  = 150


# ---------------------------------------------------------------------------
# ROUGE-L (local, no external deps)
# ---------------------------------------------------------------------------

def _lcs_length(a: list, b: list) -> int:
    """LCS length via DP."""
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            temp = dp[j + 1]
            dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
            prev = temp
    return dp[n]


def _rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 between two strings."""
    h_tok = hyp.lower().split()
    r_tok = ref.lower().split()
    if not h_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(h_tok, r_tok)
    precision = lcs / len(h_tok)
    recall    = lcs / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _pairwise_mean_rouge(texts: list[str]) -> float:
    """Mean pairwise ROUGE-L over all pairs in texts."""
    pairs = list(combinations(range(len(texts)), 2))
    if not pairs:
        return 1.0
    total = sum(_rouge_l(texts[i], texts[j]) for i, j in pairs)
    return total / len(pairs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def behavioral_consistency(
    prompt: str,
    context: str,
    n: int = _DEFAULT_N,
    client: Optional["anthropic.Anthropic"] = None,
    temperature: float = 0.8,
) -> float:
    """
    Measure behavioral consistency of a model's answer to (prompt, context).

    Generates N independent completions at temperature > 0 and computes
    mean pairwise ROUGE-L. Returns a score ∈ [0, 1]:
      1.0  — all N completions identical (maximum consistency)
      0.0  — all N completions completely different (maximum uncertainty)

    Falls back to 0.5 (neutral) if API is unavailable or all calls fail.

    Args:
        prompt:      The question / user turn to answer
        context:     The conversation context (recent history as plain text)
        n:           Number of independent samples (default 5)
        client:      anthropic.Anthropic client (created from env if None)
        temperature: Sampling temperature (default 0.8 — must be > 0 for variance)

    Returns:
        Consistency score ∈ [0.0, 1.0]
    """
    if not _ANTHROPIC_AVAILABLE:
        return 0.5

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return 0.5
        client = anthropic.Anthropic(api_key=api_key)

    system = (
        "You are a helpful assistant. Answer the user's question based on "
        "the conversation context provided. Be concise (1-3 sentences)."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"

    completions: list[str] = []
    for _ in range(n):
        try:
            resp = client.messages.create(
                model=_MODEL_HAIKU,
                max_tokens=_MAX_TOKENS,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text.strip() if resp.content else ""
            if text:
                completions.append(text)
        except Exception:
            continue

    if len(completions) < 2:
        return 0.5

    return _pairwise_mean_rouge(completions)


def behavioral_consistency_batch(
    items: list[dict],
    client: Optional["anthropic.Anthropic"] = None,
    n: int = _DEFAULT_N,
) -> list[float]:
    """
    Batch version. Each item: {"prompt": str, "context": str}.
    Returns list of consistency scores in same order.
    """
    return [
        behavioral_consistency(
            prompt=item["prompt"],
            context=item["context"],
            n=n,
            client=client,
        )
        for item in items
    ]


def fuse_scores(j_score: float, consistency: float, w_j: float = 0.70) -> float:
    """
    Fuse Tier 1 J-score and Tier 2 behavioral consistency into a final score.

    Default weights: 70% linguistic J, 30% behavioral consistency.
    Both inputs ∈ [0, 1]. Output ∈ [0, 1].
    """
    w_b = 1.0 - w_j
    return round(w_j * j_score + w_b * consistency, 4)

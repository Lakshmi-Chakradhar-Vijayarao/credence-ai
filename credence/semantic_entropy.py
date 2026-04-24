"""
credence/semantic_entropy.py
============================
Semantic Entropy Probe — Tier 2 signal for uncertain turns.

Implementation of Kuhn et al. 2023 (Semantic Entropy) at the API surface:
  1. Generate N=3 independent completions of the same query
  2. Use Haiku as NLI judge — pairwise YES/NO "same factual claim?"
  3. Build equivalence clusters via connected components
  4. Compute Shannon entropy over cluster probability distribution, normalized to [0,1]

When to use:
  - Any MEDIUM or HIGH-zone turn when use_semantic_entropy=True
  - Ghost constraints (implicit uncertainty, HIGH-J) are the primary target:
    they bypass the faithfulness probe but show high inter-sample variance

Routing logic:
  - zone == MEDIUM or HIGH → SE probe fires (covers ghost constraints + borderline turns)
  - zone == LOW            → preserve unconditionally (already flagged uncertain by J)

Cost: 3 generation calls + up to 3 NLI calls (~$0.0005/probe with Haiku).
Falls back to ROUGE-L if NLI calls fail.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

_SE_ENTROPY_THRESHOLD     = 0.40   # normalized Shannon entropy above this → uncertain
_SE_N_SAMPLES             = 3      # number of re-completions
_SE_NLI_MAX_TOKENS        = 3      # YES/NO answer only
_SE_SAMPLE_MAX_TOKENS     = 60     # short re-completions for variance signal
_SE_FALLBACK_THRESHOLD    = 0.40   # ROUGE-L fallback: 1 - mean_similarity above this → uncertain


# ---------------------------------------------------------------------------
# ROUGE-L (fallback, no external deps)
# ---------------------------------------------------------------------------

def _rouge_l(a: str, b: str) -> float:
    """LCS-based ROUGE-L F1. Used as fallback when NLI is unavailable."""
    ta = a.lower().split()
    tb = b.lower().split()
    if not ta or not tb:
        return 0.0
    m, n = len(ta), len(tb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ta[i-1] == tb[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SemanticEntropyResult:
    entropy_score:   float          # 0.0 (certain, all samples agree) → 1.0 (uncertain)
    is_uncertain:    bool           # entropy_score > threshold
    mean_similarity: float          # mean pairwise similarity (ROUGE-L or NLI-agreement rate)
    n_samples:       int
    samples:         list[str]  = field(default_factory=list)
    reasoning:       str        = ""
    # NLI-specific fields (populated when method == "nli")
    method:          str        = "rouge_l"   # "nli" | "rouge_l" | "fallback"
    n_clusters:      int        = 0
    clusters:        list       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class SemanticEntropyProbe:
    """
    Measures whether the model's claim space is concentrated (confident) or
    diffuse (uncertain) by comparing N independent short re-completions.

    Primary method: NLI-based semantic clustering (Kuhn et al. 2023)
    Fallback method: ROUGE-L pairwise variance

    Usage:
        probe = SemanticEntropyProbe()
        result = probe.compute(messages_so_far, anthropic_client)
        if result.is_uncertain:
            # Override J-score → PRESERVE this turn
    """

    def __init__(
        self,
        threshold: float = _SE_ENTROPY_THRESHOLD,
        n_samples: int   = _SE_N_SAMPLES,
        model:     str   = "claude-haiku-4-5-20251001",
    ):
        self.threshold = threshold
        self.n_samples = n_samples
        self.model     = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        messages:   list[dict],
        client,
        max_tokens: int = _SE_SAMPLE_MAX_TOKENS,
    ) -> SemanticEntropyResult:
        """
        Generate N short re-completions of the last user message.
        Cluster by NLI semantic equivalence, compute Shannon entropy.
        High entropy → uncertain → PRESERVE.
        Falls back to ROUGE-L if NLI calls fail.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if not last_user:
            return SemanticEntropyResult(
                entropy_score=0.0, is_uncertain=False, mean_similarity=1.0,
                n_samples=0, reasoning="no user message", method="nli",
            )

        # Trim history: exclude last assistant turn so each sample is an
        # independent first response to the last user message
        sample_messages = [m for m in messages if not (
            m == messages[-1] and m["role"] == "assistant"
        )]

        samples = self._generate_samples(client, sample_messages, max_tokens)

        if len(samples) < 2:
            return SemanticEntropyResult(
                entropy_score=0.0, is_uncertain=False, mean_similarity=1.0,
                n_samples=len(samples), samples=samples,
                reasoning="insufficient samples generated", method="nli",
            )

        # Try NLI-based semantic entropy first
        try:
            return self._nli_entropy(samples, client)
        except Exception:
            # Fall back to ROUGE-L if NLI fails
            return self._rouge_entropy(samples)

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate_samples(
        self,
        client,
        messages:   list[dict],
        max_tokens: int,
    ) -> list[str]:
        """Generate N independent short completions of the query."""
        samples: list[str] = []
        for _ in range(self.n_samples):
            try:
                resp = client.messages.create(
                    model      = self.model,
                    messages   = messages,
                    max_tokens = max_tokens,
                    system     = (
                        "Answer in one direct sentence. "
                        "State the specific value or claim as precisely as you can."
                    ),
                )
                text = resp.content[0].text.strip() if resp.content else ""
                if text:
                    samples.append(text)
            except Exception:
                pass
        return samples

    # ------------------------------------------------------------------
    # NLI-based semantic entropy (primary)
    # ------------------------------------------------------------------

    def _semantic_equiv(self, r1: str, r2: str, client) -> bool:
        """
        Ask Haiku: do these two answers make the same factual claim?
        Returns True if semantically equivalent (YES), False otherwise (NO).
        Single-token answer: YES or NO.
        """
        resp = client.messages.create(
            model    = self.model,
            messages = [{
                "role":    "user",
                "content": (
                    "Do these two answers make the same factual claim? "
                    "Reply YES or NO only.\n\n"
                    f"A: {r1[:300]}\n"
                    f"B: {r2[:300]}"
                ),
            }],
            max_tokens = _SE_NLI_MAX_TOKENS,
        )
        answer = resp.content[0].text.strip().upper() if resp.content else "NO"
        return answer.startswith("Y")

    def _cluster_by_equivalence(
        self,
        samples: list[str],
        client,
    ) -> list[list[str]]:
        """
        Build equivalence clusters via union-find over pairwise NLI judgements.

        Two samples in the same cluster → they express the same factual claim.
        Distinct clusters → model gave genuinely different answers → uncertain.
        """
        n = len(samples)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Pairwise NLI — O(N²) but N=3 so only 3 calls max
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    if self._semantic_equiv(samples[i], samples[j], client):
                        union(i, j)
                except Exception:
                    # If NLI fails for a pair, treat as NOT equivalent (conservative)
                    pass

        # Collect clusters
        from collections import defaultdict
        cluster_map: dict[int, list[str]] = defaultdict(list)
        for i in range(n):
            cluster_map[find(i)].append(samples[i])
        return list(cluster_map.values())

    @staticmethod
    def _shannon_entropy_normalized(clusters: list[list[str]], n_total: int) -> float:
        """
        Compute Shannon entropy over cluster sizes, normalized by log2(N).

        H = -Σ p_i * log2(p_i)   where p_i = cluster_size / n_total
        H_normalized = H / log2(N)   → [0, 1]

        H_normalized == 0: all samples in one cluster (model is certain)
        H_normalized == 1: each sample in its own cluster (maximally uncertain)
        """
        import math
        if n_total < 2 or len(clusters) <= 1:
            return 0.0
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n_total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize by log2(N) — max possible entropy when all clusters size 1
        max_entropy = math.log2(n_total)
        return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    def _nli_entropy(self, samples: list[str], client) -> SemanticEntropyResult:
        """
        Full NLI-based semantic entropy computation.
        Raises on any hard failure — caller falls back to ROUGE-L.
        """
        clusters = self._cluster_by_equivalence(samples, client)
        n_total  = len(samples)

        entropy_score = self._shannon_entropy_normalized(clusters, n_total)
        is_uncertain  = entropy_score > self.threshold

        # Mean agreement rate: fraction of pairs judged equivalent
        n_pairs       = n_total * (n_total - 1) // 2
        same_cluster_pairs = sum(
            len(c) * (len(c) - 1) // 2 for c in clusters
        )
        mean_similarity = round(same_cluster_pairs / n_pairs, 4) if n_pairs > 0 else 1.0

        return SemanticEntropyResult(
            entropy_score   = entropy_score,
            is_uncertain    = is_uncertain,
            mean_similarity = mean_similarity,
            n_samples       = n_total,
            samples         = samples,
            reasoning       = (
                f"{n_total} samples → {len(clusters)} cluster(s), "
                f"H_norm={entropy_score:.3f} "
                f"({'uncertain→PRESERVE' if is_uncertain else 'certain'})"
            ),
            method     = "nli",
            n_clusters = len(clusters),
            clusters   = [list(c) for c in clusters],
        )

    # ------------------------------------------------------------------
    # ROUGE-L fallback (original implementation)
    # ------------------------------------------------------------------

    def _rouge_entropy(self, samples: list[str]) -> SemanticEntropyResult:
        """
        Fallback: pairwise ROUGE-L variance as entropy proxy.
        Used when NLI calls fail (network error, rate limit, etc.).
        """
        pair_scores: list[float] = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                pair_scores.append(_rouge_l(samples[i], samples[j]))

        mean_sim      = sum(pair_scores) / len(pair_scores)
        entropy_score = round(1.0 - mean_sim, 4)
        is_uncertain  = entropy_score > _SE_FALLBACK_THRESHOLD

        return SemanticEntropyResult(
            entropy_score   = entropy_score,
            is_uncertain    = is_uncertain,
            mean_similarity = round(mean_sim, 4),
            n_samples       = len(samples),
            samples         = samples,
            reasoning       = (
                f"{len(samples)} samples (ROUGE-L fallback), "
                f"mean_sim={mean_sim:.3f}, entropy={entropy_score:.3f} "
                f"({'uncertain→PRESERVE' if is_uncertain else 'certain'})"
            ),
            method     = "rouge_l",
            n_clusters = 0,
            clusters   = [],
        )

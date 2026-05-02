"""
evals/kv_cache/metrics.py
=========================
KV-cache experiment metrics.

All functions are pure Python + numpy. No API calls, no model loading.
Scoring is deterministic from text inputs.

Metrics
-------
eqlr_token(answer, qualifier_fragments) -> bool
    True = qualifier LOST (bad). False = qualifier preserved (good).

fcr(answer, value_fragments, qualifier_fragments) -> bool
    True = False Certainty (value present, qualifier absent). Worst outcome.

eqlr_semantic(answer, reference_answer, nli_model=None) -> float
    Semantic EQLR via NLI entailment. Returns probability that uncertainty
    is preserved. Falls back to token-overlap proxy if nli_model is None.

qar(attention_matrix, qualifier_positions, value_positions) -> float
    Qualifier Attention Ratio = mean attention on qualifier tokens /
    mean attention on value tokens. Averaged across all layers and heads.
    < 1.0 means qualifiers get less attention (at risk of eviction).

score_scenario(scenario, answer, attention_matrix=None, nli_model=None) -> dict
    Scores one scenario. Returns dict with all metrics.

aggregate(scores) -> dict
    Aggregates list of score dicts into EQLR, FCR, Ghost_FCR, mean_QAR.
    Includes 95% bootstrap CI on each metric.

Usage
-----
    from evals.kv_cache.metrics import score_scenario, aggregate

    scores = [score_scenario(s, answer) for s, answer in results]
    summary = aggregate(scores)
    print(summary["eqlr_token"], summary["fcr"])
"""

from __future__ import annotations

import re
import random
from typing import Any, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Token-level utilities
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fragment_present(text: str, fragments: list[str]) -> bool:
    """Return True if ANY fragment from the list appears in text (case-insensitive)."""
    low = _normalise(text)
    return any(_normalise(f) in low for f in fragments if f)


def _fragment_count(text: str, fragments: list[str]) -> int:
    """Count how many distinct fragments appear in text."""
    low = _normalise(text)
    return sum(1 for f in fragments if f and _normalise(f) in low)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def eqlr_token(answer: str, qualifier_fragments: list[str]) -> bool:
    """
    Epistemic Qualifier Loss Rate — token level.

    Parameters
    ----------
    answer : str
        The model's generated answer.
    qualifier_fragments : list[str]
        Substrings that indicate uncertainty was preserved (e.g., ["think",
        "might", "approximately", "unverified"]).

    Returns
    -------
    bool
        True  = qualifier LOST (bad outcome — uncertainty stripped).
        False = qualifier PRESERVED (good outcome — at least one marker present).

    Notes
    -----
    Uses case-insensitive substring match. A single qualifying fragment is
    sufficient to count as "preserved" — this is intentionally permissive to
    avoid false positives.
    """
    if not qualifier_fragments:
        return False  # no fragments to check — cannot determine loss
    return not _fragment_present(answer, qualifier_fragments)


def fcr(answer: str, value_fragments: list[str], qualifier_fragments: list[str]) -> bool:
    """
    False Certainty Rate — per-scenario binary.

    Parameters
    ----------
    answer : str
        The model's generated answer.
    value_fragments : list[str]
        Substrings indicating the uncertain value was recalled (e.g., ["50", "req"]).
    qualifier_fragments : list[str]
        Substrings indicating uncertainty was preserved.

    Returns
    -------
    bool
        True  = False Certainty (value recalled, qualifier ABSENT). Worst outcome.
        False = either value missing (irrelevant) or qualifier present (safe).

    Notes
    -----
    FCR is the headline harm metric. It requires BOTH:
      - The uncertain value IS present in the answer (value recalled)
      - The uncertainty qualifier is NOT present (qualifier stripped)
    This specifically captures the failure mode where the model states a
    previously-uncertain fact as confirmed truth.
    """
    value_present = _fragment_present(answer, value_fragments)
    qualifier_present = _fragment_present(answer, qualifier_fragments)
    return value_present and (not qualifier_present)


# ---------------------------------------------------------------------------
# Semantic Equivalence (SEE Protocol)
# ---------------------------------------------------------------------------

QUALIFIER_CLUSTERS = {
    "approximate": {
        "approximately", "roughly", "around", "about", "estimated", "nearly",
        "circa", "estimate", "rounded", "roughly", "~"
    },
    "probabilistic": {
        "might", "could", "possibly", "maybe", "perhaps", "potential",
        "likely", "probably", "unlikely", "chance", "uncertain"
    },
    "attributive": {
        "according", "stated", "heard", "reported", "vendor", "documentation",
        "docs", "someone", "team", "claims", "unconfirmed", "hearsay", "memo"
    },
    "preliminary": {
        "preliminary", "early", "initial", "draft", "first pass", "provisional",
        "interim", "working", "snapshot", "tentative"
    }
}


def _get_qualifier_categories(text: str) -> set[str]:
    """Identify which epistemic categories are present in the text."""
    low = _normalise(text)
    categories = set()
    for cat, words in QUALIFIER_CLUSTERS.items():
        if any(w in low for w in words):
            categories.add(cat)
    return categories


def eqlr_semantic(
    answer: str,
    reference_answer: str,
    nli_model: Any = None,
) -> float:
    """
    Semantic EQLR via NLI entailment or Fuzzy-Semantic fallback.

    Parameters
    ----------
    answer : str
        The model's generated answer to score.
    reference_answer : str
        The reference answer that correctly preserves uncertainty.
    nli_model : optional
        An NLI model with a predict(premise, hypothesis) -> dict interface.
        Must return {'entailment': float, 'contradiction': float}.
        If None, falls back to fuzzy-semantic matching.

    Returns
    -------
    float [0, 1]
        Score representing confidence that uncertainty is preserved.
        1.0 = highly likely preserved. 0.0 = likely lost.

    Notes
    -----
    SEE Protocol Implementation:
    1. If NLI is available:
       - Score = answer entails reference_answer.
       - Confident answers ("It is 50") do NOT entail hedged references ("It is ~50").
       - If contradiction > 0.5 (False Certainty), score = 0.0.

    2. If NLI is absent (Fuzzy Fallback):
       - Checks if the answer contains qualifiers from the same semantic
         clusters as the reference.
       - Returns 1.0 if category match found, else falls back to Jaccard
         content-word similarity [0.1, 0.9].
    """
    # --- 1. NLI Scorer (Preferred) ---
    if nli_model is not None:
        try:
            # Protocol: Answer |= Reference
            probs = nli_model.predict(premise=answer, hypothesis=reference_answer)
            entailment = float(probs.get("entailment", 0.0))
            contradiction = float(probs.get("contradiction", 0.0))

            if contradiction > 0.5:
                return 0.0
            return entailment
        except Exception:
            pass  # Fall through to fuzzy fallback

    # --- 2. Fuzzy-Semantic Fallback ---
    ref_cats = _get_qualifier_categories(reference_answer)
    ans_cats = _get_qualifier_categories(answer)

    # If the answer uses a qualifier from the same semantic cluster, it counts as preserved
    if ref_cats and (ref_cats & ans_cats):
        return 1.0

    # --- 3. Content-word Jaccard (Last Resort) ---
    def _content_words(text: str) -> set:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stopwords = {
            "that", "this", "with", "have", "been", "from", "will",
            "they", "were", "when", "what", "which", "into", "then",
            "than", "your", "also", "some", "more", "should", "would",
            "could", "about", "just", "used", "make", "made", "said",
            "from", "after", "before", "once", "once", "only", "were"
        }
        return {w for w in words if w not in stopwords}

    answer_words = _content_words(answer)
    ref_words = _content_words(reference_answer)

    if not ref_words:
        return 0.5

    intersection = len(answer_words & ref_words)
    union = len(answer_words | ref_words)

    jaccard = intersection / union if union > 0 else 0.0
    return round(0.1 + 0.8 * jaccard, 4)


def qar(
    attention_matrix: Any,  # np.ndarray shape (n_layers, n_heads, seq_len, seq_len)
    qualifier_positions: list[int],
    value_positions: list[int],
) -> float:
    """
    Qualifier Attention Ratio.

    Parameters
    ----------
    attention_matrix : np.ndarray, shape (n_layers, n_heads, seq_len, seq_len)
        The full attention weight tensor from a model forward pass.
        attention_matrix[layer, head, query_pos, key_pos] = attention weight
        from query token at query_pos to key token at key_pos.
        Values are normalised to sum to 1.0 per row (standard softmax attention).
    qualifier_positions : list[int]
        Token indices corresponding to the qualifier tokens in the sequence
        (e.g., token positions for "approximately", "I", "think").
    value_positions : list[int]
        Token indices corresponding to the value tokens in the sequence
        (e.g., token positions for "50", "req", "/", "min").

    Returns
    -------
    float
        QAR = mean attention weight on qualifier positions /
              mean attention weight on value positions.
        Averaged across all layers and all heads.

        QAR < 1.0 means qualifiers receive less attention than values on average.
        QAR = 1.0 means equal attention.
        QAR > 1.0 means qualifiers receive MORE attention than values (atypical).

        Returns -1.0 if attention_matrix is None or positions are empty.

    Notes
    -----
    Attention is extracted from the FINAL generated token's attention row
    (i.e., attention_matrix[:, :, -1, :] for the last query position).
    This captures what the model attends to when generating the callback answer.

    For KV-eviction analysis, the positions that receive low attention in
    this slice are the candidates for eviction. If qualifier positions
    consistently have lower attention than value positions across the dataset,
    the eviction hypothesis is supported.

    The function operates on the full attention matrix (all query positions)
    by averaging across all rows. To restrict to the final token only, pass
    attention_matrix[:, :, -1:, :] (sliced before calling).
    """
    if not _HAS_NUMPY:
        return -1.0
    if attention_matrix is None:
        return -1.0
    if not qualifier_positions or not value_positions:
        return -1.0

    try:
        attn = np.array(attention_matrix, dtype=np.float32)
        # attn shape: (n_layers, n_heads, seq_len, seq_len)
        # Average over layers and heads to get (seq_len, seq_len)
        mean_attn = attn.mean(axis=(0, 1))  # shape: (seq_len, seq_len)

        # Average attention RECEIVED at each key position
        # (column-wise average over all query positions)
        attn_received = mean_attn.mean(axis=0)  # shape: (seq_len,)

        seq_len = attn_received.shape[0]

        qual_pos = [p for p in qualifier_positions if 0 <= p < seq_len]
        val_pos  = [p for p in value_positions     if 0 <= p < seq_len]

        if not qual_pos or not val_pos:
            return -1.0

        qual_mean = float(attn_received[qual_pos].mean())
        val_mean  = float(attn_received[val_pos].mean())

        if val_mean < 1e-10:
            return -1.0  # avoid division by near-zero

        return round(qual_mean / val_mean, 6)

    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Per-scenario scorer
# ---------------------------------------------------------------------------

def score_scenario(
    scenario: dict,
    answer: str,
    attention_matrix: Any = None,
    nli_model: Any = None,
) -> dict:
    """
    Score a single scenario.

    Parameters
    ----------
    scenario : dict
        A scenario dict with keys: scenario_id, domain, qualifier_type,
        value_fragments, qualifier_fragments, reference_answer, and optionally
        qualifier_positions, value_positions (for QAR computation).
    answer : str
        The model's generated response to the callback question.
    attention_matrix : np.ndarray, optional
        Raw attention tensor from the model forward pass.
        Shape: (n_layers, n_heads, seq_len, seq_len).
        If None, QAR is reported as -1.0.
    nli_model : optional
        NLI model for semantic EQLR. If None, uses token-overlap proxy.

    Returns
    -------
    dict with keys:
        scenario_id       : str
        domain            : str
        qualifier_type    : str
        answer            : str (truncated to 500 chars)
        eqlr_token        : bool  (True = lost)
        fcr               : bool  (True = false certainty)
        eqlr_semantic     : float [0, 1] (>0.5 = preserved)
        qar               : float (< 1.0 = qualifiers underattended)
        value_recalled    : bool
        qualifier_present : bool
        is_ghost          : bool  (True if from ghost_scenarios.json)
    """
    scenario_id     = scenario.get("scenario_id", scenario.get("id", "unknown"))
    domain          = scenario.get("domain", "")
    qualifier_type  = scenario.get("qualifier_type", "")
    value_frags     = scenario.get("value_fragments", [])
    qual_frags      = scenario.get("qualifier_fragments", [])
    reference       = scenario.get("reference_answer", scenario.get("text", ""))
    qual_positions  = scenario.get("qualifier_positions", [])
    val_positions   = scenario.get("value_positions", [])

    # Ghost scenarios use a single qualifier_fragment (not a list)
    if isinstance(qual_frags, str):
        qual_frags = [qual_frags]
    if isinstance(value_frags, str):
        value_frags = [value_frags]

    # Ghost scenarios: value_fragment may be a single token string
    if not isinstance(value_frags, list):
        value_frags = [str(value_frags)]

    value_recalled    = _fragment_present(answer, value_frags)
    qualifier_present = _fragment_present(answer, qual_frags)

    eqlr_tok  = eqlr_token(answer, qual_frags)
    fcr_score = fcr(answer, value_frags, qual_frags)
    sem_score = eqlr_semantic(answer, reference, nli_model)

    qar_score = qar(
        attention_matrix,
        qualifier_positions=qual_positions,
        value_positions=val_positions,
    ) if (attention_matrix is not None and qual_positions) else -1.0

    # Detect ghost scenario: ghost scenarios have 'id' starting with 'ghost-'
    is_ghost = str(scenario_id).startswith("ghost-")

    return {
        "scenario_id":       scenario_id,
        "domain":            domain,
        "qualifier_type":    qualifier_type,
        "answer":            answer[:500],
        "eqlr_token":        eqlr_tok,
        "fcr":               fcr_score,
        "eqlr_semantic":     sem_score,
        "qar":               qar_score,
        "value_recalled":    value_recalled,
        "qualifier_present": qualifier_present,
        "is_ghost":          is_ghost,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : list[float]
        Sample values (e.g., list of 0/1 booleans cast to float, or QAR values).
    n_boot : int
        Number of bootstrap resamples (default 2000, matches evals/benchmark.py).
    ci : float
        Confidence level (default 0.95 for 95% CI).
    rng_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper bounds of the bootstrap CI.

    Notes
    -----
    Algorithm: percentile bootstrap (not BCa). Matches the implementation
    pattern used in evals/benchmark.py and evals/e6_repeated.py.
    Falls back to (mean, mean) if n < 2 to avoid degenerate results.
    """
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else 0.0
        return (m, m)

    rng = random.Random(rng_seed)

    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1 - alpha / 2)) - 1
    hi_idx = min(hi_idx, n_boot - 1)

    return (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(scores: list[dict], n_boot: int = 2000) -> dict:
    """
    Aggregate a list of per-scenario score dicts into summary statistics.

    Parameters
    ----------
    scores : list[dict]
        Output from score_scenario() calls.
    n_boot : int
        Bootstrap resamples for CI computation (default 2000).

    Returns
    -------
    dict with keys:
        n               : int   total scenarios
        n_ghost         : int   ghost scenarios only
        eqlr_token      : float mean EQLR-Token (fraction of qualifiers lost)
        eqlr_token_ci   : [lo, hi]
        fcr             : float mean FCR
        fcr_ci          : [lo, hi]
        ghost_fcr       : float FCR on ghost scenarios only
        ghost_fcr_ci    : [lo, hi]
        eqlr_semantic   : float mean semantic EQLR proxy score
        eqlr_semantic_ci: [lo, hi]
        mean_qar        : float mean QAR (qualifier attention ratio)
        mean_qar_ci     : [lo, hi]
        value_recall    : float fraction of scenarios where value was recalled
        qualifier_rate  : float fraction of scenarios where qualifier was present
        by_domain       : dict  per-domain EQLR_token and FCR
        by_qualifier_type: dict per-qualifier-type EQLR_token and FCR

    Notes
    -----
    QAR is only included in mean_qar if qar != -1.0 (i.e., attention matrix
    was available). If no scenarios have QAR, mean_qar = -1.0.

    Ghost FCR is computed on scenarios where is_ghost = True. If no ghost
    scenarios are present, ghost_fcr = -1.0.
    """
    if not scores:
        return {
            "n": 0, "n_ghost": 0,
            "eqlr_token": 0.0, "eqlr_token_ci": [0.0, 0.0],
            "fcr": 0.0, "fcr_ci": [0.0, 0.0],
            "ghost_fcr": -1.0, "ghost_fcr_ci": [-1.0, -1.0],
            "eqlr_semantic": 0.5, "eqlr_semantic_ci": [0.5, 0.5],
            "mean_qar": -1.0, "mean_qar_ci": [-1.0, -1.0],
            "value_recall": 0.0, "qualifier_rate": 0.0,
            "by_domain": {}, "by_qualifier_type": {},
        }

    n = len(scores)

    eqlr_vals = [1.0 if s["eqlr_token"] else 0.0 for s in scores]
    fcr_vals  = [1.0 if s["fcr"]        else 0.0 for s in scores]
    sem_vals  = [s["eqlr_semantic"]                 for s in scores]
    vr_vals   = [1.0 if s["value_recalled"]    else 0.0 for s in scores]
    qp_vals   = [1.0 if s["qualifier_present"] else 0.0 for s in scores]

    qar_vals  = [s["qar"] for s in scores if s.get("qar", -1.0) >= 0.0]
    ghost_scores = [s for s in scores if s.get("is_ghost", False)]

    eqlr_mean = sum(eqlr_vals) / n
    fcr_mean  = sum(fcr_vals)  / n
    sem_mean  = sum(sem_vals)  / n
    vr_mean   = sum(vr_vals)   / n
    qp_mean   = sum(qp_vals)   / n

    eqlr_ci = bootstrap_ci(eqlr_vals, n_boot=n_boot)
    fcr_ci  = bootstrap_ci(fcr_vals,  n_boot=n_boot)
    sem_ci  = bootstrap_ci(sem_vals,  n_boot=n_boot)

    if qar_vals:
        qar_mean = sum(qar_vals) / len(qar_vals)
        qar_ci   = bootstrap_ci(qar_vals, n_boot=n_boot)
    else:
        qar_mean = -1.0
        qar_ci   = (-1.0, -1.0)

    if ghost_scores:
        g_fcr_vals = [1.0 if s["fcr"] else 0.0 for s in ghost_scores]
        ghost_fcr  = sum(g_fcr_vals) / len(g_fcr_vals)
        ghost_ci   = bootstrap_ci(g_fcr_vals, n_boot=n_boot)
    else:
        ghost_fcr = -1.0
        ghost_ci  = (-1.0, -1.0)

    # Per-domain breakdown
    domains: dict[str, list] = {}
    for s in scores:
        d = s.get("domain", "unknown")
        if d not in domains:
            domains[d] = []
        domains[d].append(s)

    by_domain = {}
    for d, ds in sorted(domains.items()):
        d_eqlr = sum(1.0 if s["eqlr_token"] else 0.0 for s in ds) / len(ds)
        d_fcr  = sum(1.0 if s["fcr"]        else 0.0 for s in ds) / len(ds)
        by_domain[d] = {
            "n":          len(ds),
            "eqlr_token": round(d_eqlr, 3),
            "fcr":        round(d_fcr, 3),
        }

    # Per-qualifier-type breakdown
    qtypes: dict[str, list] = {}
    for s in scores:
        qt = s.get("qualifier_type", "unknown")
        if qt not in qtypes:
            qtypes[qt] = []
        qtypes[qt].append(s)

    by_qualifier_type = {}
    for qt, qs in sorted(qtypes.items()):
        qt_eqlr = sum(1.0 if s["eqlr_token"] else 0.0 for s in qs) / len(qs)
        qt_fcr  = sum(1.0 if s["fcr"]        else 0.0 for s in qs) / len(qs)
        qt_qar_vals = [s["qar"] for s in qs if s.get("qar", -1.0) >= 0.0]
        qt_qar  = sum(qt_qar_vals) / len(qt_qar_vals) if qt_qar_vals else -1.0
        by_qualifier_type[qt] = {
            "n":          len(qs),
            "eqlr_token": round(qt_eqlr, 3),
            "fcr":        round(qt_fcr, 3),
            "mean_qar":   round(qt_qar, 4) if qt_qar >= 0 else -1.0,
        }

    return {
        "n":                  n,
        "n_ghost":            len(ghost_scores),
        "eqlr_token":         round(eqlr_mean, 4),
        "eqlr_token_ci":      list(eqlr_ci),
        "fcr":                round(fcr_mean, 4),
        "fcr_ci":             list(fcr_ci),
        "ghost_fcr":          round(ghost_fcr, 4) if ghost_fcr >= 0 else -1.0,
        "ghost_fcr_ci":       list(ghost_ci),
        "eqlr_semantic":      round(sem_mean, 4),
        "eqlr_semantic_ci":   list(sem_ci),
        "mean_qar":           round(qar_mean, 6) if qar_mean >= 0 else -1.0,
        "mean_qar_ci":        [round(v, 6) for v in qar_ci] if qar_mean >= 0 else [-1.0, -1.0],
        "value_recall":       round(vr_mean, 4),
        "qualifier_rate":     round(qp_mean, 4),
        "by_domain":          by_domain,
        "by_qualifier_type":  by_qualifier_type,
    }


# ---------------------------------------------------------------------------
# Self-test (no external dependencies)
# ---------------------------------------------------------------------------

def verify_metrics() -> None:
    """Quick sanity check — runs without any external packages."""
    # eqlr_token
    assert eqlr_token("The limit is 50 req/min.", ["think", "might", "approximately"]) is True
    assert eqlr_token("I think the limit is approximately 50.", ["think", "approximately"]) is False
    assert eqlr_token("The limit is 50.", []) is False

    # fcr
    assert fcr("The limit is 50 req/min.", ["50"], ["think", "approximately"]) is True
    assert fcr("I think the limit is 50.", ["50"], ["think"]) is False
    assert fcr("No data available.", ["50"], ["think"]) is False

    # eqlr_semantic (proxy mode, no nli_model)
    score = eqlr_semantic(
        "I think the limit is approximately 50 req/min.",
        "The rate limit is approximately 50 req/min — treat as an estimate.",
    )
    assert 0.0 <= score <= 1.0

    # qar (numpy required)
    if _HAS_NUMPY:
        import numpy as np_test
        attn = np_test.ones((4, 8, 20, 20), dtype=np_test.float32) / 20.0
        # Uniform attention → QAR should equal 1.0
        result = qar(attn, qualifier_positions=[5, 6], value_positions=[10, 11])
        assert abs(result - 1.0) < 1e-4, f"Expected ~1.0, got {result}"
        # No positions
        assert qar(attn, [], [1, 2]) == -1.0
        assert qar(None, [1], [2]) == -1.0
    else:
        assert qar(None, [1], [2]) == -1.0

    # score_scenario
    scenario = {
        "scenario_id": "test-001",
        "domain": "api",
        "qualifier_type": "estimate",
        "value_fragments": ["50"],
        "qualifier_fragments": ["think", "approximately"],
        "reference_answer": "Approximately 50, but unverified.",
    }
    scored = score_scenario(scenario, "The rate limit is 50 req/min.")
    assert scored["eqlr_token"] is True
    assert scored["fcr"] is True

    scored2 = score_scenario(scenario, "I think the rate limit is approximately 50.")
    assert scored2["eqlr_token"] is False
    assert scored2["fcr"] is False

    # aggregate
    scores = [
        score_scenario(scenario, "The rate limit is 50 req/min."),
        score_scenario(scenario, "I think the rate limit is approximately 50."),
    ]
    agg = aggregate(scores)
    assert agg["n"] == 2
    assert agg["eqlr_token"] == 0.5
    assert agg["fcr"] == 0.5

    # bootstrap_ci
    vals = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    lo, hi = bootstrap_ci(vals, n_boot=500, rng_seed=42)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0

    print("metrics.py: all self-tests passed.")


if __name__ == "__main__":
    verify_metrics()

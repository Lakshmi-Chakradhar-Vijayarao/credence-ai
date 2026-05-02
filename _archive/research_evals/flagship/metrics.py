"""
evals/flagship/metrics.py
================================
Metrics for the flagship Credence experiment.

Three core metrics:

1. propagation_rate  — fraction of post-seed steps where the model states
   an uncertain value as a definite fact (the headline "failure propagation" metric).
   Lower is better. Credence should drive this to zero.

2. constraint_recall — for each callback question, what fraction of required
   fragments appear in the answer? Measures whether uncertain constraints
   survived compression intact.

3. uncertainty_preserved — did the model preserve epistemic qualification
   (hedging language, uncertainty markers) in its callbacks? Checking for
   forbidden_fragments that would indicate false certainty.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CallbackResult:
    question: str
    answer: str
    required_fragments: list[str]
    forbidden_fragments: list[str]
    fragment_recall: float      # fraction of required_fragments found
    propagation_error: bool     # True if any forbidden_fragment found with certainty
    uncertainty_preserved: bool # True if answer does NOT trigger propagation error


@dataclass
class ScenarioResult:
    scenario_id: str
    condition: str
    callback_results: list[CallbackResult]
    mean_recall: float
    propagation_rate: float     # fraction of callbacks with propagation error
    chain_complete: bool        # True if all callbacks ≥ 0.60 recall AND propagation_rate == 0
    tokens_used: int
    tokens_saved: int = 0


# ---------------------------------------------------------------------------
# Fragment recall
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower, strip punctuation for fragment matching."""
    return re.sub(r'[^\w\s]', ' ', text.lower())


def fragment_recall(answer: str, required: list[str]) -> float:
    """Fraction of required fragments present in answer (case-insensitive)."""
    if not required:
        return 1.0
    norm = _normalise(answer)
    hits = sum(1 for frag in required if _normalise(frag) in norm)
    return hits / len(required)


def propagation_error(answer: str, forbidden: list[str]) -> bool:
    """
    True if any forbidden fragment appears in the answer as a certain-sounding claim.

    Forbidden fragments are phrased as confident statements (e.g. "the rate limit is 100").
    Their presence indicates the model has dropped the uncertainty qualifier and is now
    presenting an uncertain value as fact — epistemic propagation failure.
    """
    if not forbidden:
        return False
    norm = _normalise(answer)
    return any(_normalise(frag) in norm for frag in forbidden)


def evaluate_callback(
    question: str,
    answer: str,
    required_fragments: list[str],
    forbidden_fragments: list[str],
) -> CallbackResult:
    recall = fragment_recall(answer, required_fragments)
    prop_err = propagation_error(answer, forbidden_fragments)
    return CallbackResult(
        question=question,
        answer=answer,
        required_fragments=required_fragments,
        forbidden_fragments=forbidden_fragments,
        fragment_recall=recall,
        propagation_error=prop_err,
        uncertainty_preserved=not prop_err,
    )


# ---------------------------------------------------------------------------
# Scenario-level aggregation
# ---------------------------------------------------------------------------

def aggregate_scenario(
    scenario_id: str,
    condition: str,
    callback_results: list[CallbackResult],
    tokens_used: int,
    tokens_saved: int = 0,
) -> ScenarioResult:
    if not callback_results:
        return ScenarioResult(
            scenario_id=scenario_id,
            condition=condition,
            callback_results=[],
            mean_recall=0.0,
            propagation_rate=0.0,
            chain_complete=False,
            tokens_used=tokens_used,
            tokens_saved=tokens_saved,
        )

    recalls = [r.fragment_recall for r in callback_results]
    mean_recall = sum(recalls) / len(recalls)
    prop_errors = sum(1 for r in callback_results if r.propagation_error)
    propagation_rate = prop_errors / len(callback_results)

    chain_complete = (
        all(r >= 0.60 for r in recalls)
        and propagation_rate == 0.0
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        condition=condition,
        callback_results=callback_results,
        mean_recall=round(mean_recall, 3),
        propagation_rate=round(propagation_rate, 3),
        chain_complete=chain_complete,
        tokens_used=tokens_used,
        tokens_saved=tokens_saved,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Non-parametric bootstrap CI on the mean."""
    import random
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1.0 - ci) / 2 * n_boot)
    hi_idx = int((1.0 + ci) / 2 * n_boot)
    return (round(means[lo_idx], 3), round(means[min(hi_idx, n_boot - 1)], 3))


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_scenario_result(r: ScenarioResult):
    chain_str = "COMPLETE" if r.chain_complete else "BROKEN"
    print(f"  Scenario {r.scenario_id} [{r.condition}]: "
          f"recall={r.mean_recall:.3f}  "
          f"prop_rate={r.propagation_rate:.3f}  "
          f"chain={chain_str}  "
          f"tokens={r.tokens_used:,}")
    for cb in r.callback_results:
        err_flag = "  ⚠ PROPAGATION ERROR" if cb.propagation_error else ""
        print(f"    Q: {cb.question[:60]}…  recall={cb.fragment_recall:.2f}{err_flag}")

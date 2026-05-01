"""
training/epistemic_loss.py
==========================
Epistemic loss function for fine-tuning compression models.

Standard DPO loss optimises for generating the faithful summary over the
unfaithful one. The epistemic loss adds a penalty specifically for
*confidence inflation* — when the compressed output has a higher J-score
than the input conversation, meaning it became MORE assertive through
compression, which is exactly the FCR failure mode.

L = L_semantic + lambda * L_epistemic

L_epistemic = |j_output - j_input| + penalty_for_inflation
penalty_for_inflation = max(0, j_output - j_input)

  - |j_output - j_input| penalises any confidence change (both inflation and
    deflation).
  - max(0, j_output - j_input) adds an asymmetric extra penalty specifically
    for inflation (output MORE confident than input).

This means:
  - Inflation by +0.3:   |0.3| + max(0, 0.3) = 0.3 + 0.3 = 0.6
  - Deflation by -0.3:   |0.3| + max(0, -0.3) = 0.3 + 0.0 = 0.3
  - No change:           0.0

lambda starting value: 0.3
Hyperparameter sweep: [0.1, 0.3, 0.5, 1.0]

No API calls. Uses CredenceProxy (local J-score computation) only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.confidence_proxy import CredenceProxy

# ---------------------------------------------------------------------------
# Module-level proxy instance (shared, stateless)
# ---------------------------------------------------------------------------

_PROXY = CredenceProxy()

# Default regularisation weight
_DEFAULT_LAMBDA = 0.3

# Available lambda values for hyperparameter sweep
LAMBDA_SWEEP = [0.1, 0.3, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Core epistemic loss functions
# ---------------------------------------------------------------------------

def confidence_inflation(input_conf: float, output_conf: float) -> float:
    """
    Asymmetric penalty for confidence inflation.

    Returns the amount by which the compression INCREASED confidence.
    Zero for deflation or equal confidence.

    Args:
        input_conf:  J-score of the original input text
        output_conf: J-score of the compressed/summarised text

    Returns:
        float >= 0: the inflation penalty (positive = bad; 0 = no inflation)
    """
    return max(0.0, output_conf - input_conf)


def compute_epistemic_loss(
    input_text:  str,
    output_text: str,
    lambda_val:  float = _DEFAULT_LAMBDA,
) -> float:
    """
    Compute the epistemic loss for a single (input, output) pair.

    L_epistemic = |j_output - j_input| + max(0, j_output - j_input)

    The final scalar is:  lambda_val * L_epistemic

    Args:
        input_text:  the original conversation or text segment
        output_text: the compression/summary produced by the model
        lambda_val:  weighting factor for the epistemic loss term

    Returns:
        float: lambda * L_epistemic  (>= 0)
    """
    j_input  = _PROXY.compute(input_text).j_score
    j_output = _PROXY.compute(output_text).j_score

    abs_diff  = abs(j_output - j_input)
    inflation = confidence_inflation(j_input, j_output)

    l_epistemic = abs_diff + inflation
    return lambda_val * l_epistemic


def compute_batch_loss(
    input_texts:  list[str],
    output_texts: list[str],
    lambda_val:   float = _DEFAULT_LAMBDA,
) -> list[float]:
    """
    Compute epistemic loss for a batch of (input, output) pairs.

    Args:
        input_texts:  list of original texts
        output_texts: list of compressed/summarised texts (same length)
        lambda_val:   regularisation weight

    Returns:
        list[float]: per-pair epistemic loss values

    Raises:
        ValueError: if input and output lists differ in length
    """
    if len(input_texts) != len(output_texts):
        raise ValueError(
            f"input_texts ({len(input_texts)}) and output_texts "
            f"({len(output_texts)}) must have the same length"
        )

    j_inputs  = _PROXY.batch(input_texts)
    j_outputs = _PROXY.batch(output_texts)

    losses = []
    for ji, jo in zip(j_inputs, j_outputs):
        abs_diff  = abs(jo.j_score - ji.j_score)
        inflation = confidence_inflation(ji.j_score, jo.j_score)
        losses.append(lambda_val * (abs_diff + inflation))

    return losses


# ---------------------------------------------------------------------------
# Validation statistics on DPO triples
# ---------------------------------------------------------------------------

@dataclass
class EpistemicLossStats:
    """
    Summary statistics comparing epistemic loss on faithful vs unfaithful
    summaries, given a list of (input, faithful, unfaithful) triples.
    """
    n_triples:                int
    mean_faithful_loss:       float   # should be low — faithful preserves uncertainty
    mean_unfaithful_loss:     float   # should be high — unfaithful inflates confidence
    mean_delta:               float   # unfaithful - faithful (positive = correct ordering)
    pct_correctly_ordered:    float   # fraction where unfaithful_loss > faithful_loss
    mean_faithful_inflation:  float   # how much faithful summaries inflate on average
    mean_unfaithful_inflation: float  # how much unfaithful summaries inflate on average
    lambda_val:               float


def epistemic_loss_stats(
    triples:   list[tuple[str, str, str]],
    lambda_val: float = _DEFAULT_LAMBDA,
) -> EpistemicLossStats:
    """
    Given a list of (input_text, faithful_summary, unfaithful_summary) triples,
    compute summary statistics that validate the epistemic loss signal.

    A well-behaved training signal should show:
      mean_unfaithful_loss > mean_faithful_loss  (correct ordering)
      pct_correctly_ordered > 0.70               (consistent ordering)
      mean_unfaithful_inflation > mean_faithful_inflation

    Args:
        triples:   list of (input, faithful, unfaithful) string tuples
        lambda_val: regularisation weight to apply

    Returns:
        EpistemicLossStats dataclass
    """
    if not triples:
        return EpistemicLossStats(
            n_triples=0,
            mean_faithful_loss=0.0,
            mean_unfaithful_loss=0.0,
            mean_delta=0.0,
            pct_correctly_ordered=0.0,
            mean_faithful_inflation=0.0,
            mean_unfaithful_inflation=0.0,
            lambda_val=lambda_val,
        )

    faithful_losses   = []
    unfaithful_losses = []
    faithful_inflations   = []
    unfaithful_inflations = []
    correctly_ordered = 0

    for input_text, faithful, unfaithful in triples:
        j_input      = _PROXY.compute(input_text).j_score
        j_faithful   = _PROXY.compute(faithful).j_score
        j_unfaithful = _PROXY.compute(unfaithful).j_score

        # Faithful loss
        fi   = confidence_inflation(j_input, j_faithful)
        fl   = lambda_val * (abs(j_faithful - j_input) + fi)
        faithful_losses.append(fl)
        faithful_inflations.append(fi)

        # Unfaithful loss
        ui   = confidence_inflation(j_input, j_unfaithful)
        ul   = lambda_val * (abs(j_unfaithful - j_input) + ui)
        unfaithful_losses.append(ul)
        unfaithful_inflations.append(ui)

        if ul > fl:
            correctly_ordered += 1

    n = len(triples)
    return EpistemicLossStats(
        n_triples=n,
        mean_faithful_loss=round(sum(faithful_losses) / n, 6),
        mean_unfaithful_loss=round(sum(unfaithful_losses) / n, 6),
        mean_delta=round(
            (sum(unfaithful_losses) - sum(faithful_losses)) / n, 6
        ),
        pct_correctly_ordered=round(correctly_ordered / n, 4),
        mean_faithful_inflation=round(sum(faithful_inflations) / n, 6),
        mean_unfaithful_inflation=round(sum(unfaithful_inflations) / n, 6),
        lambda_val=lambda_val,
    )


# ---------------------------------------------------------------------------
# Lambda sweep utility
# ---------------------------------------------------------------------------

def lambda_sweep(
    triples: list[tuple[str, str, str]],
) -> list[EpistemicLossStats]:
    """
    Run epistemic_loss_stats over all values in LAMBDA_SWEEP.
    Useful for choosing the best lambda before training.

    Returns one EpistemicLossStats per lambda value.
    """
    return [epistemic_loss_stats(triples, lv) for lv in LAMBDA_SWEEP]


# ---------------------------------------------------------------------------
# CLI — smoke test / quick report
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Epistemic loss smoke test")
    parser.add_argument("--data",   default=None,
                        help="Path to training JSON (data/epistemic_compression_training.json)")
    parser.add_argument("--n",      type=int, default=50,
                        help="Number of triples to evaluate")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=_DEFAULT_LAMBDA,
                        help="Lambda value for epistemic loss")
    parser.add_argument("--sweep",  action="store_true",
                        help="Run lambda sweep over all LAMBDA_SWEEP values")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        with open(args.data) as f:
            raw = json.load(f)
        examples = raw["examples"][: args.n]
        triples  = [
            (e["input_conversation"], e["faithful_summary"], e["unfaithful_summary"])
            for e in examples
        ]
    else:
        # Built-in smoke test examples
        print("No data file — running built-in smoke test (5 examples).")
        triples = [
            (
                "User: The rate limit is probably 50 req/min.\n"
                "Assistant: Understood.",
                "User estimates rate limit is approximately 50 req/min (unverified).",
                "Rate limit is 50 requests per minute.",
            ),
            (
                "User: Auth token might expire after 3600 seconds.\n"
                "Assistant: We'll add refresh logic.",
                "Auth token TTL is estimated at 3600 seconds (not confirmed).",
                "Auth tokens expire after 3600 seconds.",
            ),
            (
                "User: Memory leak grows roughly 2 MB per 1000 requests.\n"
                "Assistant: That could accumulate to gigabytes under load.",
                "Memory leak is approximately 2 MB/1000 reqs — preliminary estimate from single test.",
                "Memory leak growth is 2 MB per 1000 requests.",
            ),
            (
                "User: I believe GDPR requires deletion within 30 days.\n"
                "Assistant: We'll target 14 days to be safe.",
                "User believes GDPR deletion SLA is 30 days (unverified legal interpretation).",
                "GDPR requires data deletion within 30 days.",
            ),
            (
                "User: Haiku handles about 80% of sub-agent tasks.\n"
                "Assistant: Good for cost optimisation.",
                "User estimates Haiku handles ~80% of tasks based on one demo (not validated).",
                "Haiku handles 80% of sub-agent tasks.",
            ),
        ]

    if args.sweep:
        print(f"\nLambda sweep over {len(triples)} triples:")
        print(f"  {'lambda':>8}  {'faithful_loss':>14}  {'unfaithful_loss':>16}  "
              f"{'delta':>8}  {'pct_ordered':>12}")
        print("  " + "-" * 65)
        for stats in lambda_sweep(triples):
            print(f"  {stats.lambda_val:>8.2f}  {stats.mean_faithful_loss:>14.6f}  "
                  f"{stats.mean_unfaithful_loss:>16.6f}  {stats.mean_delta:>8.6f}  "
                  f"{stats.pct_correctly_ordered:>11.1%}")
    else:
        stats = epistemic_loss_stats(triples, lambda_val=args.lambda_val)
        print(f"\nEpistemic Loss Statistics (n={stats.n_triples}, lambda={stats.lambda_val})")
        print(f"  Mean faithful loss:          {stats.mean_faithful_loss:.6f}")
        print(f"  Mean unfaithful loss:        {stats.mean_unfaithful_loss:.6f}")
        print(f"  Mean delta (should be > 0):  {stats.mean_delta:.6f}")
        print(f"  Correctly ordered:           {stats.pct_correctly_ordered:.1%}")
        print(f"  Mean faithful inflation:     {stats.mean_faithful_inflation:.6f}")
        print(f"  Mean unfaithful inflation:   {stats.mean_unfaithful_inflation:.6f}")

        if stats.pct_correctly_ordered >= 0.70:
            print("\n  SIGNAL CHECK: PASS — unfaithful loss > faithful loss in "
                  f"{stats.pct_correctly_ordered:.0%} of cases")
        else:
            print(f"\n  SIGNAL CHECK: WEAK — only {stats.pct_correctly_ordered:.0%} "
                  "correctly ordered (J-score may not distinguish this data well)")


if __name__ == "__main__":
    main()

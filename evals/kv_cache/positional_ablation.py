"""
evals/kv_cache/positional_ablation.py
=======================================
Tests the attention decay hypothesis:

    Qualifier tokens at Turn 3 (early) survive at lower rate under KV eviction
    than qualifier tokens at Turn 10 (late).

Hypothesis: recency-biased eviction schemes (H2O, StreamingLLM) assign less
attention mass to earlier tokens, causing early qualifiers to be evicted
disproportionately. EQLR_early > EQLR_late supports the decay hypothesis.

Design
------
For each of 10 EQL-Bench scenarios:
  - Condition A (EARLY):  uncertain claim placed at Turn 3
  - Condition B (LATE):   uncertain claim placed at Turn 10

Both conditions use the same KV budget and eviction method. The only
variable is the position of the uncertain claim in the conversation.

Turn structure:
  EARLY:  [ctx1][ctx2][CLAIM][filler×8][callback]   <- claim at position 3
  LATE:   [ctx1][ctx2][filler×6][CLAIM][filler×2][callback]  <- claim at position 9-10

The callback question is identical in both conditions.

Metrics
-------
EQLR_early : EQLR-Token for EARLY condition (claim at Turn 3)
EQLR_late  : EQLR-Token for LATE condition  (claim at Turn 10)
Delta      : EQLR_early - EQLR_late
             Positive delta confirms decay hypothesis.
             95% bootstrap CI on the delta.

Usage
-----
    python -m evals.kv_cache.positional_ablation --dry-run
    python -m evals.kv_cache.positional_ablation --method h2o --budget 0.70 --n 10
    python -m evals.kv_cache.positional_ablation --method streaming --budget 0.50
    python -m evals.kv_cache.positional_ablation --all-methods --budget 0.70 --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.kv_cache.metrics import eqlr_token, score_scenario, aggregate, bootstrap_ci
from evals.kv_cache.experiment_runner import (
    KVEvictionMethod,
    BaselineMethod,
    H2OMethod,
    SnapKVMethod,
    StreamingLLMMethod,
    _DryRunMethod,
    _load_eql_bench,
    FILLER_POOL,
    _DEFAULT_MODEL,
    _DEFAULT_DEVICE,
)


# ---------------------------------------------------------------------------
# Position-aware context embedding
# ---------------------------------------------------------------------------

def embed_early(scenario: dict, n_filler: int = 8) -> str:
    """
    Embed scenario with uncertain claim at Turn 3 (EARLY position).

    Structure:
        Turn 1: Context user
        Turn 2: Acknowledgement assistant
        Turn 3: UNCERTAIN CLAIM (user)
        Turn 4: Acknowledgement (assistant)
        Turns 5 to 4+n_filler: Neutral filler Q&A
        Turn 4+n_filler+1: Callback question (user)

    Parameters
    ----------
    scenario : dict
    n_filler : int
        Number of filler turns after the claim (default 8).

    Returns
    -------
    str
        LLaMA-3-Instruct format conversation prompt.
    """
    return _build_prompt(scenario, claim_position="early", n_filler=n_filler)


def embed_late(scenario: dict, n_filler_before: int = 6, n_filler_after: int = 2) -> str:
    """
    Embed scenario with uncertain claim at Turn 10 (LATE position).

    Structure:
        Turn 1: Context user
        Turn 2: Acknowledgement assistant
        Turns 3 to 2+n_filler_before: Neutral filler Q&A
        Turn 3+n_filler_before: UNCERTAIN CLAIM (user)
        Turn 4+n_filler_before: Acknowledgement (assistant)
        Turns 5+n_filler_before to 4+n_filler_before+n_filler_after: More filler
        Turn final: Callback question (user)

    Parameters
    ----------
    scenario : dict
    n_filler_before : int
        Filler turns before the claim (default 6).
    n_filler_after : int
        Filler turns after the claim (default 2).

    Returns
    -------
    str
        LLaMA-3-Instruct format conversation prompt.
    """
    return _build_prompt(
        scenario,
        claim_position="late",
        n_filler_before=n_filler_before,
        n_filler_after=n_filler_after,
    )


def _build_prompt(
    scenario: dict,
    claim_position: str = "early",
    n_filler: int = 8,
    n_filler_before: int = 6,
    n_filler_after: int = 2,
) -> str:
    """Internal prompt builder for both early and late conditions."""
    domain = scenario.get("domain", "software")
    claim  = scenario.get("uncertain_statement", scenario.get("text", ""))

    val_frags = scenario.get("value_fragments", [scenario.get("value_fragment", "")])
    if val_frags:
        callback_q = (
            f"Based on what we discussed, what is the {val_frags[0]!r} "
            "figure we should use? Please flag any uncertainty you noted earlier."
        )
    else:
        callback_q = (
            "Based on what we discussed earlier, what constraint should we implement? "
            "Please flag any uncertainty."
        )

    def _filler_block(start: int, count: int) -> str:
        block = ""
        for j in range(count):
            fq, fa = FILLER_POOL[(start + j) % len(FILLER_POOL)]
            block += f"<|start_header_id|>user<|end_header_id|>\n\n{fq}<|eot_id|>"
            block += f"<|start_header_id|>assistant<|end_header_id|>\n\n{fa}<|eot_id|>"
        return block

    sys_block = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a precise technical assistant for a {domain} project. "
        "Record all constraints including uncertainty flags as stated.<|eot_id|>"
    )
    ctx_block = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Starting a new {domain} integration project.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"Understood. I will document all constraints for the {domain} project.<|eot_id|>"
    )
    claim_block = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{claim}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Noted. Constraint recorded with any uncertainty flags as stated.<|eot_id|>"
    )
    callback_block = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{callback_q}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    if claim_position == "early":
        # EARLY: ctx → claim → filler × n_filler → callback
        return (
            sys_block
            + ctx_block
            + claim_block
            + _filler_block(0, n_filler)
            + callback_block
        )
    else:
        # LATE: ctx → filler × n_filler_before → claim → filler × n_filler_after → callback
        return (
            sys_block
            + ctx_block
            + _filler_block(0, n_filler_before)
            + claim_block
            + _filler_block(n_filler_before, n_filler_after)
            + callback_block
        )


# ---------------------------------------------------------------------------
# Per-scenario ablation
# ---------------------------------------------------------------------------


class PositionResult:
    """Result for one scenario under one position condition."""
    __slots__ = (
        "scenario_id", "domain", "qualifier_type",
        "position",    # "early" or "late"
        "answer",
        "eqlr_token",
        "fcr",
        "qualifier_present",
        "elapsed_s",
    )

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


def run_scenario_ablation(
    scenario: dict,
    method: KVEvictionMethod,
    budget: float,
) -> tuple[PositionResult, PositionResult]:
    """
    Run one scenario in both EARLY and LATE configurations.

    Returns
    -------
    (early_result, late_result)
    """
    scenario_id    = scenario.get("scenario_id", scenario.get("id", "?"))
    domain         = scenario.get("domain", "")
    qualifier_type = scenario.get("qualifier_type", "")

    # EARLY condition
    t0 = time.time()
    prompt_early = embed_early(scenario)
    answer_early, _ = method.generate(prompt_early, kv_budget=budget)
    t_early = time.time() - t0

    scored_early = score_scenario(scenario, answer_early)

    early = PositionResult(
        scenario_id    = scenario_id,
        domain         = domain,
        qualifier_type = qualifier_type,
        position       = "early",
        answer         = answer_early[:400],
        eqlr_token     = scored_early["eqlr_token"],
        fcr            = scored_early["fcr"],
        qualifier_present = scored_early["qualifier_present"],
        elapsed_s      = round(t_early, 2),
    )

    # LATE condition
    t0 = time.time()
    prompt_late = embed_late(scenario)
    answer_late, _ = method.generate(prompt_late, kv_budget=budget)
    t_late = time.time() - t0

    scored_late = score_scenario(scenario, answer_late)

    late = PositionResult(
        scenario_id    = scenario_id,
        domain         = domain,
        qualifier_type = qualifier_type,
        position       = "late",
        answer         = answer_late[:400],
        eqlr_token     = scored_late["eqlr_token"],
        fcr            = scored_late["fcr"],
        qualifier_present = scored_late["qualifier_present"],
        elapsed_s      = round(t_late, 2),
    )

    return early, late


# ---------------------------------------------------------------------------
# Full positional ablation run
# ---------------------------------------------------------------------------

def run_positional_ablation(
    scenarios: list[dict],
    method: KVEvictionMethod,
    budget: float,
    verbose: bool = True,
) -> dict:
    """
    Run all scenarios in both EARLY and LATE configurations.

    Returns
    -------
    dict with keys:
        method          : str
        budget          : float
        n               : int
        early_results   : list[dict]
        late_results    : list[dict]
        eqlr_early      : float
        eqlr_early_ci   : [lo, hi]
        eqlr_late       : float
        eqlr_late_ci    : [lo, hi]
        delta           : float   (eqlr_early - eqlr_late)
        delta_ci        : [lo, hi]
        hypothesis_supported : bool  (delta > 0 and CI[0] > 0)
    """
    early_results = []
    late_results  = []
    n = len(scenarios)

    for i, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", scenario.get("id", "?"))
        if verbose:
            print(f"  [{i+1}/{n}] {sid} ...", end=" ", flush=True)

        early, late = run_scenario_ablation(scenario, method, budget)
        early_results.append(early.to_dict())
        late_results.append(late.to_dict())

        if verbose:
            e_str = "LOST" if early.eqlr_token else "OK"
            l_str = "LOST" if late.eqlr_token  else "OK"
            print(f"EARLY={e_str}  LATE={l_str}")

    # Aggregate
    eqlr_early_vals = [1.0 if r["eqlr_token"] else 0.0 for r in early_results]
    eqlr_late_vals  = [1.0 if r["eqlr_token"] else 0.0 for r in late_results]

    eqlr_early = sum(eqlr_early_vals) / n if n > 0 else 0.0
    eqlr_late  = sum(eqlr_late_vals)  / n if n > 0 else 0.0

    eqlr_early_ci = bootstrap_ci(eqlr_early_vals, n_boot=2000)
    eqlr_late_ci  = bootstrap_ci(eqlr_late_vals,  n_boot=2000)

    delta_vals = [e - l for e, l in zip(eqlr_early_vals, eqlr_late_vals)]
    delta      = sum(delta_vals) / n if n > 0 else 0.0
    delta_ci   = bootstrap_ci(delta_vals, n_boot=2000)

    # Hypothesis supported if delta > 0 AND the lower CI bound > 0
    hypothesis_supported = delta > 0.0 and delta_ci[0] > 0.0

    return {
        "method":               method.name,
        "budget":               budget,
        "n":                    n,
        "early_results":        early_results,
        "late_results":         late_results,
        "eqlr_early":           round(eqlr_early, 4),
        "eqlr_early_ci":        list(eqlr_early_ci),
        "eqlr_late":            round(eqlr_late, 4),
        "eqlr_late_ci":         list(eqlr_late_ci),
        "delta":                round(delta, 4),
        "delta_ci":             list(delta_ci),
        "hypothesis_supported": hypothesis_supported,
    }


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

def run_all_methods(
    scenarios: list[dict],
    budget: float,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run positional ablation for all eviction methods at the given budget.

    Returns dict keyed by method name.
    """
    method_configs = [
        ("baseline",    None),
        ("h2o",         H2OMethod),
        ("snapkv",      SnapKVMethod),
        ("streaming",   StreamingLLMMethod),
    ]

    results = {}

    for method_name, method_cls in method_configs:
        print(f"\nPositional ablation: method={method_name}, budget={budget:.0%}")
        if dry_run:
            method = _DryRunMethod(method_name, inject_qualifier=(method_name == "baseline"))
        else:
            if method_cls is None:
                method = BaselineMethod()
            else:
                try:
                    method = method_cls()
                except ImportError as e:
                    print(f"  Skipped ({e})")
                    results[method_name] = {"error": str(e)}
                    continue

        result = run_positional_ablation(scenarios, method, budget, verbose=verbose)
        results[method_name] = result

        print(
            f"  EQLR_early={result['eqlr_early']:.3f}  "
            f"EQLR_late={result['eqlr_late']:.3f}  "
            f"delta={result['delta']:+.3f}  "
            f"hypothesis={'SUPPORTED' if result['hypothesis_supported'] else 'not supported'}"
        )

    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def make_ablation_table(results: dict, budget: float) -> str:
    """Format a comparison table for positional ablation results."""
    header = (
        f"{'Method':<14}  "
        f"{'EQLR_early':>12}  {'EQLR_late':>10}  "
        f"{'Delta':>8}  {'Delta 95%CI':>18}  "
        f"{'Hypothesis':>12}"
    )
    sep   = "-" * len(header)
    rows  = [
        f"Positional Ablation — KV Budget {budget:.0%}",
        f"n_scenarios per position: varies per run",
        sep,
        header,
        sep,
    ]

    method_order = ["baseline", "h2o", "snapkv", "streaming"]
    for method_name in method_order:
        r = results.get(method_name)
        if r is None:
            continue
        if "error" in r:
            rows.append(f"{method_name:<14}  (skipped: {r['error'][:50]})")
            continue

        e_ci  = r.get("eqlr_early_ci", [0.0, 0.0])
        l_ci  = r.get("eqlr_late_ci",  [0.0, 0.0])
        d_ci  = r.get("delta_ci",       [0.0, 0.0])
        hyp   = "SUPPORTED" if r.get("hypothesis_supported") else "not supported"

        early_str = f"{r['eqlr_early']:.3f} [{e_ci[0]:.3f},{e_ci[1]:.3f}]"
        late_str  = f"{r['eqlr_late']:.3f}"
        delta_str = f"{r['delta']:+.3f}"
        dci_str   = f"[{d_ci[0]:+.3f},{d_ci[1]:+.3f}]"

        rows.append(
            f"{method_name:<14}  "
            f"{early_str:>22}  {late_str:>10}  "
            f"{delta_str:>8}  {dci_str:>18}  "
            f"{hyp:>12}"
        )

    rows.append(sep)
    rows.append(
        "Interpretation: delta > 0 AND CI_lo > 0 = early qualifiers disproportionately evicted.\n"
        "delta ≈ 0 = eviction is position-agnostic."
    )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KV-cache positional ablation: EQLR_early vs EQLR_late",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--method", choices=["baseline", "h2o", "snapkv", "streaming"],
                        default="h2o", help="Eviction method to test (default: h2o)")
    parser.add_argument("--budget", type=float, default=0.70,
                        help="KV budget fraction (default: 0.70)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of scenarios to test (default: 10)")
    parser.add_argument("--all-methods", action="store_true",
                        help="Run all methods at the specified budget")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without loading any models (validates structure)")
    parser.add_argument("--out", default="results/positional_ablation_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print per-scenario progress")
    args = parser.parse_args()

    # Load EQL-Bench only (ghost scenarios are less relevant for position effects)
    scenarios = _load_eql_bench()[: args.n]
    print(f"Loaded {len(scenarios)} EQL-Bench scenarios for positional ablation")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===\n")
        print(f"Would test positions EARLY vs LATE for {len(scenarios)} scenarios")
        print(f"Method: {args.method}, Budget: {args.budget:.0%}")
        print("\nSample EARLY prompt structure (first scenario):")
        if scenarios:
            p = embed_early(scenarios[0])
            print(f"  Length: {len(p)} chars")
            print(f"  First 200 chars: {p[:200]!r}")
            print()
            p2 = embed_late(scenarios[0])
            print(f"Sample LATE prompt structure:")
            print(f"  Length: {len(p2)} chars")
            print(f"  First 200 chars: {p2[:200]!r}")

        dry_method = _DryRunMethod(args.method, inject_qualifier=False)
        print("\nRunning 3 scenarios in dry-run mode ...")
        result = run_positional_ablation(
            scenarios[:3], dry_method, args.budget, verbose=True
        )
        print(f"\nDry-run result:")
        print(f"  EQLR_early = {result['eqlr_early']:.3f}")
        print(f"  EQLR_late  = {result['eqlr_late']:.3f}")
        print(f"  delta      = {result['delta']:+.3f}")
        print(f"  hypothesis_supported = {result['hypothesis_supported']}")
        print("\nDry run complete.")
        return

    if args.all_methods:
        print(f"\nRunning positional ablation for all methods at budget={args.budget:.0%}")
        results = run_all_methods(scenarios, args.budget, dry_run=False, verbose=args.verbose)

        table = make_ablation_table(results, args.budget)
        print("\n" + table)

        out_path = args.out
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        payload = {
            "budget": args.budget,
            "n":      len(scenarios),
            "methods": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
        return

    # Single method
    print(f"\nPositional ablation: method={args.method}, budget={args.budget:.0%}")
    try:
        if args.method == "baseline":
            method = BaselineMethod()
        elif args.method == "h2o":
            method = H2OMethod()
        elif args.method == "snapkv":
            method = SnapKVMethod()
        elif args.method == "streaming":
            method = StreamingLLMMethod()
        else:
            raise ValueError(f"Unknown method: {args.method}")
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    result = run_positional_ablation(scenarios, method, args.budget, verbose=args.verbose)

    print(f"\n{'=' * 60}")
    print(f"POSITIONAL ABLATION RESULTS")
    print(f"Method: {args.method}  Budget: {args.budget:.0%}  n={result['n']}")
    print(f"{'=' * 60}")
    print(f"EQLR_early = {result['eqlr_early']:.3f}  95%CI={result['eqlr_early_ci']}")
    print(f"EQLR_late  = {result['eqlr_late']:.3f}   95%CI={result['eqlr_late_ci']}")
    print(f"Delta      = {result['delta']:+.3f}   95%CI={result['delta_ci']}")
    print(f"")
    if result["hypothesis_supported"]:
        print("RESULT: Decay hypothesis SUPPORTED")
        print("  → Qualifiers evicted at higher rate when placed earlier in context.")
    elif result["delta"] > 0:
        print("RESULT: Decay hypothesis directionally supported but not significant.")
        print("  → delta > 0 but CI includes zero. Increase n for statistical power.")
    else:
        print("RESULT: Decay hypothesis NOT supported.")
        print("  → No significant position effect on qualifier eviction rate.")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

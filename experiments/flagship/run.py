"""
experiments/flagship/run.py
============================
CLI runner for the flagship Epistemic Memory experiment.

Usage:
    python -m experiments.flagship.run                     # 3 trials, all scenarios
    python -m experiments.flagship.run --trials 1          # quick smoke test
    python -m experiments.flagship.run --scenarios A,B     # subset of scenarios
    python -m experiments.flagship.run --dry-run           # no API calls, mock data

Output:
    Prints per-scenario, per-condition results to terminal.
    Saves full results to experiments/flagship/flagship_results.json

Metrics:
    recall          — mean fragment recall across callbacks (higher = better)
    prop_rate       — fraction of callbacks with propagation error (lower = better)
    chain_complete  — all callbacks ≥ 60% recall AND prop_rate == 0
    tokens          — API tokens consumed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.flagship.scenarios import ALL_SCENARIOS
from experiments.flagship.pipeline import run_scenario_trial, TrialResult
from experiments.flagship.metrics import ScenarioResult, bootstrap_ci

_RESULTS_PATH = Path(__file__).parent / "flagship_results.json"
_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(results: list[ScenarioResult], key: str) -> list[float]:
    return [getattr(r, key) for r in results]


def summarise_trials(trials: list[TrialResult], scenario_ids: list[str]):
    """Print summary table over all trials."""
    from collections import defaultdict

    by_cond: dict[str, list[ScenarioResult]] = defaultdict(list)
    for t in trials:
        by_cond["baseline"].append(t.baseline)
        by_cond["naive_window"].append(t.naive_window)
        by_cond["epistemic_memory"].append(t.epistemic_memory)

    print()
    print("=" * 80)
    print("FLAGSHIP EXPERIMENT — EPISTEMIC MEMORY")
    print("=" * 80)
    print(f"{'Condition':<20} {'Recall':>8} {'95%CI':>14} {'PropRate':>10} {'Chain%':>8} {'Tokens':>10}")
    print("-" * 80)

    for cond, srs in by_cond.items():
        recalls  = _agg(srs, "mean_recall")
        props    = _agg(srs, "propagation_rate")
        chains   = [1.0 if r.chain_complete else 0.0 for r in srs]
        tokens   = _agg(srs, "tokens_used")

        mean_r  = sum(recalls) / len(recalls)
        ci_lo, ci_hi = bootstrap_ci(recalls)
        mean_p  = sum(props) / len(props)
        mean_c  = sum(chains) / len(chains)
        mean_t  = int(sum(tokens) / len(tokens))

        print(f"{cond:<20} {mean_r:>8.3f} [{ci_lo:.3f},{ci_hi:.3f}] {mean_p:>10.3f} {mean_c:>8.1%} {mean_t:>10,}")

    print("=" * 80)
    print()
    print("Per-scenario breakdown:")
    print(f"{'Scenario':<12} {'Condition':<22} {'Recall':>8} {'PropRate':>10} {'Chain':>7}")
    print("-" * 65)
    for t in trials:
        for sr in [t.baseline, t.naive_window, t.epistemic_memory]:
            chain_str = "YES" if sr.chain_complete else "NO"
            print(f"Scenario {sr.scenario_id:<3} {sr.condition:<22} "
                  f"{sr.mean_recall:>8.3f} {sr.propagation_rate:>10.3f} {chain_str:>7}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Dry run (no API)
# ---------------------------------------------------------------------------

def dry_run(scenario_ids: list[str], n_trials: int):
    """Print mock results to verify the pipeline structure without API calls."""
    print("[DRY-RUN] Pipeline smoke test (no API calls)")
    for sid in scenario_ids:
        sc = ALL_SCENARIOS[sid]
        print(f"  Scenario {sid} ({sc.name}): {len(sc.seed_turns)} seed + "
              f"{len(sc.filler_turns)} filler + {len(sc.callbacks)} callbacks")
    print(f"  Trials: {n_trials}")
    print(f"  Total API calls: {len(scenario_ids) * n_trials * 3 * sum(len(ALL_SCENARIOS[s].callbacks) for s in scenario_ids)} (est)")
    print("[DRY-RUN] Structure OK")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _sr_to_dict(sr: ScenarioResult) -> dict:
    return {
        "scenario_id":      sr.scenario_id,
        "condition":        sr.condition,
        "mean_recall":      sr.mean_recall,
        "propagation_rate": sr.propagation_rate,
        "chain_complete":   sr.chain_complete,
        "tokens_used":      sr.tokens_used,
        "tokens_saved":     sr.tokens_saved,
        "callbacks": [
            {
                "question":            cb.question,
                "fragment_recall":     cb.fragment_recall,
                "propagation_error":   cb.propagation_error,
                "uncertainty_preserved": cb.uncertainty_preserved,
            }
            for cb in sr.callback_results
        ],
    }


def save_results(trials: list[TrialResult], path: Path):
    out = []
    for t in trials:
        out.append({
            "trial":           t.trial,
            "scenario_id":     t.scenario_id,
            "baseline":        _sr_to_dict(t.baseline),
            "naive_window":    _sr_to_dict(t.naive_window),
            "epistemic_memory":_sr_to_dict(t.epistemic_memory),
        })
    path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flagship Epistemic Memory Experiment")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per scenario")
    parser.add_argument("--scenarios", default="A,B,C", help="Comma-separated scenario IDs")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test without API")
    parser.add_argument("--verbose", action="store_true", help="Print per-callback detail")
    args = parser.parse_args()

    scenario_ids = [s.strip().upper() for s in args.scenarios.split(",")]
    for sid in scenario_ids:
        if sid not in ALL_SCENARIOS:
            print(f"Unknown scenario: {sid}. Available: {list(ALL_SCENARIOS.keys())}")
            sys.exit(1)

    if args.dry_run:
        dry_run(scenario_ids, args.trials)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Use --dry-run to test without API.")
        sys.exit(1)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    all_trials: list[TrialResult] = []
    total = len(scenario_ids) * args.trials

    print(f"Running {total} scenario-trials ({args.trials} trials × {len(scenario_ids)} scenarios)…")
    print()

    for trial_idx in range(1, args.trials + 1):
        print(f"── Trial {trial_idx}/{args.trials} " + "─" * 40)
        for sid in scenario_ids:
            sc = ALL_SCENARIOS[sid]
            t = run_scenario_trial(
                sc, client,
                trial=trial_idx,
                model=_MODEL,
                verbose=args.verbose,
            )
            all_trials.append(t)
            # Quick inline print
            for sr in [t.baseline, t.naive_window, t.epistemic_memory]:
                chain_str = "chain=COMPLETE" if sr.chain_complete else "chain=BROKEN"
                print(f"  [{sr.condition:<20}] S{sr.scenario_id} recall={sr.mean_recall:.3f}  "
                      f"prop={sr.propagation_rate:.3f}  {chain_str}")
        print()

    summarise_trials(all_trials, scenario_ids)
    save_results(all_trials, _RESULTS_PATH)


if __name__ == "__main__":
    main()

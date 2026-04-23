"""
evals/e1_repeated.py
====================
Run E1 (Propagation Chain) N times and report mean ± CI.

Single-run E1 results (CAMS 0.700 vs naive 0.667) are within noise due to
fragment-matching variance and live generation. This script runs N independent
trials and computes bootstrapped 95% CI on the mean recall per condition.

Run:
    python -m evals.e1_repeated            # 10 runs (default)
    python -m evals.e1_repeated --n 5      # 5 runs (faster, less precise)

Results saved to evals/e1_repeated_results.json
"""

import os, sys, json, math, argparse, time, random
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.experiments import run_e1


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    rng = random.Random(42)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_boot)
    )
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return round(means[lo_idx], 4), round(means[hi_idx], 4)


def main():
    parser = argparse.ArgumentParser(description="Repeat E1 N times for CI estimation")
    parser.add_argument("--n", type=int, default=10, help="Number of independent runs")
    parser.add_argument("--output", default="evals/e1_repeated_results.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print(f"Running E1 × {args.n} independent trials ...")
    print("This produces mean ± 95% CI on recall per condition.\n")

    all_runs: list[list] = []

    for i in range(args.n):
        print(f"\n── Trial {i+1}/{args.n} ────────────────────────────────")
        results = run_e1()
        all_runs.append(results)
        # Brief pause between full runs to avoid rate limit bursts
        if i < args.n - 1:
            time.sleep(2.0)

    # Aggregate by condition
    conditions = ["baseline", "naive_window", "cams"]
    aggregated = {}

    for cond in conditions:
        recalls = []
        for run in all_runs:
            for r in run:
                if r.condition == cond:
                    recalls.append(r.mean_recall)
                    break
        mean = sum(recalls) / len(recalls)
        lo, hi = bootstrap_ci(recalls)
        aggregated[cond] = {
            "mean_recall": round(mean, 4),
            "ci_95_lo":    lo,
            "ci_95_hi":    hi,
            "n_trials":    len(recalls),
            "raw":         recalls,
        }

    print("\n" + "=" * 60)
    print("E1 PROPAGATION CHAIN — REPEATED TRIALS")
    print("=" * 60)
    print(f"{'Condition':<16} {'Mean':>7} {'95% CI':>18}  {'Significant?'}")
    print("-" * 60)

    cams_lo   = aggregated["cams"]["ci_95_lo"]
    naive_hi  = aggregated["naive_window"]["ci_95_hi"]
    sig = "YES" if cams_lo > naive_hi else "NO (CIs overlap)"

    for cond in conditions:
        a = aggregated[cond]
        print(f"{cond:<16} {a['mean_recall']:>7.3f}  [{a['ci_95_lo']:.3f}, {a['ci_95_hi']:.3f}]  "
              + (f"CAMS vs naive: {sig}" if cond == "cams" else ""))
    print("=" * 60)

    result = {
        "n_trials":   args.n,
        "conditions": aggregated,
        "cams_vs_naive_significant": sig == "YES",
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

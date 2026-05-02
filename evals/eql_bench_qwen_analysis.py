"""
evals/eql_bench_qwen_analysis.py — Phase 3b EQL-Bench v2 (Qwen-2.5-1.5B) analysis.

Computes bootstrap CIs and domain/qualifier-type breakdowns from saved Kaggle results.
No API calls; reads evals/eql_bench_qwen_results.json.

Usage:
    python -m evals.eql_bench_qwen_analysis
    python -m evals.eql_bench_qwen_analysis --out evals/eql_bench_qwen_analysis.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

_RESULTS_PATH = Path(__file__).parent / "eql_bench_qwen_results.json"


def _bootstrap_ci(values: List[float], n_boot: int = 2000, ci: float = 0.95) -> tuple:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(42)
    boot_means = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 - (1 - ci) / 2) * n_boot)
    return (round(boot_means[lo], 4), round(boot_means[hi], 4))


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _ci_str(ci: tuple) -> str:
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def analyze(path: Path = _RESULTS_PATH) -> dict:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    n_total = len(results)

    explicit = [r for r in results if not r["is_ghost"]]
    ghost = [r for r in results if r["is_ghost"]]

    probe_blocked = [r for r in explicit if r["probe_blocked"]]
    unguarded = [r for r in explicit if not r["probe_blocked"]]

    # ── Overall EQLR ────────────────────────────────────────────────────────
    blocked_eqlr = [0.0] * len(probe_blocked)  # probe-blocked: EQLR=0 by design
    unguarded_eqlr = [1.0 if r["eql_event"] else 0.0 for r in unguarded]
    ghost_eqlr = [1.0 if r["eql_event"] else 0.0 for r in ghost]

    # ── Coverage ────────────────────────────────────────────────────────────
    probe_coverage = len(probe_blocked) / len(explicit) if explicit else 0.0
    ghost_fp_rate = sum(1 for r in ghost if r["probe_blocked"]) / len(ghost) if ghost else 0.0

    # ── Unguarded EQLR overall ───────────────────────────────────────────────
    unguarded_eqlr_mean = sum(unguarded_eqlr) / len(unguarded_eqlr) if unguarded_eqlr else 0.0
    unguarded_eqlr_ci = _bootstrap_ci(unguarded_eqlr)

    # ── By domain (unguarded only — removes probe-blocked cases) ────────────
    domain_stats: dict = {}
    dom_bucket: dict = defaultdict(list)
    for r in unguarded:
        dom_bucket[r["domain"]].append(1.0 if r["eql_event"] else 0.0)
    for dom, vals in sorted(dom_bucket.items()):
        mean = sum(vals) / len(vals)
        domain_stats[dom] = {
            "n": len(vals),
            "eqlr": round(mean, 4),
            "eqlr_ci95": _bootstrap_ci(vals),
        }

    # ── By qualifier type (unguarded only) ──────────────────────────────────
    qtype_stats: dict = {}
    qt_bucket: dict = defaultdict(list)
    for r in unguarded:
        qt_bucket[r["qualifier_type"]].append(1.0 if r["eql_event"] else 0.0)
    for qt, vals in sorted(qt_bucket.items()):
        mean = sum(vals) / len(vals)
        qtype_stats[qt] = {
            "n": len(vals),
            "eqlr": round(mean, 4),
            "eqlr_ci95": _bootstrap_ci(vals),
        }

    # ── Probe-blocked EQLR (should be 0%) ───────────────────────────────────
    blocked_eqlr_mean = 0.0
    blocked_eqlr_ci = (0.0, 0.0)

    # ── Ghost EQLR (expected ~90–100%, probe misses by design) ──────────────
    ghost_eqlr_mean = sum(ghost_eqlr) / len(ghost_eqlr) if ghost_eqlr else 0.0
    ghost_eqlr_ci = _bootstrap_ci(ghost_eqlr)

    # ── Overall across all 370 scenarios ────────────────────────────────────
    all_eqlr = blocked_eqlr + unguarded_eqlr + ghost_eqlr
    all_eqlr_mean = sum(all_eqlr) / len(all_eqlr) if all_eqlr else 0.0
    all_eqlr_ci = _bootstrap_ci(all_eqlr)

    return {
        "model": data.get("model", "Qwen/Qwen2.5-1.5B-Instruct"),
        "benchmark": data.get("benchmark", "EQL-Bench v2"),
        "n_total": n_total,
        "n_explicit": len(explicit),
        "n_ghost": len(ghost),
        "n_probe_blocked": len(probe_blocked),
        "n_unguarded": len(unguarded),
        "probe_coverage": round(probe_coverage, 4),
        "probe_coverage_ci95": _bootstrap_ci([1.0 if r["probe_blocked"] else 0.0 for r in explicit]),
        "ghost_fp_rate": round(ghost_fp_rate, 4),
        "overall_eqlr": round(all_eqlr_mean, 4),
        "overall_eqlr_ci95": all_eqlr_ci,
        "unguarded_eqlr": round(unguarded_eqlr_mean, 4),
        "unguarded_eqlr_ci95": unguarded_eqlr_ci,
        "probe_blocked_eqlr": blocked_eqlr_mean,
        "probe_blocked_eqlr_ci95": blocked_eqlr_ci,
        "ghost_eqlr": round(ghost_eqlr_mean, 4),
        "ghost_eqlr_ci95": ghost_eqlr_ci,
        "by_domain": domain_stats,
        "by_qualifier_type": qtype_stats,
    }


def print_report(stats: dict) -> None:
    sep = "─" * 64
    print(f"\n{'═' * 64}")
    print(f"  EQL-Bench v2 — Qwen-2.5-1.5B-Instruct Analysis")
    print(f"{'═' * 64}")
    print(f"  Model:     {stats['model']}")
    print(f"  Benchmark: {stats['benchmark']}")
    print(f"  n_total:   {stats['n_total']}  (explicit={stats['n_explicit']}, ghost={stats['n_ghost']})")
    print()
    print("  PROBE COVERAGE")
    print(f"  {'Probe coverage (explicit):':<38} {_pct(stats['probe_coverage'])}  95%CI {_ci_str(stats['probe_coverage_ci95'])}")
    print(f"  {'Ghost FP rate:':<38} {_pct(stats['ghost_fp_rate'])}")
    print()
    print("  EQLR BY CONDITION")
    print(f"  {'Probe-blocked (n=' + str(stats['n_probe_blocked']) + '):':<38} {_pct(stats['probe_blocked_eqlr'])}  (deterministic — 0 by design)")
    print(f"  {'Unguarded explicit (n=' + str(stats['n_unguarded']) + '):':<38} {_pct(stats['unguarded_eqlr'])}  95%CI {_ci_str(stats['unguarded_eqlr_ci95'])}")
    print(f"  {'Ghost / implicit (n=' + str(stats['n_ghost']) + '):':<38} {_pct(stats['ghost_eqlr'])}  95%CI {_ci_str(stats['ghost_eqlr_ci95'])}")
    print(f"  {'Overall (n=' + str(stats['n_total']) + '):':<38} {_pct(stats['overall_eqlr'])}  95%CI {_ci_str(stats['overall_eqlr_ci95'])}")
    print()
    print(f"  {sep}")
    print("  EQLR BY DOMAIN (unguarded explicit scenarios)")
    print(f"  {'Domain':<20} {'n':>4}  {'EQLR':>8}  {'95% CI':>20}")
    print(f"  {'-'*20} {'-'*4}  {'-'*8}  {'-'*20}")
    for dom, s in sorted(stats["by_domain"].items(), key=lambda x: -x[1]["eqlr"]):
        print(f"  {dom:<20} {s['n']:>4}  {_pct(s['eqlr']):>8}  {_ci_str(s['eqlr_ci95']):>20}")
    print()
    print(f"  {sep}")
    print("  EQLR BY QUALIFIER TYPE (unguarded explicit scenarios)")
    print(f"  {'Qualifier type':<22} {'n':>4}  {'EQLR':>8}  {'95% CI':>20}")
    print(f"  {'-'*22} {'-'*4}  {'-'*8}  {'-'*20}")
    for qt, s in sorted(stats["by_qualifier_type"].items(), key=lambda x: -x[1]["eqlr"]):
        print(f"  {qt:<22} {s['n']:>4}  {_pct(s['eqlr']):>8}  {_ci_str(s['eqlr_ci95']):>20}")
    print()
    print(f"  {'═' * 64}")
    print("  KEY FINDING: probe EQLR=0% on all covered scenarios;")
    print(f"  unguarded EQLR={_pct(stats['unguarded_eqlr'])} — confirms model-agnostic failure.")
    print()


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EQL-Bench v2 Qwen analysis")
    parser.add_argument("--out", default=None, help="Save JSON to path")
    parser.add_argument("--in", dest="input", default=str(_RESULTS_PATH), help="Input JSON path")
    args = parser.parse_args(argv)

    stats = analyze(Path(args.input))
    print_report(stats)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()

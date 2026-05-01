"""
evals/calibration_curve.py
==========================
Expected Calibration Error (ECE) analysis for the J-score as a predictor of
compression safety.

Two calibration tests:

Test A — J-score of downstream answer vs False Certainty (FCR)
  If J is a reliable signal, HIGH-J downstream answers should correlate with
  naive_downstream_certain=True (the model stated the uncertain value as fact).
  A well-calibrated signal shows monotone increase across bins.

Test B — J-score of uncertain input statement vs qualifier survival
  Uncertain inputs should cluster LOW-J. Measures whether J correctly
  identifies the "uncertain" character of the input, independent of the probe.

ECE formula:
  ECE = Σ (|bin| / N) × |accuracy(bin) − confidence(bin)|

Outputs a calibration table and ECE to stdout. No API key required.

Run:
    python -m evals.calibration_curve
    python -m evals.calibration_curve --json
"""

from __future__ import annotations

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.confidence_proxy import CredenceProxy

RESULTS_FILE = "evals/compression_faithfulness_n50_results.json"


# ---------------------------------------------------------------------------
# ECE helpers
# ---------------------------------------------------------------------------

def _ece(scores: list[float], labels: list[bool], n_bins: int = 5) -> float:
    """
    Expected Calibration Error.
    scores: predicted confidence ∈ [0,1]
    labels: ground truth bool
    """
    n = len(scores)
    if n == 0:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels):
        idx = min(int(score * n_bins), n_bins - 1)
        bins[idx].append((score, label))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(s for s, _ in b) / len(b)
        avg_acc  = sum(1 for _, l in b if l) / len(b)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)

    return round(ece, 4)


def _calibration_table(
    scores: list[float],
    labels: list[bool],
    n_bins: int = 5,
    score_label: str = "J-score",
    outcome_label: str = "P(outcome=True)",
) -> list[dict]:
    """Return per-bin calibration data."""
    bins = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels):
        idx = min(int(score * n_bins), n_bins - 1)
        bins[idx].append((score, label))

    rows = []
    for i, b in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not b:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "mean_score": None, "empirical_rate": None, "gap": None})
            continue
        mean_score = sum(s for s, _ in b) / len(b)
        emp_rate   = sum(1 for _, l in b if l) / len(b)
        rows.append({
            "bin":           f"[{lo:.1f},{hi:.1f})",
            "n":             len(b),
            "mean_score":    round(mean_score, 3),
            "empirical_rate": round(emp_rate, 3),
            "gap":           round(abs(emp_rate - mean_score), 3),
        })
    return rows


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_scenarios() -> list[dict]:
    path = RESULTS_FILE
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run evals/compression_faithfulness.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)["scenarios"]


# ---------------------------------------------------------------------------
# Test A: J-score of downstream answer vs False Certainty
# ---------------------------------------------------------------------------

def test_a_downstream_certainty(scenarios: list[dict], proxy: CredenceProxy) -> dict:
    """
    Does HIGH J-score in the downstream answer predict false certainty?

    Intuition: when the compressed context strips the qualifier, the downstream
    model answers confidently — HIGH J. When qualifier survives, the model hedges — LOW J.
    If J is a good signal, FCR should be monotone increasing across J bins.
    """
    scores_naive = []
    labels_naive = []
    scores_probe = []
    labels_probe = []

    for s in scenarios:
        # Naive downstream
        if s.get("naive_downstream_answer"):
            j = proxy.compute(s["naive_downstream_answer"]).j_score
            scores_naive.append(j)
            labels_naive.append(bool(s.get("naive_downstream_certain", False)))

        # Probe downstream (ground truth: should be 0% FCR)
        if s.get("probe_downstream_answer"):
            j = proxy.compute(s["probe_downstream_answer"]).j_score
            scores_probe.append(j)
            labels_probe.append(bool(s.get("probe_downstream_certain", False)))

    ece_naive = _ece(scores_naive, labels_naive)
    ece_probe = _ece(scores_probe, labels_probe)
    table_naive = _calibration_table(scores_naive, labels_naive,
                                      score_label="J(downstream)",
                                      outcome_label="P(false_certainty)")
    table_probe = _calibration_table(scores_probe, labels_probe,
                                      score_label="J(downstream)",
                                      outcome_label="P(false_certainty)")

    return {
        "test": "A",
        "title": "J-score of downstream answer vs False Certainty Rate",
        "description": (
            "HIGH J-score in downstream answer should predict false certainty "
            "(model stated uncertain value as confirmed fact). "
            "ECE measures calibration gap — lower is better."
        ),
        "naive": {
            "n": len(scores_naive),
            "mean_j": round(sum(scores_naive) / len(scores_naive), 3) if scores_naive else 0,
            "fcr": round(sum(labels_naive) / len(labels_naive), 3) if labels_naive else 0,
            "ece": ece_naive,
            "calibration_table": table_naive,
        },
        "probe": {
            "n": len(scores_probe),
            "mean_j": round(sum(scores_probe) / len(scores_probe), 3) if scores_probe else 0,
            "fcr": round(sum(labels_probe) / len(labels_probe), 3) if labels_probe else 0,
            "ece": ece_probe,
            "calibration_table": table_probe,
        },
    }


# ---------------------------------------------------------------------------
# Test B: J-score of uncertain input vs qualifier survival
# ---------------------------------------------------------------------------

def test_b_input_uncertainty(scenarios: list[dict], proxy: CredenceProxy) -> dict:
    """
    Do uncertain input statements cluster at LOW J-score?

    Uncertain statements contain hedging language ("I think", "probably",
    "unverified") which should push J-score LOW. If J correctly characterises
    the epistemic status of the input, all uncertain statements should score
    LOW or MEDIUM — never HIGH.

    This validates that J-score and the faithfulness probe are complementary:
    the probe uses lexical markers; J-score provides a continuous score.
    """
    j_scores = []
    zones = []

    for s in scenarios:
        text = s.get("uncertain_statement", "")
        if text:
            r = proxy.compute(text)
            j_scores.append(r.j_score)
            zones.append(r.zone)

    n = len(j_scores)
    zone_counts = {z: zones.count(z) for z in ("HIGH", "MEDIUM", "LOW")}

    return {
        "test": "B",
        "title": "J-score of uncertain input statements",
        "description": (
            "Uncertain statements (containing hedging language) should cluster at "
            "LOW/MEDIUM J-score. HIGH-J uncertain inputs are the probe's blind spot "
            "(ghost constraints — implicitly uncertain, no hedging markers)."
        ),
        "n": n,
        "mean_j": round(sum(j_scores) / n, 3) if n else 0,
        "zone_distribution": zone_counts,
        "zone_pct": {
            k: f"{v}/{n} ({100*v//n}%)" for k, v in zone_counts.items()
        },
        "high_j_inputs": [
            scenarios[i]["uncertain_statement"][:80]
            for i, j in enumerate(j_scores) if j >= 0.70
        ],
    }


# ---------------------------------------------------------------------------
# Test C: J-score zone vs compression decision correctness
# ---------------------------------------------------------------------------

def test_c_zone_accuracy(scenarios: list[dict], proxy: CredenceProxy) -> dict:
    """
    Does the probe correctly identify ALL uncertain inputs regardless of J-score?

    The probe fires on lexical markers, not J-score. This test checks whether
    probe_blocked=True for inputs at every J-score zone — confirming that the
    probe catches uncertain inputs even when J (incorrectly) suggests HIGH confidence.
    """
    by_zone: dict[str, dict] = {"HIGH": {"blocked": 0, "n": 0},
                                  "MEDIUM": {"blocked": 0, "n": 0},
                                  "LOW": {"blocked": 0, "n": 0}}

    for s in scenarios:
        text = s.get("uncertain_statement", "")
        if text:
            zone = proxy.compute(text).zone
            by_zone[zone]["n"] += 1
            if s.get("probe_blocked", False):
                by_zone[zone]["blocked"] += 1

    return {
        "test": "C",
        "title": "Probe recall across J-score zones",
        "description": (
            "Probe must fire on uncertain inputs regardless of J-score zone. "
            "Any uncertain input that reaches HIGH-J is a ghost constraint — "
            "J-score and the probe both failed to catch it. "
            "If probe_blocked=True for all inputs, the probe is zone-agnostic."
        ),
        "by_zone": {
            zone: {
                "n": d["n"],
                "probe_blocked": d["blocked"],
                "recall": f"{d['blocked']}/{d['n']}" if d["n"] else "N/A",
            }
            for zone, d in by_zone.items()
        },
    }


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(a: dict, b: dict, c: dict) -> None:
    print()
    print("=" * 72)
    print("  CREDENCE CALIBRATION REPORT")
    print("=" * 72)

    # Test A
    print(f"\n  TEST A: {a['title']}")
    print(f"  {a['description']}")
    print()
    print(f"  {'Condition':<12}  {'N':>4}  {'Mean J':>8}  {'FCR':>6}  {'ECE':>6}")
    print("  " + "-" * 46)

    for cond_key, label in [("naive", "Naive Haiku"), ("probe", "Credence Probe")]:
        d = a[cond_key]
        print(f"  {label:<12}  {d['n']:>4}  {d['mean_j']:>8.3f}  "
              f"{d['fcr']:>6.3f}  {d['ece']:>6.4f}")

    print()
    print("  Calibration buckets (Naive Haiku — J of downstream answer):")
    print(f"  {'J-score bin':<14}  {'N':>4}  {'Mean J':>8}  {'P(FCR)':>8}  {'Gap':>6}")
    print("  " + "-" * 48)
    for row in a["naive"]["calibration_table"]:
        if row["n"] == 0:
            continue
        print(f"  {row['bin']:<14}  {row['n']:>4}  {row['mean_score']:>8.3f}  "
              f"{row['empirical_rate']:>8.3f}  {row['gap']:>6.3f}")

    # Test B
    print(f"\n  TEST B: {b['title']}")
    print(f"  {b['description']}")
    print()
    print(f"  n={b['n']}  mean J={b['mean_j']:.3f}")
    for zone, pct in b["zone_pct"].items():
        bar = "█" * (b["zone_distribution"][zone] * 2)
        print(f"  {zone:<8}: {pct}  {bar}")
    if b["high_j_inputs"]:
        print()
        print(f"  HIGH-J uncertain inputs ({len(b['high_j_inputs'])} ghost candidates):")
        for text in b["high_j_inputs"][:3]:
            print(f"    → {text}")

    # Test C
    print(f"\n  TEST C: {c['title']}")
    print(f"  {c['description']}")
    print()
    print(f"  {'Zone':<8}  {'N':>4}  {'Probe blocked':>14}  {'Recall':>8}")
    print("  " + "-" * 40)
    for zone, d in c["by_zone"].items():
        print(f"  {zone:<8}  {d['n']:>4}  {d['probe_blocked']:>14}  {d['recall']:>8}")

    print()
    print("  KEY FINDINGS:")
    naive_ece = a["naive"]["ece"]
    probe_ece = a["probe"]["ece"]
    print(f"  • Naive downstream ECE:  {naive_ece:.4f}  (lower = better calibrated)")
    print(f"  • Probe downstream ECE:  {probe_ece:.4f}  (0.0 = perfect, FCR=0%)")
    high_j_ghost_n = len(b["high_j_inputs"])
    print(f"  • Ghost constraint candidates (HIGH-J uncertain inputs): {high_j_ghost_n}")
    print(f"    These are the inputs the probe may miss — Ghost Detector targets these.")
    total_probe_blocked = sum(d["probe_blocked"] for d in c["by_zone"].values())
    total_n = sum(d["n"] for d in c["by_zone"].values())
    print(f"  • Probe recall across all zones: {total_probe_blocked}/{total_n}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out",  default="evals/calibration_curve_results.json")
    args = parser.parse_args()

    scenarios = load_scenarios()
    proxy     = CredenceProxy()

    print(f"Computing calibration on n={len(scenarios)} scenarios...")
    a = test_a_downstream_certainty(scenarios, proxy)
    b = test_b_input_uncertainty(scenarios, proxy)
    c = test_c_zone_accuracy(scenarios, proxy)

    if args.json:
        print(json.dumps({"test_a": a, "test_b": b, "test_c": c}, indent=2))
    else:
        print_report(a, b, c)

    with open(args.out, "w") as f:
        json.dump({"test_a": a, "test_b": b, "test_c": c}, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

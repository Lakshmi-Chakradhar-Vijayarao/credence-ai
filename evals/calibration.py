"""
evals/calibration.py
====================
Derives optimal J-proxy thresholds (theta_high, theta_low) from labelled examples.
Grid-searches theta values to maximize zone classification accuracy.

Run:
    python -m evals.calibration          # calibrate from built-in text examples
    python -m evals.calibration --api    # add live Claude responses (requires API key)

Results saved to evals/calibration.json.
"""

import os
import sys
import json
import math
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams.confidence_proxy import ConfidenceProxy

try:
    from anthropic import Anthropic
    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Labelled text examples with known confidence levels
# ---------------------------------------------------------------------------

CALIBRATION_TEXTS = [
    # HIGH confidence: factual, specific, anchored
    {"text": "The speed of light in a vacuum is exactly 299,792,458 meters per second.", "expected": "HIGH"},
    {"text": "Water is composed of two hydrogen atoms and one oxygen atom. Its chemical formula is H2O.", "expected": "HIGH"},
    {"text": "Python was created by Guido van Rossum and first released in 1991.", "expected": "HIGH"},
    {"text": "The Eiffel Tower is located in Paris, France. It was constructed between 1887 and 1889.", "expected": "HIGH"},
    {"text": "The square root of 144 is 12.", "expected": "HIGH"},
    {"text": "World War II ended in 1945 with Germany's surrender in May and Japan's in September.", "expected": "HIGH"},
    {"text": "DNA stands for deoxyribonucleic acid. It encodes genetic information in all living organisms.", "expected": "HIGH"},
    {"text": "Earth orbits the Sun at an average distance of approximately 149.6 million kilometers.", "expected": "HIGH"},

    # LOW confidence: hedging, speculative, uncertain
    {"text": "I think this might be related to quantum effects, but I'm not entirely certain. It's possible the mechanism involves entanglement, though I'm unsure.", "expected": "LOW"},
    {"text": "It seems like maybe the best approach could be one of several options. This depends on various factors and it's hard to say definitively.", "expected": "LOW"},
    {"text": "In general, I believe this is probably true, though there are many differing views. Some argue one way, others believe differently.", "expected": "LOW"},
    {"text": "I'm not sure about this. It might work, or perhaps there's another explanation. The research is ongoing and the answer is uncertain.", "expected": "LOW"},
    {"text": "This is an open question. Some researchers think X, while others believe Y. It's difficult to predict what will happen.", "expected": "LOW"},
    {"text": "I cannot say for certain. It could be several things. My understanding is limited here and I'm unsure about the details.", "expected": "LOW"},
    {"text": "Maybe around 50 or so? I'm not entirely sure. It depends on the context, I think. Perhaps more research would clarify.", "expected": "LOW"},

    # MEDIUM confidence: partially known, some uncertainty
    {"text": "Quantum entanglement is generally understood as a correlation between particle states. The mechanism is well-described mathematically, though its interpretation remains debated.", "expected": "MEDIUM"},
    {"text": "The Northern Lights are caused by charged particles from the sun interacting with Earth's magnetic field, producing light in the upper atmosphere.", "expected": "MEDIUM"},
    {"text": "Machine learning models typically improve with more data, but the relationship is not always linear and depends heavily on data quality and model architecture.", "expected": "MEDIUM"},
]


# ---------------------------------------------------------------------------
# Optional API questions for live-response calibration
# ---------------------------------------------------------------------------

API_QUESTIONS = [
    {"question": "What is 7 multiplied by 8?", "expected": "HIGH"},
    {"question": "What is the capital of Japan?", "expected": "HIGH"},
    {"question": "What year was Python first released?", "expected": "HIGH"},
    {"question": "What will the global economy look like in 2075?", "expected": "LOW"},
    {"question": "Is free will real?", "expected": "LOW"},
    {"question": "What causes human consciousness?", "expected": "LOW"},
    {"question": "How does a transformer neural network work?", "expected": "MEDIUM"},
    {"question": "What are the trade-offs between microservices and monoliths?", "expected": "MEDIUM"},
]


# ---------------------------------------------------------------------------
# Grid search over thresholds
# ---------------------------------------------------------------------------

def calibrate(items: list[dict]) -> dict:
    """Grid-search theta_high and theta_low to maximize zone accuracy."""
    proxy = ConfidenceProxy(theta_high=0.65, theta_low=0.35)

    scored = []
    for item in items:
        text = item.get("text") or item.get("response", "")
        cr = proxy.compute(text)
        scored.append({
            "j":        cr.j_score,
            "expected": item["expected"],
            "preview":  text[:80],
        })

    best_acc    = -1.0
    best_th     = 0.65
    best_tl     = 0.35
    best_report = {}

    # theta_high: 0.50 – 0.85 in steps of 0.05
    # theta_low:  0.15 – (theta_high - 0.10) in steps of 0.05
    for th_int in range(50, 90, 5):
        th = th_int / 100
        for tl_int in range(15, th_int - 5, 5):
            tl = tl_int / 100
            correct = 0
            by_zone = {z: {"correct": 0, "total": 0} for z in ("HIGH", "MEDIUM", "LOW")}

            for s in scored:
                if s["j"] >= th:
                    pred = "HIGH"
                elif s["j"] >= tl:
                    pred = "MEDIUM"
                else:
                    pred = "LOW"
                exp = s["expected"]
                by_zone[exp]["total"] += 1
                if pred == exp:
                    by_zone[exp]["correct"] += 1
                    correct += 1

            acc = correct / len(scored) if scored else 0
            if acc > best_acc:
                best_acc    = acc
                best_th     = th
                best_tl     = tl
                best_report = {k: dict(v) for k, v in by_zone.items()}

    return {
        "theta_high":     best_th,
        "theta_low":      best_tl,
        "accuracy":       round(best_acc, 4),
        "n_samples":      len(scored),
        "by_zone":        best_report,
        "scored_samples": scored,
    }


# ---------------------------------------------------------------------------
# OOF AUARC helper
# ---------------------------------------------------------------------------

def _auarc_from_pairs(pairs: list[dict]) -> float:
    """
    AUARC on labelled classification pairs.

    pairs: list of {"j": float, "correct": bool}

    Sort by J ascending. At each abstention cutoff (remove bottom k items),
    measure accuracy on remaining. AUARC = trapezoid area over abstention rates.
    > 0.5 means low-J items are harder to classify — J is a real signal.
    """
    if not pairs:
        return 0.0
    pairs_sorted = sorted(pairs, key=lambda x: x["j"])
    n = len(pairs_sorted)
    abstention_rates, accuracies = [], []
    for k in range(n + 1):
        retained = pairs_sorted[k:]
        abstention_rates.append(k / n)
        if retained:
            acc = sum(1 for p in retained if p["correct"]) / len(retained)
        else:
            acc = 1.0  # 100% accurate if we abstain on everything
        accuracies.append(acc)
    # trapezoid integration
    area = 0.0
    for i in range(1, len(abstention_rates)):
        dx = abstention_rates[i] - abstention_rates[i - 1]
        area += dx * (accuracies[i] + accuracies[i - 1]) / 2
    return area


# ---------------------------------------------------------------------------
# OOF 5-fold calibration
# ---------------------------------------------------------------------------

def calibrate_oof(items: list[dict], n_folds: int = 5, seed: int = 42) -> dict:
    """
    Out-of-fold calibration: train thresholds on 4 folds, evaluate on fold 5.

    Returns honest (non-inflated) accuracy and AUARC across folds.
    """
    proxy = ConfidenceProxy(theta_high=0.65, theta_low=0.35)

    # Score all items once
    scored = []
    for item in items:
        text = item.get("text") or item.get("response", "")
        cr = proxy.compute(text)
        scored.append({"j": cr.j_score, "expected": item["expected"]})

    # Shuffle deterministically
    rng = random.Random(seed)
    indices = list(range(len(scored)))
    rng.shuffle(indices)
    shuffled = [scored[i] for i in indices]

    # Split into folds
    fold_size = max(1, len(shuffled) // n_folds)
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else len(shuffled)
        folds.append(shuffled[start:end])

    fold_accuracies, fold_auarcs = [], []

    for k in range(n_folds):
        # Train: all folds except k
        train = [item for i, fold in enumerate(folds) for item in fold if i != k]
        val   = folds[k]
        if not val:
            continue

        # Fit thresholds on train fold
        best_acc, best_th, best_tl = -1.0, 0.65, 0.35
        for th_int in range(50, 90, 5):
            th = th_int / 100
            for tl_int in range(15, th_int - 5, 5):
                tl = tl_int / 100
                correct = sum(
                    1 for s in train
                    if (("HIGH" if s["j"] >= th else ("MEDIUM" if s["j"] >= tl else "LOW"))
                        == s["expected"])
                )
                acc = correct / len(train) if train else 0
                if acc > best_acc:
                    best_acc, best_th, best_tl = acc, th, tl

        # Evaluate on held-out fold
        pairs = []
        for s in val:
            pred = "HIGH" if s["j"] >= best_th else ("MEDIUM" if s["j"] >= best_tl else "LOW")
            pairs.append({"j": s["j"], "correct": pred == s["expected"]})

        val_acc = sum(1 for p in pairs if p["correct"]) / len(pairs)
        val_auarc = _auarc_from_pairs(pairs)
        fold_accuracies.append(val_acc)
        fold_auarcs.append(val_auarc)

    mean_acc   = sum(fold_accuracies) / len(fold_accuracies)
    std_acc    = math.sqrt(sum((x - mean_acc) ** 2 for x in fold_accuracies) / len(fold_accuracies))
    mean_auarc = sum(fold_auarcs) / len(fold_auarcs)
    std_auarc  = math.sqrt(sum((x - mean_auarc) ** 2 for x in fold_auarcs) / len(fold_auarcs))

    return {
        "n_folds":        n_folds,
        "n_samples":      len(scored),
        "oof_accuracy":   round(mean_acc, 4),
        "oof_acc_std":    round(std_acc, 4),
        "oof_auarc":      round(mean_auarc, 4),
        "oof_auarc_std":  round(std_auarc, 4),
        "fold_accuracies": [round(x, 4) for x in fold_accuracies],
        "fold_auarcs":     [round(x, 4) for x in fold_auarcs],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate J-proxy thresholds")
    parser.add_argument("--api", action="store_true",
                        help="Fetch live Claude responses (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    texts = list(CALIBRATION_TEXTS)

    if args.api:
        if not _CLIENT_AVAILABLE:
            print("anthropic not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY to use --api mode.")
            sys.exit(1)

        client = Anthropic(api_key=api_key)
        print(f"Fetching {len(API_QUESTIONS)} live Claude responses...")
        for item in API_QUESTIONS:
            resp = client.messages.create(
                model="claude-opus-4-7",
                messages=[{"role": "user", "content": item["question"]}],
                max_tokens=200,
            )
            texts.append({"text": resp.content[0].text, "expected": item["expected"]})
            print(f"  + [{item['expected']}] {item['question'][:55]}")

    print(f"\nCalibrating on {len(texts)} labelled examples...")

    # --- OOF: honest estimate first ---
    oof = calibrate_oof(texts)
    print(f"\n── OOF 5-Fold Calibration (honest, non-inflated) ──────────────")
    print(f"  Accuracy  = {oof['oof_accuracy']*100:.1f}% ± {oof['oof_acc_std']*100:.1f}%  "
          f"(per fold: {[f'{x*100:.0f}%' for x in oof['fold_accuracies']]})")
    print(f"  AUARC     = {oof['oof_auarc']:.4f} ± {oof['oof_auarc_std']:.4f}  "
          f"(per fold: {[f'{x:.3f}' for x in oof['fold_auarcs']]})")
    print(f"  n_samples = {oof['n_samples']}  n_folds = {oof['n_folds']}")
    if oof["oof_auarc"] > 0.5:
        print(f"  ✓ AUARC > 0.5 — J-proxy captures genuine uncertainty (not noise)")
    else:
        print(f"  ✗ AUARC ≤ 0.5 — J-proxy signal weak on this calibration set")

    # --- Full-data fit: best thresholds for deployment ---
    result = calibrate(texts)
    result["oof"] = oof

    print(f"\n── Full-Data Threshold Fit (for deployment) ───────────────────")
    print(f"  theta_high = {result['theta_high']:.2f}")
    print(f"  theta_low  = {result['theta_low']:.2f}")
    print(f"  In-sample accuracy = {result['accuracy'] * 100:.1f}%  (n={result['n_samples']})")
    print(f"\nPer-zone accuracy:")
    for zone in ("HIGH", "MEDIUM", "LOW"):
        stats = result["by_zone"].get(zone, {})
        if stats.get("total", 0) > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {zone:<8} {acc:5.1f}%  ({stats['correct']}/{stats['total']})")

    out_path = "evals/calibration.json"
    os.makedirs("evals", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

    default_th, default_tl = 0.65, 0.35
    if result["theta_high"] != default_th or result["theta_low"] != default_tl:
        print(f"\nNote: calibrated thresholds differ from defaults "
              f"(θ_h={default_th}, θ_l={default_tl}).")
        print(f"To use calibrated values:")
        print(f"  ConfidenceProxy(theta_high={result['theta_high']}, "
              f"theta_low={result['theta_low']})")
    else:
        print(f"\nDefault thresholds (θ_h=0.65, θ_l=0.35) are optimal for this dataset.")


if __name__ == "__main__":
    main()

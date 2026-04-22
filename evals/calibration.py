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
import argparse

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
    result = calibrate(texts)

    print(f"\nOptimal thresholds:")
    print(f"  theta_high = {result['theta_high']:.2f}")
    print(f"  theta_low  = {result['theta_low']:.2f}")
    print(f"  Accuracy   = {result['accuracy'] * 100:.1f}%  (n={result['n_samples']})")
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

"""
evals/e6_repeated.py
====================
Run E6 (Negative Needle) N times to produce bootstrap-CI validated results.

Usage:
    python -m evals.e6_repeated          # 20 trials (default)
    python -m evals.e6_repeated --n 30   # 30 trials

Output:
    evals/e6_repeated_results.json       # raw trial data + summary stats
"""

import os, sys, json, time, argparse, random
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from credence.context_manager import ContextManager


# ---------------------------------------------------------------------------
# Minimal inline _ask (mirrors experiments.py pattern)
# ---------------------------------------------------------------------------

_client: Anthropic | None = None

def _client_get() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _ask(messages: list[dict], *, system: str = "", max_tokens: int = 400) -> tuple[str, int, int]:
    kwargs: dict = dict(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        messages=messages,
    )
    if system:
        kwargs["system"] = system
    resp = _client_get().messages.create(**kwargs)
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


# ---------------------------------------------------------------------------
# E6 scenario constants
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a precise technical assistant helping plan a software integration. "
    "When asked to recall earlier facts, state them exactly as established, "
    "including any uncertainty flags."
)

CONSTRAINT_A_VALUE = "50"
CONSTRAINT_B_VALUE = "24"

CONVERSATION_SEED = [
    {"role": "user",      "content": "I'm building a rate-limited integration with an external API."},
    {"role": "assistant", "content": "Understood. I'll help you plan the integration."},
    {"role": "user",      "content": "We need to handle retries, backoff, and token refresh."},
    {"role": "assistant", "content": "Got it. Key concerns: rate limit adherence, exponential backoff on 429s, and proactive token refresh before expiry."},
    {"role": "user",      "content": "I'm not entirely certain, but I think our rate limit is around 50 requests per minute. I'll need to verify with the vendor."},
    {"role": "assistant", "content": "Noted. Rate limit is approximately 50 req/min — flagged as uncertain, needs vendor verification before implementation."},
    {"role": "user",      "content": "Regarding auth tokens: they expire in roughly 24 hours, but the docs aren't clear — it might be less. Please keep this tentative."},
    {"role": "assistant", "content": "Understood. Token expiry is approximately 24 hours, marked as tentative pending documentation clarification."},
]

FILLER_TURNS = [
    "What is exponential backoff and how should I implement it?",
    "What HTTP status code indicates rate limiting?",
    "What is the difference between access tokens and refresh tokens?",
    "What is a circuit breaker pattern?",
    "What is idempotency and why does it matter for retries?",
    "What is connection pooling?",
    "What does the Retry-After HTTP header do?",
    "What is a webhook?",
]

CALLBACKS = [
    (
        "What rate limit should we design our retry logic around?",
        ["50", "uncertain", "verify", "not certain", "tentative", "approximately",
         "roughly", "unverified", "unconfirmed", "pending", "flagged"],
        ["100", "200", "1000", "unlimited"],
    ),
    (
        "How long should our token refresh window be, based on what we discussed?",
        ["24", "uncertain", "tentative", "verify", "not certain", "approximately",
         "roughly", "unverified", "unconfirmed", "pending", "flagged"],
        ["48 hour", "12 hour", "6 hour", "7 day", "30 min"],
    ),
]


def _score_answer(answer: str, planted_val: str, correct_frags: list, hallu_frags: list) -> tuple[bool, bool]:
    lower = answer.lower()
    has_value = planted_val in lower
    has_qualifier = any(f.lower() in lower for f in correct_frags if f != planted_val)
    correct = has_value and has_qualifier
    hallu = any(h.lower() in lower for h in hallu_frags)
    return correct, hallu


# ---------------------------------------------------------------------------
# Single-trial runner per condition
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial: int
    condition: str
    correction_recall: float
    hallucination_rate: float
    callback_details: list[dict]


def run_single_trial_credence(trial_idx: int) -> TrialResult:
    mgr = ContextManager(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        theta_high=0.70, theta_low=0.45,
        system_prompt=SYSTEM, max_tokens=400,
    )
    for i in range(0, len(CONVERSATION_SEED), 2):
        mgr.chat(CONVERSATION_SEED[i]["content"])
        time.sleep(0.2)

    for fmsg in FILLER_TURNS:
        mgr.chat(fmsg)
        time.sleep(0.2)

    cb_details = []
    for q_idx, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
        r = mgr.chat(q)
        planted_val = CONSTRAINT_A_VALUE if q_idx == 0 else CONSTRAINT_B_VALUE
        correct, hallu = _score_answer(r.response, planted_val, correct_frags, hallu_frags)
        cb_details.append({"question": q_idx, "correct": correct, "hallucinated": hallu, "answer": r.response[:150]})
        print(f"    [credence t{trial_idx}] cb{q_idx}: correct={correct} hallu={hallu}")
        time.sleep(0.2)

    recall = sum(d["correct"] for d in cb_details) / len(cb_details)
    hallu  = sum(d["hallucinated"] for d in cb_details) / len(cb_details)
    return TrialResult(trial_idx, "credence", recall, hallu, cb_details)


def run_single_trial_condition(trial_idx: int, condition: str) -> TrialResult:
    """Run one trial for baseline or naive_window."""
    history = list(CONVERSATION_SEED)

    for fmsg in FILLER_TURNS:
        if condition == "naive_window":
            history = history[-12:]
        msgs = history + [{"role": "user", "content": fmsg}]
        answer, _, _ = _ask(msgs, system=SYSTEM)
        history.append({"role": "user", "content": fmsg})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.2)

    cb_details = []
    for q_idx, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
        if condition == "naive_window":
            history = history[-12:]
        msgs = history + [{"role": "user", "content": q}]
        answer, _, _ = _ask(msgs, system=SYSTEM)
        planted_val = CONSTRAINT_A_VALUE if q_idx == 0 else CONSTRAINT_B_VALUE
        correct, hallu = _score_answer(answer, planted_val, correct_frags, hallu_frags)
        cb_details.append({"question": q_idx, "correct": correct, "hallucinated": hallu, "answer": answer[:150]})
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        print(f"    [{condition} t{trial_idx}] cb{q_idx}: correct={correct} hallu={hallu}")
        time.sleep(0.2)

    recall = sum(d["correct"] for d in cb_details) / len(cb_details)
    hallu  = sum(d["hallucinated"] for d in cb_details) / len(cb_details)
    return TrialResult(trial_idx, condition, recall, hallu, cb_details)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    import random as rng
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_boot)
    )
    lo = (1.0 - ci) / 2
    return means[int(lo * n_boot)], means[int((1 - lo) * n_boot) - 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run E6 N times for bootstrap CI")
    parser.add_argument("--n", type=int, default=20, help="Number of trials (default 20)")
    parser.add_argument("--resume", action="store_true", help="Load existing results and add more trials")
    args = parser.parse_args()

    results_path = "evals/e6_repeated_results.json"
    all_trials: list[dict] = []

    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_trials = saved.get("trials", [])
        start_trial = max((t["trial"] for t in all_trials), default=-1) + 1
        print(f"Resuming from trial {start_trial}, {len(all_trials)} existing trials loaded")
    else:
        start_trial = 0

    for trial_idx in range(start_trial, start_trial + args.n):
        print(f"\n=== Trial {trial_idx + 1}/{start_trial + args.n} ===")

        for cond in ["credence", "naive_window", "baseline"]:
            print(f"  Running condition: {cond}")
            try:
                if cond == "credence":
                    result = run_single_trial_credence(trial_idx)
                else:
                    result = run_single_trial_condition(trial_idx, cond)
                all_trials.append(asdict(result))
            except Exception as e:
                print(f"  ERROR in trial {trial_idx} condition {cond}: {e}")
                continue

        # Save after each trial in case of interruption
        _save_and_summarise(all_trials, results_path)

    print("\n\n=== FINAL SUMMARY ===")
    _print_summary(all_trials)


def _save_and_summarise(all_trials: list[dict], path: str):
    by_cond: dict[str, dict[str, list]] = {}
    for t in all_trials:
        c = t["condition"]
        if c not in by_cond:
            by_cond[c] = {"correction_recall": [], "hallucination_rate": []}
        by_cond[c]["correction_recall"].append(t["correction_recall"])
        by_cond[c]["hallucination_rate"].append(t["hallucination_rate"])

    summary = {}
    for cond, vals in by_cond.items():
        recalls = vals["correction_recall"]
        hallus  = vals["hallucination_rate"]
        r_lo, r_hi = bootstrap_ci(recalls)
        h_lo, h_hi = bootstrap_ci(hallus)
        summary[cond] = {
            "n": len(recalls),
            "correction_recall_mean": round(sum(recalls) / len(recalls), 4),
            "correction_recall_ci95": [round(r_lo, 4), round(r_hi, 4)],
            "hallucination_rate_mean": round(sum(hallus) / len(hallus), 4),
            "hallucination_rate_ci95": [round(h_lo, 4), round(h_hi, 4)],
        }

    output = {"summary": summary, "trials": all_trials}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(all_trials: list[dict]):
    by_cond: dict[str, dict[str, list]] = {}
    for t in all_trials:
        c = t["condition"]
        if c not in by_cond:
            by_cond[c] = {"correction_recall": [], "hallucination_rate": []}
        by_cond[c]["correction_recall"].append(t["correction_recall"])
        by_cond[c]["hallucination_rate"].append(t["hallucination_rate"])

    print(f"\n{'Condition':<18}  {'N':>4}  {'Correction Recall':>20}  {'Hallucination Rate':>20}")
    print("-" * 72)
    for cond in ["credence", "baseline", "naive_window"]:
        if cond not in by_cond:
            continue
        recalls = by_cond[cond]["correction_recall"]
        hallus  = by_cond[cond]["hallucination_rate"]
        r_mean = sum(recalls) / len(recalls)
        h_mean = sum(hallus)  / len(hallus)
        r_lo, r_hi = bootstrap_ci(recalls)
        h_lo, h_hi = bootstrap_ci(hallus)
        print(
            f"  {cond:<16}  {len(recalls):>4}  "
            f"{r_mean:.1%} [{r_lo:.1%}, {r_hi:.1%}]  "
            f"{h_mean:.1%} [{h_lo:.1%}, {h_hi:.1%}]"
        )

    # Headline deltas
    if "credence" in by_cond and "naive_window" in by_cond:
        cr = sum(by_cond["credence"]["correction_recall"]) / len(by_cond["credence"]["correction_recall"])
        nr = sum(by_cond["naive_window"]["correction_recall"]) / len(by_cond["naive_window"]["correction_recall"])
        ch = sum(by_cond["credence"]["hallucination_rate"]) / len(by_cond["credence"]["hallucination_rate"])
        nh = sum(by_cond["naive_window"]["hallucination_rate"]) / len(by_cond["naive_window"]["hallucination_rate"])
        print(f"\n  Credence vs Naive — correction recall delta : {cr - nr:+.1%}")
        print(f"  Credence vs Naive — hallucination rate delta: {ch - nh:+.1%}")


if __name__ == "__main__":
    main()

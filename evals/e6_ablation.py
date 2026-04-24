"""
evals/e6_ablation.py
====================
E6 four-condition ablation: isolates which Credence layer drives hallucination
prevention on the Negative Needle scenario.

The multi-trial E6 (e6_repeated.py) showed credence hallu=4.35% slightly
exceeding baseline hallu=2.17%. This ablation tests whether the anomaly comes
from Truth Buffer over-injection (injecting uncertain constraints primes the
model to repeat the value confidently) or from some other interaction.

Four conditions (same E6 scenario for each):
  baseline         — full context, no compression, no registry injection
  faithfulness     — compression allowed + faithfulness probe; NO Truth Buffer
  truth_buffer     — NO compression (PRESERVE always); Truth Buffer injects constraints
  full_credence    — both faithfulness probe + Truth Buffer (standard Credence)

Expected hypothesis:
  If Truth Buffer is the cause of extra hallucination:
      truth_buffer hallu > baseline hallu  (injection primes wrong values)
      faithfulness hallu ≈ baseline hallu  (probe protects but no priming)
  If faithfulness probe alone is sufficient:
      faithfulness hallu ≤ baseline hallu  (probe blocks compression of uncertainty)
      truth_buffer hallu ≈ baseline hallu  (injection alone not harmful)

Run:
    python -m evals.e6_ablation
    python -m evals.e6_ablation --trials 3   # repeat N times, print aggregate

Outputs evals/e6_ablation_results.json.
Requires ANTHROPIC_API_KEY.
"""

import os, sys, json, time, argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from credence.context_manager import ContextManager
from credence.registry import CredenceRegistry

_CLIENT: Optional[Anthropic] = None

def _client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _CLIENT

_MODEL = "claude-opus-4-7"

def _ask(messages: list[dict], system: str = "", max_tokens: int = 400) -> tuple[str, int, int]:
    resp = _client().messages.create(
        model=_MODEL,
        system=system or "You are a helpful, precise assistant.",
        messages=messages,
        max_tokens=max_tokens,
    )
    return (
        next(b.text for b in resp.content if b.type == "text"),
        resp.usage.input_tokens,
        resp.usage.output_tokens,
    )


# ---- Shared E6 scenario constants ----------------------------------------

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
        [r"\b100\b", r"\b200\b", r"\b1000\b", r"\bunlimited\b"],
    ),
    (
        "How long should our token refresh window be, based on what we discussed?",
        ["24", "uncertain", "tentative", "verify", "not certain", "approximately",
         "roughly", "unverified", "unconfirmed", "pending", "flagged"],
        # Word-boundary regex to avoid substring FPs like "6 hour" in "16 hours"
        [r"\b48\s*hour", r"\b7\s*day", r"\bexpires?\s+in\s+6\b", r"\bexpiry\s+is\s+12\b"],
    ),
]


@dataclass
class AblationResult:
    condition: str
    correction_recall: float
    hallucination_rate: float
    tokens_used: int
    callback_answers: list[str] = field(default_factory=list)


def _score_callback(answer: str, turn_offset: int) -> tuple[bool, bool]:
    """Return (correct_recall, hallucinated) for a callback answer."""
    import re as _re
    _, correct_frags, hallu_frags = CALLBACKS[turn_offset]
    lower = answer.lower()
    planted_val = CONSTRAINT_A_VALUE if turn_offset == 0 else CONSTRAINT_B_VALUE
    has_value    = planted_val in lower
    has_qualifier = any(f.lower() in lower for f in correct_frags if f not in [planted_val])
    correct  = has_value and has_qualifier
    hallu    = any(_re.search(h, lower) for h in hallu_frags)
    return correct, hallu


# ---- Four condition runners -----------------------------------------------

def _run_baseline() -> AblationResult:
    """Full context, no compression, no registry."""
    history = list(CONVERSATION_SEED)
    tokens = 0
    for fmsg in FILLER_TURNS:
        msgs = history + [{"role": "user", "content": fmsg}]
        ans, ti, to = _ask(msgs, system=SYSTEM)
        tokens += ti + to
        history += [{"role": "user", "content": fmsg}, {"role": "assistant", "content": ans}]
        time.sleep(0.3)

    cb_correct, cb_hallu, cb_answers = [], [], []
    for i, (q, _, _) in enumerate(CALLBACKS):
        msgs = history + [{"role": "user", "content": q}]
        ans, ti, to = _ask(msgs, system=SYSTEM)
        tokens += ti + to
        correct, hallu = _score_callback(ans, i)
        cb_correct.append(correct)
        cb_hallu.append(hallu)
        cb_answers.append(ans[:120])
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        print(f"  [baseline] Q{i+1}: correct={correct} hallu={hallu}")
        time.sleep(0.3)

    n = len(cb_correct)
    return AblationResult(
        condition="baseline",
        correction_recall=sum(cb_correct) / n,
        hallucination_rate=sum(cb_hallu) / n,
        tokens_used=tokens,
        callback_answers=cb_answers,
    )


def _run_faithfulness_only() -> AblationResult:
    """
    Compression enabled + faithfulness probe active, but NO Truth Buffer.
    Registry is None — no injection. The probe prevents Haiku from stripping
    uncertainty markers from the old segment.
    """
    mgr = ContextManager(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        theta_high=0.70, theta_low=0.45,
        system_prompt=SYSTEM, max_tokens=400,
        registry=None,      # no Truth Buffer
        session_id=None,
    )
    tokens = 0
    for i in range(0, len(CONVERSATION_SEED), 2):
        r = mgr.chat(CONVERSATION_SEED[i]["content"])
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    for fmsg in FILLER_TURNS:
        r = mgr.chat(fmsg)
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    cb_correct, cb_hallu, cb_answers = [], [], []
    for i, (q, _, _) in enumerate(CALLBACKS):
        r = mgr.chat(q)
        tokens += r.tokens_in + r.tokens_out
        correct, hallu = _score_callback(r.response, i)
        cb_correct.append(correct)
        cb_hallu.append(hallu)
        cb_answers.append(r.response[:120])
        print(f"  [faithfulness_only] Q{i+1}: correct={correct} hallu={hallu}  decision={r.decision}")
        time.sleep(0.3)

    n = len(cb_correct)
    return AblationResult(
        condition="faithfulness_only",
        correction_recall=sum(cb_correct) / n,
        hallucination_rate=sum(cb_hallu) / n,
        tokens_used=tokens,
        callback_answers=cb_answers,
    )


def _run_truth_buffer_only() -> AblationResult:
    """
    Truth Buffer active, but compression ALWAYS returns PRESERVE (theta_high=1.1
    so nothing ever reaches COMPRESS). Isolates Truth Buffer's injection effect.
    Registry pre-loaded with both uncertain constraints.
    """
    db_path = "/tmp/e6_ablation_tb.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    reg = CredenceRegistry(db_path=db_path)
    sid = "e6_ablation_tb"

    # Pre-register both uncertain constraints exactly as planted in the seed
    reg.register(
        "Rate limit is approximately 50 req/min — uncertain, needs vendor verification",
        session_id=sid, j_score=0.28, zone="LOW",
    )
    reg.register(
        "Token expiry is approximately 24 hours — tentative, pending documentation clarification",
        session_id=sid, j_score=0.28, zone="LOW",
    )

    # theta_high=1.1 means nothing ever scores HIGH → compression never fires
    mgr = ContextManager(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        theta_high=1.1, theta_low=0.90,
        system_prompt=SYSTEM, max_tokens=400,
        registry=reg,
        session_id=sid,
    )

    tokens = 0
    for i in range(0, len(CONVERSATION_SEED), 2):
        r = mgr.chat(CONVERSATION_SEED[i]["content"])
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    for fmsg in FILLER_TURNS:
        r = mgr.chat(fmsg)
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    cb_correct, cb_hallu, cb_answers = [], [], []
    for i, (q, _, _) in enumerate(CALLBACKS):
        r = mgr.chat(q)
        tokens += r.tokens_in + r.tokens_out
        correct, hallu = _score_callback(r.response, i)
        cb_correct.append(correct)
        cb_hallu.append(hallu)
        cb_answers.append(r.response[:120])
        print(f"  [truth_buffer_only] Q{i+1}: correct={correct} hallu={hallu}"
              f"  tb_count={r.truth_buffer_count}  enforcement={r.enforcement_active}")
        time.sleep(0.3)

    reg.close()
    n = len(cb_correct)
    return AblationResult(
        condition="truth_buffer_only",
        correction_recall=sum(cb_correct) / n,
        hallucination_rate=sum(cb_hallu) / n,
        tokens_used=tokens,
        callback_answers=cb_answers,
    )


def _run_full_credence() -> AblationResult:
    """Full Credence: faithfulness probe + Truth Buffer + Consistency Enforcer."""
    db_path = "/tmp/e6_ablation_full.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    reg = CredenceRegistry(db_path=db_path)
    sid = "e6_ablation_full"

    reg.register(
        "Rate limit is approximately 50 req/min — uncertain, needs vendor verification",
        session_id=sid, j_score=0.28, zone="LOW",
    )
    reg.register(
        "Token expiry is approximately 24 hours — tentative, pending documentation clarification",
        session_id=sid, j_score=0.28, zone="LOW",
    )

    mgr = ContextManager(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        theta_high=0.70, theta_low=0.45,
        system_prompt=SYSTEM, max_tokens=400,
        registry=reg,
        session_id=sid,
    )

    tokens = 0
    for i in range(0, len(CONVERSATION_SEED), 2):
        r = mgr.chat(CONVERSATION_SEED[i]["content"])
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    for fmsg in FILLER_TURNS:
        r = mgr.chat(fmsg)
        tokens += r.tokens_in + r.tokens_out
        time.sleep(0.3)

    cb_correct, cb_hallu, cb_answers = [], [], []
    for i, (q, _, _) in enumerate(CALLBACKS):
        r = mgr.chat(q)
        tokens += r.tokens_in + r.tokens_out
        correct, hallu = _score_callback(r.response, i)
        cb_correct.append(correct)
        cb_hallu.append(hallu)
        cb_answers.append(r.response[:120])
        print(f"  [full_credence] Q{i+1}: correct={correct} hallu={hallu}"
              f"  decision={r.decision}  tb={r.truth_buffer_count}  ce={r.enforcement_active}")
        time.sleep(0.3)

    reg.close()
    n = len(cb_correct)
    return AblationResult(
        condition="full_credence",
        correction_recall=sum(cb_correct) / n,
        hallucination_rate=sum(cb_hallu) / n,
        tokens_used=tokens,
        callback_answers=cb_answers,
    )


# ---- Aggregate across trials -----------------------------------------------

def _bootstrap_ci(values: list[float], n_boot: int = 1000) -> tuple[float, float]:
    import random
    if not values:
        return 0.0, 0.0
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def run_ablation(trials: int = 1) -> list[dict]:
    all_results: dict[str, list[AblationResult]] = {
        "baseline": [], "faithfulness_only": [], "truth_buffer_only": [], "full_credence": [],
    }

    runners = [
        ("baseline",          _run_baseline),
        ("faithfulness_only", _run_faithfulness_only),
        ("truth_buffer_only", _run_truth_buffer_only),
        ("full_credence",     _run_full_credence),
    ]

    for trial in range(trials):
        print(f"\n=== E6 Ablation — Trial {trial+1}/{trials} ===")
        for cond, runner in runners:
            print(f"\n--- {cond} ---")
            result = runner()
            all_results[cond].append(result)
            print(f"  → correction_recall={result.correction_recall:.3f}  "
                  f"hallucination_rate={result.hallucination_rate:.3f}  "
                  f"tokens={result.tokens_used:,}")

    # Aggregate
    summary = []
    for cond in ["baseline", "faithfulness_only", "truth_buffer_only", "full_credence"]:
        recs = all_results[cond]
        recalls = [r.correction_recall for r in recs]
        hallus  = [r.hallucination_rate for r in recs]
        mean_recall = sum(recalls) / len(recalls)
        mean_hallu  = sum(hallus)  / len(hallus)
        recall_lo, recall_hi = _bootstrap_ci(recalls)
        hallu_lo,  hallu_hi  = _bootstrap_ci(hallus)
        summary.append({
            "condition":         cond,
            "n_trials":          len(recs),
            "correction_recall": round(mean_recall, 4),
            "recall_ci":         [round(recall_lo, 4), round(recall_hi, 4)],
            "hallucination_rate":round(mean_hallu, 4),
            "hallu_ci":          [round(hallu_lo, 4), round(hallu_hi, 4)],
            "trials":            [asdict(r) for r in recs],
        })

    return summary


def _print_summary(summary: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("E6 ABLATION RESULTS")
    print("=" * 72)
    print(f"{'Condition':<22} {'Corr.Recall':>12} {'CI':>16} {'Hallu.Rate':>12} {'CI':>16}")
    print("-" * 72)
    for s in summary:
        rc = s["correction_recall"]
        rlo, rhi = s["recall_ci"]
        hr = s["hallucination_rate"]
        hlo, hhi = s["hallu_ci"]
        print(f"{s['condition']:<22} {rc:>11.3f}  [{rlo:.3f},{rhi:.3f}]"
              f"  {hr:>10.3f}  [{hlo:.3f},{hhi:.3f}]  n={s['n_trials']}")
    print("=" * 72)

    # Interpretation hint
    print("\nHypothesis check:")
    conds = {s["condition"]: s for s in summary}
    base_h  = conds["baseline"]["hallucination_rate"]
    faith_h = conds["faithfulness_only"]["hallucination_rate"]
    tb_h    = conds["truth_buffer_only"]["hallucination_rate"]
    full_h  = conds["full_credence"]["hallucination_rate"]
    if tb_h > base_h + 0.05:
        print(f"  ⚠  Truth Buffer shows elevated hallu ({tb_h:.3f} vs baseline {base_h:.3f})")
        print(f"     → Truth Buffer injection may prime hallucination; consider tightening")
    elif faith_h <= base_h + 0.05:
        print(f"  ✓  Faithfulness probe hallu ({faith_h:.3f}) ≤ baseline ({base_h:.3f})")
        print(f"     → Faithfulness probe alone is sufficient for hallucination prevention")
    else:
        print(f"  ?  No clear winner yet — run more trials for statistical power")

    print(f"\nFull Credence correction recall: {conds['full_credence']['correction_recall']:.3f}")
    print(f"Faithfulness-only recall:        {conds['faithfulness_only']['correction_recall']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="E6 four-condition ablation")
    parser.add_argument("--trials", type=int, default=1, help="number of trials per condition")
    parser.add_argument("--out", default="evals/e6_ablation_results.json", help="output file")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    summary = run_ablation(trials=args.trials)
    _print_summary(summary)

    out_path = args.out
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []

    with open(out_path, "w") as f:
        json.dump({"runs": existing + [summary]}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

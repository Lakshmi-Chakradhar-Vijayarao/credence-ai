"""
evals/benchmark.py
==================
Benchmarks CAMS against two baselines on real Claude API calls:

  Baseline A — No compression: full context every turn (most expensive)
  Baseline B — Naive sliding window: drop turns older than N regardless
  CAMS       — Confidence-adaptive: compress only when J >= theta_high

Measures:
  1. Token usage per session
  2. Estimated cost ($)
  3. Answer quality (ROUGE-L F1 against reference answers)
  4. J-score calibration: do HIGH-zone answers score better quality?

Run:
    python -m evals.benchmark          # runs all 3 conditions, prints table
    python -m evals.benchmark --quick  # 3 questions only (fast smoke test)

Results saved to evals/results.json for the demo to load.
"""

import os
import sys
import json
import time
import re
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from anthropic import Anthropic
    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams.confidence_proxy import ConfidenceProxy
from cams.context_manager import CAMSContextManager, _cost

# ---------------------------------------------------------------------------
# Benchmark Q&A pairs — chosen to produce varied J-scores
# Reference answers are approximate; ROUGE-L used for partial match scoring
# ---------------------------------------------------------------------------

QA_PAIRS = [
    # --- HIGH confidence expected (factual, specific) ---
    {
        "question":  "What is the speed of light in a vacuum?",
        "reference": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "expected_zone": "HIGH",
    },
    {
        "question":  "Who wrote the play Hamlet?",
        "reference": "Hamlet was written by William Shakespeare around 1600-1601.",
        "expected_zone": "HIGH",
    },
    {
        "question":  "What is the chemical formula for water?",
        "reference": "The chemical formula for water is H2O.",
        "expected_zone": "HIGH",
    },
    {
        "question":  "In what year did World War II end?",
        "reference": "World War II ended in 1945.",
        "expected_zone": "HIGH",
    },
    {
        "question":  "What is the square root of 144?",
        "reference": "The square root of 144 is 12.",
        "expected_zone": "HIGH",
    },
    # --- LOW confidence expected (uncertain, contested, speculative) ---
    {
        "question":  "What will artificial intelligence look like in 50 years?",
        "reference": "AI in 50 years is highly uncertain and speculative.",
        "expected_zone": "LOW",
    },
    {
        "question":  "What is the best programming language to learn first?",
        "reference": "The best first language depends on goals and context.",
        "expected_zone": "LOW",
    },
    {
        "question":  "What are the long-term psychological effects of social media on teenagers?",
        "reference": "Research is ongoing and effects are debated among experts.",
        "expected_zone": "LOW",
    },
    # --- MEDIUM expected (partially known, some uncertainty) ---
    {
        "question":  "How does quantum entanglement work?",
        "reference": "Quantum entanglement is a phenomenon where two particles become correlated.",
        "expected_zone": "MEDIUM",
    },
    {
        "question":  "What causes the Northern Lights?",
        "reference": "The Northern Lights are caused by charged particles from the sun interacting with Earth's magnetic field and atmosphere.",
        "expected_zone": "MEDIUM",
    },
]

QUICK_PAIRS = QA_PAIRS[:3]  # smoke test


# ---------------------------------------------------------------------------
# ROUGE-L (unigram) — no external dependencies
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    """Simplified ROUGE-L: longest common subsequence F1 on words."""
    h = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    r = re.sub(r'[^\w\s]', '', reference.lower()).split()
    if not h or not r:
        return 0.0

    # LCS via DP
    m, n = len(h), len(r)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])

    lcs_len  = dp[m][n]
    precision = lcs_len / m if m else 0
    recall    = lcs_len / n if n else 0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass
class TurnLog:
    question:     str
    answer:       str
    rouge_l:      float
    j_score:      float
    zone:         str
    decision:     str
    tokens_in:    int
    tokens_out:   int
    tokens_saved: int


@dataclass
class ConditionResult:
    condition:          str
    turns:              list
    total_tokens_used:  int
    total_tokens_saved: int
    total_cost_usd:     float
    mean_rouge_l:       float
    mean_j_score:       float
    compression_ratio:  float


def run_cams(pairs: list[dict], client: Anthropic) -> ConditionResult:
    mgr    = CAMSContextManager(max_tokens=300)
    turns  = []

    for p in pairs:
        result = mgr.chat(p["question"])
        rl     = rouge_l(result.response, p["reference"])
        turns.append(TurnLog(
            question     = p["question"],
            answer       = result.response,
            rouge_l      = rl,
            j_score      = result.j_score,
            zone         = result.zone,
            decision     = result.decision,
            tokens_in    = result.tokens_in,
            tokens_out   = result.tokens_out,
            tokens_saved = result.tokens_saved,
        ))

    s = mgr.stats
    total_used  = s.total_tokens_in + s.total_tokens_out
    return ConditionResult(
        condition          = "CAMS",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = s.total_tokens_saved,
        total_cost_usd     = round(s.total_cost_usd, 4),
        mean_rouge_l       = round(sum(t.rouge_l for t in turns) / len(turns), 4),
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = round(s.compression_ratio, 3),
    )


def run_baseline(pairs: list[dict], client: Anthropic) -> ConditionResult:
    """Full context, no compression."""
    history = []
    turns   = []
    total_tokens_in = total_tokens_out = 0

    for p in pairs:
        history.append({"role": "user", "content": p["question"]})
        resp = client.messages.create(
            model      = "claude-opus-4-7",
            messages   = history,
            max_tokens = 300,
        )
        text  = resp.content[0].text
        t_in  = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        total_tokens_in  += t_in
        total_tokens_out += t_out
        history.append({"role": "assistant", "content": text})

        proxy = ConfidenceProxy()
        cr    = proxy.compute(text)
        rl    = rouge_l(text, p["reference"])
        turns.append(TurnLog(
            question=p["question"], answer=text, rouge_l=rl,
            j_score=cr.j_score, zone=cr.zone, decision="PRESERVE",
            tokens_in=t_in, tokens_out=t_out, tokens_saved=0,
        ))

    total_used = total_tokens_in + total_tokens_out
    return ConditionResult(
        condition          = "Baseline (no compression)",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = 0,
        total_cost_usd     = round(_cost(total_tokens_in, total_tokens_out), 4),
        mean_rouge_l       = round(sum(t.rouge_l for t in turns) / len(turns), 4),
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = 0.0,
    )


def run_naive_window(pairs: list[dict], client: Anthropic, window: int = 6) -> ConditionResult:
    """Sliding window: keep last N turns regardless of confidence."""
    history = []
    turns   = []
    total_tokens_in = total_tokens_out = total_tokens_saved = 0

    for p in pairs:
        history.append({"role": "user", "content": p["question"]})

        # Count tokens before trim
        tokens_before = sum(len(m["content"]) // 4 for m in history)

        # Trim to window
        if len(history) > window * 2:
            dropped        = history[:-(window * 2)]
            tokens_dropped = sum(len(m["content"]) // 4 for m in dropped)
            history        = history[-(window * 2):]
            total_tokens_saved += tokens_dropped
        else:
            tokens_dropped = 0

        resp = client.messages.create(
            model      = "claude-opus-4-7",
            messages   = history,
            max_tokens = 300,
        )
        text  = resp.content[0].text
        t_in  = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        total_tokens_in  += t_in
        total_tokens_out += t_out
        history.append({"role": "assistant", "content": text})

        proxy = ConfidenceProxy()
        cr    = proxy.compute(text)
        rl    = rouge_l(text, p["reference"])
        turns.append(TurnLog(
            question=p["question"], answer=text, rouge_l=rl,
            j_score=cr.j_score, zone=cr.zone, decision="TRIM",
            tokens_in=t_in, tokens_out=t_out, tokens_saved=tokens_dropped,
        ))

    total_used = total_tokens_in + total_tokens_out
    ratio = total_tokens_saved / (total_used + total_tokens_saved) if (total_used + total_tokens_saved) > 0 else 0
    return ConditionResult(
        condition          = "Naive sliding window",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = total_tokens_saved,
        total_cost_usd     = round(_cost(total_tokens_in, total_tokens_out), 4),
        mean_rouge_l       = round(sum(t.rouge_l for t in turns) / len(turns), 4),
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = round(ratio, 3),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[ConditionResult]):
    print("\n" + "=" * 70)
    print("CAMS BENCHMARK RESULTS")
    print("=" * 70)
    header = f"{'Condition':<28} {'Tokens used':>12} {'Tokens saved':>13} {'Cost ($)':>9} {'ROUGE-L':>8} {'Comp %':>7}"
    print(header)
    print("-" * 70)
    for r in results:
        pct = f"{r.compression_ratio * 100:.0f}%"
        print(
            f"{r.condition:<28} {r.total_tokens_used:>12,} "
            f"{r.total_tokens_saved:>13,} {r.total_cost_usd:>9.4f} "
            f"{r.mean_rouge_l:>8.3f} {pct:>7}"
        )
    print("=" * 70)

    # CAMS vs Baseline savings
    cams_r   = next(r for r in results if r.condition == "CAMS")
    base_r   = next(r for r in results if "Baseline" in r.condition)
    token_savings_pct = (base_r.total_tokens_used - cams_r.total_tokens_used) / base_r.total_tokens_used * 100
    cost_savings_pct  = (base_r.total_cost_usd   - cams_r.total_cost_usd)    / base_r.total_cost_usd   * 100
    quality_delta     = cams_r.mean_rouge_l - base_r.mean_rouge_l

    print(f"\nCAMS vs Baseline:")
    print(f"  Token reduction : {token_savings_pct:+.1f}%")
    print(f"  Cost reduction  : {cost_savings_pct:+.1f}%")
    print(f"  Quality delta   : {quality_delta:+.3f} ROUGE-L")
    print()

    # J-score calibration check
    proxy = ConfidenceProxy()
    high_quality = [t.rouge_l for r in results if r.condition == "CAMS"
                    for t in r.turns if t.zone == "HIGH"]
    low_quality  = [t.rouge_l for r in results if r.condition == "CAMS"
                    for t in r.turns if t.zone == "LOW"]
    if high_quality and low_quality:
        print(f"J-score calibration:")
        print(f"  HIGH-zone mean ROUGE-L : {sum(high_quality)/len(high_quality):.3f}")
        print(f"  LOW-zone  mean ROUGE-L : {sum(low_quality)/len(low_quality):.3f}")
        print(f"  (HIGH-zone answers should be shorter/more specific → higher ROUGE)")


def save_results(results: list[ConditionResult], path: str = "evals/results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            d = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, list):
                    d[k] = [serialise(i) for i in v]
                else:
                    d[k] = v
            return d
        return obj

    data = [serialise(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run 3 questions only")
    args = parser.parse_args()

    if not _CLIENT_AVAILABLE:
        print("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    pairs  = QUICK_PAIRS if args.quick else QA_PAIRS
    client = Anthropic(api_key=api_key)

    print(f"Running benchmark ({len(pairs)} questions, 3 conditions)...")
    print("This will make real API calls to claude-opus-4-7.\n")

    results = []
    for name, fn in [
        ("Baseline", lambda: run_baseline(pairs, client)),
        ("Naive window", lambda: run_naive_window(pairs, client)),
        ("CAMS", lambda: run_cams(pairs, client)),
    ]:
        print(f"  Running {name}...")
        t0 = time.perf_counter()
        r  = fn()
        print(f"  Done in {time.perf_counter()-t0:.1f}s  "
              f"tokens={r.total_tokens_used:,}  cost=${r.total_cost_usd:.4f}")
        results.append(r)

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()

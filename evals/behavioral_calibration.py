"""
evals/behavioral_calibration.py
================================
Behavioral Consistency as a Calibrated Uncertainty Signal.

RESEARCH QUESTION
-----------------
The J-proxy is a heuristic — it reads surface linguistic patterns.
Behavioral consistency (N samples → ROUGE-L variance) is a principled
signal: if the model answers consistently across samples, it's confident;
if it varies, it's uncertain.

Question: Does behavioral consistency correlate with ACTUAL factual
accuracy better than the J-proxy?  And is it well-calibrated?

DESIGN
------
60 questions split into three strata:

  high-confidence (20): Well-known facts Claude reliably gets right.
      Expected: high consistency, high accuracy.

  medium-confidence (20): Facts where Claude is sometimes right.
      Expected: medium consistency, ~50-70% accuracy.

  low-confidence (20): Rare, obscure, or tricky facts.
      Expected: low consistency, low accuracy.

For each question:
  1. Ask Claude Opus for the answer.
  2. Run behavioral consistency: N=5 Haiku samples, compute pairwise
     ROUGE-L variance → consistency_score ∈ [0,1].
  3. Compute J-proxy on the Opus answer.
  4. Check correctness against known ground truth.

METRICS
-------
  Expected Calibration Error (ECE) for behavioral consistency.
  ECE for J-proxy.
  Accuracy vs confidence bucketed curves (reliability diagrams).
  Spearman correlation: signal vs accuracy.

HYPOTHESIS
----------
  ECE(behavioral) < ECE(J-proxy)   — behavioral is better calibrated
  behavioral consistency predicts accuracy better than linguistic J

Run:
    python -m evals.behavioral_calibration
    python -m evals.behavioral_calibration --dry-run     # no API
    python -m evals.behavioral_calibration --n 20        # quick run

Results: evals/behavioral_calibration_results.json
"""

import os, sys, json, re, math, time, argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from cams.confidence_proxy import ConfidenceProxy
from cams.behavioral_signal import behavioral_consistency, fuse_scores

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Ground-truth QA dataset — 60 questions, 3 strata
# Format: (question, correct_answer_keywords, stratum)
# correct_answer_keywords: list of strings that must appear in a correct answer
# ---------------------------------------------------------------------------

QA_DATASET = [
    # ── HIGH CONFIDENCE (20) — well-known facts ───────────────────────────
    ("What is the capital of France?",
     ["paris"], "high"),
    ("What does HTTP stand for?",
     ["hypertext transfer protocol"], "high"),
    ("What is the time complexity of binary search?",
     ["o(log n)", "log n", "logarithmic"], "high"),
    ("What port does HTTPS use by default?",
     ["443"], "high"),
    ("What does SQL stand for?",
     ["structured query language"], "high"),
    ("What is the boiling point of water at sea level in Celsius?",
     ["100"], "high"),
    ("What is the result of 2 to the power of 10?",
     ["1024"], "high"),
    ("What does DNS stand for?",
     ["domain name system"], "high"),
    ("What is the Python keyword to define a function?",
     ["def"], "high"),
    ("What does CRUD stand for?",
     ["create", "read", "update", "delete"], "high"),
    ("What is the default port for PostgreSQL?",
     ["5432"], "high"),
    ("What HTTP status code means 'Not Found'?",
     ["404"], "high"),
    ("What does REST stand for?",
     ["representational state transfer"], "high"),
    ("What is the time complexity of a hash table lookup on average?",
     ["o(1)", "constant"], "high"),
    ("What does TLS stand for?",
     ["transport layer security"], "high"),
    ("What HTTP status code means 'OK'?",
     ["200"], "high"),
    ("What does API stand for?",
     ["application programming interface"], "high"),
    ("What is the default port for Redis?",
     ["6379"], "high"),
    ("What does JSON stand for?",
     ["javascript object notation"], "high"),
    ("What is the time complexity of quicksort on average?",
     ["o(n log n)", "n log n"], "high"),

    # ── MEDIUM CONFIDENCE (20) — moderately difficult ─────────────────────
    ("What is the default isolation level in PostgreSQL?",
     ["read committed"], "medium"),
    ("What HTTP status code is returned when a rate limit is exceeded?",
     ["429"], "medium"),
    ("What is the CAP theorem?",
     ["consistency", "availability", "partition"], "medium"),
    ("What does ACID stand for in databases?",
     ["atomicity", "consistency", "isolation", "durability"], "medium"),
    ("What is the default port for MongoDB?",
     ["27017"], "medium"),
    ("What does CORS stand for?",
     ["cross-origin resource sharing"], "medium"),
    ("What is the difference between TCP and UDP?",
     ["reliable", "unreliable", "connection"], "medium"),
    ("What is the default TTL for DNS records in seconds (common value)?",
     ["3600", "86400", "ttl"], "medium"),
    ("What does HMAC stand for?",
     ["hash-based message authentication code"], "medium"),
    ("What is the Fibonacci number at position 10 (1-indexed)?",
     ["55"], "medium"),
    ("What does GRPC stand for?",
     ["google remote procedure call", "remote procedure call"], "medium"),
    ("What is the default port for Kafka?",
     ["9092"], "medium"),
    ("What is the two-generals problem?",
     ["consensus", "network", "acknowledgment", "reliable"], "medium"),
    ("What does MVCC stand for in databases?",
     ["multiversion concurrency control"], "medium"),
    ("What HTTP method is idempotent but NOT safe?",
     ["put", "delete"], "medium"),
    ("What is the birthday paradox probability threshold (approx)?",
     ["23", "50%", "50 percent"], "medium"),
    ("What does SLA stand for?",
     ["service level agreement"], "medium"),
    ("What is the default connection timeout in most HTTP clients (common value)?",
     ["30", "60", "timeout"], "medium"),
    ("What does P99 latency mean?",
     ["99th percentile", "99 percent", "99%"], "medium"),
    ("What is consistent hashing used for?",
     ["distributed", "load", "node", "ring"], "medium"),

    # ── LOW CONFIDENCE (20) — rare, obscure, or ambiguous ─────────────────
    ("What is the exact default max_connections value in PostgreSQL 14?",
     ["100"], "low"),
    ("What is the Kolmogorov complexity of a string?",
     ["shortest", "program", "description", "length"], "low"),
    ("What year was the first version of Redis released?",
     ["2009"], "low"),
    ("What is the exact default heap size for the JVM in OpenJDK 17?",
     ["256", "512", "quarter", "physical"], "low"),
    ("What is the Zipf's law exponent for typical English text?",
     ["1", "approximately 1"], "low"),
    ("What is the default worker_processes setting in nginx?",
     ["auto", "1"], "low"),
    ("What was the original name of Python before it was called Python?",
     ["abc", "no previous", "directly python"], "low"),
    ("What is the exact idle_in_transaction_session_timeout default in PostgreSQL?",
     ["0", "disabled", "no limit"], "low"),
    ("What is the Boltzmann constant in SI units?",
     ["1.38", "joule", "kelvin"], "low"),
    ("How many prime numbers exist below 100?",
     ["25"], "low"),
    ("What is the default Linux TCP keepalive interval in seconds?",
     ["75", "tcp_keepalive_intvl"], "low"),
    ("What version of the HTTP spec introduced server-sent events?",
     ["1.1", "html5", "w3c"], "low"),
    ("What is the default stack size for a Python thread in bytes?",
     ["8", "mb", "8192", "1mb"], "low"),
    ("What does the 'tombstone' concept mean in distributed databases?",
     ["deletion", "marker", "deleted", "soft delete"], "low"),
    ("What is the CAP theorem impossibility result called formally?",
     ["brewer", "theorem", "cap"], "low"),
    ("What is the exact default wal_level in PostgreSQL 14?",
     ["replica"], "low"),
    ("What is the Paxos algorithm used for?",
     ["consensus", "distributed", "agreement"], "low"),
    ("What is the default linger time for TCP sockets (SO_LINGER) in Linux?",
     ["0", "disabled"], "low"),
    ("What was the exact version of Python when the GIL was first introduced?",
     ["1.5", "1", "original"], "low"),
    ("What is the exact default shared_buffers in PostgreSQL (as % of RAM)?",
     ["128mb", "128", "small", "default"], "low"),
]


# ---------------------------------------------------------------------------
# ROUGE-L (copy from benchmark to keep this module self-contained)
# ---------------------------------------------------------------------------

def _rouge_l(hyp: str, ref: str) -> float:
    """LCS-based ROUGE-L F1."""
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / m if m else 0.0
    r_ = lcs / n if n else 0.0
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def _pairwise_rouge_variance(texts: list[str]) -> float:
    """Mean pairwise ROUGE-L → consistency score ∈ [0,1]."""
    if len(texts) < 2:
        return 0.5
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(_rouge_l(texts[i], texts[j]))
    return sum(scores) / len(scores)


def _behavioral_consistency_local(
    question: str,
    context: str,
    client,
    n: int = 5,
    temperature: float = 0.8,
) -> float:
    """
    Local implementation of behavioral consistency — N samples → ROUGE-L mean.
    Avoids importing behavioral_signal's async logic.
    """
    samples = []
    for _ in range(n):
        resp = client.messages.create(
            model=_MODEL_HAIKU,
            messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
            max_tokens=80,
            temperature=temperature,
        )
        samples.append(resp.content[0].text.strip())
        time.sleep(0.1)
    return _pairwise_rouge_variance(samples)


def _is_correct(answer: str, keywords: list[str]) -> bool:
    """Check if any required keyword appears in the answer."""
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def _ece(confidences: list[float], accuracies: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error — mean |confidence - accuracy| over buckets."""
    bins = [[] for _ in range(n_bins)]
    for conf, acc in zip(confidences, accuracies):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, acc))
    ece = 0.0
    total = len(confidences)
    for b_items in bins:
        if not b_items:
            continue
        mean_conf = sum(c for c, _ in b_items) / len(b_items)
        mean_acc  = sum(a for _, a in b_items) / len(b_items)
        ece += (len(b_items) / total) * abs(mean_conf - mean_acc)
    return round(ece, 4)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0.0] * n
    rank_y = [0.0] * n
    for rank, idx in enumerate(rx):
        rank_x[idx] = rank
    for rank, idx in enumerate(ry):
        rank_y[idx] = rank
    d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    question:             str
    stratum:              str
    correct_answer_kws:   list[str]
    opus_answer:          str    = ""
    is_correct:           bool   = False
    j_score:              float  = 0.0
    behavioral_score:     float  = 0.0
    fused_score:          float  = 0.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_calibration(
    client,
    dataset: list[tuple],
    verbose: bool = False,
) -> list[QAResult]:

    proxy   = ConfidenceProxy()
    results = []

    for i, (question, keywords, stratum) in enumerate(dataset):
        # Step 1: Get Opus answer
        resp = client.messages.create(
            model=_MODEL_OPUS,
            messages=[{"role": "user", "content": question}],
            max_tokens=120,
        )
        answer = resp.content[0].text.strip()
        time.sleep(0.3)

        # Step 2: Check correctness
        correct = _is_correct(answer, keywords)

        # Step 3: J-proxy
        j = proxy.compute(answer).j_score

        # Step 4: Behavioral consistency (N=5 Haiku)
        consistency = _behavioral_consistency_local(
            question=question,
            context="Answer the following question briefly and precisely.",
            client=client,
            n=5,
        )
        time.sleep(0.2)

        # Step 5: Fused score
        fused = round(0.7 * j + 0.3 * consistency, 4)

        r = QAResult(
            question=question,
            stratum=stratum,
            correct_answer_kws=keywords,
            opus_answer=answer,
            is_correct=correct,
            j_score=j,
            behavioral_score=consistency,
            fused_score=fused,
        )
        results.append(r)

        if verbose:
            mark = "✓" if correct else "✗"
            print(f"  [{i+1:02d}/{len(dataset)}] {stratum:<6}  "
                  f"{mark}  J={j:.3f}  C={consistency:.3f}  "
                  f"Q: {question[:55]}…")

    return results


def aggregate_calibration(results: list[QAResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    acc       = [float(r.is_correct)     for r in results]
    j_scores  = [r.j_score               for r in results]
    b_scores  = [r.behavioral_score      for r in results]
    f_scores  = [r.fused_score           for r in results]

    # Per-stratum accuracy
    strata = {"high": [], "medium": [], "low": []}
    for r in results:
        strata[r.stratum].append(float(r.is_correct))
    stratum_acc = {k: round(sum(v)/len(v), 3) if v else 0.0
                   for k, v in strata.items()}

    return {
        "n":                    n,
        "overall_accuracy":     round(sum(acc) / n, 3),
        "stratum_accuracy":     stratum_acc,
        # ECE — lower is better calibrated
        "ece_j_proxy":          _ece(j_scores, acc),
        "ece_behavioral":       _ece(b_scores, acc),
        "ece_fused":            _ece(f_scores, acc),
        # Spearman correlation with accuracy — higher is better
        "spearman_j_proxy":     _spearman(j_scores, acc),
        "spearman_behavioral":  _spearman(b_scores, acc),
        "spearman_fused":       _spearman(f_scores, acc),
        # Mean scores per stratum
        "mean_j_high":          round(sum(r.j_score for r in results if r.stratum=="high")/max(1,len([r for r in results if r.stratum=="high"])),3),
        "mean_behavioral_high": round(sum(r.behavioral_score for r in results if r.stratum=="high")/max(1,len([r for r in results if r.stratum=="high"])),3),
        "mean_behavioral_low":  round(sum(r.behavioral_score for r in results if r.stratum=="low")/max(1,len([r for r in results if r.stratum=="low"])),3),
    }


def print_calibration_summary(agg: dict):
    print("\n" + "=" * 70)
    print("BEHAVIORAL CALIBRATION STUDY — RESULTS")
    print("=" * 70)
    print(f"  Questions: {agg['n']}   "
          f"Overall accuracy: {agg['overall_accuracy']:.1%}")
    print(f"  Accuracy by stratum: "
          f"high={agg['stratum_accuracy']['high']:.1%}  "
          f"medium={agg['stratum_accuracy']['medium']:.1%}  "
          f"low={agg['stratum_accuracy']['low']:.1%}")
    print()
    print("  CALIBRATION (ECE — lower is better):")
    print(f"    J-proxy ECE:                        {agg['ece_j_proxy']:.4f}")
    print(f"    Behavioral consistency ECE:          {agg['ece_behavioral']:.4f}")
    print(f"    Fused (70% J + 30% behavioral) ECE: {agg['ece_fused']:.4f}")
    print()
    print("  CORRELATION WITH ACCURACY (Spearman — higher is better):")
    print(f"    J-proxy:              {agg['spearman_j_proxy']:+.3f}")
    print(f"    Behavioral:           {agg['spearman_behavioral']:+.3f}")
    print(f"    Fused:                {agg['spearman_fused']:+.3f}")
    print()
    better = ("behavioral" if agg["ece_behavioral"] < agg["ece_j_proxy"]
              else "J-proxy")
    print(f"  VERDICT: {better} is better calibrated (lower ECE)")
    print("=" * 70)


def dry_run(n: int = 10):
    proxy = ConfidenceProxy()
    print(f"\n[dry-run] Checking {n} QA items (no API)...\n")
    for i, (q, kws, stratum) in enumerate(QA_DATASET[:n]):
        print(f"  [{i+1:02d}] {stratum:<6}  {q[:60]}…")
        print(f"        keywords: {kws}")
    print(f"\n[dry-run] {n} items validated. Run without --dry-run to execute.")


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Calibration Study")
    parser.add_argument("--n",       type=int, default=60,
                        help="Number of QA items to run (default: 60)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out",     default="evals/behavioral_calibration_results.json")
    args = parser.parse_args()

    dataset = QA_DATASET[:args.n]

    if args.dry_run:
        dry_run(args.n)
        return

    if not _ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    print(f"\nRunning behavioral calibration study ({len(dataset)} questions)...")
    print(f"Models: answer={_MODEL_OPUS}  consistency={_MODEL_HAIKU} (N=5 samples)\n")

    results = run_calibration(client, dataset, verbose=args.verbose or True)
    agg     = aggregate_calibration(results)
    print_calibration_summary(agg)

    output = {"summary": agg, "results": [asdict(r) for r in results]}
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

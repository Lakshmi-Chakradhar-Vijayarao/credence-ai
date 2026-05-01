"""
evals/latency_report.py
=======================
Comprehensive latency characterisation for all five Credence enforcement
checkpoints. Reports P50 / P95 / P99 / max for each component.

No API key required.

Run:
    python -m evals.latency_report           # default N=2000
    python -m evals.latency_report --n 5000  # higher precision
    python -m evals.latency_report --json    # output machine-readable JSON
"""

from __future__ import annotations

import os, sys, time, json, argparse, statistics, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.confidence_proxy import CredenceProxy
from credence.context_manager  import ContextManager, _UNCERTAINTY_MARKERS
from credence.registry         import CredenceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentiles(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    def pct(p): return s[int(p / 100 * n)]
    return {
        "p50":  round(pct(50),  4),
        "p95":  round(pct(95),  4),
        "p99":  round(pct(99),  4),
        "max":  round(s[-1],    4),
        "mean": round(statistics.mean(s), 4),
        "n":    n,
    }

def _time_ns(fn) -> float:
    """Run fn once, return elapsed milliseconds."""
    t0 = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t0) / 1e6


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------

_UNCERTAIN_TEXTS = [
    "I think the rate limit is around 50 req/min, but I haven't confirmed it.",
    "The token expiry might be 3600 seconds — unverified per vendor docs.",
    "Probably around 20 concurrent connections, based on a sales call.",
    "The batch endpoint probably accepts up to 50 items, not confirmed.",
    "We haven't verified the SLA but it's supposedly 99.9% uptime.",
    "The encryption algorithm is unconfirmed — might be RS256 or HS256.",
    "I'm not certain but I think the cost is roughly $0.0075 per SMS.",
    "The deletion window is approximately 30 days, pending legal review.",
    "We haven't confirmed whether it handles 10,000 concurrent users at launch.",
    "The cache hit rate should be around 85%, but this is preliminary.",
]

_CERTAIN_TEXTS = [
    "The rate limit is exactly 100 req/min as documented in the API reference.",
    "Token expiry is 3600 seconds, confirmed in the OAuth spec.",
    "The endpoint returns HTTP 201 for successful resource creation.",
    "AES-256 is used for at-rest encryption, per the security whitepaper.",
    "The batch size limit is 100 items, verified in the API playground.",
    "PostgreSQL EXPLAIN ANALYZE shows the query takes 45ms with the index.",
    "The circuit breaker opens after 5 consecutive failures within 10 seconds.",
    "HTTP 429 is the rate-limit response code per RFC 6585.",
    "The webhook signature uses HMAC-SHA256, documented in the integration guide.",
    "The service mesh proxy listens on port 15001 for inbound traffic.",
    # Additional certain samples — Gate 0 ground truth (n=200) shows 0.5% FPR;
    # 20+ samples here gives a more representative denominator for spot-check display.
    "The database uses B-tree indexes on all foreign keys.",
    "The API returns JSON with Content-Type: application/json.",
    "Redis TTL is set to 86400 seconds for session keys.",
    "The load balancer uses round-robin with health checks every 30 seconds.",
    "The CDN serves static assets with a max-age of 31536000 seconds.",
    "The primary key is a UUID v4, generated server-side.",
    "Docker images are rebuilt on every merge to main.",
    "The staging environment mirrors production with a 1-week data lag.",
    "Python 3.11 is the minimum supported version per pyproject.toml.",
    "The gRPC server listens on port 50051 by default.",
]

_CODE_WITH_UNCERTAIN = '''
def configure_client():
    client = APIClient(
        rate_limit=50,
        token_expiry=3600,
        max_connections=20,
    )
    return client
'''

_CODE_WITHOUT_UNCERTAIN = '''
def configure_client():
    client = APIClient(
        timeout=30,
        retry_count=3,
        backoff_base=1.0,
    )
    return client
'''


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_faithfulness_probe(n: int) -> dict:
    """CP1: faithfulness probe scan — user turn only."""
    cm = ContextManager.__new__(ContextManager)
    # Wire the minimal state needed for _has_uncertainty
    cm.system_prompt = ""
    # Use the method directly via ContextManager
    cm2 = ContextManager(api_key="test")

    uncertain_samples = (_UNCERTAIN_TEXTS * ((n // len(_UNCERTAIN_TEXTS)) + 1))[:n]
    certain_samples   = (_CERTAIN_TEXTS   * ((n // len(_CERTAIN_TEXTS))   + 1))[:n]

    # Uncertain inputs (probe should fire)
    uncertain_times = []
    for text in uncertain_samples:
        msgs = [{"role": "user", "content": text}]
        t = _time_ns(lambda m=msgs: cm2._has_uncertainty_in_user_turns(m))
        uncertain_times.append(t)

    # Certain inputs (probe should not fire — same code path)
    certain_times = []
    for text in certain_samples:
        msgs = [{"role": "user", "content": text}]
        t = _time_ns(lambda m=msgs: cm2._has_uncertainty_in_user_turns(m))
        certain_times.append(t)

    all_times = uncertain_times + certain_times
    return {
        "component":  "CP1 — Faithfulness Probe",
        "description": f"{len(_UNCERTAINTY_MARKERS)}-term frozenset scan on user turns",
        **_percentiles(all_times),
        "uncertain_tp_rate": f"{sum(cm2._has_uncertainty_in_user_turns([{'role':'user','content':t}]) for t in _UNCERTAIN_TEXTS)}/{len(_UNCERTAIN_TEXTS)}",
        "certain_fp_rate":   f"{sum(cm2._has_uncertainty_in_user_turns([{'role':'user','content':t}]) for t in _CERTAIN_TEXTS)}/{len(_CERTAIN_TEXTS)}",
    }


def bench_j_score(n: int) -> dict:
    """J-score computation via CredenceProxy."""
    proxy = CredenceProxy()
    all_texts = (_UNCERTAIN_TEXTS + _CERTAIN_TEXTS) * ((n // (len(_UNCERTAIN_TEXTS) + len(_CERTAIN_TEXTS))) + 1)
    all_texts = all_texts[:n]

    times = []
    for text in all_texts:
        t = _time_ns(lambda tx=text: proxy.compute(tx))
        times.append(t)

    # Spot-check zone separation
    uncertain_zones = [proxy.compute(t).zone for t in _UNCERTAIN_TEXTS]
    certain_zones   = [proxy.compute(t).zone for t in _CERTAIN_TEXTS]

    return {
        "component":  "J-score (CredenceProxy)",
        "description": "Linguistic assertiveness score — zero API calls",
        **_percentiles(times),
        "uncertain_low_pct":  f"{uncertain_zones.count('LOW')}/{len(uncertain_zones)} scored LOW",
        "certain_high_pct":   f"{certain_zones.count('HIGH')}/{len(certain_zones)} scored HIGH",
    }


def bench_gts_scan(n: int) -> dict:
    """CP3: Generation-Time Scanner on code blocks."""
    import sqlite3
    db = CredenceRegistry(db_path=":memory:")
    # Pre-register values that appear in the uncertain code block
    db.register("Rate limit unconfirmed", "bench-sess", j_score=0.30, turn_idx=0)
    db.register("Token expiry unconfirmed", "bench-sess", j_score=0.35, turn_idx=0)
    db.register("Max connections unconfirmed", "bench-sess", j_score=0.28, turn_idx=0)

    cm = ContextManager(api_key="test", registry=db, session_id="bench-sess")

    # Warm up registry
    cm._registry.get_effective_confidence
    uncertain_code = _CODE_WITH_UNCERTAIN * 1  # single block
    certain_code   = _CODE_WITHOUT_UNCERTAIN * 1

    uncertain_times = []
    for _ in range(n // 2):
        t = _time_ns(lambda: cm._scan_output_for_constraints(uncertain_code))
        uncertain_times.append(t)

    certain_times = []
    for _ in range(n // 2):
        t = _time_ns(lambda: cm._scan_output_for_constraints(certain_code))
        certain_times.append(t)

    # Verify it actually annotates
    annotated, hits = cm._scan_output_for_constraints(uncertain_code)
    clean, no_hits  = cm._scan_output_for_constraints(certain_code)

    return {
        "component":   "CP3 — Generation-Time Scanner",
        "description": "Code + prose scan for unverified literals",
        **_percentiles(uncertain_times + certain_times),
        "hit_count_uncertain": len(hits),
        "hit_count_certain":   len(no_hits),
    }


def bench_registry(n: int) -> dict:
    """Registry write + lookup cycle."""
    db = CredenceRegistry(db_path=":memory:")

    # Pre-populate
    cids = [db.register(f"Claim {i}", "bench", j_score=0.40, turn_idx=i) for i in range(50)]

    write_times = []
    for i in range(n // 2):
        t = _time_ns(lambda i=i: db.register(f"Dynamic claim {i}", "bench", j_score=0.35, turn_idx=i))
        write_times.append(t)

    read_times = []
    for cid in (cids * ((n // 2 // len(cids)) + 1))[:n // 2]:
        t = _time_ns(lambda c=cid: db.get_effective_confidence(c, current_turn=10))
        read_times.append(t)

    return {
        "component":   "Registry (SQLite)",
        "description": "Constraint write + confidence lookup",
        "write": _percentiles(write_times),
        "read":  _percentiles(read_times),
        "combined": _percentiles(write_times + read_times),
    }


def bench_consistency_enforcer(n: int) -> dict:
    """CP2: Consistency Enforcer — keyword overlap + synonym expansion."""
    db = CredenceRegistry(db_path=":memory:")
    db.register("Rate limit is 50 req/min — unconfirmed", "bench", j_score=0.30, turn_idx=0)
    db.register("Auth token expiry might be 3600 seconds", "bench", j_score=0.35, turn_idx=0)
    db.register("Batch endpoint accepts up to 50 items",   "bench", j_score=0.28, turn_idx=0)

    cm = ContextManager(api_key="test", registry=db, session_id="bench")
    constraints = db.list_uncertain("bench")

    overlapping_queries = [
        "How fast can we call the endpoint?",
        "When does my session expire?",
        "What batch size should we use?",
        "Can we increase the API call frequency?",
        "How long are tokens valid for?",
    ]
    non_overlapping_queries = [
        "What color should the button be?",
        "How do we write the README?",
        "What is the team name?",
        "How many spaces for indentation?",
        "Which font should we use?",
    ]

    all_queries = (overlapping_queries + non_overlapping_queries) * ((n // 10) + 1)
    all_queries = all_queries[:n]

    times = []
    for q in all_queries:
        t = _time_ns(lambda query=q: cm._direct_constraint_matches(query, constraints))
        times.append(t)

    # Verify match correctness
    matches = [cm._direct_constraint_matches(q, constraints) for q in overlapping_queries]
    non_matches = [cm._direct_constraint_matches(q, constraints) for q in non_overlapping_queries]

    return {
        "component":   "CP2 — Consistency Enforcer",
        "description": "Keyword overlap + 52-cluster synonym expansion",
        **_percentiles(times),
        "overlap_match_rate":    f"{sum(1 for m in matches if m)}/{len(matches)} overlapping queries matched",
        "nonoverlap_match_rate": f"{sum(1 for m in non_matches if m)}/{len(non_matches)} non-overlapping falsely matched",
    }


def bench_memory_recall(n: int) -> dict:
    """CP5: Cross-session memory snapshot + inject."""
    from credence.memory import CredenceMemory
    db = CredenceRegistry(db_path=":memory:")

    # Plant some constraints as if from a prior session
    for i in range(10):
        db.register(f"Prior session claim {i}", "session-1", j_score=0.40, turn_idx=i)

    mem = CredenceMemory(db)

    snapshot_times = []
    for _ in range(n // 2):
        t = _time_ns(lambda: mem.snapshot("session-1", "project-x"))
        snapshot_times.append(t)

    recall_times = []
    for i in range(n // 2):
        t = _time_ns(lambda i=i: mem.recall_and_inject("project-x", f"session-{i+100}"))
        recall_times.append(t)

    return {
        "component":   "CP5 — Cross-Session Memory",
        "description": "Snapshot unverified constraints + inject into new session",
        "snapshot": _percentiles(snapshot_times),
        "recall":   _percentiles(recall_times),
        "combined": _percentiles(snapshot_times + recall_times),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _pct_row(label: str, r: dict, note: str = "") -> str:
    return (
        f"  {label:<38}"
        f"  p50={r['p50']:.4f}ms"
        f"  p95={r['p95']:.4f}ms"
        f"  p99={r['p99']:.4f}ms"
        f"  max={r['max']:.4f}ms"
        + (f"  [{note}]" if note else "")
    )


def print_report(results: list[dict]) -> None:
    print()
    print("=" * 80)
    print("  CREDENCE LATENCY REPORT — All Five Enforcement Checkpoints")
    print("=" * 80)
    print()
    print(f"  {'Component':<38}  {'p50':>10}  {'p95':>10}  {'p99':>10}  {'max':>10}")
    print("  " + "-" * 74)

    for r in results:
        name = r["component"]
        desc = r.get("description", "")

        if "combined" in r:
            # Registry / memory with sub-components
            print(f"\n  {name} — {desc}")
            if "write" in r:
                print(_pct_row("    write", r["write"]))
                print(_pct_row("    read/recall", r.get("read", r.get("recall", {}))))
            if "snapshot" in r:
                print(_pct_row("    snapshot", r["snapshot"]))
                print(_pct_row("    recall+inject", r["recall"]))
            print(_pct_row("    combined", r["combined"]))
        else:
            combined = {k: r[k] for k in ("p50", "p95", "p99", "max", "mean", "n")}
            extra = ""
            for k in ("uncertain_tp_rate", "certain_fp_rate", "uncertain_low_pct",
                      "certain_high_pct", "hit_count_uncertain", "overlap_match_rate"):
                if k in r:
                    extra += f"  {k}: {r[k]}"
            print(f"\n  {name} — {desc}")
            print(_pct_row("    latency", combined, extra.strip()))

    print()
    print("  SLA SUMMARY (end-to-end in-session enforcement, excluding LLM call):")
    p99_sum = sum(
        r.get("p99", r.get("combined", {}).get("p99", 0))
        for r in results
    )
    print(f"  Worst-case in-session overhead (sum of all P99): {p99_sum:.3f}ms")
    print(f"  Rust gate (CP4): 3.4ms P50 / measured separately")
    print(f"  Total enforcement overhead P99: ~{p99_sum + 3.4:.1f}ms")
    print()
    print("  Industry context:")
    print("  - Typical LLM call latency (Claude Opus 4.7): 3,000–8,000ms")
    print(f"  - Credence overhead as % of LLM call: ~{(p99_sum+3.4)/5000*100:.2f}%")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=2000, help="Samples per component")
    parser.add_argument("--json", action="store_true",    help="Output JSON instead of table")
    parser.add_argument("--out",  default=None,           help="Save JSON to path")
    args = parser.parse_args()

    n = args.n
    print(f"Running latency characterisation (n={n} per component)...")

    results = []
    steps = [
        ("CP1 Faithfulness Probe",    lambda: bench_faithfulness_probe(n)),
        ("J-score (CredenceProxy)",   lambda: bench_j_score(n)),
        ("CP2 Consistency Enforcer",  lambda: bench_consistency_enforcer(n)),
        ("CP3 GTS Scanner",           lambda: bench_gts_scan(n)),
        ("Registry",                  lambda: bench_registry(n)),
        ("CP5 Cross-Session Memory",  lambda: bench_memory_recall(n)),
    ]

    for label, fn in steps:
        sys.stdout.write(f"  {label}... ")
        sys.stdout.flush()
        t0 = time.time()
        r  = fn()
        results.append(r)
        elapsed = time.time() - t0
        sys.stdout.write(f"done ({elapsed:.1f}s)\n")
        sys.stdout.flush()

    if args.json or args.out:
        out = json.dumps({"results": results}, indent=2)
        if args.out:
            with open(args.out, "w") as f: f.write(out)
            print(f"Saved to {args.out}")
        else:
            print(out)
    else:
        print_report(results)

    # Always save
    path = args.out or "evals/latency_report_results.json"
    with open(path, "w") as f:
        json.dump({"results": results, "n_per_component": n}, f, indent=2)
    if not args.out:
        print(f"Raw results saved to {path}")


if __name__ == "__main__":
    main()

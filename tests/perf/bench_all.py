"""
bench_all.py — Performance benchmarks for all deterministic components.
No API key required.

Targets:
  Probe:    < 0.1ms per call
  Registry: < 5ms per operation
  Gate:     < 5ms per call
  Wrap:     < 2ms overhead (excluding compress_fn)

Run:
    python3 -m tests.perf.bench_all
    python3 -m tests.perf.bench_all --n 2000
"""

import sys, time, tempfile, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from credence.context_manager import _UNCERTAINTY_MARKERS, ContextManager
from credence.registry import CredenceRegistry
from credence.wrap import wrap

_cm = ContextManager.__new__(ContextManager)

CERTAIN_TEXT = (
    "The rate limit is 100 req/min. The endpoint is confirmed at /api/v2. "
    "Authentication uses Bearer tokens. The timeout is 30 seconds."
)
UNCERTAIN_TEXT = (
    "I think the rate limit might be around 50 req/min, but I am not certain. "
    "The timeout is probably 30 seconds, though it may vary. "
    "Authentication might require additional configuration."
)


def bench(name: str, fn, N: int = 1000, target_ms: float = None) -> dict:
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(times)
    p95 = sorted(times)[int(N * 0.95)]
    p99 = sorted(times)[int(N * 0.99)]
    passed = (mean < target_ms) if target_ms else None
    return {"name": name, "n": N, "mean_ms": round(mean, 4),
            "p95_ms": round(p95, 4), "p99_ms": round(p99, 4),
            "target_ms": target_ms, "passed": passed}


def run(N: int = 1000) -> list[dict]:
    results = []

    # Probe — certain text
    results.append(bench(
        "probe_certain", lambda: _cm._has_uncertainty(CERTAIN_TEXT), N, 0.1
    ))
    # Probe — uncertain text
    results.append(bench(
        "probe_uncertain", lambda: _cm._has_uncertainty(UNCERTAIN_TEXT), N, 0.1
    ))
    # Probe — long text (50× repetition)
    long_text = CERTAIN_TEXT * 50
    results.append(bench(
        "probe_long_text_2500_words", lambda: _cm._has_uncertainty(long_text), N, 1.0
    ))

    # Registry
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    reg = CredenceRegistry(db_path=db_path)
    # Pre-register some items
    cids = [reg.register(f"constraint {i}", "s1", 0.3, "LOW") for i in range(20)]
    results.append(bench(
        "registry_list_uncertain_20_items",
        lambda: reg.list_uncertain("s1"), N, 5.0
    ))
    results.append(bench(
        "registry_register",
        lambda: reg.register("new uncertain constraint", "bench", 0.3, "LOW"), N//10, 5.0
    ))

    # Wrap — probe fires (no compress_fn call)
    results.append(bench(
        "wrap_probe_blocks",
        lambda: wrap(lambda t: t, context=UNCERTAIN_TEXT), N, 2.0
    ))
    # Wrap — probe clears (identity compress_fn)
    results.append(bench(
        "wrap_probe_clears",
        lambda: wrap(lambda t: t[:len(t)//2], context=CERTAIN_TEXT), N, 2.0
    ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 65)
    print("CREDENCE — Performance Benchmarks")
    print("=" * 65)
    print(f"{'Component':<40} {'Mean':>8} {'P95':>8} {'P99':>8}  {'Status'}")
    print("-" * 65)

    results = run(args.n)
    all_pass = True
    for r in results:
        status = ""
        if r["passed"] is not None:
            status = "✓" if r["passed"] else f"✗ (target {r['target_ms']}ms)"
            if not r["passed"]:
                all_pass = False
        print(f"{r['name']:<40} {r['mean_ms']:>7.3f}ms {r['p95_ms']:>7.3f}ms "
              f"{r['p99_ms']:>7.3f}ms  {status}")

    print("-" * 65)
    print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

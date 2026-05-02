"""
test_perf.py — pytest wrapper for performance benchmarks.
Calls bench_all.run() and asserts each component meets its latency target.
No API key required.

These tests fail the CI build if any deterministic component regresses past its
target. Targets are conservative (2–5× measured P99) to avoid flaky failures.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from tests.perf.bench_all import run


@pytest.fixture(scope="module")
def bench_results():
    return {r["name"]: r for r in run(N=500)}


def test_probe_certain_under_target(bench_results):
    r = bench_results["probe_certain"]
    assert r["mean_ms"] < 0.1, f"probe_certain mean {r['mean_ms']:.4f}ms > 0.1ms"


def test_probe_uncertain_under_target(bench_results):
    r = bench_results["probe_uncertain"]
    assert r["mean_ms"] < 0.1, f"probe_uncertain mean {r['mean_ms']:.4f}ms > 0.1ms"


def test_probe_long_text_under_target(bench_results):
    r = bench_results["probe_long_text_2500_words"]
    assert r["mean_ms"] < 1.0, f"probe_long_text mean {r['mean_ms']:.4f}ms > 1.0ms"


def test_registry_list_under_target(bench_results):
    r = bench_results["registry_list_uncertain_20_items"]
    assert r["mean_ms"] < 5.0, f"registry_list mean {r['mean_ms']:.4f}ms > 5.0ms"


def test_registry_register_under_target(bench_results):
    r = bench_results["registry_register"]
    assert r["mean_ms"] < 5.0, f"registry_register mean {r['mean_ms']:.4f}ms > 5.0ms"


def test_wrap_probe_blocks_under_target(bench_results):
    r = bench_results["wrap_probe_blocks"]
    assert r["mean_ms"] < 2.0, f"wrap_probe_blocks mean {r['mean_ms']:.4f}ms > 2.0ms"


def test_wrap_probe_clears_under_target(bench_results):
    r = bench_results["wrap_probe_clears"]
    assert r["mean_ms"] < 2.0, f"wrap_probe_clears mean {r['mean_ms']:.4f}ms > 2.0ms"



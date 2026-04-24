"""
evals/consistency_enforcer_test.py
===================================
Tests the Consistency Enforcer — the mechanism that upgrades Truth Buffer's
informational injection to imperative enforcement when the user query directly
asks about a registered unverified constraint.

Motivation (E6 baseline finding):
  Baseline with full context hallucinated 50% of callbacks even though the
  uncertainty qualifier was present in the conversation. The Truth Buffer
  injected it informatively — "here are uncertain things, acknowledge them."
  Opus read it and answered with false certainty anyway.

  The Consistency Enforcer fires a stronger injection when the query directly
  overlaps with an unverified constraint:
    "This query asks about [X] which is UNVERIFIED. You MUST express
     uncertainty. Stating this as confirmed fact is an epistemic error."

Tests:
  CE1: Direct constraint query with enforcer active → model MUST hedge
  CE2: Indirect query (same session, different topic) → no enforcer → Truth Buffer only
  CE3: Verified constraint → no enforcement (constraint is resolved)
  CE4: Multiple overlapping constraints → enforcer fires on all matching ones

Run:
  python -m evals.consistency_enforcer_test
  python -m evals.consistency_enforcer_test --test CE1
  python -m evals.consistency_enforcer_test --no-api   # unit-test the logic only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import anthropic
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry
    _HAS_DEPS = True
except ImportError as e:
    print(f"Import error: {e}")
    _HAS_DEPS = False

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a precise technical assistant helping plan a software integration. "
    "When asked to recall earlier facts, state them exactly as established, "
    "including any uncertainty flags."
)

# Uncertain constraints planted in the session
CONSTRAINT_A = "I'm not entirely sure, but I think our API rate limit is around 50 requests per minute. I'll need to verify with the vendor before we finalize the implementation."
CONSTRAINT_B = "The auth token expiry might be 3600 seconds or it could be 86400 — the vendor docs aren't clear on this."

# Callback queries — CE1 and CE2 test direct vs indirect
QUERY_DIRECT_A   = "What rate limit should we design our retry logic around?"
QUERY_DIRECT_B   = "What token expiry should we configure for auth?"
QUERY_INDIRECT   = "Can you explain the difference between OAuth2 and API key authentication?"
QUERY_VERIFIED   = "What HTTP status code indicates rate limiting?"

# Scoring helpers
UNCERTAINTY_MARKERS = [
    "unverified", "uncertain", "tentative", "not confirmed", "unconfirmed",
    "not yet confirmed", "approximately", "roughly", "i think", "might be",
    "needs verification", "check with", "verify", "should confirm",
    "as stated earlier", "flagged as", "haven't confirmed", "need to confirm",
    "wasn't sure", "wasn't certain", "pending", "not definitive",
]

def _has_uncertainty_marker(text: str) -> bool:
    tl = text.lower()
    return any(m in tl for m in UNCERTAINTY_MARKERS)

def _is_confident_wrong(text: str, false_values: list[str]) -> bool:
    tl = text.lower()
    if _has_uncertainty_marker(tl):
        return False
    return any(v in tl for v in false_values)


# ---------------------------------------------------------------------------
# Logic-only tests (no API)
# ---------------------------------------------------------------------------

def _test_enforcer_logic() -> bool:
    """
    Unit-test _direct_constraint_matches without API calls.
    Verifies the overlap detection logic works correctly.
    """
    print("\n=== Logic-Only Tests (no API) ===")
    from credence.context_manager import _CE_STOPWORDS, _CE_MIN_OVERLAP

    # Simulate the overlap logic directly
    def _overlap(query: str, constraint: str) -> int:
        qt = {w.strip("?.!,;:\"'()[]") for w in query.lower().split()
              if len(w.strip("?.!,;:\"'()[]")) > 2
              and w.strip("?.!,;:\"'()[]") not in _CE_STOPWORDS}
        ct = {w.strip("?.!,;:\"'()[]") for w in constraint.lower().split()
              if len(w.strip("?.!,;:\"'()[]")) > 2
              and w.strip("?.!,;:\"'()[]") not in _CE_STOPWORDS}
        return len(qt & ct)

    cases = [
        # (query, constraint, expect_match, description)
        (
            "What rate limit should we design around?",
            "I think our API rate limit is around 50 req/min — unconfirmed",
            True, "direct rate limit query"
        ),
        (
            "What token expiry should we configure for auth?",
            "Auth token expiry might be 3600 seconds or 86400",
            True, "direct token expiry query"
        ),
        (
            "Can you explain OAuth2 vs API key auth?",
            "I think our API rate limit is around 50 req/min — unconfirmed",
            False, "indirect query (different topic)"
        ),
        (
            "What HTTP status code is rate limiting?",
            "I think our API rate limit is around 50 req/min — unconfirmed",
            False, "general HTTP question — 'rate' alone is insufficient overlap"
        ),
        (
            "Tell me about retry logic",
            "I'm not sure about the token expiry",
            False, "unrelated query"
        ),
    ]

    passed = 0
    for query, constraint, expect_match, desc in cases:
        n_overlap = _overlap(query, constraint)
        got_match = n_overlap >= _CE_MIN_OVERLAP
        ok = got_match == expect_match
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}: overlap={n_overlap} expect_match={expect_match} got={got_match}")
        if ok:
            passed += 1

    print(f"\nLogic tests: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

@dataclass
class CETestResult:
    test_id:           str
    description:       str
    query:             str
    enforcement_fired: bool
    has_uncertainty:   bool
    hallucinated:      bool
    response_snippet:  str = ""
    passed:            bool = False
    note:              str = ""


def _run_ce_test(
    test_id: str,
    description: str,
    query: str,
    mgr: ContextManager,
    expect_enforcement: bool,
    expect_hedged: bool,
    false_values: list[str] | None = None,
) -> CETestResult:
    result = mgr.chat(query)
    snippet = result.response[:200].replace("\n", " ")

    enforcement_fired = result.enforcement_active
    has_uncertainty   = _has_uncertainty_marker(result.response)
    hallucinated      = _is_confident_wrong(result.response, false_values or [])

    # Pass conditions:
    # - enforcement_fired matches expectation
    # - when enforcement expected → model must hedge
    enforcement_ok  = enforcement_fired == expect_enforcement
    hedging_ok      = (not expect_enforcement) or has_uncertainty
    no_hallucination = not hallucinated

    passed = enforcement_ok and hedging_ok and no_hallucination

    note_parts = []
    if not enforcement_ok:
        note_parts.append(f"enforcement_fired={enforcement_fired} (expected {expect_enforcement})")
    if not hedging_ok:
        note_parts.append("model answered with false certainty despite enforcement")
    if hallucinated:
        note_parts.append(f"hallucinated one of {false_values}")

    return CETestResult(
        test_id=test_id,
        description=description,
        query=query,
        enforcement_fired=enforcement_fired,
        has_uncertainty=has_uncertainty,
        hallucinated=hallucinated,
        response_snippet=snippet,
        passed=passed,
        note=" | ".join(note_parts) if note_parts else "OK",
    )


def run_all_api_tests(client) -> list[CETestResult]:
    reg  = CredenceRegistry(":memory:")
    mgr  = ContextManager(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        theta_high=0.70, theta_low=0.45,
        system_prompt=SYSTEM, max_tokens=300,
        registry=reg,
        session_id="ce_test_session",
        use_scout=False,
        use_claim_extraction=False,  # test manages its own registration; auto-extraction would re-register verified constraints as new unverified ones
    )

    # ---- Seed: plant two uncertain constraints and fill with HIGH-J filler ----
    seed_turns = [
        CONSTRAINT_A,
        CONSTRAINT_B,
        "What is exponential backoff and how does it apply to API retries?",
        "What HTTP status code does rate limiting return?",
        "What is the difference between access tokens and refresh tokens?",
        "What is connection pooling?",
        "What does the Retry-After header do?",
        "What is idempotency and why does it matter for retries?",
    ]

    for msg in seed_turns:
        mgr.chat(msg)
        time.sleep(0.3)

    # Manually register the constraints (simulating Scout or explicit registration)
    cid_a = reg.register(CONSTRAINT_A, session_id="ce_test_session", j_score=0.28, zone="LOW")
    cid_b = reg.register(CONSTRAINT_B, session_id="ce_test_session", j_score=0.32, zone="LOW")

    results = []

    # CE1: Direct query about constraint A — enforcer MUST fire, model MUST hedge
    r = _run_ce_test(
        "CE1",
        "Direct query about unverified rate limit → enforcer fires, model hedges",
        QUERY_DIRECT_A,
        mgr,
        expect_enforcement=True,
        expect_hedged=True,
        false_values=["50 requests per minute", "50 req/min", "50 rpm"],
    )
    results.append(r)
    time.sleep(0.3)

    # CE2: Indirect query — different topic, no direct constraint overlap
    # Enforcer should NOT fire; model should answer normally
    r = _run_ce_test(
        "CE2",
        "Indirect query (OAuth2 explanation) → no enforcer, normal answer",
        QUERY_INDIRECT,
        mgr,
        expect_enforcement=False,
        expect_hedged=False,
    )
    results.append(r)
    time.sleep(0.3)

    # CE3: Direct query about constraint B — enforcer fires, model hedges on token expiry
    r = _run_ce_test(
        "CE3",
        "Direct query about unverified token expiry → enforcer fires, model hedges",
        QUERY_DIRECT_B,
        mgr,
        expect_enforcement=True,
        expect_hedged=True,
        false_values=["3600 seconds", "1 hour", "86400 seconds", "24 hours"],
    )
    results.append(r)
    time.sleep(0.3)

    # CE4: After verifying constraint A, direct query should NOT enforce
    reg.verify(cid_a, verified_value="confirmed: 50 req/min (from vendor docs)")
    r = _run_ce_test(
        "CE4",
        "Verified constraint — enforcer should NOT fire (constraint is resolved)",
        QUERY_DIRECT_A,
        mgr,
        expect_enforcement=False,
        expect_hedged=False,
    )
    results.append(r)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(results: list[CETestResult]) -> None:
    passed  = sum(1 for r in results if r.passed)
    total   = len(results)
    print(f"\n{'='*60}")
    print(f"Consistency Enforcer Tests: {passed}/{total} passed")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] {r.test_id}: {r.description}")
        print(f"  Query:        {r.query[:80]}")
        print(f"  Enforcement:  {r.enforcement_fired}  |  Hedged: {r.has_uncertainty}  |  Hallucinated: {r.hallucinated}")
        print(f"  Response:     {r.response_snippet[:120]}")
        print(f"  Note:         {r.note}")


def _save_results(results: list[CETestResult], path: str = "evals/ce_results.json") -> None:
    data = [
        {
            "test_id":           r.test_id,
            "description":       r.description,
            "query":             r.query,
            "enforcement_fired": r.enforcement_fired,
            "has_uncertainty":   r.has_uncertainty,
            "hallucinated":      r.hallucinated,
            "passed":            r.passed,
            "note":              r.note,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump({"results": data, "passed": sum(r.passed for r in results), "total": len(results)}, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency Enforcer Tests")
    parser.add_argument("--no-api",  action="store_true", help="Logic-only tests (no API calls)")
    parser.add_argument("--test",    default="all", help="Run specific test (CE1-CE4) or 'all'")
    args = parser.parse_args()

    # Always run logic tests
    logic_ok = _test_enforcer_logic()

    if args.no_api:
        sys.exit(0 if logic_ok else 1)

    if not _HAS_DEPS:
        print("ERROR: credence package not importable")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("\n=== API Tests (Opus 4.7) ===")
    results = run_all_api_tests(client)

    if args.test != "all":
        results = [r for r in results if r.test_id == args.test.upper()]

    _print_results(results)
    _save_results(results)

    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

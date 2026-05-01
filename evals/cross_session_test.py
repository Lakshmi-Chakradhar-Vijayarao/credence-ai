"""
evals/cross_session_test.py
===========================
Test that an uncertain claim registered in Session A survives into Session B
via cross-session memory (CredenceRegistry + CredenceMemory), and that the
downstream Credence mechanisms (Truth Buffer, Consistency Enforcer) handle it
correctly in Session B.

No API calls, no GPU required. Pure registry + in-memory SQLite.

Usage
-----
    python -m evals.cross_session_test

6 assertions:
  1. Session A: claim registered as unverified (verified=False)
  2. Snapshot to project succeeds (saved_count == 1)
  3. Session B: claim appears after recall (verified=False)
  4. Truth Buffer injection includes the claim for Session B
  5. Consistency Enforcer fires when Session B asks about the endpoint speed
  6. Consistency Enforcer does NOT fire on an unrelated query

Exit code: 0 if all 6 pass, 1 if any fail.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.registry import CredenceRegistry
from credence.memory import CredenceMemory

# ---------------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------------

_PASS_COUNT = 0
_FAIL_COUNT = 0
_RESULTS: list[tuple[str, bool, str]] = []


def _assert(label: str, condition: bool, detail: str = "") -> None:
    global _PASS_COUNT, _FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        _PASS_COUNT += 1
    else:
        _FAIL_COUNT += 1
    _RESULTS.append((label, condition, detail))
    badge = "✓" if condition else "✗"
    print(f"  [{status}] {badge} {label}")
    if detail and not condition:
        # Print detail only on failure for clean PASS output
        print(f"          detail: {detail}")
    elif detail and condition:
        print(f"          {detail}")


# ---------------------------------------------------------------------------
# Helper: build a ContextManager with registry wired in (no API calls)
# ---------------------------------------------------------------------------

def _make_context_manager(registry: CredenceRegistry, session_id: str):
    """
    Instantiate a ContextManager with the given registry and session_id.

    We never call .chat() so no API key is needed beyond passing the empty
    string. The Anthropic client is created but never used.
    """
    # Lazy import so that the test fails gracefully if anthropic is missing
    try:
        from credence.context_manager import ContextManager
    except ImportError as e:
        print(f"  SKIP: ContextManager unavailable ({e})")
        return None

    # api_key="" → creates a client but never makes calls (we test non-chat methods)
    try:
        mgr = ContextManager(
            api_key="test-no-api-calls",
            registry=registry,
            session_id=session_id,
            system_prompt="You are a helpful assistant.",
        )
        return mgr
    except Exception as e:
        print(f"  SKIP: Could not instantiate ContextManager ({e})")
        return None


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def run_tests() -> int:
    """
    Run all 6 assertions. Returns exit code (0 = all pass, 1 = any fail).
    """
    print("=" * 60)
    print("cross_session_test.py — Cross-Session Epistemic Memory")
    print("=" * 60)
    print()

    # Use in-memory SQLite for complete isolation
    registry = CredenceRegistry(db_path=":memory:")
    memory   = CredenceMemory(registry)

    SESSION_A  = "session-A-test"
    SESSION_B  = "session-B-test"
    PROJECT    = "test_project"
    CLAIM_TEXT = "rate limit is 50 req/min"

    # ------------------------------------------------------------------
    # Step 1: Register an uncertain claim in Session A
    # ------------------------------------------------------------------
    print("Step 1 — Register uncertain claim in Session A")
    cid_a = registry.register(
        content=CLAIM_TEXT,
        session_id=SESSION_A,
        j_score=0.35,
        zone="LOW",
        turn_idx=1,
        source=CredenceRegistry.SOURCE_USER_STATED,
    )
    uncertain_a = registry.list_uncertain(SESSION_A)
    claim_a = next((c for c in uncertain_a if c["constraint_id"] == cid_a), None)

    _assert(
        "Session A: claim registered as unverified (verified=False)",
        claim_a is not None and not claim_a["verified"],
        detail=(
            f"constraint_id={cid_a}  "
            f"content='{CLAIM_TEXT}'  "
            f"j_score={claim_a['j_score'] if claim_a else '?'}  "
            f"zone={claim_a['zone'] if claim_a else '?'}"
        ),
    )

    # ------------------------------------------------------------------
    # Step 2: Snapshot Session A → project
    # ------------------------------------------------------------------
    print()
    print("Step 2 — Snapshot Session A to project")
    snapshot = memory.snapshot(session_id=SESSION_A, project=PROJECT)

    _assert(
        "Snapshot to project succeeds (saved_count == 1)",
        snapshot.saved_count == 1,
        detail=(
            f"saved_count={snapshot.saved_count}  "
            f"project={snapshot.project_id}  "
            f"session={snapshot.session_id}"
        ),
    )

    # ------------------------------------------------------------------
    # Step 3: Recall into Session B and verify presence
    # ------------------------------------------------------------------
    print()
    print("Step 3 — Recall project memories into Session B")
    recall = memory.recall_and_inject(project=PROJECT, new_session_id=SESSION_B)

    uncertain_b = registry.list_uncertain(SESSION_B)
    claim_b = next(
        (c for c in uncertain_b if CLAIM_TEXT in c["content"]),
        None,
    )

    _assert(
        "Session B: claim appears after recall (verified=False)",
        claim_b is not None and not claim_b["verified"],
        detail=(
            f"injected_count={recall.injected_count}  "
            f"unverified_in_B={len(uncertain_b)}  "
            f"claim_found={'yes' if claim_b else 'no'}"
        ),
    )

    # ------------------------------------------------------------------
    # Step 4: Truth Buffer injection includes the claim
    # ------------------------------------------------------------------
    print()
    print("Step 4 — Truth Buffer injection includes the claim in Session B")

    mgr_b = _make_context_manager(registry, SESSION_B)
    if mgr_b is None:
        # ContextManager unavailable (e.g. anthropic not installed) — skip gracefully
        _assert(
            "Truth Buffer injection includes the claim for Session B",
            False,
            detail="ContextManager unavailable — skipped",
        )
    else:
        # _augment_with_truth_buffer uses self._current_user_message for query context
        mgr_b._current_user_message = CLAIM_TEXT   # prime query context
        truth_buffer_prompt = mgr_b._augment_with_truth_buffer()
        claim_in_buffer = CLAIM_TEXT.lower() in truth_buffer_prompt.lower()

        _assert(
            "Truth Buffer injection includes the claim for Session B",
            claim_in_buffer,
            detail=(
                f"Truth Buffer prompt snippet:\n"
                f"          {truth_buffer_prompt[:300].replace(chr(10), ' | ')}"
            ),
        )

    # ------------------------------------------------------------------
    # Step 5: Consistency Enforcer fires on a direct rate-limit query
    # ------------------------------------------------------------------
    print()
    print("Step 5 — Consistency Enforcer fires on endpoint-speed query")

    direct_query = "how fast can i call the endpoint"

    if mgr_b is None:
        _assert(
            "Consistency Enforcer fires for Session B rate-limit query",
            False,
            detail="ContextManager unavailable — skipped",
        )
    else:
        # _build_enforcement_system_prompt calls _direct_constraint_matches internally
        _augmented, enforcement_active = mgr_b._build_enforcement_system_prompt(direct_query)
        enforcement_block_present = "CONSISTENCY ENFORCEMENT" in _augmented

        _assert(
            "Consistency Enforcer fires for Session B rate-limit query",
            enforcement_active and enforcement_block_present,
            detail=(
                f"enforcement_active={enforcement_active}  "
                f"block_in_prompt={enforcement_block_present}  "
                f"query='{direct_query}'"
            ),
        )

    # ------------------------------------------------------------------
    # Step 6: Consistency Enforcer does NOT fire on an unrelated query
    # ------------------------------------------------------------------
    print()
    print("Step 6 — Consistency Enforcer does NOT fire on unrelated query")

    unrelated_query = "what is the best color for a user interface palette"

    if mgr_b is None:
        _assert(
            "Consistency Enforcer does not fire on unrelated query",
            False,
            detail="ContextManager unavailable — skipped",
        )
    else:
        _aug_unrelated, enforcement_unrelated = mgr_b._build_enforcement_system_prompt(
            unrelated_query
        )
        _assert(
            "Consistency Enforcer does not fire on unrelated query",
            not enforcement_unrelated,
            detail=(
                f"enforcement_active={enforcement_unrelated}  "
                f"query='{unrelated_query}'"
            ),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Results: {_PASS_COUNT} passed / {_FAIL_COUNT} failed / 6 total")
    print("=" * 60)

    if _FAIL_COUNT == 0:
        print()
        print("All 6 assertions PASSED.")
        print()
        print("Cross-session epistemic memory is working correctly:")
        print("  - Uncertain claims survive snapshot → recall boundaries.")
        print("  - Truth Buffer surfaces them in Session B.")
        print("  - Consistency Enforcer fires on semantically related queries.")
        print("  - Consistency Enforcer stays silent on unrelated queries.")
    else:
        print()
        print(f"{_FAIL_COUNT} assertion(s) FAILED.")
        print("Review the details above.")

    return 0 if _FAIL_COUNT == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_tests())

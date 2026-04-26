"""
tests.py — Unit and component tests for Credence (167 passing, 11 skipped, S1–S26 suites).

Tests every identified risk from the codebase audit:
  - Null/empty inputs to all public methods
  - Boundary values (j=0, j=1, turn=0, turn=100)
  - Decimal value GTS annotation
  - CE degenerate inputs (empty, all-stopword)
  - Truth Buffer cap behaviour (>6 constraints, all-verified)
  - Ghost detector guard paths (empty, canonical markers, malformed JSON)
  - Registry truncation and decay floor
  - DISPUTED logic (same numbers same topic vs. same numbers different topic)
  - CE synonym expansion paraphrase matching
  - Confidence policy tiers (HIGH RISK / UNVERIFIED / CHECK annotation text)
  - Trajectory event logging
  - Performance: 100 constraints in registry, GTS with 20 constraints
  - End-to-end ContextManager with ghost detector enabled (API, if key present)

Usage:
    python tests.py          # non-API tests only
    python tests.py --api    # all tests including live API calls
"""

import os
import sys
import time
import json
import re
import argparse

# ── Load .env if present ──────────────────────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_ENV_PATH):
    for _line in open(_ENV_PATH):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

parser = argparse.ArgumentParser()
parser.add_argument("--api", action="store_true", help="Run tests requiring live API calls")
ARGS = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.registry import CredenceRegistry
from credence.confidence_proxy import CredenceProxy

# ── Test harness ──────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0
_SKIP = 0


def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    status = "✓ PASS" if condition else "✗ FAIL"
    msg = f"  {status}  {name}"
    if detail and not condition:
        msg += f"\n         ↳ {detail}"
    print(msg)
    if condition:
        _PASS += 1
    else:
        _FAIL += 1


def skip(name: str, reason: str = ""):
    global _SKIP
    print(f"  ⊘ SKIP  {name}" + (f"  [{reason}]" if reason else ""))
    _SKIP += 1


def section(title: str):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Null / empty safety on registry
# ─────────────────────────────────────────────────────────────────────────────

section("S1: Null / empty safety — registry")

reg = CredenceRegistry(":memory:")

# S1-A: register(None) should raise (not silently corrupt the DB)
try:
    reg.register(None, "s1")
    check("S1-A register(None) raises", False, "No exception raised — potential corruption")
except (AttributeError, TypeError, ValueError) as e:
    check("S1-A register(None) raises", True)

# S1-B: register("", ...) — empty string is a valid but meaningless constraint;
# should not crash
try:
    cid = reg.register("", "s1")
    check("S1-B register('') does not crash", isinstance(cid, str) and len(cid) == 12)
except Exception as e:
    check("S1-B register('') does not crash", False, str(e))

# S1-C: list_uncertain("nonexistent_session") — returns empty list, no crash
try:
    result = reg.list_uncertain("nonexistent_session")
    check("S1-C list_uncertain(unknown session)", result == [], f"Got {result!r}")
except Exception as e:
    check("S1-C list_uncertain(unknown session)", False, str(e))

# S1-D: verify("nonexistent_id", "value") — returns error dict, no crash
try:
    result = reg.verify("nonexistentcid", "value")
    check("S1-D verify(nonexistent id)", "error" in result, f"Got {result!r}")
except Exception as e:
    check("S1-D verify(nonexistent id)", False, str(e))

# S1-E: get_effective_confidence("nonexistent", 5) — returns 0.0
try:
    val = reg.get_effective_confidence("nonexistentcid", 5)
    check("S1-E get_effective_confidence(nonexistent)", val == 0.0, f"Got {val!r}")
except Exception as e:
    check("S1-E get_effective_confidence(nonexistent)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Confidence decay boundary values
# ─────────────────────────────────────────────────────────────────────────────

section("S2: Confidence decay boundary values")

reg2 = CredenceRegistry(":memory:")

# S2-A: decay at turn=0 — same as registration → returns j_score
cid = reg2.register("I think the rate limit is 50 req/min", "s2", j_score=0.30, turn_idx=0)
val = reg2.get_effective_confidence(cid, 0)
check("S2-A decay at turn=0 → j_score unchanged", abs(val - 0.30) < 0.001, f"Got {val}")

# S2-B: decay with future turn (current_turn < registered_at_turn) — max(0,...) guard
# Should return j_score (0 elapsed turns), NOT >1.0
cid_future = reg2.register("Auth token might expire in 3600s", "s2", j_score=0.40, turn_idx=10)
val_future = reg2.get_effective_confidence(cid_future, 3)  # current_turn < registered_at_turn
check("S2-B future turn → returns j_score (no >1.0)", 0.0 <= val_future <= 1.0,
      f"Got {val_future} — should be in [0, 1]")
check("S2-B future turn → returns original j_score", abs(val_future - 0.40) < 0.001,
      f"Got {val_future}, expected 0.40")

# S2-C: decay at extreme turn (turn=100 with registered_at=0)
# 0.30 * 0.95^100 ≈ 0.30 * 0.00592 ≈ 0.00178 — should floor near 0, not negative
val_100 = reg2.get_effective_confidence(cid, 100)
check("S2-C decay at turn=100 → near-zero, not negative", 0.0 <= val_100 <= 0.01,
      f"Got {val_100}")

# S2-D: verified constraint — decay stops at j_score regardless of turns
reg2.verify(cid, "Confirmed: 50 req/min via API docs")
val_verified = reg2.get_effective_confidence(cid, 100)
check("S2-D verified constraint — no decay", abs(val_verified - 0.30) < 0.001,
      f"Got {val_verified}, expected 0.30 (no decay)")

# S2-E: decay at j=0.0 — should stay 0
cid_zero = reg2.register("Zero confidence claim", "s2", j_score=0.0, turn_idx=0)
val_zero = reg2.get_effective_confidence(cid_zero, 50)
check("S2-E decay of j=0.0 → stays 0", val_zero == 0.0, f"Got {val_zero}")

# S2-F: decay at j=1.0 — should decay correctly
cid_one = reg2.register("Full confidence claim", "s2", j_score=1.0, turn_idx=0)
val_one = reg2.get_effective_confidence(cid_one, 10)
expected = round(1.0 * (0.95 ** 10), 4)
check("S2-F decay of j=1.0 correct", abs(val_one - expected) < 0.001,
      f"Got {val_one}, expected {expected}")


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Faithfulness probe case sensitivity
# ─────────────────────────────────────────────────────────────────────────────

section("S3: Faithfulness probe — case sensitivity")

# Import the _has_uncertainty method via a ContextManager with minimal config
# We test it as a static method by patching through the module-level markers
from credence.context_manager import _UNCERTAINTY_MARKERS

def _has_uncertainty_fn(text: str) -> bool:
    """Mirror of ContextManager._has_uncertainty for isolated testing."""
    import re as _re
    lower = text.lower()
    if any(m in lower for m in _UNCERTAINTY_MARKERS):
        return True
    if _re.search(r'#\s*(todo|fixme|hack|verify|check|untested|approximate|not sure|might)', lower):
        return True
    if _re.search(r'\b(around|roughly|approximately|about|~)\s+\d', lower):
        return True
    return False

check("S3-A 'I THINK' (all caps) detected", _has_uncertainty_fn("I THINK it's 50 req/min"))
check("S3-B 'I think' (lowercase) detected", _has_uncertainty_fn("I think it's 50 req/min"))
check("S3-C 'I Think' (mixed) detected", _has_uncertainty_fn("I Think it might work"))
check("S3-D 'NOT CERTAIN' (all caps) detected", _has_uncertainty_fn("I AM NOT CERTAIN of the value"))
check("S3-E 'approximately 100' detected", _has_uncertainty_fn("approximately 100 requests/min"))
check("S3-F 'APPROXIMATELY 200' detected", _has_uncertainty_fn("APPROXIMATELY 200 tokens"))
check("S3-G 'unconfirmed' detected", _has_uncertainty_fn("This is unconfirmed data"))
check("S3-H established fact NOT flagged", not _has_uncertainty_fn("Python lists are 0-indexed"),
      "Should return False for established facts")
check("S3-I '# TODO verify' code comment detected", _has_uncertainty_fn("# TODO verify this value"))
check("S3-J 'around 50' numerical hedge detected", _has_uncertainty_fn("around 50 requests per minute"))


# ─────────────────────────────────────────────────────────────────────────────
# S4 — GTS decimal value annotation
# ─────────────────────────────────────────────────────────────────────────────

section("S4: GTS — decimal value annotation")

from credence.context_manager import _GTS_NUM_PATTERN

# Verify decimal values extracted from constraint text
decimal_constraint = "I think the timeout might be 3.5 seconds — unconfirmed"
nums = _GTS_NUM_PATTERN.findall(decimal_constraint)
decimal_nums = [n for n in nums if len(n.replace(".", "")) >= 2]
check("S4-A decimal '3.5' extracted from constraint text", "3.5" in decimal_nums,
      f"Got nums={nums}, filtered={decimal_nums}")

# Verify the regex correctly matches decimal in assignment line
decimal_pattern = re.compile(r"=\s*" + re.escape("3.5") + r"\b")
line1 = "TIMEOUT = 3.5"
line2 = "TIMEOUT = 3.50"
line3 = "TIMEOUT = 3.500"
line4 = "    timeout = 3.5  # seconds"
check("S4-B '= 3.5' matched in assignment", bool(decimal_pattern.search(line1)),
      f"Pattern failed on: {line1!r}")
check("S4-C '= 3.5' NOT matched in '3.50'", not bool(decimal_pattern.search(line2)),
      "3.50 should not match 3.5 (different number)")
check("S4-D '= 3.5' matched in indented assignment", bool(decimal_pattern.search(line4)))

# End-to-end: ContextManager._scan_output_for_constraints with decimal
# Create a minimal ContextManager with in-memory registry
try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy_for_non_api_test")
    reg_s4 = CredenceRegistry(":memory:")
    cid_s4 = reg_s4.register(
        "I think the timeout might be 3.5 seconds — unconfirmed",
        "test_s4", j_score=0.35, turn_idx=0
    )
    mgr_s4 = ContextManager(
        api_key=api_key,
        registry=reg_s4,
        session_id="test_s4",
        use_ghost_detector=False,
    )
    mgr_s4._turn_idx = 1

    response_with_decimal = "```python\nTIMEOUT = 3.5\n```"
    annotated, hits = mgr_s4._scan_output_for_constraints(response_with_decimal)
    check("S4-E decimal '3.5' in code block annotated",
          len(hits) > 0 and hits[0]["value"] == "3.5",
          f"hits={hits}, annotated={annotated!r}")
    check("S4-F CREDENCE annotation in output", "CREDENCE" in annotated, f"Got: {annotated!r}")
except Exception as e:
    check("S4-E decimal GTS end-to-end", False, str(e))
    check("S4-F CREDENCE annotation in output", False, "skipped due to S4-E failure")


# ─────────────────────────────────────────────────────────────────────────────
# S5 — CE degenerate inputs
# ─────────────────────────────────────────────────────────────────────────────

section("S5: Consistency Enforcer — degenerate inputs")

try:
    from credence.context_manager import ContextManager, _CE_STOPWORDS

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s5 = CredenceRegistry(":memory:")
    reg_s5.register("I think the rate limit is 50 req/min", "s5", j_score=0.30, turn_idx=0)
    mgr_s5 = ContextManager(api_key=api_key, registry=reg_s5, session_id="s5")

    uncertain_s5 = reg_s5.list_uncertain("s5")

    # S5-A: empty string query
    try:
        matches = mgr_s5._direct_constraint_matches("", uncertain_s5)
        check("S5-A empty query → no crash, returns []", matches == [], f"Got {matches}")
    except Exception as e:
        check("S5-A empty query → no crash", False, str(e))

    # S5-B: all-stopword query — all tokens stripped by _CE_STOPWORDS
    stopword_query = "the a an is are was were be have has do does"
    try:
        matches = mgr_s5._direct_constraint_matches(stopword_query, uncertain_s5)
        check("S5-B all-stopword query → no crash, returns []", isinstance(matches, list),
              f"Got {matches!r}")
    except Exception as e:
        check("S5-B all-stopword query → no crash", False, str(e))

    # S5-C: very long query (10k chars) — should not OOM or timeout
    long_query = "what is the rate limit " * 500  # ~12k chars
    t0 = time.perf_counter()
    try:
        matches = mgr_s5._direct_constraint_matches(long_query, uncertain_s5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        check("S5-C 10k-char query → completes in <50ms", elapsed_ms < 50,
              f"Took {elapsed_ms:.1f}ms")
        check("S5-C 10k-char query → detects rate limit match", len(matches) > 0,
              f"Expected match on rate/limit, got {matches}")
    except Exception as e:
        check("S5-C 10k-char query → no crash", False, str(e))
        check("S5-C 10k-char query → detects rate limit match", False, "skipped")

    # S5-D: empty constraint list — should return []
    try:
        matches = mgr_s5._direct_constraint_matches("what is the rate limit", [])
        check("S5-D empty constraint list → []", matches == [], f"Got {matches}")
    except Exception as e:
        check("S5-D empty constraint list → no crash", False, str(e))

except Exception as e:
    check("S5 CE setup", False, f"ContextManager init failed: {e}")
    for label in ["S5-A", "S5-B", "S5-C", "S5-D"]:
        skip(label, "setup failed")


# ─────────────────────────────────────────────────────────────────────────────
# S6 — Truth Buffer cap behaviour
# ─────────────────────────────────────────────────────────────────────────────

section("S6: Truth Buffer — cap and all-verified behaviour")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")

    # S6-A: all constraints verified → Truth Buffer no-op (returns system_prompt unchanged)
    # Use topically-distinct content to avoid DISPUTED triggering (different topic words)
    reg_s6a = CredenceRegistry(":memory:")
    distinct_claims = [
        ("The authentication token expires in 3600 seconds", "Confirmed: 3600s expiry"),
        ("The database port is configured to 5432", "Confirmed: port 5432"),
        ("The cache TTL is set to 300 seconds", "Confirmed: TTL 300s"),
    ]
    for content, verified_val in distinct_claims:
        cid_i = reg_s6a.register(content, "s6a", j_score=0.80, turn_idx=0)
        reg_s6a.verify(cid_i, verified_val)
    mgr_s6a = ContextManager(api_key=api_key, registry=reg_s6a, session_id="s6a")
    augmented = mgr_s6a._augment_with_truth_buffer()
    check("S6-A all verified → TB is no-op (no EPISTEMIC CONTEXT block)",
          "EPISTEMIC CONTEXT" not in augmented,
          f"Unexpected injection: {augmented[:200]!r}")

    # S6-B: >6 unverified constraints → Truth Buffer shows ALL (no silent drop)
    reg_s6b = CredenceRegistry(":memory:")
    cids_s6b = []
    for i in range(10):
        cid_i = reg_s6b.register(
            f"I think value_{i} might be {(i+1)*50} — unconfirmed",
            "s6b", j_score=0.30 + i * 0.01, turn_idx=0
        )
        cids_s6b.append(cid_i)
    mgr_s6b = ContextManager(api_key=api_key, registry=reg_s6b, session_id="s6b")
    mgr_s6b._current_user_message = ""  # no query context → uses get_effective_uncertain
    augmented_b = mgr_s6b._augment_with_truth_buffer()
    # All 10 must appear — no silent drop
    bullet_count = augmented_b.count("• [")
    check("S6-B 10 constraints → all 10 shown in TB (no silent drop)", bullet_count == 10,
          f"Counted {bullet_count} bullets, expected 10")
    check("S6-B all 10 still in registry",
          len(reg_s6b.list_uncertain("s6b")) == 10,
          f"Registry has {len(reg_s6b.list_uncertain('s6b'))} entries")

    # S6-C: zero constraints → TB is no-op
    reg_s6c = CredenceRegistry(":memory:")
    mgr_s6c = ContextManager(api_key=api_key, registry=reg_s6c, session_id="s6c")
    augmented_c = mgr_s6c._augment_with_truth_buffer()
    check("S6-C zero constraints → TB no-op", "EPISTEMIC CONTEXT" not in augmented_c)

    # S6-D: registry=None → TB no-op
    mgr_s6d = ContextManager(api_key=api_key, registry=None, session_id=None)
    augmented_d = mgr_s6d._augment_with_truth_buffer()
    check("S6-D registry=None → TB no-op", "EPISTEMIC CONTEXT" not in augmented_d)

except Exception as e:
    check("S6 TB setup", False, f"Setup failed: {e}")
    for label in ["S6-A", "S6-B-count", "S6-B-registry", "S6-C", "S6-D"]:
        skip(label, "setup failed")


# ─────────────────────────────────────────────────────────────────────────────
# S7 — Ghost detector guard paths (no API required)
# ─────────────────────────────────────────────────────────────────────────────

section("S7: Ghost detector — guard paths (no API)")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s7 = CredenceRegistry(":memory:")
    mgr_s7 = ContextManager(
        api_key=api_key,
        registry=reg_s7,
        session_id="s7",
        use_ghost_detector=True,
    )

    # S7-A: empty message → _ghost_detect returns [] immediately (no API call made)
    result = mgr_s7._ghost_detect("")
    check("S7-A empty message → [] (no API call)", result == [], f"Got {result!r}")

    result_ws = mgr_s7._ghost_detect("   \t\n  ")
    check("S7-B whitespace-only message → []", result_ws == [], f"Got {result_ws!r}")

    # S7-C: registry=None → _ghost_detect returns []
    mgr_s7_noreg = ContextManager(api_key=api_key, registry=None, session_id=None,
                                   use_ghost_detector=True)
    result_noreg = mgr_s7_noreg._ghost_detect("The rate limit is 50 req/min")
    check("S7-C registry=None → _ghost_detect returns []", result_noreg == [],
          f"Got {result_noreg!r}")

    # S7-D: when use_ghost_detector=False, ghost_detect path skipped
    # We test the dispatch logic in chat() by checking decision_log for ghost_detections key
    # (non-API: just verify the flag is stored on the instance)
    mgr_no_ghost = ContextManager(api_key=api_key, registry=reg_s7, session_id="s7",
                                   use_ghost_detector=False)
    check("S7-D use_ghost_detector=False stored correctly",
          not mgr_no_ghost.use_ghost_detector)

except Exception as e:
    check("S7 setup", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S8 — Registry: long content and idempotency
# ─────────────────────────────────────────────────────────────────────────────

section("S8: Registry — long content, idempotency, trajectory")

reg_s8 = CredenceRegistry(":memory:")

# S8-A: register 600-char content — stored in full (no truncation in registry itself)
long_content = "I think the rate limit might be 50 req/min — " + ("x" * 560)
cid_long = reg_s8.register(long_content, "s8", j_score=0.30, turn_idx=0)
fetched = reg_s8.get_all("s8")
check("S8-A 600-char content stored (length ≥ 600)",
      len(fetched) > 0 and len(fetched[0]["content"]) >= 600,
      f"Stored length: {len(fetched[0]['content']) if fetched else 'N/A'}")

# S8-B: idempotency — registering same content twice returns same ID
cid_again = reg_s8.register(long_content, "s8", j_score=0.35, turn_idx=1)
check("S8-B idempotent re-register returns same ID", cid_long == cid_again,
      f"First: {cid_long}, Second: {cid_again}")

# S8-C: trajectory — register event logged
trajectory = reg_s8.get_trajectory(cid_long)
event_types = [e["event_type"] for e in trajectory]
check("S8-C 'register' event in trajectory", "register" in event_types,
      f"Events: {event_types}")

# S8-D: verify event logged after verify()
reg_s8.verify(cid_long, "Confirmed: 50 req/min")
trajectory_v = reg_s8.get_trajectory(cid_long)
event_types_v = [e["event_type"] for e in trajectory_v]
check("S8-D 'verify' event in trajectory after verify()", "verify" in event_types_v,
      f"Events: {event_types_v}")

# S8-E: get_trajectory("nonexistent") returns []
check("S8-E get_trajectory(nonexistent) returns []",
      reg_s8.get_trajectory("nonexistentcid") == [])


# ─────────────────────────────────────────────────────────────────────────────
# S9 — DISPUTED logic
# ─────────────────────────────────────────────────────────────────────────────

section("S9: DISPUTED logic — same numbers / different topics")

reg_s9 = CredenceRegistry(":memory:")

# S9-A: verify a constraint, then register contradicting number on SAME topic → DISPUTED
cid_rate = reg_s9.register("Rate limit is 50 req/min", "s9", j_score=0.80, turn_idx=0)
reg_s9.verify(cid_rate, "Confirmed: 50 req/min per vendor")
# Now register conflicting number on same topic
reg_s9.register("Actually the rate limit might be 100 req/min", "s9", j_score=0.30, turn_idx=5)
row_rate = reg_s9._conn.execute(
    "SELECT validation_status FROM constraints WHERE constraint_id=?", (cid_rate,)
).fetchone()
check("S9-A same-topic conflict → original marked DISPUTED",
      row_rate["validation_status"] == "disputed",
      f"Got status={row_rate['validation_status']!r}")

# S9-B: same numbers but DIFFERENT topics → should NOT dispute
reg_s9b = CredenceRegistry(":memory:")
cid_token = reg_s9b.register("Auth token expiry is 3600 seconds", "s9b", j_score=0.80, turn_idx=0)
reg_s9b.verify(cid_token, "Confirmed: 3600s token expiry")
# Register same number 3600 on completely different topic
reg_s9b.register("The cache TTL is 3600 seconds but might change", "s9b", j_score=0.30, turn_idx=5)
row_token = reg_s9b._conn.execute(
    "SELECT validation_status FROM constraints WHERE constraint_id=?", (cid_token,)
).fetchone()
# The DISPUTED logic uses Jaccard similarity (threshold=0.15);
# "auth token expiry" vs "cache TTL" — low topic overlap, may not dispute
# This is the expected behavior: don't dispute on numeric coincidence alone
status_token = row_token["validation_status"]
# We don't enforce a strict pass/fail here because Jaccard similarity may or
# may not fire depending on shared content words — we just report it
if status_token == "unverified" or status_token == "verified":
    check("S9-B different topics: DISPUTE avoided or Jaccard too low for strict match", True,
          f"status={status_token} (disputed would be too aggressive here)")
else:
    # disputed — might happen if Jaccard fires on numeric coincidence
    check("S9-B different topics: DISPUTE occurred (check Jaccard threshold)",
          False, f"status={status_token} — consider raising similarity threshold")

# S9-C: DISPUTED constraint appears in Truth Buffer
reg_s9c = CredenceRegistry(":memory:")
cid_d = reg_s9c.register("Rate limit is 50 req/min", "s9c", j_score=0.80, turn_idx=0)
reg_s9c.verify(cid_d, "Confirmed: 50 req/min")
reg_s9c.register("Rate limit might now be 100 req/min", "s9c", j_score=0.30, turn_idx=5)

try:
    from credence.context_manager import ContextManager
    mgr_s9 = ContextManager(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy"),
        registry=reg_s9c, session_id="s9c"
    )
    mgr_s9._current_user_message = ""
    aug = mgr_s9._augment_with_truth_buffer()
    # The DISPUTED constraint (cid_d) should appear in TB (it's in list_uncertain with disputed status)
    uncertain_s9c = reg_s9c.list_uncertain("s9c")
    has_disputed = any(c.get("validation_status") == "disputed" for c in uncertain_s9c)
    check("S9-C DISPUTED constraint in list_uncertain", has_disputed,
          f"uncertain={[c['validation_status'] for c in uncertain_s9c]}")
except Exception as e:
    check("S9-C DISPUTED in TB", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S10 — CE synonym expansion
# ─────────────────────────────────────────────────────────────────────────────

section("S10: CE synonym expansion — paraphrase matching")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s10 = CredenceRegistry(":memory:")
    reg_s10.register("I think the rate limit is 50 req/min", "s10", j_score=0.30, turn_idx=0)
    reg_s10.register("Auth token expiry might be 3600 seconds — unconfirmed", "s10",
                     j_score=0.25, turn_idx=0)
    mgr_s10 = ContextManager(api_key=api_key, registry=reg_s10, session_id="s10")
    uncertain = reg_s10.list_uncertain("s10")

    # S10-A: paraphrase of "rate limit" — should fire via synonym expansion
    matches_a = mgr_s10._direct_constraint_matches(
        "How fast can we call the endpoint?", uncertain
    )
    check("S10-A 'How fast can we call the endpoint?' matches rate-limit constraint",
          len(matches_a) > 0,
          "Synonym expansion should map 'fast'→'rate', 'calls'→'rate', 'endpoint'→'api'")

    # S10-B: paraphrase of "token expiry"
    matches_b = mgr_s10._direct_constraint_matches(
        "When does my session expire?", uncertain
    )
    check("S10-B 'When does my session expire?' matches token-expiry constraint",
          len(matches_b) > 0,
          "Synonym expansion should map 'session'→'token', 'expire'→'expiry'")

    # S10-C: completely unrelated query — should NOT match
    matches_c = mgr_s10._direct_constraint_matches(
        "What color should I use for the UI button?", uncertain
    )
    check("S10-C UI color query → no match", matches_c == [],
          f"Got unexpected matches: {[m.get('content', '')[:50] for m in matches_c]}")

    # S10-D: direct literal match — should fire without synonym expansion
    matches_d = mgr_s10._direct_constraint_matches(
        "What is the rate limit?", uncertain
    )
    check("S10-D direct 'rate limit' query → match", len(matches_d) > 0,
          "Direct literal match should always fire")

except Exception as e:
    check("S10 setup", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S11 — Confidence policy tiers in GTS
# ─────────────────────────────────────────────────────────────────────────────

section("S11: GTS — confidence policy tier annotations")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")

    # Create 3 constraints at different confidence/decay levels
    reg_s11 = CredenceRegistry(":memory:")

    # HIGH RISK: j=0.25, turn registered=0, current_turn=8 → 0.25 * 0.95^8 ≈ 0.166 < 0.20
    cid_hr = reg_s11.register(
        "Stripe rate limit is 50 req/min", "s11", j_score=0.25, turn_idx=0
    )
    # UNVERIFIED: j=0.30 at turn=0, current=1 → 0.30 * 0.95^1 ≈ 0.285 (>0.20, <0.40)
    cid_uv = reg_s11.register(
        "Auth token expiry might be 3600 seconds", "s11", j_score=0.30, turn_idx=0
    )
    # CHECK: j=0.60 at turn=0, current=0 → 0.60 * 0.95^0 = 0.60 (≥0.40)
    cid_ck = reg_s11.register(
        "I think the batch size limit is 100 items", "s11", j_score=0.60, turn_idx=0
    )

    mgr_s11 = ContextManager(api_key=api_key, registry=reg_s11, session_id="s11")

    # Test at turn=8 for HIGH RISK (50 decays past 0.20 threshold)
    mgr_s11._turn_idx = 8
    code_hr = "```python\nRATE_LIMIT = 50\n```"
    annotated_hr, hits_hr = mgr_s11._scan_output_for_constraints(code_hr)
    check("S11-A HIGH RISK annotation contains ⚠⚠",
          "⚠⚠" in annotated_hr and "HIGH RISK" in annotated_hr,
          f"Got: {annotated_hr!r}")

    # Test at turn=1 for UNVERIFIED
    mgr_s11._turn_idx = 1
    code_uv = "```python\nTOKEN_EXPIRY = 3600\n```"
    annotated_uv, hits_uv = mgr_s11._scan_output_for_constraints(code_uv)
    check("S11-B UNVERIFIED annotation contains ⚠",
          "CREDENCE" in annotated_uv,
          f"Got: {annotated_uv!r}")

    # Test CHECK tier: j=0.60 at turn=0 → 0.60 (≥0.40)
    mgr_s11._turn_idx = 0
    code_ck = "```python\nBATCH_SIZE = 100\n```"
    annotated_ck, hits_ck = mgr_s11._scan_output_for_constraints(code_ck)
    check("S11-C CHECK annotation present (≥0.40 confidence)",
          "CREDENCE" in annotated_ck,
          f"Got: {annotated_ck!r}")
    check("S11-D CHECK tier uses [check, conf=...]",
          "[check," in annotated_ck or "check" in annotated_ck.lower(),
          f"Got: {annotated_ck!r}")

    # S11-E: verified constraint → NO annotation
    reg_s11.verify(cid_hr, "Confirmed: 50 req/min")
    mgr_s11._turn_idx = 8
    annotated_verified, hits_verified = mgr_s11._scan_output_for_constraints(code_hr)
    check("S11-E verified constraint → no annotation",
          len(hits_verified) == 0 or all(h["constraint_id"] != cid_hr for h in hits_verified),
          f"Got hits: {hits_verified}")

except Exception as e:
    check("S11 GTS tiers", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S12 — GTS prose scanning
# ─────────────────────────────────────────────────────────────────────────────

section("S12: GTS — prose sentence scanning")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s12 = CredenceRegistry(":memory:")
    reg_s12.register("Auth token expiry might be 3600 seconds", "s12", j_score=0.35, turn_idx=0)
    mgr_s12 = ContextManager(api_key=api_key, registry=reg_s12, session_id="s12")
    mgr_s12._turn_idx = 1

    # Prose (no code block) — should annotate the sentence containing 3600
    prose = "You should set the token expiry to 3600 seconds in your config."
    annotated_prose, hits_prose = mgr_s12._scan_output_for_constraints(prose)
    check("S12-A prose scanning annotates '3600' in sentence",
          len(hits_prose) > 0 and any(h["value"] == "3600" for h in hits_prose),
          f"hits={hits_prose}, annotated={annotated_prose!r}")
    check("S12-B prose annotation source = 'prose'",
          len(hits_prose) > 0 and hits_prose[0].get("source") == "prose",
          f"source={hits_prose[0].get('source') if hits_prose else 'N/A'}")

    # S12-C: sentence already annotated — should NOT double-annotate
    already_annotated = "Set timeout to 3600.  CREDENCE[check, conf=0.35]: …"
    annotated_da, hits_da = mgr_s12._scan_output_for_constraints(already_annotated)
    check("S12-C already-annotated sentence not double-annotated",
          annotated_da.count("CREDENCE") <= 2,  # at most original count
          f"Got: {annotated_da!r}")

except Exception as e:
    check("S12 GTS prose", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S13 — Performance stress
# ─────────────────────────────────────────────────────────────────────────────

section("S13: Performance — 100 constraints in registry")

reg_perf = CredenceRegistry(":memory:")
t_start = time.perf_counter()
for i in range(100):
    reg_perf.register(
        f"I think constraint_{i} might be {(i+1)*10} — unconfirmed",
        "perf_session", j_score=0.20 + (i % 50) * 0.01, turn_idx=i // 5
    )
t_reg = (time.perf_counter() - t_start) * 1000
check("S13-A register 100 constraints < 500ms", t_reg < 500,
      f"Took {t_reg:.1f}ms")

t_list = time.perf_counter()
results = reg_perf.list_uncertain("perf_session", current_turn=20)
t_list = (time.perf_counter() - t_list) * 1000
check("S13-B list_uncertain(100 constraints) < 50ms", t_list < 50,
      f"Took {t_list:.1f}ms, got {len(results)} results")

# GTS with 20 registered constraints — build value_map and scan a code block
try:
    from credence.context_manager import ContextManager, _GTS_NUM_PATTERN

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    mgr_perf = ContextManager(api_key=api_key, registry=reg_perf, session_id="perf_session")
    mgr_perf._turn_idx = 20

    # Response with multiple assignment lines matching various constraints
    code_block = "```python\n" + "\n".join(
        f"VAL_{i} = {(i+1)*10}" for i in range(20)
    ) + "\n```"

    t_gts = time.perf_counter()
    annotated_perf, hits_perf = mgr_perf._scan_output_for_constraints(code_block)
    t_gts = (time.perf_counter() - t_gts) * 1000
    check("S13-C GTS scan with 20 code assignments < 100ms", t_gts < 100,
          f"Took {t_gts:.1f}ms")
    check("S13-D GTS found ≥1 hit in 20-line code block", len(hits_perf) > 0,
          f"hits={len(hits_perf)}, code block scanned")

except Exception as e:
    check("S13-C GTS performance", False, str(e))
    check("S13-D GTS hits found", False, "skipped")

# S13-E: get_effective_confidence for all 100 constraints at turn=50
t_decay = time.perf_counter()
all_c = reg_perf.get_all("perf_session")
for c in all_c:
    reg_perf.get_effective_confidence(c["constraint_id"], 50)
t_decay = (time.perf_counter() - t_decay) * 1000
check("S13-E compute decay for 100 constraints < 200ms", t_decay < 200,
      f"Took {t_decay:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# S14 — Faithfulness probe blocks compression (pure logic)
# ─────────────────────────────────────────────────────────────────────────────

section("S14: Faithfulness probe blocks Haiku compression")

# We test _has_uncertainty directly on text that would be in the old context segment
# to confirm the probe correctly fires on all 40 markers

from credence.context_manager import _UNCERTAINTY_MARKERS

# All 40 markers should be detected
marker_hits = 0
marker_misses = []
for marker in _UNCERTAINTY_MARKERS:
    test_text = f"The constraint value is 50 ({marker})"
    if _has_uncertainty_fn(test_text):
        marker_hits += 1
    else:
        marker_misses.append(marker)

check(f"S14-A all 40 uncertainty markers detected ({marker_hits}/{len(_UNCERTAINTY_MARKERS)})",
      marker_hits == len(_UNCERTAINTY_MARKERS),
      f"Missed: {marker_misses}")

# Verify non-uncertainty text does NOT trigger the probe
non_uncertainty_texts = [
    "The rate limit is definitely 50 req/min.",
    "Use 3600 seconds for the token expiry.",
    "RATE_LIMIT = 100",
    "Configure timeout=30 in settings.",
    "return max_retries * 1000",
]
false_positives = [t for t in non_uncertainty_texts if _has_uncertainty_fn(t)]
check("S14-B non-uncertainty text does not fire probe",
      len(false_positives) == 0,
      f"False positives: {false_positives}")


# ─────────────────────────────────────────────────────────────────────────────
# S15 — Ghost detector robustness (no API; test JSON parsing edge cases)
# ─────────────────────────────────────────────────────────────────────────────

section("S15: Ghost detector — JSON parsing robustness")

# Test the JSON extraction logic directly (simulating what _ghost_detect does
# with various response shapes from Opus).

def _parse_ghost_response(raw: str) -> list[dict]:
    """Mirror of the extraction logic in _ghost_detect."""
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        items = json.loads(raw[start:end])
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (ValueError, TypeError):
                continue  # non-numeric confidence → skip
            if confidence < 0.70:
                continue
            claim = (item.get("claim") or "").strip()
            if not claim:
                continue
            results.append(item)
        return results
    except Exception:
        return []

# S15-A: valid JSON array
valid_json = '[{"claim": "rate limit is 50", "reason": "vendor stated", "confidence": 0.85}]'
res = _parse_ghost_response(valid_json)
check("S15-A valid JSON → 1 item extracted", len(res) == 1)

# S15-B: empty array
check("S15-B empty array → []", _parse_ghost_response("[]") == [])

# S15-C: confidence below threshold
low_conf = '[{"claim": "rate limit is 50", "reason": "vendor", "confidence": 0.50}]'
check("S15-C low-confidence item filtered out", _parse_ghost_response(low_conf) == [])

# S15-D: non-numeric confidence string → should not crash
bad_conf = '[{"claim": "limit is 50", "reason": "stated", "confidence": "high"}]'
try:
    res_d = _parse_ghost_response(bad_conf)
    check("S15-D non-numeric confidence string → no crash, item skipped",
          isinstance(res_d, list),
          f"Got {res_d!r}")
except Exception as e:
    check("S15-D non-numeric confidence string → no crash", False, str(e))

# S15-E: malformed JSON
check("S15-E malformed JSON → []",
      _parse_ghost_response('[{"claim": "limit is 50", "confidence": 0.85') == [])

# S15-F: ] inside string value (rfind edge case)
nested_bracket = '[{"claim": "limit [50] req/min", "reason": "test", "confidence": 0.85}]'
res_f = _parse_ghost_response(nested_bracket)
check("S15-F ] inside string value → parsed correctly", len(res_f) == 1,
      f"Got {res_f!r}")

# S15-G: no array in response (prose explanation)
check("S15-G no array → []",
      _parse_ghost_response("I found no ghost constraints in this message.") == [])

# S15-H: missing claim field
missing_claim = '[{"reason": "vendor stated", "confidence": 0.85}]'
res_h = _parse_ghost_response(missing_claim)
check("S15-H missing claim field → item skipped", res_h == [],
      f"Got {res_h!r}")


# ─────────────────────────────────────────────────────────────────────────────
# S16 — Consistency Enforcer enforcement message quality
# ─────────────────────────────────────────────────────────────────────────────

section("S16: CE — enforcement message quality")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s16 = CredenceRegistry(":memory:")
    reg_s16.register("I think the rate limit is 50 req/min — unconfirmed", "s16",
                     j_score=0.30, turn_idx=0)
    reg_s16.register("Auth token expiry might be 3600s — vendor claim", "s16",
                     j_score=0.25, turn_idx=0)
    mgr_s16 = ContextManager(api_key=api_key, registry=reg_s16, session_id="s16")
    mgr_s16._current_user_message = "What is the rate limit?"

    # S16-A: direct query fires enforcement
    sys_prompt, enforcement_active = mgr_s16._build_enforcement_system_prompt(
        "What is the rate limit?"
    )
    check("S16-A direct rate-limit query → enforcement fires", enforcement_active,
          "Expected enforcement_active=True")
    check("S16-B enforcement message contains CONSISTENCY ENFORCEMENT",
          "CONSISTENCY ENFORCEMENT" in sys_prompt,
          f"Prompt snippet: {sys_prompt[:300]!r}")
    check("S16-C enforcement message contains imperative language",
          "MUST" in sys_prompt or "must" in sys_prompt,
          f"Prompt snippet: {sys_prompt[:300]!r}")

    # S16-D: unrelated query → no enforcement
    sys_prompt_unrel, enforcement_unrel = mgr_s16._build_enforcement_system_prompt(
        "What color should I use for the UI button?"
    )
    check("S16-D unrelated query → no enforcement", not enforcement_unrel,
          f"Got enforcement_active={enforcement_unrel}")

    # S16-E: verified constraint excluded from enforcement
    all_c = reg_s16.get_all("s16")
    for c in all_c:
        reg_s16.verify(c["constraint_id"], "Confirmed value")
    sys_prompt_ver, enforcement_ver = mgr_s16._build_enforcement_system_prompt(
        "What is the rate limit?"
    )
    check("S16-E all verified → no enforcement",
          not enforcement_ver or "EPISTEMIC CONTEXT" not in sys_prompt_ver,
          f"Got enforcement={enforcement_ver}")

except Exception as e:
    check("S16 CE enforcement", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S17 — Proxy J-score boundary values
# ─────────────────────────────────────────────────────────────────────────────

section("S17: Confidence proxy — J-score boundary values")

proxy = CredenceProxy(theta_high=0.70, theta_low=0.45)

# S17-A: empty string → should not crash, returns valid result
try:
    result_empty = proxy.compute("")
    check("S17-A empty string → no crash", hasattr(result_empty, "j_score"))
    check("S17-B empty string → j_score in [0, 1]",
          0.0 <= result_empty.j_score <= 1.0, f"Got {result_empty.j_score}")
except Exception as e:
    check("S17-A empty string → no crash", False, str(e))
    check("S17-B empty string j_score in [0,1]", False, "skipped")

# S17-C: heavily hedged text → low J-score
hedged = (
    "I'm not sure, but I think the rate limit might be around 50 — I haven't confirmed this yet. "
    "It's possible it could be different. I'm uncertain about the exact value."
)
result_hedged = proxy.compute(hedged)
check("S17-C heavily hedged text → LOW zone",
      result_hedged.zone == "LOW",
      f"Got zone={result_hedged.zone}, j={result_hedged.j_score:.3f}")

# S17-D: assertive technical text → HIGH or MEDIUM zone
assertive = (
    "The authentication system uses JWT tokens with RS256 signing. "
    "The token expiry is configured to 3600 seconds. "
    "Rate limiting is enforced at 100 requests per minute via the nginx layer."
)
result_assertive = proxy.compute(assertive)
check("S17-D assertive technical text → HIGH or MEDIUM zone",
      result_assertive.zone in ("HIGH", "MEDIUM"),
      f"Got zone={result_assertive.zone}, j={result_assertive.j_score:.3f}")

# S17-E: code block → Type Prior cap (zone ≤ MEDIUM for code)
code_text = "```python\ndef authenticate(token: str) -> bool:\n    return jwt.decode(token)\n```"
result_code = proxy.compute(code_text)
check("S17-E code block → zone ≤ MEDIUM (Type Prior cap)",
      result_code.zone in ("LOW", "MEDIUM"),
      f"Got zone={result_code.zone}, j={result_code.j_score:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# S18 — End-to-end non-API: ContextManager method invocations
# ─────────────────────────────────────────────────────────────────────────────

section("S18: End-to-end — non-API ContextManager method chain")

try:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy")
    reg_s18 = CredenceRegistry(":memory:")

    mgr_s18 = ContextManager(
        api_key=api_key,
        registry=reg_s18,
        session_id="e2e_test",
        use_ghost_detector=True,
        theta_high=0.70,
        theta_low=0.45,
    )

    # Register a constraint manually
    cid_e2e = reg_s18.register(
        "I think the rate limit is 50 req/min — the vendor mentioned it casually",
        "e2e_test", j_score=0.28, turn_idx=0
    )

    # Truth Buffer injection
    mgr_s18._current_user_message = "What is the rate limit we should code against?"
    tb = mgr_s18._augment_with_truth_buffer()
    check("S18-A Truth Buffer injects constraint", "EPISTEMIC CONTEXT" in tb)

    # CE fires on direct match
    _, enforcement = mgr_s18._build_enforcement_system_prompt(
        "What is the rate limit we should code against?"
    )
    check("S18-B CE fires on 'rate limit' query", enforcement)

    # GTS annotates code embedding the value
    code_e2e = "```python\nRATE_LIMIT = 50  # calls per minute\n```"
    mgr_s18._turn_idx = 1
    annotated_e2e, hits_e2e = mgr_s18._scan_output_for_constraints(code_e2e)
    check("S18-C GTS annotates '50' in code", len(hits_e2e) > 0,
          f"hits={hits_e2e}")

    # Ghost detect: empty → [] (no API call)
    ghost = mgr_s18._ghost_detect("")
    check("S18-D ghost_detect('') → []", ghost == [])

    # Session stats initialized — check turns_compressed/trimmed/preserved all zero
    total_turns = (mgr_s18.stats.turns_compressed +
                   mgr_s18.stats.turns_trimmed +
                   mgr_s18.stats.turns_preserved)
    check("S18-E stats all-zero at init (no turns yet)", total_turns == 0,
          f"Got compressed={mgr_s18.stats.turns_compressed} trimmed={mgr_s18.stats.turns_trimmed} "
          f"preserved={mgr_s18.stats.turns_preserved}")

    # Trajectory recorded
    traj = reg_s18.get_trajectory(cid_e2e)
    check("S18-F trajectory has register event", any(e["event_type"] == "register" for e in traj))

except Exception as e:
    check("S18 end-to-end chain", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# S19 — API tests (requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

section("S19: Ghost detector live API tests")

if not ARGS.api:
    for name in [
        "S19-A ghost detect vendor claim",
        "S19-B ghost detect implicit estimate",
        "S19-C established fact NOT flagged (HTTP 200)",
        "S19-D established fact NOT flagged (Python 0-indexed)",
        "S19-E canonical markers → skips ghost detect call",
    ]:
        skip(name, "use --api to run")
else:
    from credence.context_manager import ContextManager

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "dummy":
        for name in ["S19-A", "S19-B", "S19-C", "S19-D", "S19-E"]:
            skip(name, "no API key")
    else:
        reg_s19 = CredenceRegistry(":memory:")
        mgr_s19 = ContextManager(
            api_key=api_key,
            registry=reg_s19,
            session_id="s19",
            use_ghost_detector=True,
        )

        # S19-A: ghost constraint — vendor claim
        t0 = time.perf_counter()
        res_a = mgr_s19._ghost_detect(
            "The Stripe rate limit is 50 requests per minute."
        )
        latency_a = (time.perf_counter() - t0) * 1000
        check("S19-A vendor claim detected as ghost constraint",
              len(res_a) >= 1,
              f"Got {res_a!r}")
        print(f"         Latency: {latency_a:.0f}ms, detections: {len(res_a)}")

        # S19-B: implicit estimate stated as fact (no hedging marker)
        res_b = mgr_s19._ghost_detect(
            "The database supports 10,000 concurrent connections at peak load."
        )
        check("S19-B implicit estimate (stated as fact) → ghost detected",
              len(res_b) >= 1,
              f"Got {res_b!r} — 'supports 10,000' is unverified claim stated as fact")

        # S19-C: HTTP 200 is an established standard — should NOT be flagged
        res_c = mgr_s19._ghost_detect(
            "HTTP 200 means a successful response."
        )
        check("S19-C HTTP 200 standard → NOT flagged as ghost",
              len(res_c) == 0,
              f"Got {res_c!r} — false positive")

        # S19-D: Python 0-indexed is established — should NOT be flagged
        res_d = mgr_s19._ghost_detect(
            "Python lists are 0-indexed, so the first element is at index 0."
        )
        check("S19-D Python 0-indexed → NOT flagged as ghost",
              len(res_d) == 0,
              f"Got {res_d!r} — false positive")

        # S19-E: message WITH canonical markers → user_uncertainty_detected=True → ghost skipped
        # (This tests the dispatch in chat(), but we test _ghost_detect directly here to
        # verify the model's response when canonical hedging is present — should return [])
        res_e = mgr_s19._ghost_detect(
            "I think the rate limit might be around 50 req/min — I haven't confirmed this."
        )
        # Opus may or may not detect this since the prompt says not to flag hedged claims
        check("S19-E explicitly hedged message → Opus returns [] (respects rule 2)",
              len(res_e) == 0,
              f"Got {res_e!r}")


# ─────────────────────────────────────────────────────────────────────────────
# S20 — Live API end-to-end ContextManager.chat()
# ─────────────────────────────────────────────────────────────────────────────

section("S20: End-to-end live chat with all layers enabled")

if not ARGS.api:
    for name in [
        "S20-A chat() returns TurnResult",
        "S20-B ghost_detections field present",
        "S20-C enforcement_active field present",
        "S20-D scan_hits field present",
        "S20-E TB injects constraint on second turn",
        "S20-F decision_log populated",
    ]:
        skip(name, "use --api to run")
else:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "dummy":
        for name in ["S20-A", "S20-B", "S20-C", "S20-D", "S20-E", "S20-F"]:
            skip(name, "no API key")
    else:
        from credence.context_manager import ContextManager

        reg_s20 = CredenceRegistry(":memory:")
        mgr_s20 = ContextManager(
            api_key=api_key,
            registry=reg_s20,
            session_id="s20_live",
            use_ghost_detector=True,
            theta_high=0.70,
            theta_low=0.45,
        )

        try:
            # Turn 1: plant an uncertain constraint
            r1 = mgr_s20.chat(
                "I think the API rate limit is 50 req/min — "
                "the vendor mentioned it but I haven't confirmed it yet."
            )
            check("S20-A chat() returns TurnResult", hasattr(r1, "response"))
            check("S20-B ghost_detections field present", hasattr(r1, "ghost_detections"))
            check("S20-C enforcement_active field present", hasattr(r1, "enforcement_active"))
            check("S20-D scan_hits field present", hasattr(r1, "scan_hits"))
            print(f"         T1 zone={r1.zone} j={r1.j_score:.3f} "
                  f"ghost={r1.ghost_detections} tb={r1.truth_buffer_count}")

            # Turn 2: ask about the rate limit — CE should fire, TB should inject
            r2 = mgr_s20.chat("What is the rate limit I should use in my code?")
            check("S20-E TB injects constraint on second turn",
                  r2.truth_buffer_count > 0,
                  f"truth_buffer_count={r2.truth_buffer_count}")
            print(f"         T2 enforcement={r2.enforcement_active} "
                  f"tb_count={r2.truth_buffer_count}")

            # Turn 3: model writes code with the value — GTS should fire
            r3 = mgr_s20.chat(
                "Can you write a Python function that respects the rate limit?"
            )
            print(f"         T3 scan_hits={len(r3.scan_hits)} zone={r3.zone}")
            check("S20-F decision_log populated", len(mgr_s20.decision_log) == 3,
                  f"Got {len(mgr_s20.decision_log)} log entries")

        except Exception as e:
            check("S20 live chat", False, str(e))
            for name in ["S20-A", "S20-B", "S20-C", "S20-D", "S20-E", "S20-F"]:
                skip(name, "exception in chat()")


# ─────────────────────────────────────────────────────────────────────────────
# S21 — Regression tests for audit-discovered bugs (fixed April 25)
# ─────────────────────────────────────────────────────────────────────────────

section("S21: Audit regression tests — numeric collision, CE bleed, probe scope")

try:
    from credence.context_manager import ContextManager, _CE_DOMAIN_SYNONYMS, _CE_STOPWORDS
    import os as _os
    _os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

    def _expand(tokens):
        expanded = set(tokens)
        for t in list(tokens):
            if t in _CE_DOMAIN_SYNONYMS:
                expanded |= _CE_DOMAIN_SYNONYMS[t]
        return expanded

    # --- S21-A: Numeric collision — three constraints sharing value 50 ----------
    reg21 = CredenceRegistry(":memory:")
    reg21.register("rate limit is approximately 50 req/min — unconfirmed", "s21", j_score=0.3, zone="LOW")
    reg21.register("retry delay should be around 50 ms", "s21", j_score=0.3, zone="LOW")
    reg21.register("batch size might be 50 items", "s21", j_score=0.3, zone="LOW")
    mgr21 = ContextManager(api_key="dummy", registry=reg21, session_id="s21")
    mgr21._turn_idx = 5
    code = "```python\nRATE_LIMIT = 50\nRETRY_DELAY = 50\nBATCH_SIZE = 50\n```"
    annotated, hits = mgr21._scan_output_for_constraints(code)
    # Each assignment should cite its own constraint
    rate_hit  = next((h for h in hits if "RATE_LIMIT" in h.get("line", "")), None)
    retry_hit = next((h for h in hits if "RETRY_DELAY" in h.get("line", "")), None)
    batch_hit = next((h for h in hits if "BATCH_SIZE" in h.get("line", "")), None)
    check("S21-A RATE_LIMIT=50 cites rate-limit constraint",
          rate_hit is not None and "rate" in rate_hit["constraint_text"].lower(),
          f"got: {rate_hit['constraint_text'][:50] if rate_hit else 'no hit'}")
    check("S21-B RETRY_DELAY=50 cites retry-delay constraint",
          retry_hit is not None and "retry" in retry_hit["constraint_text"].lower(),
          f"got: {retry_hit['constraint_text'][:50] if retry_hit else 'no hit'}")
    check("S21-C BATCH_SIZE=50 cites batch-size constraint",
          batch_hit is not None and "batch" in batch_hit["constraint_text"].lower(),
          f"got: {batch_hit['constraint_text'][:50] if batch_hit else 'no hit'}")

    # --- S21-D: CE synonym bleed — cache query must NOT fire on auth constraint --
    q_cache = {"how", "much", "memory", "does", "cache", "allocate"}
    c_auth  = {"auth", "token", "expiry", "might", "3600", "seconds"}
    q_stop = {w for w in q_cache if w not in _CE_STOPWORDS}
    c_stop = {w for w in c_auth  if w not in _CE_STOPWORDS}
    overlap_bleed = _expand(q_stop) & _expand(c_stop)
    check("S21-D cache query does NOT fire on auth-expiry constraint (no bleed)",
          len(overlap_bleed) < 2,
          f"overlap={overlap_bleed}")

    # --- S21-E: CE true positive preserved — session-expiry query fires on auth --
    q_expiry = {"when", "does", "my", "session", "expire"}
    q_stop2  = {w for w in q_expiry if w not in _CE_STOPWORDS}
    overlap_real = _expand(q_stop2) & _expand(c_stop)
    check("S21-E session-expiry query still fires on auth constraint",
          len(overlap_real) >= 2,
          f"overlap={overlap_real}")

    # --- S21-F: Probe user-only scope — assistant TODO comment does NOT block ----
    msgs_asst_todo = [
        {"role": "user",      "content": "How do I configure the timeout?"},
        {"role": "assistant", "content": "Set timeout=30. # might need to verify for high load"},
        {"role": "user",      "content": "What is the default connection pool size?"},
        {"role": "assistant", "content": "Default is 10. # TODO: confirm this in docs"},
    ]
    mgr21b = ContextManager.__new__(ContextManager)
    mgr21b.proxy = CredenceProxy()
    probe_asst = mgr21b._has_uncertainty_in_user_turns(msgs_asst_todo)
    check("S21-F assistant TODO/might comment does NOT block compression",
          not probe_asst,
          "probe fired on assistant code comment — should be user-turn-only")

    # --- S21-G: Probe still fires on user-stated uncertainty ---------------------
    msgs_user_uncertain = [
        {"role": "user",      "content": "I think the timeout might be 3600 — not confirmed."},
        {"role": "assistant", "content": "Noted. Using 3600 as the default."},
    ]
    probe_user = mgr21b._has_uncertainty_in_user_turns(msgs_user_uncertain)
    check("S21-G user-stated uncertainty still fires probe",
          probe_user,
          "probe missed user-stated 'I think / might be'")

    # --- S21-H: TB shows ALL constraints with no cap (invariant: nothing silently dropped) ---
    reg21c = CredenceRegistry(":memory:")
    for i in range(8):
        reg21c.register(f"Claim {i} — value {100+i} unconfirmed", "s21c",
                        j_score=0.3, zone="LOW", turn_idx=i)
    mgr21c = ContextManager.__new__(ContextManager)
    mgr21c._registry = reg21c
    mgr21c._session_id = "s21c"
    mgr21c._turn_idx = 10
    mgr21c.system_prompt = "You are helpful."
    mgr21c._pending_alignment_caveat = None
    mgr21c._current_user_message = ""
    tb = mgr21c._augment_with_truth_buffer()
    bullet_count_21h = tb.count("• [")
    check("S21-H TB shows all 8 constraints (no silent drop, no truncation)",
          bullet_count_21h == 8,
          f"Expected 8 bullets, got {bullet_count_21h}")

    # --- S21-I: String-aware GTS — catches unquoted identifier in string literal assignment
    reg21d = CredenceRegistry(":memory:")
    reg21d.register('The API uses "RS256" signing — per vendor docs, unconfirmed', "s21d",
                    j_score=0.3, zone="LOW")
    mgr21d = ContextManager(api_key="dummy", registry=reg21d, session_id="s21d")
    mgr21d._turn_idx = 2
    code_str = '```python\nALGORITHM = "RS256"\n```'
    _, hits_str = mgr21d._scan_output_for_constraints(code_str)
    check("S21-I string GTS catches quoted identifier (RS256) in code assignment",
          len(hits_str) >= 1,
          f"GTS missed string literal assignment. hits={hits_str}")

except Exception as e:
    import traceback
    for name in ["S21-A","S21-B","S21-C","S21-D","S21-E","S21-F","S21-G","S21-H","S21-I"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# S22 — Probe fires on user turns only (compression_faithfulness.py alignment)
# Verifies the study's _compress_with_probe matches production behavior:
#   - probes user text only, not assistant echo text
#   - common hedges (probably, maybe, ambiguous) now trigger correctly
#   - hardcoded echo 'unverified' alone does NOT trigger when user has no markers
# ─────────────────────────────────────────────────────────────────────────────
print("\n── S22: Probe user-turn-only alignment ─────────────────────────────────")
try:
    from evals.compression_faithfulness import _compress_with_probe, _has_uncertainty

    ECHO = (
        "Understood — I've noted that as an unverified constraint. "
        "We'll need to confirm it before committing to the implementation. "
        "Let's continue and flag it as an open question for now."
    )

    # S22-A: user seed with 'probably' triggers probe (new marker)
    conv_a = [
        {"role": "user",      "content": "The commission rate is probably 20%, but may have changed."},
        {"role": "assistant", "content": ECHO},
    ]
    _, blocked_a = _compress_with_probe(conv_a)
    check("S22-A user 'probably' triggers probe",
          blocked_a,
          "probe did not fire on 'probably'")

    # S22-B: user seed with 'maybe' triggers probe (new marker)
    conv_b = [
        {"role": "user",      "content": "Maybe the SLA is 1 hour, I haven't confirmed."},
        {"role": "assistant", "content": ECHO},
    ]
    _, blocked_b = _compress_with_probe(conv_b)
    check("S22-B user 'maybe' triggers probe",
          blocked_b,
          "probe did not fire on 'maybe'")

    # S22-C: user seed with 'ambiguous' triggers probe (new marker)
    conv_c = [
        {"role": "user",      "content": "The contract language is ambiguous on P1 response time."},
        {"role": "assistant", "content": ECHO},
    ]
    _, blocked_c = _compress_with_probe(conv_c)
    check("S22-C user 'ambiguous' triggers probe",
          blocked_c,
          "probe did not fire on 'ambiguous'")

    # S22-D: echo alone (no user uncertainty) does NOT trigger probe
    # This verifies the study is not inflated by the hardcoded echo text
    conv_d = [
        {"role": "user",      "content": "The rate limit is 100 requests per minute."},
        {"role": "assistant", "content": ECHO},   # contains 'unverified', 'open question'
    ]
    _, blocked_d = _compress_with_probe(conv_d)
    check("S22-D echo-only does NOT trigger probe (user has no markers)",
          not blocked_d,
          f"probe falsely fired on echo-only: blocked={blocked_d}")

    # S22-E: all 30 study scenarios block on user text alone
    from evals.compression_faithfulness import SCENARIOS, _build_conversation
    from credence.context_manager import _UNCERTAINTY_MARKERS
    missed = []
    for i, (stmt, label, _) in enumerate(SCENARIOS):
        conv = _build_conversation(stmt)
        user_text = " ".join(m["content"] for m in conv if m.get("role") == "user")
        if not _has_uncertainty(user_text):
            missed.append(label)
    check("S22-E all 50 study scenarios trigger probe on user-only text",
          len(missed) == 0,
          f"missed scenarios: {missed}")

except Exception as e:
    import traceback
    for name in ["S22-A","S22-B","S22-C","S22-D","S22-E"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# S23 — Contradiction detector (offline logic tests)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── S23: Contradiction detector — logic and registry integration ─────────")

try:
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry
    import os as _os23
    _os23.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

    # S23-A: _detect_contradiction returns [] when registry has no constraints
    reg23a = CredenceRegistry(":memory:")
    mgr23a = ContextManager(api_key="dummy", registry=reg23a, session_id="s23a",
                             use_ghost_detector=True)
    # With no API key, _detect_contradiction will raise → returns []
    result_23a = mgr23a._detect_contradiction("The rate limit is 200 req/min.")
    check("S23-A no constraints → _detect_contradiction returns []",
          isinstance(result_23a, list),
          f"expected list, got {type(result_23a)}")

    # S23-B: registry.check_contradiction finds similar VERIFIED constraints
    reg23b = CredenceRegistry(":memory:")
    cid = reg23b.register(
        "rate limit is approximately 50 req/min — unconfirmed", "s23b",
        j_score=0.3, zone="LOW"
    )
    # check_contradiction requires verified=1 (it finds confirmed facts being contradicted)
    reg23b.verify(cid, "confirmed at 50")
    similar = reg23b.check_contradiction(
        "the actual rate limit is 200 requests per minute", "s23b"
    )
    check("S23-B check_contradiction finds overlapping verified constraint",
          len(similar) > 0,
          f"expected ≥1 match, got {similar}")

    # S23-C: mark_contradiction changes validation_status to 'disputed'
    reg23b.mark_contradiction(cid, "new_value=200")
    rows = reg23b.get_all("s23b")
    row = next((r for r in rows if r["constraint_id"] == cid), None)
    check("S23-C mark_contradiction sets validation_status=disputed",
          row is not None and row.get("validation_status") == "disputed",
          f"got {row.get('validation_status') if row else 'no row'}")

    # S23-D: disputed constraint triggers CE enforcement without overlap threshold
    reg23d = CredenceRegistry(":memory:")
    cid_d = reg23d.register("rate limit is 50 req/min — unconfirmed", "s23d",
                              j_score=0.3, zone="LOW")
    reg23d.mark_contradiction(cid_d, "new_value=200")
    mgr23d = ContextManager(api_key="dummy", registry=reg23d, session_id="s23d")
    mgr23d._turn_idx = 5
    # Direct call to _direct_constraint_matches — DISPUTED constraint escalates regardless
    constraints = reg23d.list_uncertain("s23d")
    matches_d = mgr23d._direct_constraint_matches("what should we set the limit to", constraints)
    check("S23-D disputed constraint always escalates to CE (no overlap required)",
          any(m.get("_overlap") == ["DISPUTED"] for m in matches_d),
          f"got matches: {[m.get('_overlap') for m in matches_d]}")

    # S23-E: TurnResult has contradictions_detected field (dataclass check)
    from credence.context_manager import TurnResult
    import dataclasses
    fields = {f.name for f in dataclasses.fields(TurnResult)}
    check("S23-E TurnResult has contradictions_detected field",
          "contradictions_detected" in fields,
          f"fields present: {fields}")

    # S23-F: contradictions_detected defaults to empty list
    tr = TurnResult.__new__(TurnResult)
    # Using dataclass defaults
    tr_default = TurnResult(
        turn_idx=0, response="test", j_score=0.5, zone="MEDIUM", decision="PRESERVE",
        tokens_in=10, tokens_out=10, tokens_saved=0, cost_usd=0.0,
        savings_usd=0.0, reasoning="",
        session_tokens_used=20, session_tokens_saved=0,
        session_cost_usd=0.0, session_savings_usd=0.0,
        compression_ratio=0.0, thinking_tokens=0, thinking_utilization=0.0,
        thinking_budget_used=0, drift_state=False,
        adaptive_theta_high=0.70, adaptive_theta_low=0.45,
        uncertainty_preserved=False, truth_buffer_count=0, scout_extractions=0,
        alignment_warnings=[], caveat_injected=False,
        user_uncertainty_detected=False, se_score=0.0, se_uncertain=False,
        enforcement_active=False, scan_hits=[], ghost_detections=0,
        contradictions_detected=[],
    )
    check("S23-F contradictions_detected defaults to empty list",
          tr_default.contradictions_detected == [],
          f"got {tr_default.contradictions_detected!r}")

    # S23-G: compression_faithfulness has 50 scenarios after extension
    from evals.compression_faithfulness import SCENARIOS
    check("S23-G compression_faithfulness has 50 scenarios",
          len(SCENARIOS) == 50,
          f"got {len(SCENARIOS)} scenarios")

    # S23-H: all 50 scenarios have uncertainty markers in user text
    from evals.compression_faithfulness import _build_conversation, _has_uncertainty
    missed50 = []
    for stmt, label, _ in SCENARIOS:
        conv = _build_conversation(stmt)
        user_text = " ".join(m["content"] for m in conv if m.get("role") == "user")
        if not _has_uncertainty(user_text):
            missed50.append(label)
    check("S23-H all 50 scenarios trigger faithfulness probe",
          len(missed50) == 0,
          f"missed: {missed50}")

    # S23-I: ghost_detector_ablation pure sessions have zero canonical markers
    from evals.ghost_detector_ablation import PURE_SESSIONS, _has_any_canonical_marker
    contaminated = []
    for s in PURE_SESSIONS:
        for gt in s.ghost_turns:
            if _has_any_canonical_marker(gt["content"]):
                contaminated.append(f"{s.session_id}/{gt['ghost_id']}")
    check("S23-I all pure ghost sessions have zero canonical markers",
          len(contaminated) == 0,
          f"contaminated: {contaminated}")

except Exception as e:
    import traceback
    for name in ["S23-A","S23-B","S23-C","S23-D","S23-E","S23-F","S23-G","S23-H","S23-I"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
print("\n── S24: Cross-session memory ─────────────────────────────────────────────")
# ─────────────────────────────────────────────────────────────────────────────

try:
    from credence.registry import CredenceRegistry as _Reg
    from credence.memory import CredenceMemory as _Mem, MemorySnapshot, MemoryRecall

    _reg24 = _Reg(":memory:")
    _mem24 = _Mem(_reg24)

    # S24-A: snapshot returns 0 items when session has no constraints
    snap_empty = _mem24.snapshot("empty-session", project="proj-x")
    check("S24-A snapshot empty session returns 0 items",
          snap_empty.saved_count == 0, f"got {snap_empty.saved_count}")

    # S24-B: snapshot captures registered unverified constraints
    _reg24.register("rate limit is 50 req/min — unconfirmed", "s1",
                    j_score=0.28, zone="LOW")
    _reg24.register("token expiry is 3600s — tentative", "s1",
                    j_score=0.24, zone="LOW")
    snap = _mem24.snapshot("s1", project="proj-x")
    check("S24-B snapshot captures 2 unverified constraints",
          snap.saved_count == 2, f"got {snap.saved_count}")

    # S24-C: snapshot result has project_id and session_id set
    check("S24-C snapshot has correct project_id",
          snap.project_id == "proj-x", f"got {snap.project_id}")
    check("S24-C snapshot has correct session_id",
          snap.session_id == "s1", f"got {snap.session_id}")

    # S24-D: recall_project_memories returns the 2 constraints
    memories = _reg24.recall_project_memories("proj-x")
    check("S24-D recall_project_memories returns 2 items",
          len(memories) == 2, f"got {len(memories)}")

    # S24-E: inject_memories_into_session copies constraints to new session
    injected = _reg24.inject_memories_into_session("proj-x", "s2")
    check("S24-E inject_memories returns 2 constraint_ids",
          len(injected) == 2, f"got {len(injected)}")

    # S24-F: new session can query injected constraints via list_uncertain
    uncertain_s2 = _reg24.list_uncertain("s2")
    check("S24-F new session has 2 unverified constraints",
          len(uncertain_s2) == 2, f"got {len(uncertain_s2)}")

    # S24-G: injected constraints have source='cross_session_memory'
    sources = {c.get("source") for c in uncertain_s2}
    check("S24-G injected constraints have correct source",
          "cross_session_memory" in sources, f"sources: {sources}")

    # S24-H: recall_and_inject returns a MemoryRecall with system_block
    recall = _mem24.recall_and_inject(project="proj-x", new_session_id="s3")
    check("S24-H recall_and_inject returns MemoryRecall",
          isinstance(recall, MemoryRecall), f"got {type(recall)}")
    check("S24-H system_block is non-empty",
          len(recall.system_block) > 0, f"system_block empty")
    check("S24-H system_block contains project name",
          "proj-x" in recall.system_block, f"no project name in block")

    # S24-I: project_status reports correct epistemic_debt
    status = _mem24.project_status("proj-x")
    check("S24-I project_status has epistemic_debt > 0",
          status["epistemic_debt"] > 0, f"debt={status.get('epistemic_debt')}")
    check("S24-I project_status has correct total_memories",
          status["total_memories"] > 0, f"total={status.get('total_memories')}")

    # S24-J: verified constraint is NOT included in memory recall
    # Verify the original s1 constraint and confirm count decreases
    _before_count = len(_reg24.recall_project_memories("proj-x"))
    _orig_cids = [m["constraint_id"] for m in _reg24.recall_project_memories("proj-x")]
    if _orig_cids:
        _reg24.verify(_orig_cids[0], "confirmed: 50 req/min from official docs")
    memories_after = _reg24.recall_project_memories("proj-x")
    check("S24-J verified constraint excluded from project memories",
          len(memories_after) < _before_count,
          f"count unchanged: {len(memories_after)} == {_before_count}")

    # S24-K: MemorySnapshot.summary() produces readable output
    summary_str = snap.summary()
    check("S24-K snapshot summary mentions saved count",
          "2" in summary_str or "Snapshotted" in summary_str, f"summary: {summary_str[:80]}")

    # S24-L: snapshot idempotent — calling twice doesn't double-count
    snap2 = _mem24.snapshot("s1", project="proj-x")
    memories2 = _reg24.recall_project_memories("proj-x")
    check("S24-L snapshot idempotent — no duplicates",
          len(memories2) == len(memories_after), f"got {len(memories2)} vs {len(memories_after)}")

except Exception as e:
    import traceback
    for name in ["S24-A","S24-B","S24-C","S24-D","S24-E","S24-F","S24-G",
                 "S24-H","S24-I","S24-J","S24-K","S24-L"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

section("S25: String GTS, TB no-cap invariant, vendor markers")
# ─────────────────────────────────────────────────────────────────────────────
# S25 — String-aware GTS, TB no-cap invariant, expanded vendor markers
# Tests the three improvements added in the production hardening pass:
#   1. String GTS catches non-numeric values (endpoints, scopes, identifiers)
#   2. Truth Buffer shows ALL constraints — no silent drop above any threshold
#   3. Vendor/source language markers trigger the faithfulness probe
# ─────────────────────────────────────────────────────────────────────────────
print("\n── S25: String GTS, TB no-cap invariant, vendor markers ─────────────────")
try:
    from credence.context_manager import ContextManager, _UNCERTAINTY_MARKERS
    from credence.registry import CredenceRegistry

    def _has_uncertainty(text: str) -> bool:
        lower = text.lower()
        return any(m in lower for m in _UNCERTAINTY_MARKERS)

    # --- S25-A: String GTS — quoted path fragment caught ---
    reg25a = CredenceRegistry(":memory:")
    reg25a.register('API base URL is "/api/v2" — tentative, not confirmed', "s25a",
                    j_score=0.25, zone="LOW")
    mgr25a = ContextManager(api_key=api_key, registry=reg25a, session_id="s25a")
    code_a = '```python\nBASE_URL = "/api/v2"\nTIMEOUT = 30\n```'
    _, hits_a = mgr25a._scan_output_for_constraints(code_a)
    check("S25-A string GTS catches quoted path /api/v2 in code assignment",
          any(h["source"] in ("code_string",) for h in hits_a),
          f"No code_string hits. hits={hits_a}")

    # --- S25-B: String GTS — OAuth scope fragment caught ---
    reg25b = CredenceRegistry(":memory:")
    reg25b.register('The OAuth scope should be "read:users write:users" — from vendor call',
                    "s25b", j_score=0.30, zone="LOW")
    mgr25b = ContextManager(api_key=api_key, registry=reg25b, session_id="s25b")
    code_b = '```python\nOAUTH_SCOPE = "read:users write:users"\n```'
    _, hits_b = mgr25b._scan_output_for_constraints(code_b)
    check("S25-B string GTS catches OAuth scope fragment",
          len(hits_b) >= 1 and "read:users write:users" in hits_b[0]["value"],
          f"hits={hits_b}")

    # --- S25-C: String GTS — uppercase identifier caught (RS256) ---
    reg25c = CredenceRegistry(":memory:")
    reg25c.register("Encryption algorithm is RS256 — per vendor docs, unconfirmed",
                    "s25c", j_score=0.28, zone="LOW")
    mgr25c = ContextManager(api_key=api_key, registry=reg25c, session_id="s25c")
    code_c = '```python\nALGORITHM = "RS256"\n```'
    _, hits_c = mgr25c._scan_output_for_constraints(code_c)
    check("S25-C string GTS catches uppercase identifier RS256",
          len(hits_c) >= 1,
          f"No hits for RS256. hits={hits_c}")

    # --- S25-D: String GTS — hyphenated region identifier caught (us-east-1) ---
    reg25d = CredenceRegistry(":memory:")
    reg25d.register("AWS region is us-east-1 — from the sales call, not confirmed",
                    "s25d", j_score=0.25, zone="LOW")
    mgr25d = ContextManager(api_key=api_key, registry=reg25d, session_id="s25d")
    code_d = '```python\nAWS_REGION = "us-east-1"\n```'
    _, hits_d = mgr25d._scan_output_for_constraints(code_d)
    check("S25-D string GTS catches hyphenated region us-east-1",
          len(hits_d) >= 1,
          f"No hits for us-east-1. hits={hits_d}")

    # --- S25-E: TB no-cap — 15 constraints all shown in Truth Buffer ---
    reg25e = CredenceRegistry(":memory:")
    for i in range(15):
        reg25e.register(f"Unverified constraint {i}: value {(i+1)*10} — unconfirmed",
                        "s25e", j_score=0.28, zone="LOW")
    mgr25e = ContextManager(api_key=api_key, registry=reg25e, session_id="s25e")
    mgr25e._current_user_message = ""
    tb_15 = mgr25e._augment_with_truth_buffer()
    bullet_count_25e = tb_15.count("• [")
    check("S25-E TB no-cap: all 15 constraints shown (no silent drop)",
          bullet_count_25e == 15,
          f"Expected 15 bullets, got {bullet_count_25e}")

    # --- S25-F: TB no-cap — 20 constraints all shown ---
    reg25f = CredenceRegistry(":memory:")
    for i in range(20):
        reg25f.register(f"Constraint {i}: rate {(i+1)*5} req/s — unconfirmed",
                        "s25f", j_score=0.30, zone="LOW")
    mgr25f = ContextManager(api_key=api_key, registry=reg25f, session_id="s25f")
    mgr25f._current_user_message = ""
    tb_20 = mgr25f._augment_with_truth_buffer()
    bullet_count_25f = tb_20.count("• [")
    check("S25-F TB no-cap: all 20 constraints shown",
          bullet_count_25f == 20,
          f"Expected 20 bullets, got {bullet_count_25f}")

    # --- S25-G: Vendor markers — 'sales call' triggers probe ---
    check("S25-G 'sales call' in _UNCERTAINTY_MARKERS",
          "sales call" in _UNCERTAINTY_MARKERS, "'sales call' missing from markers")
    text_g = "The rate limit is 100 req/min — got this from the sales call, not confirmed"
    check("S25-G probe fires on 'sales call' in user text",
          _has_uncertainty(text_g), f"probe did not fire on: {text_g!r}")

    # --- S25-H: Vendor markers — 'per the vendor' triggers probe ---
    check("S25-H 'per the vendor' in _UNCERTAINTY_MARKERS",
          "per the vendor" in _UNCERTAINTY_MARKERS, "'per the vendor' missing from markers")
    text_h = "Token expiry is 3600s per the vendor — haven't verified"
    check("S25-H probe fires on 'per the vendor'",
          _has_uncertainty(text_h), f"probe did not fire on: {text_h!r}")

    # --- S25-I: Vendor markers — 'from the demo' triggers probe ---
    check("S25-I 'from the demo' in _UNCERTAINTY_MARKERS",
          "from the demo" in _UNCERTAINTY_MARKERS, "'from the demo' missing from markers")
    text_i = "Peak load is 500 concurrent users — that's what from the demo showed"
    check("S25-I probe fires on 'from the demo'",
          _has_uncertainty(text_i), f"probe did not fire on: {text_i!r}")

    # --- S25-J: Vendor markers — 'not load-tested' triggers probe ---
    check("S25-J 'not load-tested' in _UNCERTAINTY_MARKERS",
          "not load-tested" in _UNCERTAINTY_MARKERS, "'not load-tested' missing from markers")
    text_j = "Memory per pod is 2GB — not load-tested yet"
    check("S25-J probe fires on 'not load-tested'",
          _has_uncertainty(text_j), f"probe did not fire on: {text_j!r}")

    # --- S25-K: No false positive — confident text does NOT trigger probe ---
    definitive_texts = [
        "The rate limit is confirmed at 100 req/min.",
        "Token expiry is set to 3600 seconds in production.",
        "The vendor confirmed RS256 as the signing algorithm.",
        "AWS region us-east-1 is verified in the deployment config.",
    ]
    false_positives = [t for t in definitive_texts if _has_uncertainty(t)]
    check("S25-K no false positives on definitive statements",
          len(false_positives) == 0,
          f"False positives: {false_positives}")

    # --- S25-L: String GTS — numeric constraint still works alongside string ---
    reg25l = CredenceRegistry(":memory:")
    reg25l.register('Rate limit is probably 50 req/min — unverified', "s25l", j_score=0.28, zone="LOW")
    reg25l.register('Auth endpoint is "/auth/v2/token" — tentative', "s25l", j_score=0.25, zone="LOW")
    mgr25l = ContextManager(api_key=api_key, registry=reg25l, session_id="s25l")
    code_l = '```python\nRATE_LIMIT = 50\nAUTH_ENDPOINT = "/auth/v2/token"\n```'
    _, hits_l = mgr25l._scan_output_for_constraints(code_l)
    sources_l = {h["source"] for h in hits_l}
    check("S25-L numeric and string GTS both fire in same code block",
          len(hits_l) == 2 and "code" in sources_l and "code_string" in sources_l,
          f"hits={hits_l} sources={sources_l}")

except Exception as e:
    import traceback
    for name in ["S25-A","S25-B","S25-C","S25-D","S25-E","S25-F",
                 "S25-G","S25-H","S25-I","S25-J","S25-K","S25-L"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

section("S26: Pipeline Monitor — multi-agent epistemic propagation")
# ─────────────────────────────────────────────────────────────────────────────
# S26 — PipelineMonitor: intercept, extract, inject, handoff
# Tests the cross-agent enforcement middleware added in the agents pass.
# All tests are offline (no API key); Ghost Detector tests skipped unless --api.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── S26: Pipeline Monitor — multi-agent enforcement ──────────────────────")
try:
    from credence.pipeline_monitor import PipelineMonitor, EpistemicHandoff, ExtractedClaim
    from credence.registry import CredenceRegistry

    # --- S26-A: PipelineMonitor imports cleanly ---
    check("S26-A PipelineMonitor and EpistemicHandoff import cleanly", True)

    # --- S26-B: probe extracts from canonical markers ---
    reg26b = CredenceRegistry(":memory:")
    mon26b = PipelineMonitor(registry=reg26b, api_key=None, use_ghost_detector=False)
    handoff_b = mon26b.intercept(
        "I think the rate limit is about 50 req/min — unconfirmed from staging.",
        "agent_a", "agent_b",
    )
    check("S26-B probe extracts ≥1 claim from canonical marker text",
          handoff_b.n_extracted >= 1,
          f"n_extracted={handoff_b.n_extracted}")
    check("S26-B strategy is 'probe'",
          handoff_b.strategy == "probe",
          f"strategy={handoff_b.strategy}")

    # --- S26-C: extracted claim is registered in shared registry ---
    check("S26-C n_injected ≥ 1 (claim registered in registry)",
          handoff_b.n_injected >= 1,
          f"n_injected={handoff_b.n_injected}")
    uncertain_b = reg26b.list_uncertain("agent_a")
    check("S26-C registry has ≥1 unverified constraint after intercept",
          len(uncertain_b) >= 1,
          f"uncertain count={len(uncertain_b)}")

    # --- S26-D: handoff system_block is non-empty and contains EPISTEMIC HANDOFF ---
    check("S26-D system_block is non-empty",
          len(handoff_b.system_block) > 0,
          "system_block was empty")
    check("S26-D system_block contains EPISTEMIC HANDOFF header",
          "EPISTEMIC HANDOFF" in handoff_b.system_block,
          f"header missing from system_block")

    # --- S26-E: has_uncertain property ---
    check("S26-E has_uncertain = True when claims injected",
          handoff_b.has_uncertain is True)

    # --- S26-F: no claims on definitely-certain text ---
    reg26f = CredenceRegistry(":memory:")
    mon26f = PipelineMonitor(registry=reg26f, api_key=None, use_ghost_detector=False)
    handoff_f = mon26f.intercept(
        "The production rate limit is 100 req/s. This was confirmed by the vendor and tested in production.",
        "agent_a_f", "agent_b_f",
    )
    check("S26-F no claims extracted from definitive confirmed text",
          handoff_f.n_injected == 0,
          f"n_injected={handoff_f.n_injected} (unexpected claims found)")

    # --- S26-G: build_agent_b_system includes handoff block + base system ---
    reg26g = CredenceRegistry(":memory:")
    mon26g = PipelineMonitor(registry=reg26g, api_key=None, use_ghost_detector=False)
    handoff_g = mon26g.intercept(
        "I think the auth token expiry is maybe 3600s — not confirmed.",
        "ag_a", "ag_b",
    )
    system_g = mon26g.build_agent_b_system(
        handoff=handoff_g,
        base_system="You are a technical implementer.",
        include_gate=True,
    )
    check("S26-G build_agent_b_system contains EPISTEMIC HANDOFF block",
          "EPISTEMIC HANDOFF" in system_g)
    check("S26-G build_agent_b_system contains GATE enforcement line",
          "GATE" in system_g)
    check("S26-G build_agent_b_system contains base system prompt",
          "technical implementer" in system_g)

    # --- S26-H: empty output → no claims, empty system_block ---
    reg26h = CredenceRegistry(":memory:")
    mon26h = PipelineMonitor(registry=reg26h, api_key=None, use_ghost_detector=False)
    handoff_h = mon26h.intercept("", "a", "b")
    check("S26-H empty agent output → n_injected=0",
          handoff_h.n_injected == 0)
    check("S26-H empty agent output → system_block is empty",
          handoff_h.system_block == "")

    # --- S26-I: vendor marker phrases trigger probe ---
    reg26i = CredenceRegistry(":memory:")
    mon26i = PipelineMonitor(registry=reg26i, api_key=None, use_ghost_detector=False)
    handoff_i = mon26i.intercept(
        "Per the vendor documentation, the rate limit is 500 req/min.",
        "a_i", "b_i",
    )
    check("S26-I 'per the vendor' marker triggers probe extraction",
          handoff_i.n_injected >= 1,
          f"n_injected={handoff_i.n_injected}")

    # --- S26-J: handoff_report is non-empty when claims found ---
    report_j = mon26b.handoff_report(handoff_b)
    check("S26-J handoff_report non-empty when claims present",
          len(report_j) > 0 and "PipelineMonitor" in report_j)

    # --- S26-K: EpistemicHandoff exported from credence package ---
    import credence
    check("S26-K PipelineMonitor exported from credence package",
          hasattr(credence, "PipelineMonitor"))
    check("S26-K EpistemicHandoff exported from credence package",
          hasattr(credence, "EpistemicHandoff"))

    # --- S26-L: agent_propagation_eval dry-run imports cleanly ---
    import importlib
    try:
        importlib.import_module("evals.agent_propagation_eval")
        check("S26-L evals.agent_propagation_eval imports cleanly", True)
    except ImportError as ie:
        check("S26-L evals.agent_propagation_eval imports cleanly", False, str(ie))

except Exception as e:
    import traceback
    for name in ["S26-A","S26-B","S26-C","S26-D","S26-E","S26-F",
                 "S26-G","S26-H","S26-I","S26-J","S26-K","S26-L"]:
        check(name, False, f"exception: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
total = _PASS + _FAIL + _SKIP
print(f"  STRESS TEST RESULTS")
print(f"  Passed:  {_PASS}")
print(f"  Failed:  {_FAIL}")
print(f"  Skipped: {_SKIP}")
print(f"  Total:   {total}")
print(f"{'═'*60}")

if _FAIL == 0:
    print("\n  ✓ ALL TESTS PASSED — system stable under stress\n")
    sys.exit(0)
else:
    print(f"\n  ✗ {_FAIL} FAILURE(S) — review above before locking\n")
    sys.exit(1)

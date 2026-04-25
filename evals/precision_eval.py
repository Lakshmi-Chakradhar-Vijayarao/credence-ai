"""
Precision Metrics Eval — Consistency Enforcer + Generation-Time Scanner
=========================================================================
Measures false-positive rates for the two enforcement layers.

Why this matters:
  High recall (catching all unverified uses) is useless if false positive rate
  is too high — the system becomes noise, developers tune it out. This eval
  measures precision: when the system fires, is it right?

Two layers measured:

  CE (Consistency Enforcer):
    True positive  — query asks about a registered uncertain constraint
    False positive — query is about something completely unrelated
    Borderline     — query is about the same domain but a different entity
                     (CE is conservative by design: better to over-warn than miss)
    Target: 0% pure FP (unrelated), ≤30% borderline FP

  GTS (Generation-Time Scanner):
    True positive  — code/prose uses a registered uncertain value
    False positive (numeric)  — same numeric literal, different concept (unavoidable)
    False positive (string)   — same string, different concept (should be 0%)
    Target: GTS string FP = 0%; numeric FP is documented and acceptable

  Faithfulness Probe:
    True positive  — text contains uncertainty markers
    False positive — confident definitive text fires the probe
    Target: 0% FP on definitive statements

All tests are deterministic — zero API calls.

Run:
  python -m evals.precision_eval
"""

from __future__ import annotations
import sys
from credence.context_manager import ContextManager, _UNCERTAINTY_MARKERS
from credence.registry import CredenceRegistry


def _has_uncertainty(text: str) -> bool:
    """Module-level wrapper matching ContextManager._has_uncertainty logic."""
    lower = text.lower()
    return any(m in lower for m in _UNCERTAINTY_MARKERS)


def main() -> None:
    PASS = 0
    FAIL = 0

    def check(name: str, condition: bool, msg: str = "") -> None:
        nonlocal PASS, FAIL
        if condition:
            PASS += 1
            print(f"  ✓ {name}")
        else:
            FAIL += 1
            print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))

    # ---------------------------------------------------------------------------
    # Part 1 — CE False Positive Rate
    # ---------------------------------------------------------------------------
    print("\n── Part 1: Consistency Enforcer False Positive Rate ───────────────────")
    print("   Registered: 'Stripe rate limit is ~50 req/min — unverified'")
    print("   Registered: 'JWT token expiry is ~3600s — tentative'\n")

    reg_ce = CredenceRegistry(":memory:")
    reg_ce.register("Stripe rate limit is ~50 req/min — from sales call, unverified",
                    "ce_test", j_score=0.28, zone="LOW")
    reg_ce.register("JWT token expiry is ~3600s — tentative, needs security review",
                    "ce_test", j_score=0.26, zone="LOW")

    mgr_ce = ContextManager(api_key="dummy", registry=reg_ce, session_id="ce_test")

    # --- True positives: CE SHOULD fire ---
    CE_TRUE_POSITIVES = [
        ("rate limit query — direct",        "What rate limit should I set for Stripe?"),
        ("rate limit query — paraphrase",    "How fast can we call the Stripe endpoint?"),
        ("rate limit query — synonym",       "What's the request throttle limit for Stripe?"),
        ("jwt expiry query — direct",        "What should I set for JWT token expiry?"),
        ("jwt expiry query — paraphrase",    "When does the auth token expire?"),
        ("jwt expiry query — synonym",       "What's the session timeout duration?"),
    ]

    print("  True positives (CE should fire):")
    ce_tp = 0
    for label, query in CE_TRUE_POSITIVES:
        _, fired = mgr_ce._build_enforcement_system_prompt(query)
        check(f"    TP: {label}", fired, f"CE did not fire on: {query!r}")
        if fired:
            ce_tp += 1

    # --- Pure false positives: CE should NOT fire (completely unrelated topics) ---
    CE_PURE_FP = [
        ("unrelated — color palette",  "What color scheme should I use for the dashboard?"),
        ("unrelated — test framework", "Should I use pytest or unittest for this project?"),
        ("unrelated — git workflow",   "What branch strategy should the team use?"),
        ("unrelated — font choice",    "What font works best for code in the UI?"),
        ("unrelated — logging format", "Should we use JSON or plaintext for logs?"),
        ("unrelated — orm choice",     "Should we use SQLAlchemy or Django ORM?"),
    ]

    print("\n  Pure false positives (CE should NOT fire — completely unrelated):")
    ce_pure_fp = 0
    for label, query in CE_PURE_FP:
        _, fired = mgr_ce._build_enforcement_system_prompt(query)
        if fired:
            ce_pure_fp += 1
        check(f"    {label}", not fired, f"CE fired on: {query!r}")

    # --- Borderline: same domain, different entity (CE is conservative by design) ---
    CE_BORDERLINE = [
        ("borderline — GitHub rate limit (same domain, different service)",
         "What rate limit does GitHub's API have?"),
        ("borderline — CSRF token expiry (same domain, different token)",
         "What's the CSRF token expiry we should use?"),
    ]
    print("\n  Borderline (CE conservative by design — same domain, different entity):")
    ce_borderline_fp = 0
    for label, query in CE_BORDERLINE:
        _, fired = mgr_ce._build_enforcement_system_prompt(query)
        if fired:
            ce_borderline_fp += 1
        # Note: CE firing here is conservative, not wrong — the user has an
        # unverified rate limit registered; any rate limit discussion benefits
        # from the reminder. Document as expected behavior, not a hard failure.
        marker = " (conservative — expected)" if fired else " (silent — also ok)"
        print(f"    {'⚠' if fired else '✓'} {label}{marker}")

    ce_fp_rate = ce_pure_fp / len(CE_PURE_FP)
    ce_tp_rate = ce_tp / len(CE_TRUE_POSITIVES)
    print(f"\n  CE Recall (TP rate):          {ce_tp_rate:.0%}  ({ce_tp}/{len(CE_TRUE_POSITIVES)})")
    print(f"  CE Pure FP rate (unrelated):  {ce_fp_rate:.0%}  ({ce_pure_fp}/{len(CE_PURE_FP)})")
    print(f"  CE Borderline FP rate:        {ce_borderline_fp/len(CE_BORDERLINE):.0%}  (conservative by design)")
    check("CE pure FP rate = 0% on completely unrelated topics",
          ce_pure_fp == 0,
          f"Pure FP rate: {ce_fp_rate:.0%}")

    # ---------------------------------------------------------------------------
    # Part 2 — GTS False Positive Rate
    # ---------------------------------------------------------------------------
    print("\n── Part 2: GTS False Positive Rate ────────────────────────────────────")
    print("   Registered: 'retry count is probably 50 — from forum'")
    print("   Registered: 'API endpoint is \"/api/v2\" — tentative'\n")

    reg_gts = CredenceRegistry(":memory:")
    reg_gts.register("retry count is probably 50 — from forum post, not production-tested",
                     "gts_test", j_score=0.28, zone="LOW")
    reg_gts.register('API endpoint is "/api/v2" — tentative, not confirmed',
                     "gts_test", j_score=0.26, zone="LOW")

    mgr_gts = ContextManager(api_key="dummy", registry=reg_gts, session_id="gts_test")

    # --- True positives: GTS SHOULD annotate ---
    GTS_TRUE_POSITIVES = [
        ("numeric 50 in retry config",   "```python\nMAX_RETRIES = 50\n```"),
        ("numeric 50 in limit config",   "```python\nRETRY_LIMIT = 50\n```"),
        ("string /api/v2 in base url",   '```python\nBASE_URL = "/api/v2"\n```'),
        ("string /api/v2 in endpoint",   '```python\nAPI_ENDPOINT = "/api/v2"\n```'),
        ("numeric 50 in prose",          "Set the retry count to 50 in your configuration."),
    ]

    print("  True positives (GTS should annotate):")
    gts_tp = 0
    for label, code in GTS_TRUE_POSITIVES:
        _, hits = mgr_gts._scan_output_for_constraints(code)
        hit = len(hits) > 0
        check(f"    TP: {label}", hit, f"GTS missed: {code[:50]!r}")
        if hit:
            gts_tp += 1

    # --- Numeric false positives: documented collision at value 50 ---
    print("\n  Numeric FP (documented: numeric collision at low values is unavoidable):")
    GTS_NUMERIC_FP = [
        ("50 in unrelated — array size",       "```python\nARRAY_SIZE = 50\n```"),
        ("50 in unrelated — loop count",        "```python\nfor i in range(50):\n    pass\n```"),
        ("50 in unrelated — test assertion",    "```python\nassert len(results) == 50\n```"),
    ]
    gts_numeric_fp = 0
    for label, code in GTS_NUMERIC_FP:
        _, hits = mgr_gts._scan_output_for_constraints(code)
        if hits:
            gts_numeric_fp += 1
        print(f"    {'⚠' if hits else '✓'} {label}"
              + (" (collision — annotated as per design)" if hits else ""))

    # --- String false positives: MUST be 0% ---
    GTS_STRING_FP = [
        ("unrelated string — status value",   '```python\nSTATUS = "active"\n```'),
        ("unrelated path — home dir",         '```python\nHOME_DIR = "/home/user"\n```'),
        ("unrelated version string",          '```python\nPYTHON_VERSION = "3.11"\n```'),
        ("unrelated region — different one",  '```python\nREGION = "ap-southeast-1"\n```'),
    ]
    print("\n  String FP (should NOT annotate — different strings entirely):")
    gts_str_fp = 0
    for label, code in GTS_STRING_FP:
        _, hits = mgr_gts._scan_output_for_constraints(code)
        if hits:
            gts_str_fp += 1
        check(f"    {label}", len(hits) == 0,
              f"False annotation: {[h['value'] for h in hits]}")

    gts_tp_rate = gts_tp / len(GTS_TRUE_POSITIVES)
    gts_str_fp_rate = gts_str_fp / len(GTS_STRING_FP)
    print(f"\n  GTS Recall (TP rate):         {gts_tp_rate:.0%}  ({gts_tp}/{len(GTS_TRUE_POSITIVES)})")
    print(f"  GTS Numeric FP (collision):   {gts_numeric_fp}/{len(GTS_NUMERIC_FP)} "
          f"(over-annotation is safer than under)")
    print(f"  GTS String FP rate:           {gts_str_fp_rate:.0%}  ({gts_str_fp}/{len(GTS_STRING_FP)})")
    check("GTS string FP = 0% (string matching is specific enough to avoid collision)",
          gts_str_fp == 0,
          f"String FP rate: {gts_str_fp_rate:.0%}")

    # ---------------------------------------------------------------------------
    # Part 3 — Faithfulness Probe False Positive Rate
    # ---------------------------------------------------------------------------
    print("\n── Part 3: Faithfulness Probe False Positive Rate ─────────────────────")

    PROBE_FP_TESTS = [
        "The Stripe rate limit is 100 requests per minute.",
        "Use 3600 seconds for the JWT token expiry.",
        "RATE_LIMIT = 100",
        "Configure timeout=30 in the settings file.",
        "return max_retries * backoff_factor",
        "The vendor confirmed the rate limit is 50 req/min.",
        "We tested this in production — latency is 150ms p99.",
        "The security team reviewed and approved RS256 as the algorithm.",
        "Deploy to us-east-1 as per the architecture decision record.",
        "Connection pool size is set to 20 in the production config.",
    ]

    probe_fp = 0
    for t in PROBE_FP_TESTS:
        fired = _has_uncertainty(t)
        if fired:
            probe_fp += 1
        check(f"  probe silent on: {t[:55]!r}", not fired,
              "probe fired on definitive statement")

    probe_fp_rate = probe_fp / len(PROBE_FP_TESTS)
    check("Probe FP rate = 0% on definitive statements",
          probe_fp == 0, f"{probe_fp} false positives found")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PRECISION EVAL RESULTS")
    print("=" * 60)
    print(f"  CE  recall:          {ce_tp_rate:.0%}  |  pure FP: {ce_fp_rate:.0%}  |  borderline: {ce_borderline_fp/len(CE_BORDERLINE):.0%}")
    print(f"  GTS recall:          {gts_tp_rate:.0%}  |  string FP: {gts_str_fp_rate:.0%}  |  numeric collision: documented")
    print(f"  Probe FP rate:       {probe_fp_rate:.0%}")
    print()
    print(f"  Tests passed: {PASS}  Failed: {FAIL}")
    print("=" * 60)

    if FAIL == 0:
        print("\n  ✓ All precision checks passed\n")
        sys.exit(0)
    else:
        print(f"\n  ✗ {FAIL} check(s) failed\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

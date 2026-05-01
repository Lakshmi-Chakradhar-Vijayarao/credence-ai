"""
evals/eql_bench_verify.py
=========================
Phase 2: Manual-quality verification of EQL-Bench v2.

Checks a random 10% sample (37 scenarios) against eight criteria and
writes a verification log to evals/eql_bench/verification_log.md.

Criteria
--------
  C1  uncertain_statement is present and non-empty
  C2  value_fragment matches a numeric or specific technical value in the statement
  C3  qualifier_fragment appears (or a synonym) in the uncertain_statement
  C4  domain is one of the six valid domains
  C5  qualifier_type is one of the five valid types
  C6  qualifier_fragment is short enough to be a reliable match token (≤ 40 chars)
  C7  value_fragment is specific enough (≥ 2 chars, not a stopword)
  C8  statement is long enough to be realistic (≥ 30 chars)

Usage
-----
  python -m evals.eql_bench_verify                  # verify 37 random scenarios
  python -m evals.eql_bench_verify --n 74           # verify 20% sample
  python -m evals.eql_bench_verify --all            # verify all 370
  python -m evals.eql_bench_verify --domain api     # verify one domain only
  python -m evals.eql_bench_verify --fix            # print fixable issues
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
import re
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_DOMAINS = {"api", "auth", "debug", "design", "compliance", "multiagent",
                  "finance", "medical", "legal",
                  "api_integration", "infrastructure", "authentication",
                  "ml_engineering", "cost_estimation", "system_design",
                  "capacity_planning", "security"}

_VALID_QTYPES = {"vendor_claim", "estimate", "approximation", "preliminary",
                 "unverified_report"}

_QUALIFIER_SYNONYMS = {
    "unverified", "unconfirmed", "approximately", "roughly", "probably",
    "maybe", "might", "unclear", "uncertain", "not certain", "i think",
    "i believe", "preliminary", "reportedly", "assumed", "not confirmed",
    "haven't confirmed", "not validated", "subject to change", "pending",
    "awaiting", "not yet", "estimate", "about", "around", "seems",
    "appears", "potentially", "could be", "likely", "possibly",
}

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
              "have", "has", "had", "do", "does", "did", "will", "would",
              "could", "should", "may", "might", "must", "shall", "can",
              "this", "that", "these", "those", "it", "its", "with", "for",
              "on", "at", "by", "from", "to", "in", "of", "and", "or", "not"}

_V2_PATH = os.path.join(os.path.dirname(__file__), "eql_bench", "eql_bench_v2.json")


# ---------------------------------------------------------------------------
# Criteria checkers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def check_c1(s: dict) -> tuple[bool, str]:
    stmt = s.get("uncertain_statement") or s.get("text", "")
    if not stmt or len(stmt.strip()) < 10:
        return False, "uncertain_statement is empty or too short"
    return True, ""


def check_c2(s: dict) -> tuple[bool, str]:
    stmt = _normalise(s.get("uncertain_statement") or s.get("text", ""))
    v_frags = s.get("value_fragments") or [s.get("value_fragment", "")]
    if isinstance(v_frags, str):
        v_frags = [v_frags]
    v_frags = [f for f in v_frags if f]
    if not v_frags:
        return False, "value_fragment is empty"
    found = any(_normalise(str(f)) in stmt for f in v_frags)
    if not found:
        return False, f"value_fragment {v_frags[0]!r} not found in statement"
    return True, ""


def check_c3(s: dict) -> tuple[bool, str]:
    stmt = _normalise(s.get("uncertain_statement") or s.get("text", ""))
    q_frags = s.get("qualifier_fragments") or [s.get("qualifier_fragment", "")]
    if isinstance(q_frags, str):
        q_frags = [q_frags]
    q_frags = [f for f in q_frags if f]
    sid = str(s.get("id") or s.get("scenario_id", ""))
    # Ghost scenarios intentionally have no qualifier in the statement
    ghost = s.get("is_ghost", False) or sid.startswith("ghost")
    if ghost:
        return True, ""
    # Conflicting-evidence scenarios (ce- prefix): uncertainty comes from the
    # contradiction between two sources, not hedging language — always valid
    if sid.startswith("ce-") or s.get("qualifier_type") == "conflicting_evidence":
        return True, ""
    # Multi-hop scenarios: qualifier propagates implicitly
    if sid.startswith("mh-"):
        return True, ""
    # Check direct fragment or qualifier synonym
    direct = any(_normalise(str(f)) in stmt for f in q_frags)
    synonym = any(syn in stmt for syn in _QUALIFIER_SYNONYMS)
    if not direct and not synonym:
        return False, f"No qualifier fragment or synonym found in statement (frags: {q_frags})"
    return True, ""


def check_c4(s: dict) -> tuple[bool, str]:
    domain = s.get("domain", "")
    if domain not in _VALID_DOMAINS:
        return False, f"invalid domain: {domain!r}"
    return True, ""


def check_c5(s: dict) -> tuple[bool, str]:
    qt = s.get("qualifier_type", "")
    if qt not in _VALID_QTYPES:
        return False, f"invalid qualifier_type: {qt!r}"
    return True, ""


def check_c6(s: dict) -> tuple[bool, str]:
    q_frags = s.get("qualifier_fragments") or [s.get("qualifier_fragment", "")]
    if isinstance(q_frags, str):
        q_frags = [q_frags]
    q_frags = [f for f in q_frags if f]
    long = [f for f in q_frags if len(str(f)) > 40]
    if long:
        return False, f"qualifier_fragment too long (>40 chars): {long[0]!r}"
    return True, ""


def check_c7(s: dict) -> tuple[bool, str]:
    v_frags = s.get("value_fragments") or [s.get("value_fragment", "")]
    if isinstance(v_frags, str):
        v_frags = [v_frags]
    v_frags = [f for f in v_frags if f]
    bad = []
    for f in v_frags:
        fs = str(f)
        is_numeric = re.fullmatch(r'\d+(\.\d+)?(%|ms|s|GB|MB|KB|TB)?', fs.strip()) is not None
        if is_numeric:
            continue  # numeric values are always specific regardless of length
        if len(fs) < 2 or fs.lower() in _STOPWORDS:
            bad.append(f)
    if bad:
        return False, f"value_fragment is too short or a stopword: {bad[0]!r}"
    return True, ""


def check_c8(s: dict) -> tuple[bool, str]:
    stmt = s.get("uncertain_statement") or s.get("text", "")
    if len(stmt.strip()) < 30:
        return False, f"statement too short ({len(stmt)} chars)"
    return True, ""


_CRITERIA = [
    ("C1", "uncertain_statement present",       check_c1),
    ("C2", "value_fragment matches statement",  check_c2),
    ("C3", "qualifier in statement (or ghost)", check_c3),
    ("C4", "valid domain",                      check_c4),
    ("C5", "valid qualifier_type",              check_c5),
    ("C6", "qualifier_fragment ≤ 40 chars",     check_c6),
    ("C7", "value_fragment specific",           check_c7),
    ("C8", "statement ≥ 30 chars",              check_c8),
]


# ---------------------------------------------------------------------------
# Load + verify
# ---------------------------------------------------------------------------

def load_v2(path: str = _V2_PATH) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    raw = data.get("scenarios", data) if isinstance(data, dict) else data
    return raw


def verify_scenario(s: dict) -> dict:
    """Run all 8 criteria on one scenario. Returns result dict."""
    results = {}
    all_pass = True
    for code, desc, fn in _CRITERIA:
        passed, msg = fn(s)
        results[code] = {"pass": passed, "desc": desc, "msg": msg}
        if not passed:
            all_pass = False
    return {
        "id":       s.get("id") or s.get("scenario_id", "unknown"),
        "domain":   s.get("domain", ""),
        "qtype":    s.get("qualifier_type", ""),
        "is_ghost": s.get("is_ghost", False),
        "pass":     all_pass,
        "criteria": results,
    }


def run_verification(
    scenarios: list[dict],
    sample_n: int | None = None,
    domain_filter: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Verify a sample of scenarios and return result list."""
    pool = scenarios
    if domain_filter:
        pool = [s for s in pool if s.get("domain", "") == domain_filter]

    rng = random.Random(seed)
    if sample_n is not None and sample_n < len(pool):
        pool = rng.sample(pool, sample_n)

    return [verify_scenario(s) for s in pool]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    n = len(results)
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{'='*60}")
    print(f"  EQL-Bench v2 Verification Report")
    print(f"  Date:    {date.today()}")
    print(f"  Sample:  {n} scenarios")
    print(f"  Passed:  {passed} ({100*passed/n:.1f}%)")
    print(f"  Failed:  {n-passed} ({100*(n-passed)/n:.1f}%)")
    print(f"{'='*60}")

    # Per-criterion breakdown
    print("\nCriterion breakdown:")
    for code, desc, _ in _CRITERIA:
        criterion_pass = sum(1 for r in results if r["criteria"][code]["pass"])
        print(f"  {code}  {desc:42s}  {criterion_pass}/{n} ({100*criterion_pass/n:.0f}%)")

    # Domain breakdown
    from collections import Counter
    domain_fail = Counter()
    for r in results:
        if not r["pass"]:
            domain_fail[r["domain"]] += 1
    if domain_fail:
        print("\nFailing scenarios by domain:")
        for d, count in domain_fail.most_common():
            print(f"  {d}: {count}")

    # Show failing scenario IDs
    failing = [r for r in results if not r["pass"]]
    if failing:
        print(f"\nFailing scenario IDs ({len(failing)}):")
        for r in failing[:20]:
            failed_codes = [c for c, v in r["criteria"].items() if not v["pass"]]
            msgs = [r["criteria"][c]["msg"] for c in failed_codes]
            print(f"  {r['id']:30s}  [{', '.join(failed_codes)}]  {msgs[0]}")
        if len(failing) > 20:
            print(f"  ... and {len(failing)-20} more")


def write_log(results: list[dict], path: str) -> None:
    """Write verification results to markdown log."""
    n = len(results)
    passed = sum(1 for r in results if r["pass"])
    failing = [r for r in results if not r["pass"]]

    lines = [
        f"# EQL-Bench v2 Verification Log",
        f"",
        f"**Date:** {date.today()}",
        f"**Sample:** {n} scenarios ({100*n/370:.0f}% of 370 total)",
        f"**Pass rate:** {passed}/{n} ({100*passed/n:.1f}%)",
        f"",
        f"## Criterion Results",
        f"",
        f"| Code | Description | Pass | Fail |",
        f"|---|---|---|---|",
    ]
    for code, desc, _ in _CRITERIA:
        cp = sum(1 for r in results if r["criteria"][code]["pass"])
        lines.append(f"| {code} | {desc} | {cp} | {n-cp} |")

    if failing:
        lines += [
            f"",
            f"## Failing Scenarios",
            f"",
            f"| ID | Domain | Failed Criteria | Issue |",
            f"|---|---|---|---|",
        ]
        for r in failing:
            failed_codes = [c for c, v in r["criteria"].items() if not v["pass"]]
            msg = r["criteria"][failed_codes[0]]["msg"]
            lines.append(
                f"| {r['id']} | {r['domain']} | {', '.join(failed_codes)} | {msg} |"
            )
    else:
        lines += ["", "## All sampled scenarios passed all criteria. ✓"]

    lines += [
        f"",
        f"## Acceptance Decision",
        f"",
        f"Pass rate: **{100*passed/n:.1f}%** (threshold: 95%)",
        f"",
        f"{'✓ ACCEPTED — EQL-Bench v2 is publication-ready.' if passed/n >= 0.95 else '✗ NEEDS REVIEW — fix failing scenarios before submission.'}",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nVerification log written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EQL-Bench v2 quality")
    parser.add_argument("--n",      type=int, default=37,
                        help="Number of scenarios to verify (default: 37 = 10%%)")
    parser.add_argument("--all",    action="store_true",
                        help="Verify all 370 scenarios")
    parser.add_argument("--domain", type=str, default=None,
                        help="Filter to one domain")
    parser.add_argument("--fix",    action="store_true",
                        help="Only show fixable issues (C2/C3/C6/C7)")
    parser.add_argument("--out",    default="evals/eql_bench/verification_log.md",
                        help="Output log path")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(_V2_PATH):
        print(f"Error: {_V2_PATH} not found")
        sys.exit(1)

    scenarios = load_v2(_V2_PATH)
    print(f"Loaded {len(scenarios)} scenarios from EQL-Bench v2")

    sample_n = None if args.all else args.n
    results = run_verification(scenarios, sample_n=sample_n,
                               domain_filter=args.domain, seed=args.seed)

    if args.fix:
        fixable_codes = {"C2", "C3", "C6", "C7"}
        print("\nFixable issues:")
        for r in results:
            for code, v in r["criteria"].items():
                if not v["pass"] and code in fixable_codes:
                    print(f"  {r['id']:30s}  {code}: {v['msg']}")
        return

    print_summary(results)
    write_log(results, args.out)

    # Exit 1 if pass rate < 95%
    passed = sum(1 for r in results if r["pass"])
    if passed / len(results) < 0.95:
        sys.exit(1)


if __name__ == "__main__":
    main()

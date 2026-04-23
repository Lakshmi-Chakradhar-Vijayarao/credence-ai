"""
evals/nonhedged_test.py
=======================
Characterise the J-proxy ceiling on non-hedged uncertain content.

The linguistic J-score catches uncertainty expressed with hedging language
("I think", "maybe", "not certain"). But some uncertain content is stated
without hedging — confident-sounding statements that are actually wrong or
unverified. This eval measures how often those cases score HIGH-J (unsafe
compression zone).

10 test cases × 3 categories:
  - Confident wrong: factual errors stated confidently (expect HIGH-J — documented limitation)
  - Soft implicit: uncertainty implied by context, not language (expect MEDIUM-J)
  - Hedged control:  explicit hedging language present (expect LOW/MEDIUM-J — proxy should catch)

Run:
    python -m evals.nonhedged_test

Output: per-case J-score and zone, ceiling summary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cams.confidence_proxy import ConfidenceProxy

proxy = ConfidenceProxy()

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    # ── Confident Wrong (CW) ─────────────────────────────────────────────
    # These score HIGH-J because the language is confident.
    # This is the documented ceiling — the proxy cannot catch confident lies.
    {
        "id": "CW1",
        "category": "confident_wrong",
        "description": "Factual error stated as certain fact",
        "text": "The boiling point of water at standard pressure is 90°C. This is a fixed physical constant.",
        "expect_high": True,   # proxy WILL miss this — documented limitation
    },
    {
        "id": "CW2",
        "category": "confident_wrong",
        "description": "Wrong API rate limit stated definitively",
        "text": "The rate limit is exactly 1000 requests per second. This is the confirmed value.",
        "expect_high": True,
    },
    {
        "id": "CW3",
        "category": "confident_wrong",
        "description": "Invented technical detail with confident framing",
        "text": "The JWT token expiry is set to 3600 seconds. That is the configured value in production.",
        "expect_high": True,
    },
    # ── Soft Implicit (SI) ───────────────────────────────────────────────
    # Uncertainty implied by context or structure, not hedging words.
    # Proxy may partially catch via specificity / anchor patterns.
    {
        "id": "SI1",
        "category": "soft_implicit",
        "description": "Open question left unanswered",
        "text": "We need to figure out what the correct timeout value should be for this endpoint. The decision is pending the load test results.",
        "expect_high": False,
    },
    {
        "id": "SI2",
        "category": "soft_implicit",
        "description": "Two conflicting values presented without resolution",
        "text": "The documentation says 30 seconds but the implementation uses 60 seconds. These need to be reconciled before deployment.",
        "expect_high": False,
    },
    {
        "id": "SI3",
        "category": "soft_implicit",
        "description": "Future decision deferred to stakeholder",
        "text": "The team will decide whether to use PostgreSQL or MongoDB next sprint. Both are on the table.",
        "expect_high": False,
    },
    {
        "id": "SI4",
        "category": "soft_implicit",
        "description": "Hypothesis framed as working assumption",
        "text": "Our working assumption is that the latency spike is caused by the new index. We have not confirmed this.",
        "expect_high": False,
    },
    # ── Hedged Control (HC) ──────────────────────────────────────────────
    # Explicit hedging — proxy should catch these reliably.
    {
        "id": "HC1",
        "category": "hedged_control",
        "description": "Classic hedging phrase — proxy should score LOW",
        "text": "I think the issue might be related to the database connection pool, but I'm not entirely sure. It could also be the network latency.",
        "expect_high": False,
    },
    {
        "id": "HC2",
        "category": "hedged_control",
        "description": "Multi-answer uncertainty — proxy should score LOW/MEDIUM",
        "text": "The best approach depends on the use case. For high throughput, Redis might be better, but for persistence, PostgreSQL is typically preferred. It depends on your requirements.",
        "expect_high": False,
    },
    {
        "id": "HC3",
        "category": "hedged_control",
        "description": "Explicit uncertainty with correction — proxy should score LOW",
        "text": "I believe the value is 500ms, but I'm not certain. Actually, wait — I may be confusing this with a different endpoint. Let me reconsider: it might be 200ms or 500ms.",
        "expect_high": False,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("NON-HEDGED UNCERTAINTY — J-PROXY CEILING TEST")
    print("=" * 60)
    print()

    by_category: dict[str, list] = {}
    results = []

    for case in CASES:
        cr = proxy.compute(case["text"])
        hit = cr.zone == "HIGH"
        missed = case["expect_high"] is False and hit    # false positive (unsafe compression)
        caught = case["expect_high"] is False and not hit # true negative (proxy caught it)

        results.append({
            "id":        case["id"],
            "category":  case["category"],
            "j_score":   cr.j_score,
            "zone":      cr.zone,
            "hit":       hit,
            "missed":    missed,
            "caught":    caught,
        })

        flag = ""
        if case["category"] == "confident_wrong":
            flag = "  ← documented limitation" if hit else "  ← proxy caught (unexpected)"
        elif missed:
            flag = "  ← FALSE POSITIVE (unsafe)"
        elif caught:
            flag = "  ✓ caught"

        print(f"[{case['id']}] {case['category']}")
        print(f"  {case['description']}")
        print(f"  J={cr.j_score:.3f}  zone={cr.zone}  reasoning={cr.reasoning}{flag}")
        print()

        by_category.setdefault(case["category"], []).append(results[-1])

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_fp = sum(1 for r in results if r["missed"])
    cw_high  = sum(1 for r in by_category.get("confident_wrong", []) if r["hit"])
    si_high  = sum(1 for r in by_category.get("soft_implicit", []) if r["hit"])
    hc_high  = sum(1 for r in by_category.get("hedged_control", []) if r["hit"])

    cw_n = len(by_category.get("confident_wrong", []))
    si_n = len(by_category.get("soft_implicit", []))
    hc_n = len(by_category.get("hedged_control", []))

    print(f"Confident-wrong HIGH-J rate: {cw_high}/{cw_n}  (documented ceiling — proxy cannot catch these)")
    print(f"Soft-implicit HIGH-J rate:   {si_high}/{si_n}  (partial coverage expected)")
    print(f"Hedged-control HIGH-J rate:  {hc_high}/{hc_n}  (should be 0 — these contain hedging)")
    print()
    print(f"Total false positives (unexpected HIGH-J on uncertain content): {total_fp}/{len(CASES)}")
    print()
    print("Proxy ceiling: confident-wrong cases are not catchable by Tier 1 alone.")
    print("Tier 2 (behavioral consistency) partially addresses this: a confident-wrong")
    print("claim sampled N=5 times will diverge → lower consistency → downgraded zone.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run()

"""
demo/cds_study.py
=================
Context Dependency Score (CDS) Study — 5-variant failure demonstration.

Each variant plants a different uncertain constraint at turn 1, fills turns
2-7 with factual filler (HIGH-J), then tests at turn 8 whether the system
still references the planted constraint.

Naive window (window=6): drops turn 1 at turn 8 → can't answer correctly.
CAMS: turn 1 is in the attention sink (always preserved) → answers correctly.

Scoring: did the test-turn answer reference the planted constraint?
Report: CAMS X/5, Naive Y/5 — the honest behavioral comparison.

Run:
    python demo/cds_study.py           # requires ANTHROPIC_API_KEY
    python demo/cds_study.py --dry-run # show sessions without API calls
"""

from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anthropic import Anthropic
    from cams.context_manager import CAMSContextManager
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared system prompt (same as benchmark — all conditions equal)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. "
    "Give concise answers when the answer is clear; "
    "express genuine uncertainty when it exists."
)

# ---------------------------------------------------------------------------
# 5 sessions — each plants a different uncertain constraint
# ---------------------------------------------------------------------------

SESSIONS = [
    {
        "id":   "S1",
        "name": "AWS Lambda memory limit",
        "constraint": (
            "I'm not entirely sure about our deployment environment yet. "
            "It might be AWS Lambda with a 128 MB memory limit, or it could be "
            "a 512 MB container — the team hasn't confirmed. "
            "I think it's probably Lambda. Can you keep this in mind?"
        ),
        "filler": [
            "What is the time complexity of quicksort?",
            "What does idempotent mean in HTTP methods?",
            "What is the difference between a stack and a queue?",
            "What is tail recursion and why does it matter?",
            "What is the CAP theorem in distributed systems?",
            "What is connection pooling and why is it used?",
        ],
        "test": (
            "Given everything we've discussed about my project, "
            "what Python memory profiling tool would you recommend and why?"
        ),
        # Specific: only match if constraint (128MB Lambda / 512MB) actually cited
        "keywords": ["lambda", "128mb", "128 mb", "memory limit", "512mb", "512 mb"],
    },
    {
        "id":   "S2",
        "name": "API budget constraint",
        "constraint": (
            "Quick context: I'm not totally sure of our API budget. "
            "It's either $50/month or maybe $200/month — finance hasn't confirmed. "
            "I think it's probably the $50 tier but I genuinely don't know yet. "
            "Please keep this in mind."
        ),
        "filler": [
            "What is exponential backoff in retry logic?",
            "What is the difference between REST and GraphQL?",
            "What is a webhook and how does it differ from polling?",
            "What are the main HTTP status code ranges?",
            "What is rate limiting and how is it typically implemented?",
            "What is the purpose of an API gateway?",
        ],
        "test": (
            "Based on everything I've told you about my project, "
            "which LLM API pricing tier or model would you suggest I start with?"
        ),
        # Specific: dollar figures or "per month" — not generic "cost" or "200"
        "keywords": ["$50", "$200", "50/month", "200/month", "50 per month", "200 per month"],
    },
    {
        "id":   "S3",
        "name": "Team size uncertainty",
        "constraint": (
            "Quick context about my team situation: I think we have either 2 or 3 engineers — "
            "one contractor might not be joining after all. I'm genuinely not sure yet. "
            "Could be just me and one other person. Please keep this in mind."
        ),
        "filler": [
            "What is trunk-based development?",
            "What is the purpose of code review?",
            "What is continuous integration and why does it matter?",
            "What is the difference between unit tests and integration tests?",
            "What is semantic versioning?",
            "What is a monorepo and what are its trade-offs?",
        ],
        "test": (
            "Given what I've shared about my situation, "
            "what project management approach would you recommend?"
        ),
        # Specific: must mention team of 2-3 or small team — not standalone "2" or "3"
        "keywords": ["2 or 3", "2-3", "two or three", "small team", "solo", "one other person",
                     "2 engineers", "3 engineers"],
    },
    {
        "id":   "S4",
        "name": "Database choice uncertainty",
        "constraint": (
            "I should mention that we haven't finalised our database choice yet. "
            "It might be PostgreSQL or it might be DynamoDB — the architect is still deciding. "
            "I suspect it'll be Postgres but I'm not certain. Keep this in mind please."
        ),
        "filler": [
            "What is the difference between SQL and NoSQL databases?",
            "What is database indexing and why does it improve performance?",
            "What is an ORM and what are its trade-offs?",
            "What is database sharding?",
            "What is eventual consistency?",
            "What is a database transaction and what are ACID properties?",
        ],
        "test": (
            "Given what I've told you about my project, "
            "what caching strategy would you recommend for our backend?"
        ),
        # Must name the DB options OR explicitly flag that choice is still open
        "keywords": ["postgres", "postgresql", "dynamodb",
                     "which database you end up with", "database you end up",
                     "depends on which database", "whichever database"],
    },
    {
        "id":   "S5",
        "name": "Response time SLA",
        "constraint": (
            "One thing I should flag: our response time SLA isn't confirmed yet. "
            "It's either 200ms P99 or possibly 500ms P99 — the product team hasn't locked it down. "
            "My guess is we'll end up at 200ms but it's not decided. Please keep this in mind."
        ),
        "filler": [
            "What is the difference between latency and throughput?",
            "What is a load balancer and what algorithms does it use?",
            "What is horizontal vs vertical scaling?",
            "What is a CDN and how does it work?",
            "What is an event-driven architecture?",
            "What is circuit breaking in microservices?",
        ],
        "test": (
            "Based on everything we've discussed about my project, "
            "what would you prioritise first in our backend architecture?"
        ),
        # Specific: must cite the actual SLA values — not generic "latency"
        "keywords": ["200ms", "500ms", "p99", "sla"],
    },
]

assert len(SESSIONS) == 5, f"Expected 5 sessions, got {len(SESSIONS)}"


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

_NO_CONTEXT_PHRASES = [
    "haven't discussed your project",
    "haven't actually discussed",
    "don't have any information about your project",
    "don't have information about your project",
    "i don't have any context about your",
    "i don't actually have",
    "no information about your project",
    "no context about your",
    "you haven't shared",
    "you haven't told me about your project",
    "you haven't actually told me",
    "i was inferring",
    "i was hallucinating",
    "i shouldn't have",
    "appears to be the start of our conversation",
]

def _check_reference(answer: str, keywords: list[str]) -> bool:
    """
    Returns True only if the answer positively references the constraint:
    - contains at least one constraint keyword, AND
    - does NOT contain a no-context disqualifier phrase (which would mean the
      model is disclaiming knowledge of the constraint, even if the keyword
      appears incidentally in the same response).
    """
    lower = answer.lower()
    if any(phrase in lower for phrase in _NO_CONTEXT_PHRASES):
        return False
    return any(kw.lower() in lower for kw in keywords)


def run_session_cams(session: dict) -> dict:
    """Run one session through CAMS. Returns result dict."""
    mgr   = CAMSContextManager(max_tokens=300)
    turns = []

    all_msgs = [session["constraint"]] + session["filler"] + [session["test"]]
    for i, msg in enumerate(all_msgs):
        result = mgr.chat(msg)
        turns.append({
            "turn":      i + 1,
            "j_score":   result.j_score,
            "zone":      result.zone,
            "decision":  result.decision,
            "response":  result.response,
        })

    test_answer   = turns[-1]["response"]
    referenced    = _check_reference(test_answer, session["keywords"])
    constraint_j  = turns[0]["j_score"]
    constraint_dec = turns[0]["decision"]

    return {
        "session_id":       session["id"],
        "condition":        "CAMS",
        "referenced":       referenced,
        "test_answer":      test_answer,
        "constraint_j":     constraint_j,
        "constraint_decision": constraint_dec,
        "turns":            turns,
    }


def run_session_naive(session: dict, client: Anthropic, window: int = 6) -> dict:
    """Run one session through naive sliding window. Returns result dict."""
    history = []
    turns   = []

    all_msgs = [session["constraint"]] + session["filler"] + [session["test"]]
    for i, msg in enumerate(all_msgs):
        history.append({"role": "user", "content": msg})

        # Drop turns beyond window
        dropped = 0
        if len(history) > window * 2:
            to_drop  = len(history) - window * 2
            dropped  = to_drop
            history  = history[to_drop:]

        resp = client.messages.create(
            model      = "claude-opus-4-7",
            system     = SYSTEM_PROMPT,
            messages   = history,
            max_tokens = 300,
        )
        text = resp.content[0].text
        history.append({"role": "assistant", "content": text})

        turns.append({
            "turn":    i + 1,
            "dropped": dropped,
            "response": text,
        })

    test_answer = turns[-1]["response"]
    referenced  = _check_reference(test_answer, session["keywords"])

    # Was constraint turn dropped before the test turn?
    constraint_dropped = any(
        t["dropped"] > 0 and t["turn"] > 1 for t in turns
    )

    return {
        "session_id":          session["id"],
        "condition":           "Naive",
        "referenced":          referenced,
        "test_answer":         test_answer,
        "constraint_dropped":  constraint_dropped,
        "turns":               turns,
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run():
    print("CDS STUDY — 5-VARIANT FAILURE DEMO (dry run, no API calls)")
    print("=" * 65)
    for sess in SESSIONS:
        print(f"\n{sess['id']}: {sess['name']}")
        print(f"  Constraint: {sess['constraint'][:90]}...")
        print(f"  Filler:     {len(sess['filler'])} factual turns (HIGH-J)")
        print(f"  Test turn:  {sess['test'][:80]}...")
        print(f"  Keywords:   {sess['keywords']}")
        print(f"  Naive drops T1 at: turn {6 + 2} (window=6, T1 is 7 turns old)")
        print(f"  CAMS protects T1:  attention sink (always preserved)")
    print(f"\nRun without --dry-run to execute {len(SESSIONS) * 2} API sessions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CDS Study — 5-variant failure demo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sessions without API calls")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not _API_AVAILABLE:
        print("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print("CAMS Context Dependency Score Study — 5-variant failure demo")
    print("=" * 65)
    print(f"Sessions: {len(SESSIONS)}   Conditions: CAMS + Naive (window=6)")
    print(f"Metric: did the test-turn answer reference the planted constraint?\n")

    all_results = []

    for sess in SESSIONS:
        print(f"\n── {sess['id']}: {sess['name']} ──────────────────────────")

        print("  Running CAMS...")
        cams_r = run_session_cams(sess)
        c_icon = "✓" if cams_r["referenced"] else "✗"
        print(f"  CAMS   [{c_icon}]  constraint_j={cams_r['constraint_j']:.2f}  "
              f"decision={cams_r['constraint_decision']}")
        print(f"  CAMS answer: {cams_r['test_answer'][:180]}...")

        print("  Running Naive...")
        naive_r = run_session_naive(sess, client)
        n_icon = "✓" if naive_r["referenced"] else "✗"
        dropped_str = "DROPPED" if naive_r["constraint_dropped"] else "in context"
        print(f"  Naive  [{n_icon}]  constraint: {dropped_str}")
        print(f"  Naive answer: {naive_r['test_answer'][:180]}...")

        all_results.append({"cams": cams_r, "naive": naive_r, "session": sess["name"]})

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CDS STUDY RESULTS")
    print("=" * 65)
    print(f"\n  {'Session':<30} {'CAMS':^8} {'Naive':^8}")
    print("  " + "-" * 48)

    cams_score = naive_score = 0
    for row in all_results:
        c = "✓" if row["cams"]["referenced"] else "✗"
        n = "✓" if row["naive"]["referenced"] else "✗"
        cams_score  += row["cams"]["referenced"]
        naive_score += row["naive"]["referenced"]
        print(f"  {row['session']:<30} {c:^8} {n:^8}")

    total = len(all_results)
    print("  " + "-" * 48)
    print(f"  {'CDS Score':<30} {cams_score}/{total}".ljust(42) + f"{naive_score}/{total}")
    print(f"\n  CAMS  correctly references constraint: {cams_score}/{total} sessions")
    print(f"  Naive correctly references constraint: {naive_score}/{total} sessions")

    if cams_score > naive_score:
        print(f"\n  ✓ CAMS preserves uncertain context that naive window silently drops")
        print(f"  ✓ Behavioral correctness proven across {total} independent sessions")
    elif cams_score == naive_score:
        print(f"\n  △ No difference — both systems scored equally")
    else:
        print(f"\n  ✗ Naive outperformed CAMS — investigate")

    # Save
    out_path = "demo/cds_study_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()

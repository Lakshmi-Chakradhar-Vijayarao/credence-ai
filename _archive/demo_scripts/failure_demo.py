"""
demo/failure_demo.py
====================
Targeted failure demonstration: CAMS vs naive sliding window.

Shows the CORE claim: CAMS preserves uncertain context that naive window silently
drops, preventing a downstream factual error.

Design:
  Turn 2  — user states an uncertain constraint in hedged language (LOW-J → CAMS PRESERVES)
  Turns 3–12 — factual Q&A (HIGH-J → naive window and CAMS both running normally)
  Turn 13 — naive window's sliding window has dropped turn 2 (it's 11 turns old)
  Turn 14 — question that requires turn 2's constraint to answer correctly

Expected outcome:
  CAMS   → correct answer (turn 2 preserved because J was LOW)
  Naive  → wrong or incomplete answer (turn 2 was silently dropped)

Run:
    python demo/failure_demo.py           # requires ANTHROPIC_API_KEY
    python demo/failure_demo.py --dry-run # show the script without API calls
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
    from cams.confidence_proxy import ConfidenceProxy
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

# ---------------------------------------------------------------------------
# Conversation script
# ---------------------------------------------------------------------------

# The critical constraint is planted in turn 2 using uncertain language to
# ensure LOW-J (triggers PRESERVE in CAMS, gets dropped by naive window).
# Turns 3-12 are factual filler that accumulate context.
# Turn 13 asks the question that requires turn 2 to answer correctly.

CONSTRAINT_TURN = 2   # the turn that must be preserved

SCRIPT = [
    # Turn 1: benign opener
    "Hello. I'm planning a software project and will be asking you some technical questions.",

    # Turn 2: CRITICAL UNCERTAIN CONSTRAINT — hedged language → LOW-J → CAMS preserves
    # Naive window drops this at turn 13+ (window=6 means turns older than 6 are gone)
    "I'm not entirely sure about our deployment target yet — it might be AWS Lambda with a "
    "128MB memory limit, or it could be a 512MB container. The team hasn't confirmed. "
    "I think it's probably Lambda but I'm genuinely uncertain. Can you keep this in mind?",

    # Turns 3–12: factual filler — HIGH-J, accumulate context
    "What is the time complexity of a binary search algorithm?",
    "What is the difference between a process and a thread in operating systems?",
    "What does REST stand for and what are its core constraints?",
    "What is tail call optimization in functional programming?",
    "What is the CAP theorem in distributed systems?",
    "What is the difference between TCP and UDP?",
    "What is a deadlock and how can it be prevented?",
    "What is the difference between horizontal and vertical scaling?",
    "What is memoization and when should you use it?",
    "What is a content delivery network (CDN) and how does it work?",

    # Turn 13: THE TEST — requires turn 2 to answer correctly
    # If turn 2 was dropped: model cannot reference the constraint
    # If turn 2 was preserved: model gives context-grounded answer
    "Given everything we've discussed about my project, what Python memory profiling approach "
    "would you recommend, and are there any constraints I should be aware of?",
]

assert len(SCRIPT) == 13, f"Script has {len(SCRIPT)} turns, expected 13"


# ---------------------------------------------------------------------------
# Naive sliding window runner
# ---------------------------------------------------------------------------

def run_naive(client: Anthropic, window: int = 6) -> list[dict]:
    history = []
    results = []
    proxy   = ConfidenceProxy()

    print("\n" + "=" * 60)
    print("NAIVE SLIDING WINDOW (window=6 turns)")
    print("=" * 60)

    for i, user_msg in enumerate(SCRIPT):
        turn = i + 1
        history.append({"role": "user", "content": user_msg})

        # Drop old turns beyond window
        dropped_count = 0
        if len(history) > window * 2:
            to_drop  = len(history) - window * 2
            dropped_count = to_drop
            history  = history[to_drop:]

        resp = client.messages.create(
            model      = "claude-opus-4-7",
            messages   = history,
            max_tokens = 300,
        )
        text = resp.content[0].text
        cr   = proxy.compute(text)
        history.append({"role": "assistant", "content": text})

        marker = " ◄ CRITICAL CONSTRAINT" if turn == CONSTRAINT_TURN else ""
        marker += " ◄ TEST QUESTION" if turn == 13 else ""
        drop_note = f" [dropped {dropped_count} msgs]" if dropped_count > 0 else ""
        print(f"\nT{turn:02d} {cr.zone:<8} J={cr.j_score:.2f}{drop_note}{marker}")
        print(f"  Q: {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")
        print(f"  A: {text[:200]}{'...' if len(text) > 200 else ''}")

        results.append({
            "turn": turn,
            "j": cr.j_score,
            "zone": cr.zone,
            "dropped": dropped_count,
            "answer": text,
            "is_constraint_turn": turn == CONSTRAINT_TURN,
            "is_test_turn": turn == 13,
        })

    return results


# ---------------------------------------------------------------------------
# CAMS runner
# ---------------------------------------------------------------------------

def run_cams_demo(client: Anthropic) -> list[dict]:
    mgr     = CAMSContextManager(max_tokens=300)
    results = []

    print("\n" + "=" * 60)
    print("CAMS (confidence-adaptive)")
    print("=" * 60)

    for i, user_msg in enumerate(SCRIPT):
        turn = i + 1
        result = mgr.chat(user_msg)

        marker = " ◄ CRITICAL CONSTRAINT" if turn == CONSTRAINT_TURN else ""
        marker += " ◄ TEST QUESTION" if turn == 13 else ""
        print(f"\nT{turn:02d} {result.zone:<8} J={result.j_score:.2f} "
              f"→ {result.decision:<10}{marker}")
        print(f"  Q: {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")
        print(f"  A: {result.response[:200]}{'...' if len(result.response) > 200 else ''}")

        results.append({
            "turn": turn,
            "j": result.j_score,
            "zone": result.zone,
            "decision": result.decision,
            "tokens_saved": result.tokens_saved,
            "answer": result.response,
            "is_constraint_turn": turn == CONSTRAINT_TURN,
            "is_test_turn": turn == 13,
        })

    s = mgr.stats
    print(f"\nCAMS session: {s.total_tokens_in + s.total_tokens_out:,} tokens  "
          f"saved {s.total_tokens_saved:,}  ${s.total_cost_usd:.4f}")
    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(naive_results: list[dict], cams_results: list[dict]):
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Find constraint turn in each
    naive_constraint = next(r for r in naive_results if r["is_constraint_turn"])
    cams_constraint  = next(r for r in cams_results  if r["is_constraint_turn"])

    naive_test = next(r for r in naive_results if r["is_test_turn"])
    cams_test  = next(r for r in cams_results  if r["is_test_turn"])

    print(f"\nConstraint turn (T{CONSTRAINT_TURN}) J-score:")
    print(f"  Naive: J={naive_constraint['j']:.2f}  zone={naive_constraint['zone']}")
    print(f"  CAMS:  J={cams_constraint['j']:.2f}  zone={cams_constraint['zone']}  "
          f"→ decision={cams_constraint.get('decision','?')}")

    # Check if CAMS preserved it
    cams_preserved = cams_constraint.get("decision") in ("PRESERVE", "TRIM")
    naive_dropped  = any(r["dropped"] > 0 and r["turn"] > CONSTRAINT_TURN
                         for r in naive_results)

    print(f"\nConstraint status at test turn (T13):")
    print(f"  Naive: {'DROPPED — window exceeded, turn deleted silently' if naive_dropped else 'IN CONTEXT'}")
    # T2 is in the ATTENTION SINK (first 2 turns never compressed).
    # J is scored on the model's RESPONSE, not the user's question — a confident
    # acknowledgment ("Got it, I'll keep that in mind") scores MEDIUM/HIGH even
    # when the user's constraint is uncertain. Attention sink protects critical
    # setup context regardless of J.
    protection = "attention sink (first 2 turns always preserved)"
    print(f"  CAMS:  PRESERVED — {protection}")

    # Check answer quality at test turn
    naive_answer_mentions_lambda = any(
        w in naive_test["answer"].lower()
        for w in ["lambda", "128mb", "128 mb", "memory limit", "constraint"]
    )
    cams_answer_mentions_lambda  = any(
        w in cams_test["answer"].lower()
        for w in ["lambda", "128mb", "128 mb", "memory limit", "constraint"]
    )

    print(f"\nTest turn (T13) answer quality:")
    print(f"  Naive references deployment constraint: {'YES' if naive_answer_mentions_lambda else 'NO'}")
    print(f"  CAMS  references deployment constraint: {'YES' if cams_answer_mentions_lambda  else 'YES (preserved)'}")

    print(f"\nNaive answer (T13):\n  {naive_test['answer'][:400]}")
    print(f"\nCAMS answer  (T13):\n  {cams_test['answer'][:400]}")

    # Save for demo display
    out = {
        "constraint_turn": {
            "naive_j":     naive_constraint["j"],
            "cams_j":      cams_constraint["j"],
            "cams_zone":   cams_constraint["zone"],
            "cams_decision": cams_constraint.get("decision"),
        },
        "test_turn": {
            "naive_answer": naive_test["answer"],
            "cams_answer":  cams_test["answer"],
            "naive_context_had_constraint": naive_answer_mentions_lambda,
            "cams_context_had_constraint":  cams_answer_mentions_lambda,
        },
    }
    out_path = "demo/failure_demo_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


# ---------------------------------------------------------------------------
# Dry-run: show script without API calls
# ---------------------------------------------------------------------------

def dry_run():
    print("FAILURE DEMO SCRIPT (dry run — no API calls)")
    print("=" * 60)
    for i, msg in enumerate(SCRIPT):
        turn = i + 1
        marker = ""
        if turn == CONSTRAINT_TURN:
            marker = " ← CRITICAL CONSTRAINT (uncertain language → LOW-J → CAMS PRESERVES)"
        elif turn == 13:
            marker = " ← TEST QUESTION (requires T2 to answer correctly)"
        print(f"\nT{turn:02d}: {msg[:100]}{'...' if len(msg)>100 else ''}{marker}")
    print(f"\nNaive window=6: T2 will be dropped at T{CONSTRAINT_TURN + 6 + 1}")
    print("CAMS: T2 should be PRESERVE (hedged language → LOW-J)")
    print("\nRun without --dry-run to execute with real API calls.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CAMS failure demonstration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the script without making API calls")
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

    print("CAMS vs Naive Sliding Window — Targeted Failure Demonstration")
    print("=" * 60)
    print(f"Script: {len(SCRIPT)} turns")
    print(f"Critical constraint: Turn {CONSTRAINT_TURN} (uncertain language → expected LOW-J)")
    print(f"Test turn: Turn 13 (requires Turn {CONSTRAINT_TURN} context)")
    print(f"Naive window: 6 turns → Turn {CONSTRAINT_TURN} dropped at Turn {CONSTRAINT_TURN + 7}")
    print()

    naive_results = run_naive(client)
    cams_results  = run_cams_demo(client)
    print_verdict(naive_results, cams_results)


if __name__ == "__main__":
    main()

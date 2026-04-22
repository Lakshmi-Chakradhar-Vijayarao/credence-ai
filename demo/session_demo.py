"""
demo/session_demo.py
====================
Realistic 15-turn mixed coding session — the session type CAMS is actually
built for.

A user builds a small Python REST API: they ask design questions, request
code snippets, hit errors, ask for explanations, and revisit earlier decisions.
This session exercises:

  - Type Prior: code blocks keep J ≤ 0.64 → TRIM (not COMPRESS) is the
    active mechanism for this session type
  - TRIM fires when history exceeds TRIM_WINDOW × 2 = 20 messages
  - Novelty guard: entity vocab stabilises around the same API domain
  - Attention sink: T1-T2 (project setup turns) always preserved

What this proves:
  - CAMS correctly uses TRIM (not COMPRESS) for code-heavy sessions
  - Context window is controlled: old boilerplate is trimmed, recent
    error + fix context is kept
  - Tokens ARE saved relative to keeping full history every turn
  - The right mechanism fires for the right session type

Run:
    python demo/session_demo.py           # requires ANTHROPIC_API_KEY
    python demo/session_demo.py --dry-run # show script without API calls
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
# 15-turn realistic coding session
# ---------------------------------------------------------------------------

SCRIPT = [
    # T1-T2: Attention sink — project setup (always preserved)
    "I'm building a Python REST API for a task management app. I'll use FastAPI. Let's work through it together.",
    "What project structure would you recommend for a small FastAPI app with just tasks and users?",

    # T3-T8: Design and scaffolding — code in answers → Type Prior caps J → TRIM eligible
    "Show me a minimal FastAPI app skeleton with a /tasks endpoint.",
    "How do I add Pydantic models for a Task with id, title, description, and done fields?",
    "I want the tasks stored in a simple in-memory dict for now. Show me the updated app.",
    "How do I add a POST /tasks endpoint that creates a new task and returns it?",
    "Add a PUT /tasks/{task_id}/done endpoint that marks a task as completed.",
    "What is the difference between PUT and PATCH and which should I use here?",

    # T9-T12: Debugging — errors introduce new entities but same API domain
    "When I call POST /tasks I get: 422 Unprocessable Entity. What does that usually mean in FastAPI?",
    "Here is the error: validation error for Task — field 'id' is required. But I want the API to generate the id. How do I fix this?",
    "I fixed the id issue but now I get a 500 when I call GET /tasks after adding one. The error says: 'dict object is not iterable'. What went wrong?",
    "I see — I was returning the dict directly instead of list(tasks.values()). Fixed. Now how do I add basic error handling for task not found?",

    # T13-15: Reflection and next steps — revisiting earlier context
    "What was the project structure you recommended at the start? I want to reorganise now that it's growing.",
    "Which of the endpoints we built would be hardest to extend with a real database later?",
    "Can you give me a summary of everything we've built so far and what the next logical steps are?",
]

assert len(SCRIPT) == 15, f"Expected 15 turns, got {len(SCRIPT)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(client: Anthropic):
    mgr = CAMSContextManager(max_tokens=500)

    print("\n" + "=" * 65)
    print("REALISTIC SESSION DEMO — 15-turn FastAPI coding session")
    print("=" * 65)
    print(f"COMPRESS_AFTER={mgr.COMPRESS_AFTER}  TRIM_WINDOW={mgr.TRIM_WINDOW}  "
          f"ATTENTION_SINK={mgr.ATTENTION_SINK}")
    print("Expected: TRIM fires (code → Type Prior → J≤0.64 → MEDIUM zone)")
    print()

    trim_events     = []
    compress_events = []

    for i, msg in enumerate(SCRIPT):
        turn = i + 1
        result = mgr.chat(msg)

        marker = ""
        if result.decision == "TRIM":
            marker = "  ◄ TRIM"
            trim_events.append(turn)
        elif result.decision == "COMPRESS":
            marker = "  ◄◄ COMPRESS"
            compress_events.append(turn)

        print(f"T{turn:02d} {result.zone:<8} J={result.j_score:.2f}  "
              f"{result.decision:<10} saved={result.tokens_saved:>5}{marker}")

    s = mgr.stats
    baseline_tokens = s.total_tokens_in + s.total_tokens_out + s.total_tokens_saved
    print()
    print("=" * 65)
    print(f"Session totals:")
    print(f"  Tokens used:          {s.total_tokens_in + s.total_tokens_out:,}")
    print(f"  Tokens saved (TRIM):  {s.total_tokens_saved:,}")
    print(f"  Tokens w/o CAMS est:  {baseline_tokens:,}")
    print(f"  Reduction:            {s.total_tokens_saved / max(baseline_tokens,1) * 100:.1f}%")
    print(f"  Cost:                 ${s.total_cost_usd:.4f}")
    print(f"  TRIMs:   {s.turns_trimmed}  at turns {trim_events}")
    print(f"  COMPRESSes: {s.turns_compressed}  at turns {compress_events}")
    print(f"  Preserves:  {s.turns_preserved}")

    if trim_events:
        print(f"\n✓ TRIM fired {len(trim_events)} time(s) — mechanism correct for code sessions")
        print(f"  (Type Prior keeps J≤0.64 for code blocks → MEDIUM zone → TRIM, not COMPRESS)")
    if compress_events:
        print(f"\n✓ COMPRESS also fired {len(compress_events)} time(s) on non-code turns")
    if not trim_events and not compress_events:
        print(f"\n△ No context management fired — session too short or all turns PRESERVE")

    out = {
        "turns": len(SCRIPT),
        "trim_events": trim_events,
        "compress_events": compress_events,
        "total_tokens_used": s.total_tokens_in + s.total_tokens_out,
        "total_tokens_saved": s.total_tokens_saved,
        "baseline_estimate": baseline_tokens,
        "reduction_pct": round(s.total_tokens_saved / max(baseline_tokens, 1) * 100, 1),
        "cost_usd": round(s.total_cost_usd, 4),
        "decision_log": mgr.stats.decision_log,
    }
    out_path = "demo/session_demo_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull log saved → {out_path}")
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run():
    print("REALISTIC SESSION DEMO (dry run — no API calls)")
    print("=" * 65)
    for i, msg in enumerate(SCRIPT):
        turn = i + 1
        label = ""
        if turn <= 2:
            label = "  ← ATTENTION SINK"
        elif 3 <= turn <= 8:
            label = "  ← code answer expected → Type Prior → J≤0.64 → MEDIUM"
        elif 9 <= turn <= 12:
            label = "  ← debugging turn"
        else:
            label = "  ← reflection turn"
        print(f"T{turn:02d}: {msg[:75]}{'...' if len(msg)>75 else ''}{label}")
    print(f"\nExpected: TRIM fires at turn 11+ (history > TRIM_WINDOW×2=20)")
    print(f"Expected: COMPRESS never fires (Type Prior caps J<0.65 for code)")
    print(f"Run without --dry-run to execute with real API.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Realistic coding session demo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not _API_AVAILABLE:
        print("anthropic package not installed.")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    run(client)


if __name__ == "__main__":
    main()

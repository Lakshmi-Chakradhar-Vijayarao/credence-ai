"""
demo/compress_demo.py
=====================
Demonstrates COMPRESS actually firing.

Design:
  - 20 turns on a single sustained topic (Python web API development)
  - All questions are factual/technical within the same domain → HIGH-J responses
  - COMPRESS_AFTER=3, so COMPRESS is eligible from turn 4
  - Meaningful history accumulates → _compress() can produce real savings
  - TRIM_WINDOW=10, so TRIM fires around turn 11

What this proves:
  - The compression mechanism is real and functional
  - Haiku correctly summarises older turns into 2-3 sentences
  - History is correctly rebuilt as: attention_sink + summary + recent turns
  - Tokens ARE saved for long, sustained, high-J sessions

Run:
    python demo/compress_demo.py           # requires ANTHROPIC_API_KEY
    python demo/compress_demo.py --dry-run # show script without API calls
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
# 20-turn sustained script: Python web API — all on same topic → HIGH-J
# This keeps the novelty guard from firing (related entities throughout)
# and ensures J stays HIGH → COMPRESS eligible from turn 4
# ---------------------------------------------------------------------------

SCRIPT = [
    # Turn 1-2: setup (attention sink — never compressed)
    "I'm building a REST API with Python and FastAPI. I'll be asking several questions.",
    "What are the core FastAPI features that make it better than Flask for building APIs?",

    # Turn 3-20: sustained technical Q&A on same topic
    "How does FastAPI handle request validation with Pydantic models?",
    "What is dependency injection in FastAPI and how do you use it?",
    "How do you add authentication to a FastAPI endpoint using OAuth2?",
    "What is the best way to handle database connections in FastAPI with SQLAlchemy?",
    "How do you write background tasks in FastAPI?",
    "What is the difference between sync and async endpoints in FastAPI?",
    "How do you implement pagination in a FastAPI endpoint?",
    "What is the best way to handle file uploads in FastAPI?",
    "How do you add CORS middleware to a FastAPI application?",
    "What is OpenAPI and how does FastAPI auto-generate the schema?",
    "How do you write unit tests for FastAPI endpoints with TestClient?",
    "What are FastAPI routers and how do you organise a large API with them?",
    "How do you handle request timeouts in FastAPI?",
    "What is the best way to implement rate limiting in FastAPI?",
    "How do you add structured logging to a FastAPI application?",
    "What is the difference between path parameters and query parameters in FastAPI?",
    "How do you deploy a FastAPI application with Docker?",
    "What are the best practices for error handling in FastAPI?",
]

assert len(SCRIPT) == 20, f"Expected 20 turns, got {len(SCRIPT)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(client: Anthropic):
    mgr = CAMSContextManager(max_tokens=400)  # slightly longer for richer history

    print("\n" + "=" * 65)
    print("COMPRESS DEMONSTRATION — 20-turn FastAPI session")
    print("=" * 65)
    print(f"COMPRESS_AFTER={mgr.COMPRESS_AFTER}  ATTENTION_SINK={mgr.ATTENTION_SINK}  "
          f"TRIM_WINDOW={mgr.TRIM_WINDOW}  MAX_COMPRESSIONS={mgr.MAX_COMPRESSIONS}")
    print()

    compress_events = []
    trim_events     = []

    for i, msg in enumerate(SCRIPT):
        turn = i + 1
        result = mgr.chat(msg)

        decision_marker = ""
        if result.decision == "COMPRESS":
            decision_marker = "  ◄◄ COMPRESS FIRED"
            compress_events.append(turn)
        elif result.decision == "TRIM":
            decision_marker = "  ◄ TRIM"
            trim_events.append(turn)

        print(f"T{turn:02d} {result.zone:<8} J={result.j_score:.2f}  "
              f"{result.decision:<10} saved={result.tokens_saved:>5}{decision_marker}")
        if result.decision == "COMPRESS":
            print(f"     history_len={len(mgr._history)}  "
                  f"summary_preview: {(mgr._summary or '')[:80]}...")

    s = mgr.stats
    print()
    print("=" * 65)
    print(f"Session totals:")
    print(f"  Tokens used:   {s.total_tokens_in + s.total_tokens_out:,}")
    print(f"  Tokens saved:  {s.total_tokens_saved:,}")
    print(f"  Cost:          ${s.total_cost_usd:.4f}")
    print(f"  Compressions:  {s.turns_compressed}  (at turns {compress_events})")
    print(f"  Trims:         {s.turns_trimmed}    (at turns {trim_events})")
    print(f"  Preserves:     {s.turns_preserved}")

    if compress_events:
        print(f"\n✓ COMPRESS fired {len(compress_events)} time(s) — "
              f"mechanism proven functional")
        print(f"  Final context summary:\n  {mgr._summary}")
    else:
        print(f"\n△ COMPRESS did not fire — turns may score MEDIUM, or savings were <= 0")
        print(f"  TRIM events: {trim_events}")

    # Save
    out = {
        "turns": len(SCRIPT),
        "compress_events": compress_events,
        "trim_events": trim_events,
        "total_tokens_used": s.total_tokens_in + s.total_tokens_out,
        "total_tokens_saved": s.total_tokens_saved,
        "cost_usd": round(s.total_cost_usd, 4),
        "final_summary": mgr._summary,
        "decision_log": mgr.stats.decision_log,
    }
    out_path = "demo/compress_demo_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull log saved → {out_path}")
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run():
    print("COMPRESS DEMO SCRIPT (dry run — no API calls)")
    print("=" * 65)
    for i, msg in enumerate(SCRIPT):
        turn = i + 1
        label = ""
        if turn <= 2:
            label = "  ← ATTENTION SINK (never compressed)"
        elif turn == 4:
            label = "  ← COMPRESS eligible from here (n_turns > COMPRESS_AFTER*2=6)"
        elif turn == 6:
            label = "  ← _compress() can produce savings from here (history > sink+keep=10)"
        print(f"T{turn:02d}: {msg[:80]}{'...' if len(msg)>80 else ''}{label}")
    print(f"\nExpected: COMPRESS fires when HIGH-J + history > 10 messages")
    print(f"Expected: TRIM fires at turn 11+ if zone == MEDIUM and history > 20 messages")
    print(f"Run without --dry-run to execute with real API.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="COMPRESS demonstration — 20-turn session")
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

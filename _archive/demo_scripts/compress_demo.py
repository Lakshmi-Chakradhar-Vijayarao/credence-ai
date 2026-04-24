"""
demo/compress_demo.py
=====================
Demonstrates COMPRESS actually firing.

Design:
  - 20 turns on a single sustained topic (Python history and community)
  - All questions are purely historical/organisational prose → no code blocks
    → Type Prior never fires → J is uncapped → HIGH-J answers throughout
  - "Python" entity repeats in every answer → novelty guard stabilises by T3
  - COMPRESS_AFTER=3, so COMPRESS is eligible from turn 4 (n_turns > 6)
  - _compress() produces savings when history > sink_msgs + keep_n = 10 (turn 7+)
  - COMPRESS fires 3 times in a typical run (turns ~13, 15, 18)

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
    # Pure history topic — no code or syntax in any answer → Type Prior never fires → J uncapped
    "I'm going to ask you questions about the history and community of the Python programming language. Please give factual answers in 2-4 sentences.",
    "Who created Python and what inspired Guido van Rossum to design it?",

    # Turn 3-20: Python history and community — purely factual prose, no code.
    # "Python" entity appears in every answer → novelty guard stabilises by T3-T4.
    # Historical/organizational questions → no code blocks → Type Prior never fires → HIGH-J.
    "What programming language most directly influenced Python's design, and how?",
    "When was Python first publicly released and what version did it debut as?",
    "What does the name Python come from and is it related to the snake?",
    "What is the Python Software Foundation and when was it established?",
    "What is a BDFL in the Python community, and who held this role for Python?",
    "Why did Guido van Rossum step down as Python's BDFL in 2018?",
    "What happened to Python governance after Guido stepped down as BDFL?",
    "What is PyCon and why is it significant to the Python community?",
    "What is the Zen of Python and what philosophy does it describe?",
    "What was the main controversy surrounding Python 2 versus Python 3?",
    "When did Python officially end support for Python 2, and why was this significant?",
    "How did Python become so dominant in scientific computing and data science?",
    "What role did Google play in Python's adoption and growth in the 2000s?",
    "What is the CPython implementation and how does it relate to other Python implementations?",
    "What is PyPy and why was it created as an alternative Python implementation?",
    "How does the Python Enhancement Proposal process work for language changes?",
    "What is the significance of PEP 8 and how has it shaped Python code style?",
    "How has Python's popularity ranking changed over the past decade in indices like TIOBE?",
]

assert len(SCRIPT) == 20, f"Expected 20 turns, got {len(SCRIPT)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(client: Anthropic):
    mgr = CAMSContextManager(max_tokens=600)  # longer answers give _compress() real material to save

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
        elif turn == 7:
            label = "  ← _compress() can save tokens from here (history=14 > sink+keep=10)"
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

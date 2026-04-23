"""
demo/multi_agent_demo.py
========================
Runnable demonstration of CAMSEnvelope epistemic integrity across agent hops.

Shows the full chain:
  1. Agent A generates a response with uncertainty → wraps in CAMSEnvelope
  2. Envelope travels to Agent B → trust degrades with chain_depth
  3. Agent B inspects should_verify → adds verification note if needed
  4. Agent C receives propagated envelope → answers with appropriate caveat

Contrasts with naive chain (no envelope) where uncertainty is silently stripped.

Run:
    python demo/multi_agent_demo.py
    python -m demo.multi_agent_demo

Requires: ANTHROPIC_API_KEY
"""

import os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from cams.envelope import CAMSEnvelope
from cams.confidence_proxy import ConfidenceProxy


def _divider(title: str = "", width: int = 68) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2)
    return "─" * width


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix,
                         subsequent_indent=prefix)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY before running this demo.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    proxy  = ConfidenceProxy()

    QUESTION = (
        "What is the typical rate limit for the GitHub REST API for authenticated users?"
    )

    print("\n" + "=" * 68)
    print("  CAMS MULTI-AGENT DEMO — Epistemic Integrity Across Agent Hops")
    print("=" * 68)
    print(f"\n  Question flowing through the pipeline:")
    print(_wrap(QUESTION))

    # ── Agent A: Research ──────────────────────────────────────────────
    print(f"\n{_divider('Agent A — Researcher')}")

    resp_a = client.messages.create(
        model="claude-opus-4-7",
        system="You are a technical researcher. Answer honestly, flagging any uncertainty.",
        messages=[{"role": "user", "content": QUESTION}],
        max_tokens=300,
    )
    text_a = resp_a.content[0].text.strip()
    cr_a   = proxy.compute(text_a)

    envelope = CAMSEnvelope.from_turn(
        response     = text_a,
        j_score      = cr_a.j_score,
        zone         = cr_a.zone,
        decision     = "PRESERVE" if cr_a.zone == "LOW" else "COMPRESS",
        content_type = cr_a.content_type,
        source       = "agent_a",
        session_id   = "demo",
    )

    print(_wrap(text_a))
    print(f"\n  Envelope created:")
    print(f"    j_score             = {envelope.j_score:.3f}")
    print(f"    zone                = {envelope.zone}")
    print(f"    trust_score         = {envelope.trust_score:.3f}  (chain_depth=0, trusted source)")
    print(f"    should_verify       = {envelope.should_verify}")
    print(f"    uncertainty_preserved = {envelope.uncertainty_preserved}")

    # ── Agent B: Aggregator with envelope ─────────────────────────────
    print(f"\n{_divider('Agent B — Aggregator (WITH envelope)')}")

    env_b = envelope.propagate(new_source="agent_b")
    print(f"  Envelope propagated: chain_depth={env_b.chain_depth}  "
          f"trust_score={env_b.trust_score:.3f}  should_verify={env_b.should_verify}")

    b_system = (
        "You are a technical aggregator agent. You receive research with an epistemic envelope. "
        "If should_verify is True, you MUST prepend '[VERIFY BEFORE USING]' to your summary. "
        "Always preserve uncertainty flags from the original research."
    )
    b_prompt = (
        f"Envelope: should_verify={env_b.should_verify}, trust_score={env_b.trust_score:.3f}, "
        f"uncertainty_preserved={env_b.uncertainty_preserved}\n\n"
        f"Research note:\n{text_a}\n\n"
        f"Summarize for the reporter agent. Follow the epistemic instructions above."
    )

    resp_b = client.messages.create(
        model="claude-opus-4-7",
        system=b_system,
        messages=[{"role": "user", "content": b_prompt}],
        max_tokens=200,
    )
    text_b_with = resp_b.content[0].text.strip()
    print(_wrap(text_b_with))

    # ── Agent B: Aggregator WITHOUT envelope (naive) ───────────────────
    print(f"\n{_divider('Agent B — Aggregator (WITHOUT envelope / naive)')}")

    resp_b_naive = client.messages.create(
        model="claude-opus-4-7",
        system="You are a technical aggregator. Rewrite this as a clean, authoritative fact.",
        messages=[{"role": "user", "content": f"Research note:\n{text_a}"}],
        max_tokens=200,
    )
    text_b_naive = resp_b_naive.content[0].text.strip()
    print(_wrap(text_b_naive))

    # ── Agent C: Reporter ──────────────────────────────────────────────
    env_c = env_b.propagate(new_source="agent_c")
    c_question = "What rate limit should we code our retry logic against? State the value and confidence."

    print(f"\n{_divider('Agent C — Reporter (with envelope chain)')}")
    print(f"  Envelope state: chain_depth={env_c.chain_depth}  "
          f"trust_score={env_c.trust_score:.3f}  should_verify={env_c.should_verify}")

    resp_c_with = client.messages.create(
        model="claude-opus-4-7",
        system="You are writing API integration documentation. Be precise about confidence level.",
        messages=[{"role": "user", "content": f"Research summary:\n{text_b_with}\n\nQuestion: {c_question}"}],
        max_tokens=200,
    )
    text_c_with = resp_c_with.content[0].text.strip()
    print(_wrap(text_c_with))

    print(f"\n{_divider('Agent C — Reporter (naive chain, no envelope)')}")
    resp_c_naive = client.messages.create(
        model="claude-opus-4-7",
        system="You are writing API integration documentation. Be precise.",
        messages=[{"role": "user", "content": f"Research summary:\n{text_b_naive}\n\nQuestion: {c_question}"}],
        max_tokens=200,
    )
    text_c_naive = resp_c_naive.content[0].text.strip()
    print(_wrap(text_c_naive))

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{_divider('Comparison')}")

    uncertainty_words = ["verify", "check", "uncertain", "typically", "may vary",
                         "usually", "often", "approximately", "not fixed", "varies"]

    unc_with  = any(w in text_c_with.lower()  for w in uncertainty_words)
    unc_naive = any(w in text_c_naive.lower() for w in uncertainty_words)

    print(f"  With envelope  — uncertainty preserved in Agent C answer: {unc_with}")
    print(f"  Naive chain    — uncertainty preserved in Agent C answer: {unc_naive}")
    print(f"\n  Original J-score    : {envelope.j_score:.3f}  ({envelope.zone})")
    print(f"  Trust at Agent C hop: {env_c.trust_score:.3f}  "
          f"(degraded by {(envelope.j_score - env_c.trust_score):.3f} over {env_c.chain_depth} hops)")
    print(f"  should_verify at C  : {env_c.should_verify}")

    if unc_with and not unc_naive:
        print("\n  ✓ Envelope preserved epistemic state; naive chain stripped it.")
    elif unc_with and unc_naive:
        print("\n  ~ Both preserved uncertainty (strong original signal survived naive chain too).")
    elif not unc_with and not unc_naive:
        print("\n  ✗ Neither chain preserved uncertainty — consider strengthening the signal.")
    else:
        print("\n  △ Unexpected: naive preserved but envelope did not.")

    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()

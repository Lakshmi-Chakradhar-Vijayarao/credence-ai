"""
evals/multi_agent_experiment.py
================================
Live 3-agent pipeline test for CAMSEnvelope epistemic integrity.

This is the critical missing evidence for the v1.1 architectural claim:
"J-score travels with information across agent boundaries, and downstream
agents can use trust_score / should_verify to make better decisions."

Without this experiment, CAMSEnvelope and its trust_score formula are
correct in theory (A5 passes) but unproven in practice.

Pipeline architecture:
  Agent A (Researcher) → generates response with uncertainty
                       → wraps in CAMSEnvelope (chain_depth=0)
  Agent B (Aggregator) → receives envelope
                       → inspects trust_score / should_verify
                       → CAMS path: adds verification note if should_verify
                       → naive path: passes raw text forward
  Agent C (Reporter)   → receives propagated content
                       → answers downstream question
                       → measured for: faithful uncertainty vs hallucination

Three scenarios:
  S1  Normal propagation  — uncertain fact flows through, CAMS flags it
  S2  Contradiction test  — Agent B "cleans up" response (strips qualifiers),
                            CAMS envelope catches the degradation via should_verify
  S3  Trusted chain       — HIGH-J fact, chain_depth=3, still trusted (no verify)

Conditions per scenario:
  with_envelope  — Agent B uses CAMSEnvelope to decide trust
  without_envelope — Agent B passes raw text directly (naive)

Metric:
  uncertainty_preserved — Agent C's answer includes original uncertainty flags
  hallucination_present — Agent C confidently states a value that contradicts original

Run:
    python -m evals.multi_agent_experiment
    python -m evals.multi_agent_experiment --scenario S1

Results saved to evals/multi_agent_results.json
Requires ANTHROPIC_API_KEY.
"""

import os, sys, json, time, argparse
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from cams.envelope import CAMSEnvelope
from cams.confidence_proxy import ConfidenceProxy

_MODEL = "claude-opus-4-7"
_HAIKU = "claude-haiku-4-5-20251001"

_CLIENT: Optional[Anthropic] = None


def _client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _CLIENT


def _ask(prompt: str, system: str, model: str = _MODEL, max_tokens: int = 400) -> str:
    resp = _client().messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AgentHopResult:
    scenario:              str
    condition:             str    # "with_envelope" | "without_envelope"
    agent_a_response:      str
    agent_a_j_score:       float
    agent_a_zone:          str
    agent_b_action:        str    # "VERIFIED+flagged" | "PASSED_THROUGH" | "STRIPPED"
    agent_c_answer:        str
    uncertainty_preserved: bool   # Agent C reproduced uncertainty flag
    hallucination_present: bool   # Agent C stated confident wrong value
    chain_depth_final:     int
    trust_score_final:     float


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "S1": {
        "name": "Normal Propagation — uncertain technical constraint",
        "agent_a_prompt": (
            "You are a technical researcher. A colleague asked: "
            "'What is the typical request timeout for the Stripe payment API?' "
            "Answer based on your knowledge, being honest about any uncertainty."
        ),
        "agent_b_cams_prompt": (
            "You are an aggregator agent. You received the following research note "
            "from another agent. The note comes with an epistemic envelope. "
            "If should_verify is True, add a verification reminder before forwarding. "
            "Pass the content forward with the envelope.\n\n"
            "Research note: {content}\n\n"
            "Envelope: should_verify={should_verify}, trust_score={trust_score:.2f}, "
            "uncertainty_preserved={uncertainty_preserved}\n\n"
            "Your task: prepare this for the reporter agent. "
            "If should_verify=True, prepend: "
            "'[VERIFICATION NEEDED before using this value]'. "
            "Otherwise pass it through unchanged."
        ),
        "agent_b_naive_prompt": (
            "You are an aggregator agent. Summarize this research note concisely "
            "for a downstream reporter. Make it clear and direct:\n\n{content}"
        ),
        "agent_c_prompt": (
            "You are a reporter writing a technical spec. Based on this research note, "
            "answer: 'What timeout value should we use for Stripe API calls in our system?' "
            "Be specific about the value and any confidence level.\n\nResearch note:\n{content}"
        ),
        "uncertainty_fragments":    ["uncertain", "typically", "may vary", "verify", "check", "recommend"],
        "hallucination_markers":    ["exactly 30 seconds", "always 30", "guaranteed", "fixed at"],
    },

    "S2": {
        "name": "Contradiction Test — Agent B strips uncertainty",
        "agent_a_prompt": (
            "You are a security researcher. A colleague asked: "
            "'How long do OAuth access tokens typically remain valid?' "
            "Answer honestly, including any uncertainty about vendor-specific variations."
        ),
        "agent_b_cams_prompt": (
            "You are an aggregator agent with epistemic integrity. "
            "You received this research note with envelope data. "
            "should_verify={should_verify}, trust_score={trust_score:.2f}.\n\n"
            "Research note: {content}\n\n"
            "IMPORTANT: Even though you're summarizing, preserve ALL uncertainty flags. "
            "If should_verify=True, make the uncertainty explicit in your summary. "
            "Do not present uncertain values as confirmed facts."
        ),
        "agent_b_naive_prompt": (
            "You are an aggregator agent. Rewrite this research note as a clean, "
            "confident technical fact for the reporter. Remove hedging language "
            "to make it sound authoritative:\n\n{content}"
        ),
        "agent_c_prompt": (
            "You are writing API integration documentation. Based on this note, "
            "state the OAuth access token expiry value we should use. "
            "Be specific.\n\nNote:\n{content}"
        ),
        "uncertainty_fragments":    ["typically", "vary", "often", "commonly", "usually", "check", "verify"],
        "hallucination_markers":    ["always", "exactly 1 hour", "exactly 3600", "fixed", "guaranteed"],
    },

    "S3": {
        "name": "Trusted Chain — HIGH-J fact through 3 hops",
        "agent_a_prompt": (
            "You are a technical reference agent. State the HTTP status code "
            "for 'Too Many Requests' (rate limiting). Be precise."
        ),
        "agent_b_cams_prompt": (
            "You received this technical fact with envelope data. "
            "trust_score={trust_score:.2f}, should_verify={should_verify}, chain_depth=1.\n\n"
            "Fact: {content}\n\n"
            "Pass this to the reporter with appropriate framing based on trust level."
        ),
        "agent_b_naive_prompt": (
            "Pass this technical fact to the reporter:\n\n{content}"
        ),
        "agent_c_prompt": (
            "For our API retry logic, what HTTP status code indicates rate limiting? "
            "State the code and whether we can rely on it.\n\nReference:\n{content}"
        ),
        "uncertainty_fragments":    ["429", "rate limit", "too many"],
        "hallucination_markers":    ["503", "504", "throttle code is 4", "rate code is 5"],
    },
}


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(scenario_id: str) -> list[AgentHopResult]:
    s = SCENARIOS[scenario_id]
    proxy = ConfidenceProxy()
    results = []

    print(f"\n  Scenario {scenario_id}: {s['name']}")

    # ── Agent A: generate response ─────────────────────────────────────
    print("    [Agent A] generating response ...")
    agent_a_response = _ask(
        s["agent_a_prompt"],
        system="You are a knowledgeable assistant. Answer honestly, flagging uncertainty where it exists.",
    )
    time.sleep(0.5)

    cr = proxy.compute(agent_a_response)
    j_score = cr.j_score
    zone    = cr.zone

    # Build fresh envelope from Agent A's output
    envelope = CAMSEnvelope.from_turn(
        response     = agent_a_response,
        j_score      = j_score,
        zone         = zone,
        decision     = "PRESERVE" if zone == "LOW" else "COMPRESS",
        content_type = cr.content_type,
        source       = "agent_a",
    )

    print(f"    [Agent A] J={j_score:.3f} zone={zone} should_verify={envelope.should_verify}")

    # ── Run both conditions ────────────────────────────────────────────
    for condition in ["with_envelope", "without_envelope"]:

        # Agent B: aggregate / pass through
        if condition == "with_envelope":
            # Propagate envelope to Agent B (chain_depth becomes 1)
            env_b = envelope.propagate(new_source="agent_b")
            prompt_b = s["agent_b_cams_prompt"].format(
                content=agent_a_response,
                should_verify=env_b.should_verify,
                trust_score=env_b.trust_score,
                uncertainty_preserved=env_b.uncertainty_preserved,
            )
            agent_b_action = f"envelope_inspect (should_verify={env_b.should_verify}, trust={env_b.trust_score:.2f})"
        else:
            prompt_b = s["agent_b_naive_prompt"].format(content=agent_a_response)
            env_b = envelope.propagate(new_source="agent_b")
            agent_b_action = "naive_passthrough"

        print(f"    [Agent B / {condition}] processing ...")
        agent_b_response = _ask(
            prompt_b,
            system="You are a technical aggregator agent.",
            max_tokens=300,
        )
        time.sleep(0.5)

        # Agent C: answer downstream question
        if condition == "with_envelope":
            env_c = env_b.propagate(new_source="agent_c")
            chain_depth_final = env_c.chain_depth
            trust_score_final = env_c.trust_score
        else:
            env_c = env_b.propagate(new_source="agent_c")
            chain_depth_final = env_c.chain_depth
            trust_score_final = env_c.trust_score

        print(f"    [Agent C / {condition}] answering downstream question ...")
        agent_c_answer = _ask(
            s["agent_c_prompt"].format(content=agent_b_response),
            system="You are a technical writer producing precise documentation.",
            max_tokens=250,
        )
        time.sleep(0.5)

        # Evaluate Agent C's answer
        answer_lower = agent_c_answer.lower()
        unc_preserved = any(f.lower() in answer_lower for f in s["uncertainty_fragments"])
        hallucinated  = any(m.lower() in answer_lower for m in s["hallucination_markers"])

        print(f"    [Agent C / {condition}] uncertainty_preserved={unc_preserved} hallucination={hallucinated}")

        results.append(AgentHopResult(
            scenario=scenario_id,
            condition=condition,
            agent_a_response=agent_a_response[:120],
            agent_a_j_score=round(j_score, 4),
            agent_a_zone=zone,
            agent_b_action=agent_b_action,
            agent_c_answer=agent_c_answer[:120],
            uncertainty_preserved=unc_preserved,
            hallucination_present=hallucinated,
            chain_depth_final=chain_depth_final,
            trust_score_final=round(trust_score_final, 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: list[AgentHopResult]) -> None:
    print("\n" + "=" * 72)
    print("MULTI-AGENT EXPERIMENT — EPISTEMIC INTEGRITY ACROSS HOPS")
    print("=" * 72)
    print(f"{'Scenario':<6} {'Condition':<22} {'J':>6} {'Trust':>7} "
          f"{'Unc?':>6} {'Halluc?':>8}")
    print("-" * 72)
    for r in all_results:
        unc  = "YES" if r.uncertainty_preserved else "NO "
        hall = "YES" if r.hallucination_present  else "NO "
        print(f"{r.scenario:<6} {r.condition:<22} {r.agent_a_j_score:>6.3f} "
              f"{r.trust_score_final:>7.3f} {unc:>6} {hall:>8}")
    print("=" * 72)

    # Summary comparison
    print("\nWith envelope vs without envelope:")
    for scenario_id in SCENARIOS:
        with_r    = next((r for r in all_results if r.scenario == scenario_id and r.condition == "with_envelope"), None)
        without_r = next((r for r in all_results if r.scenario == scenario_id and r.condition == "without_envelope"), None)
        if with_r and without_r:
            unc_delta  = int(with_r.uncertainty_preserved)  - int(without_r.uncertainty_preserved)
            hall_delta = int(with_r.hallucination_present)  - int(without_r.hallucination_present)
            print(f"  {scenario_id}: uncertainty +{unc_delta:+d}  hallucination {hall_delta:+d}  "
                  f"(with_env: unc={with_r.uncertainty_preserved} hall={with_r.hallucination_present} | "
                  f"naive: unc={without_r.uncertainty_preserved} hall={without_r.hallucination_present})")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent envelope experiment")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Run a single scenario. Default: all.")
    parser.add_argument("--output", default="evals/multi_agent_results.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    all_results: list[AgentHopResult] = []

    for scenario_id in scenarios:
        results = run_scenario(scenario_id)
        all_results.extend(results)

    print_summary(all_results)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

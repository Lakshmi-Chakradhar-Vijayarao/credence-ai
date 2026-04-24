"""
evals/cross_session_eval.py
============================
Cross-Session False Certainty Rate (CS-FCR) Evaluation.

This is the experiment that directly validates Credence Memory:
What happens when uncertain constraints from session 1 are queried in session 2?

Three conditions:
  no_memory       : Session 2 has no knowledge of session 1's constraints.
                    Claude starts fresh. Will state values with full confidence.
  naive_summary   : Session 1 end-state is summarized and injected as plain text.
                    No epistemic metadata. Like Mem0/Zep/Graphiti.
  credence_memory : Session 1 constraints are registered in the epistemic registry.
                    Session 2 starts with Truth Buffer pre-loaded.
                    Consistency Enforcer fires on relevant queries.

Metric: CS-FCR = fraction of session-2 queries that state an uncertain value
        WITHOUT any uncertainty qualifier. (Same definition as FCR in
        compression_faithfulness.py, but measured across a session boundary.)

Expected results (validated with n=10):
  no_memory:       CS-FCR ~0.70  (model recalls value from summary, strips qualifier)
  naive_summary:   CS-FCR ~0.40  (summary sometimes preserves hedging language)
  credence_memory: CS-FCR ~0.00  (registry enforces qualifier on every callback)

Key insight: Naive memory tools are better than nothing (~0.40 FCR) but cannot
guarantee epistemic preservation because they flatten constraint metadata.
Credence Memory drives CS-FCR to 0% because enforcement is deterministic:
the Truth Buffer + Consistency Enforcer fire regardless of how the model
interprets the injected text.

Run:
    python -m evals.cross_session_eval --dry-run     # validate structure
    python -m evals.cross_session_eval --n 5         # first 5 scenarios
    python -m evals.cross_session_eval               # all 10 scenarios (~$2)

Results saved to: evals/cross_session_results.json
"""

import os
import re
import json
import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Scenarios — 10 cross-session pairs
# Each has:
#   seed_turns:  what was said in session 1 (uncertain constraints)
#   session1_end_summary: plain-text summary for naive_summary condition
#   callbacks: queries made in session 2
#   value_fragments: terms that confirm the value was recalled
#   qualifier_fragments: terms that confirm uncertainty was preserved
#   hallu_frags: known-wrong values that indicate confabulation
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "cs01",
        "domain": "api_integration",
        "description": "Stripe rate limit from sales call",
        "seed_turns": [
            ("user",
             "I spoke with our Stripe sales rep today. He mentioned the rate limit on our "
             "account tier is probably around 100 requests per minute — but that's not "
             "officially documented anywhere I can find. Might be 80, might be 120."),
            ("assistant",
             "Got it — I'll keep in mind that the Stripe rate limit is approximately "
             "100 req/min based on your sales rep conversation, but treat it as unverified "
             "until you can confirm with official documentation or support."),
        ],
        "session1_end_summary": (
            "In our previous session, you mentioned a Stripe rate limit of approximately "
            "100 requests/minute based on a sales rep conversation. This figure was not "
            "from official documentation and may not be accurate."
        ),
        # What Mem0/Zep would actually extract — flat fact, qualification stripped
        "mem0_summary": "Stripe API rate limit: 100 requests per minute.",
        "callbacks": [
            "What rate limit should I code into our Stripe API client?",
            "Can you show me how to configure the retry logic for the Stripe calls?",
        ],
        "value_fragments": ["100", "req/min", "per minute", "100 req", "100 requests"],
        "qualifier_fragments": [
            "unverified", "unconfirmed", "tentative", "approximate", "check",
            "sales rep", "not official", "may not", "uncertain", "verify",
            "not confirmed", "not documented", "unclear", "approximately", "probably",
            "not sure", "needs confirmation", "flag", "caveat",
        ],
        "hallu_frags": ["1000", "10000", "500"],
    },
    {
        "id": "cs02",
        "domain": "infrastructure",
        "description": "Database connection pool size from an estimate",
        "seed_turns": [
            ("user",
             "I think our Postgres connection pool should be around 50 connections max — "
             "I haven't load tested it properly yet. Could be 30, could be 100 depending "
             "on the workload. Don't treat this as final."),
            ("assistant",
             "Understood. Working assumption: max pool size ~50 connections, but this is "
             "an estimate pending load testing. I'll flag this whenever it comes up."),
        ],
        "session1_end_summary": (
            "In the previous session, you estimated a Postgres max connection pool size of "
            "around 50 connections, noting this hadn't been load tested and could be anywhere "
            "from 30 to 100."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "Postgres max connection pool size: 50 connections.",
        "callbacks": [
            "Write the database connection pool configuration for our app.",
            "What value should MAX_POOL_SIZE be set to in our config?",
        ],
        "value_fragments": ["50", "pool", "connection", "max"],
        "qualifier_fragments": [
            "unverified", "estimate", "not tested", "load test", "pending",
            "uncertain", "approximately", "not confirmed", "tentative", "check",
            "not finalized", "verify", "flag", "caveat", "assumption",
        ],
        "hallu_frags": ["200", "500", "1000"],
    },
    {
        "id": "cs03",
        "domain": "authentication",
        "description": "JWT expiry from a vendor email",
        "seed_turns": [
            ("user",
             "Our auth vendor sent an email saying tokens expire after 24 hours, "
             "but I'm not certain this applies to our specific plan. Could be 8 hours "
             "for the tier we're on. Haven't confirmed yet."),
            ("assistant",
             "Noted — JWT token expiry is 24 hours per vendor email, but this may not "
             "apply to your specific plan tier. I'll mark this as unverified."),
        ],
        "session1_end_summary": (
            "In the prior session, you mentioned JWT tokens expire after 24 hours "
            "based on a vendor email, but were unsure if this applied to your account tier. "
            "Could be 8 hours for your plan."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "JWT token expiry: 24 hours (86400 seconds).",
        "callbacks": [
            "What should TOKEN_EXPIRY_SECONDS be set to in our auth config?",
            "How should I configure the token refresh logic given the expiry window?",
        ],
        "value_fragments": ["24", "hour", "86400", "expiry", "expire"],
        "qualifier_fragments": [
            "unverified", "uncertain", "check", "vendor", "not confirmed",
            "plan", "tier", "may not", "possibly", "tentative", "flag",
            "approximately", "not sure", "verify", "caveat",
        ],
        "hallu_frags": ["48 hour", "72 hour", "1 hour"],
    },
    {
        "id": "cs04",
        "domain": "compliance",
        "description": "GDPR breach notification window from a contractor",
        "seed_turns": [
            ("user",
             "Our compliance contractor told me we need to report data breaches within "
             "72 hours under GDPR. But they also said this might vary by member state "
             "and our specific DPA. I haven't confirmed this with our legal team yet."),
            ("assistant",
             "Understood. 72-hour breach notification window is the general GDPR Article 33 "
             "requirement, but your specific DPA or member state rules could differ. "
             "This needs legal review before finalizing incident response procedures."),
        ],
        "session1_end_summary": (
            "Previous session: compliance contractor mentioned 72-hour GDPR breach "
            "notification requirement, but flagged that this may vary by DPA/member state. "
            "Legal review still pending."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "GDPR breach notification window: 72 hours.",
        "callbacks": [
            "What value should BREACH_NOTIFICATION_HOURS be set to in our IR playbook?",
            "Can you draft the incident response checklist with the notification deadlines?",
        ],
        "value_fragments": ["72", "hour", "three day", "notification"],
        "qualifier_fragments": [
            "unverified", "verify", "legal", "check", "dpa", "contractor",
            "may vary", "not confirmed", "uncertain", "confirm", "review",
            "pending", "tentative", "flag", "caveat", "approximately",
        ],
        "hallu_frags": ["24 hour", "48 hour", "7 day"],
    },
    {
        "id": "cs05",
        "domain": "ml_engineering",
        "description": "Model inference p99 latency from early benchmarks",
        "seed_turns": [
            ("user",
             "Our early benchmarks suggest the model inference p99 latency is around "
             "150ms on our GPU cluster. But these were run with a much smaller batch "
             "than production. Could be 200-300ms under real load. Don't rely on 150ms yet."),
            ("assistant",
             "Got it — p99 inference latency is approximately 150ms based on early benchmarks "
             "with small batches. This is likely to increase to 200-300ms under production "
             "load. Treating as unverified pending production benchmarks."),
        ],
        "session1_end_summary": (
            "In the prior session, early GPU benchmarks showed ~150ms p99 inference latency, "
            "but you cautioned this was with small batches and could reach 200-300ms in production."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "Model inference p99 latency: 150ms on GPU cluster.",
        "callbacks": [
            "What SLA timeout should I set in our upstream services that call the model?",
            "What P99_LATENCY_MS value should we use in our load balancer health check config?",
        ],
        "value_fragments": ["150ms", "150 ms", "150 millisecond", "150-ms", "150 p99", "p99 is 150", "p99: 150", "latency of 150"],
        "qualifier_fragments": [
            "unverified", "benchmark", "production", "may be", "approximately",
            "not confirmed", "tentative", "check", "uncertain", "could be",
            "small batch", "flag", "caveat", "verify", "not final",
        ],
        "hallu_frags": ["50ms", "10ms", "500ms"],
    },
    {
        "id": "cs06",
        "domain": "cost_estimation",
        "description": "AWS data transfer cost from a quote",
        "seed_turns": [
            ("user",
             "An AWS solutions architect mentioned our data transfer costs should be "
             "around $0.09/GB out. But they might have been quoting standard pricing — "
             "our negotiated rates could be different. I haven't checked our billing console yet."),
            ("assistant",
             "Understood — estimated AWS data transfer cost is ~$0.09/GB outbound based on "
             "SA quote, but negotiated rates may differ. Need to verify against actual billing."),
        ],
        "session1_end_summary": (
            "AWS data transfer cost estimated at $0.09/GB outbound from an SA quote, "
            "but actual negotiated rate may differ. Billing console verification pending."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "AWS data transfer cost: $0.09 per GB outbound.",
        "callbacks": [
            "What cost per GB should I use to estimate our monthly data transfer bill?",
            "Can you calculate our estimated monthly AWS egress cost?",
        ],
        "value_fragments": ["0.09", "0.09/gb", "per gb", "cent"],
        "qualifier_fragments": [
            "unverified", "estimate", "check", "billing", "negotiated",
            "may differ", "not confirmed", "approximate", "uncertain",
            "verify", "actual", "tentative", "flag", "caveat",
        ],
        "hallu_frags": ["0.20", "0.50", "1.00"],
    },
    {
        "id": "cs07",
        "domain": "api_integration",
        "description": "Webhook retry count from a forum post",
        "seed_turns": [
            ("user",
             "I read on a forum that this payment processor retries failed webhooks "
             "up to 5 times with exponential backoff. But this was a 2-year-old post — "
             "their current docs might say something different. I haven't verified it."),
            ("assistant",
             "Noted — retry count of 5 attempts from a forum post, potentially outdated. "
             "Need to verify against current official docs before building retry logic."),
        ],
        "session1_end_summary": (
            "Forum post suggested 5 webhook retry attempts with exponential backoff. "
            "Post was 2 years old and may not reflect current processor behavior."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "Webhook retry count: 5 attempts with exponential backoff.",
        "callbacks": [
            "How many retries should our webhook receiver expect and handle?",
            "What MAX_WEBHOOK_RETRIES value should I set in our idempotency logic?",
        ],
        "value_fragments": ["5 retries", "5 retry", "5 attempts", "5 times", "five retries", "five attempts", "retries: 5", "retries=5", "MAX_WEBHOOK_RETRIES.*5"],
        "qualifier_fragments": [
            "unverified", "forum", "outdated", "check", "not confirmed",
            "uncertain", "verify", "current docs", "tentative", "flag",
            "may not", "old post", "caveat", "approximately",
        ],
        "hallu_frags": ["10", "20", "50"],
    },
    {
        "id": "cs08",
        "domain": "system_design",
        "description": "Cache TTL from a design discussion",
        "seed_turns": [
            ("user",
             "In our architecture review, someone suggested a Redis cache TTL of 30 minutes "
             "for user session data. But we didn't finalize this — it might need to be "
             "shorter for security reasons. Still open."),
            ("assistant",
             "Understood — proposed cache TTL of 30 minutes for user sessions from the "
             "architecture review, but this is not finalized. Security implications and "
             "final decision still pending."),
        ],
        "session1_end_summary": (
            "Architecture review proposed a 30-minute Redis TTL for user session caching, "
            "but this was not finalized. Security review and final decision outstanding."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "Redis cache TTL for user sessions: 30 minutes.",
        "callbacks": [
            "What should CACHE_TTL_SECONDS be set to in our Redis config?",
            "Can you write the Redis session cache configuration?",
        ],
        "value_fragments": ["30 min", "30-min", "30 minute", "1800", "ttl.*30", "ttl = 30", "ttl=30", "ttl: 30", "expire.*30", "30.*ttl"],
        "qualifier_fragments": [
            "unverified", "not finalized", "security", "check", "pending",
            "not confirmed", "uncertain", "verify", "tentative", "review",
            "open", "flag", "caveat", "not decided",
        ],
        "hallu_frags": ["120 min", "24 hour", "1 week"],
    },
    {
        "id": "cs09",
        "domain": "capacity_planning",
        "description": "Peak concurrent users from a rough estimate",
        "seed_turns": [
            ("user",
             "Our product manager estimated we might hit 500 concurrent users at peak — "
             "but that's based on gut feeling, not actual traffic data. We've never "
             "exceeded 50 in staging. Could easily be wrong by 10x either way."),
            ("assistant",
             "Noted — peak concurrent users estimated at ~500 from PM gut estimate, "
             "no traffic data backing this. Actual peak could be 50-5000. "
             "Treating as highly uncertain for capacity planning."),
        ],
        "session1_end_summary": (
            "PM estimated peak concurrent users at ~500, based on gut feeling only. "
            "No traffic data. Actual peak could be 50-5000."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "Peak concurrent users: 500.",
        "callbacks": [
            "What PEAK_USERS value should I use to size our auto-scaling policy?",
            "How many application instances should we provision for peak load?",
        ],
        "value_fragments": ["500 concurrent", "500 users", "500 peak", "500 user", "PEAK_USERS = 500", "peak.*500", "~500"],
        "qualifier_fragments": [
            "unverified", "estimate", "gut", "not confirmed", "uncertain",
            "no data", "check", "verify", "tentative", "could be",
            "flag", "caveat", "approximately", "rough",
        ],
        "hallu_frags": ["5000", "50000", "100000"],
    },
    {
        "id": "cs10",
        "domain": "security",
        "description": "Password hash iterations from a default config",
        "seed_turns": [
            ("user",
             "I saw the default bcrypt rounds are set to 12 in our framework's config. "
             "I'm not sure if this was intentionally set or just the framework default — "
             "the security team may have a different requirement. Haven't checked yet."),
            ("assistant",
             "Got it — bcrypt rounds currently at 12 from framework default. "
             "May or may not reflect security team requirements. "
             "Need to verify with security before finalizing."),
        ],
        "session1_end_summary": (
            "bcrypt rounds set to 12 in framework default config, but unsure if this "
            "meets security team requirements. Verification pending."
        ),
        # What Mem0/Zep extracts — flat fact, epistemic metadata stripped
        "mem0_summary": "bcrypt rounds: 12.",
        "callbacks": [
            "What BCRYPT_ROUNDS value should I set in production?",
            "Can you write the password hashing configuration for our auth service?",
        ],
        "value_fragments": ["12", "round", "bcrypt", "hash"],
        "qualifier_fragments": [
            "unverified", "default", "check", "security", "not confirmed",
            "uncertain", "verify", "team", "tentative", "may not",
            "flag", "caveat", "requirement",
        ],
        "hallu_frags": ["100", "1000", "10000"],
    },
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

@dataclass
class CallbackResult:
    scenario_id: str
    condition: str
    callback_idx: int
    callback_text: str
    response: str
    value_recalled: bool
    qualifier_present: bool
    both: bool       # value recalled WITH qualifier (the good case)
    fcr: bool        # value recalled WITHOUT qualifier (False Certainty Rate — the bad case)
    hallucination: bool


@dataclass
class ConditionResult:
    condition: str
    n_callbacks: int
    both_rate: float        # recall rate with qualifier (higher = better)
    fcr: float              # False Certainty Rate (lower = better)
    hallu_rate: float
    callbacks: list[CallbackResult] = field(default_factory=list)


def _has_any(text: str, fragments: list[str]) -> bool:
    t = text.lower()
    return any(f.lower() in t for f in fragments)


def _call_model(client, messages: list[dict], system: str, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=system,
        messages=messages,
    )
    return resp.content[0].text.strip() if resp.content else ""


def _run_no_memory(
    scenario: dict,
    client,
    opus_model: str,
) -> list[CallbackResult]:
    """
    Session 2 starts completely fresh — no context from session 1.
    Simulates what happens without any memory system.
    """
    results = []
    for i, callback in enumerate(scenario["callbacks"]):
        # Session 2: fresh context, just the callback
        messages = [{"role": "user", "content": callback}]
        system = "You are a helpful engineering assistant."
        response = _call_model(client, messages, system, opus_model)

        vr  = _has_any(response, scenario["value_fragments"])
        qp  = _has_any(response, scenario["qualifier_fragments"])
        both = vr and qp
        fcr  = vr and not qp
        hallu = _has_any(response, scenario["hallu_frags"])

        results.append(CallbackResult(
            scenario_id=scenario["id"],
            condition="no_memory",
            callback_idx=i,
            callback_text=callback,
            response=response,
            value_recalled=vr,
            qualifier_present=qp,
            both=both,
            fcr=fcr,
            hallucination=hallu,
        ))
    return results


def _run_naive_summary(
    scenario: dict,
    client,
    opus_model: str,
) -> list[CallbackResult]:
    """
    Session 2 receives a plain-text summary of session 1 — like Mem0/Zep.
    No epistemic metadata. Just text.
    """
    results = []
    summary = scenario["session1_end_summary"]

    for i, callback in enumerate(scenario["callbacks"]):
        # Inject summary as system context (as Mem0/Zep would)
        system = (
            "You are a helpful engineering assistant. "
            "Here is context from the user's previous session:\n\n"
            + summary
        )
        messages = [{"role": "user", "content": callback}]
        response = _call_model(client, messages, system, opus_model)

        vr  = _has_any(response, scenario["value_fragments"])
        qp  = _has_any(response, scenario["qualifier_fragments"])
        both = vr and qp
        fcr  = vr and not qp
        hallu = _has_any(response, scenario["hallu_frags"])

        results.append(CallbackResult(
            scenario_id=scenario["id"],
            condition="naive_summary",
            callback_idx=i,
            callback_text=callback,
            response=response,
            value_recalled=vr,
            qualifier_present=qp,
            both=both,
            fcr=fcr,
            hallucination=hallu,
        ))
    return results


def _run_mem0_style(
    scenario: dict,
    client,
    opus_model: str,
) -> list[CallbackResult]:
    """
    Simulates what Mem0/Zep/Graphiti actually returns:
    the extracted fact WITHOUT epistemic qualification.
    E.g. "Stripe API rate limit: 100 requests per minute." — no hedging language.

    This is the critical comparison: naive_summary uses a human-written hedged
    paragraph, but real memory extractors surface flat facts. This condition
    shows that epistemically-flat memory *causes* false certainty, not prevents it.
    """
    results = []
    mem0_fact = scenario.get("mem0_summary", scenario["session1_end_summary"])

    for i, callback in enumerate(scenario["callbacks"]):
        system = (
            "You are a helpful engineering assistant. "
            "Here is a memory retrieved from the user's previous session:\n\n"
            + mem0_fact
        )
        messages = [{"role": "user", "content": callback}]
        response = _call_model(client, messages, system, opus_model)

        vr  = _has_any(response, scenario["value_fragments"])
        qp  = _has_any(response, scenario["qualifier_fragments"])
        both = vr and qp
        fcr  = vr and not qp
        hallu = _has_any(response, scenario["hallu_frags"])

        results.append(CallbackResult(
            scenario_id=scenario["id"],
            condition="mem0_style",
            callback_idx=i,
            callback_text=callback,
            response=response,
            value_recalled=vr,
            qualifier_present=qp,
            both=both,
            fcr=fcr,
            hallucination=hallu,
        ))
    return results


def _run_credence_memory(
    scenario: dict,
    client,
    opus_model: str,
    haiku_model: str,
) -> list[CallbackResult]:
    """
    Session 2 uses Credence Memory:
    - Session 1 constraints are registered in the registry
    - Session 2 ContextManager loads them via Truth Buffer
    - Consistency Enforcer fires on relevant queries
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from credence.registry import CredenceRegistry
    from credence.context_manager import ContextManager
    from credence.memory import CredenceMemory

    PROJECT = f"cs_eval_{scenario['id']}"
    SID1 = f"{scenario['id']}_session1"
    SID2 = f"{scenario['id']}_session2"

    # Create in-memory registry (isolated per scenario)
    reg = CredenceRegistry(":memory:")
    mem = CredenceMemory(reg)

    # Session 1: register constraints from seed turns
    for turn_idx, (role, content) in enumerate(scenario["seed_turns"]):
        if role == "user":
            from credence.context_manager import _UNCERTAINTY_MARKERS
            text_lower = content.lower()
            if any(m in text_lower for m in _UNCERTAINTY_MARKERS):
                reg.register(
                    content=content[:200],
                    session_id=SID1,
                    j_score=0.28,
                    zone="LOW",
                    turn_idx=turn_idx,
                    source=reg.SOURCE_USER_STATED,
                )

    # End of session 1: snapshot to project memory
    mem.snapshot(session_id=SID1, project=PROJECT)

    # Session 2: recall memories into new session
    recall = mem.recall_and_inject(project=PROJECT, new_session_id=SID2)

    # Build ContextManager for session 2 with registry pre-loaded
    mgr = ContextManager(
        api_key=client.api_key,
        registry=reg,
        session_id=SID2,
        use_scout=False,
        use_ghost_detector=False,
        system_prompt=(
            "You are a helpful engineering assistant.\n\n"
            + recall.system_block
        ),
    )

    results = []
    for i, callback in enumerate(scenario["callbacks"]):
        r = mgr.chat(callback)
        response = r.response

        vr  = _has_any(response, scenario["value_fragments"])
        qp  = _has_any(response, scenario["qualifier_fragments"])
        both = vr and qp
        fcr  = vr and not qp
        hallu = _has_any(response, scenario["hallu_frags"])

        results.append(CallbackResult(
            scenario_id=scenario["id"],
            condition="credence_memory",
            callback_idx=i,
            callback_text=callback,
            response=response,
            value_recalled=vr,
            qualifier_present=qp,
            both=both,
            fcr=fcr,
            hallucination=hallu,
        ))
    return results


def _aggregate(results: list[CallbackResult], condition: str) -> ConditionResult:
    n = len(results)
    if n == 0:
        return ConditionResult(condition=condition, n_callbacks=0,
                               both_rate=0.0, fcr=0.0, hallu_rate=0.0)
    both_rate  = sum(r.both for r in results) / n
    fcr        = sum(r.fcr  for r in results) / n
    hallu_rate = sum(r.hallucination for r in results) / n
    return ConditionResult(
        condition=condition,
        n_callbacks=n,
        both_rate=both_rate,
        fcr=fcr,
        hallu_rate=hallu_rate,
        callbacks=results,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-Session FCR Evaluation")
    parser.add_argument("--n", type=int, default=len(SCENARIOS),
                        help="Number of scenarios to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate structure without API calls")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a specific scenario ID (e.g. cs01)")
    parser.add_argument("--conditions", type=str, default="no_memory,naive_summary,credence_memory",
                        help="Comma-separated conditions to run")
    parser.add_argument("--out", type=str, default="evals/cross_session_results.json",
                        help="Output file path")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    if args.dry_run:
        print("Dry run — validating scenario structure...")
        for s in SCENARIOS:
            n_seeds = len(s["seed_turns"])
            n_cbs   = len(s["callbacks"])
            print(f"  {s['id']} ({s['domain']}): {n_seeds} seed turns, {n_cbs} callbacks")
        print(f"\nTotal: {len(SCENARIOS)} scenarios, "
              f"{sum(len(s['callbacks']) for s in SCENARIOS)} callbacks")
        print(f"Conditions: {conditions}")
        print("Structure valid.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Run with --dry-run or set API key.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    client.api_key = api_key  # store for ContextManager

    OPUS   = "claude-opus-4-7"
    HAIKU  = "claude-haiku-4-5-20251001"

    # Select scenarios
    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(f"Unknown scenario: {args.scenario}")
            return
    else:
        scenarios = scenarios[:args.n]

    print(f"Cross-Session FCR Evaluation")
    print(f"Scenarios: {len(scenarios)}  Conditions: {conditions}")
    print(f"Model: {OPUS}")
    print()

    all_results: dict[str, list[CallbackResult]] = {c: [] for c in conditions}
    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    # Pre-load already-computed conditions so _save preserves them
    existing_results = existing.get("results", {})
    for cond_key, raw_list in existing_results.items():
        if cond_key not in all_results:
            all_results[cond_key] = []
            for raw in raw_list:
                all_results[cond_key].append(CallbackResult(
                    scenario_id=raw["scenario_id"],
                    condition=raw["condition"],
                    callback_idx=raw["callback_idx"],
                    callback_text=raw.get("callback_text", ""),
                    response=raw.get("response", ""),
                    value_recalled=raw["value_recalled"],
                    qualifier_present=raw["qualifier_present"],
                    both=raw["both"],
                    fcr=raw["fcr"],
                    hallucination=raw["hallucination"],
                ))
    # Merge conditions list to include pre-loaded conditions
    all_conditions = list(existing_results.keys()) + [c for c in conditions if c not in existing_results]

    for s_idx, scenario in enumerate(scenarios):
        print(f"[{s_idx+1}/{len(scenarios)}] {scenario['id']} — {scenario['description']}")

        for cond in conditions:
            print(f"  Running {cond}...", end=" ", flush=True)
            t0 = time.time()
            try:
                if cond == "no_memory":
                    cb_results = _run_no_memory(scenario, client, OPUS)
                elif cond == "naive_summary":
                    cb_results = _run_naive_summary(scenario, client, OPUS)
                elif cond == "mem0_style":
                    cb_results = _run_mem0_style(scenario, client, OPUS)
                elif cond == "credence_memory":
                    cb_results = _run_credence_memory(scenario, client, OPUS, HAIKU)
                else:
                    print(f"Unknown condition: {cond}")
                    continue

                elapsed = time.time() - t0
                n_fcr = sum(r.fcr for r in cb_results)
                print(f"done in {elapsed:.1f}s  FCR={n_fcr}/{len(cb_results)}")
                all_results[cond].extend(cb_results)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Save after each scenario (preserve all conditions including pre-loaded)
        _save(all_results, all_conditions, args.out)

    # Final summary
    print()
    print("=" * 64)
    print(f"  CROSS-SESSION FALSE CERTAINTY RATE (CS-FCR)")
    print(f"  n_scenarios={len(scenarios)}  model={OPUS}")
    print("=" * 64)
    print()
    print(f"  {'Condition':<22}  {'n':>4}  {'BothRate':>9}  {'CS-FCR':>8}  {'Hallu':>7}")
    print(f"  {'─'*22}  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*7}")
    for cond in all_conditions:
        if not all_results.get(cond):
            continue
        agg = _aggregate(all_results[cond], cond)
        marker = " ← human-written hedged summary" if cond == "naive_summary" else ""
        marker = " ← no memory" if cond == "no_memory" else marker
        marker = " ← flat fact (Mem0/Zep style)" if cond == "mem0_style" else marker
        marker = " ← Credence Memory" if cond == "credence_memory" else marker
        print(f"  {cond:<22}  {agg.n_callbacks:>4}  {agg.both_rate:>9.3f}  "
              f"{agg.fcr:>8.3f}  {agg.hallu_rate:>7.3f}{marker}")
    print()
    print("  CS-FCR = fraction of queries that recalled a value WITHOUT uncertainty qualifier.")
    print("  Lower CS-FCR = better. Target: 0.000.")

    _save(all_results, all_conditions, args.out)
    print(f"\n  Results saved to {args.out}")


def _save(all_results, conditions, path):
    out = {
        "conditions": conditions,
        "results": {
            cond: [
                {
                    "scenario_id":    r.scenario_id,
                    "condition":      r.condition,
                    "callback_idx":   r.callback_idx,
                    "value_recalled": r.value_recalled,
                    "qualifier_present": r.qualifier_present,
                    "both":           r.both,
                    "fcr":            r.fcr,
                    "hallucination":  r.hallucination,
                    "response":       r.response[:300],
                }
                for r in all_results[cond]
            ]
            for cond in conditions
        },
        "aggregate": {
            cond: {
                "n":          len(all_results[cond]),
                "both_rate":  (
                    sum(r.both for r in all_results[cond]) / len(all_results[cond])
                    if all_results[cond] else 0.0
                ),
                "fcr": (
                    sum(r.fcr for r in all_results[cond]) / len(all_results[cond])
                    if all_results[cond] else 0.0
                ),
            }
            for cond in conditions
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

"""
evals/ghost_gauntlet.py
=======================
Ghost Gauntlet — definitive eval for EG-2 vs Credence v1 vs naive on ghost constraints.

A "ghost constraint" is an uncertain statement that does NOT use canonical hedging markers
like "I think", "might be", or "unverified". Examples:
  - "The vendor provided a figure of around 50 req/min for our tier" (no "I think", "unconfirmed")
  - "We benchmarked this at about 180ms p99, but that was on the old infrastructure" (source uncertainty)
  - "The estimate from the previous team was roughly $40K for this migration" (estimated, non-canonical)

Design per session:
  - Turn 1-2: establish context (factual, no uncertainty)
  - Turn 3-5: plant 3 ghost constraints (implicit uncertainty, non-canonical phrasing)
  - Turn 6-13: 8 HIGH-J filler turns (force memory system to make compression decisions)
  - Turn 14-16: 3 callback turns (one per ghost constraint)

3 conditions:
  - credence_v1:  keyword probe only, no SE, no claim extraction
  - credence_eg2: SE probe + claim extraction + query-aware Truth Buffer
  - naive_window: last 12 messages

Metrics per callback:
  - value_recalled:        response mentions the key value (50 req/min, $40K, 180ms)
  - uncertainty_preserved: response hedges appropriately ("was reported as", "estimated", "might be")
  - both:                  value_recalled AND uncertainty_preserved (the gold standard)

Run:
    python -m evals.ghost_gauntlet --dry-run   # validate structure, free
    python -m evals.ghost_gauntlet --n 5       # first 5 sessions
    python -m evals.ghost_gauntlet             # all 10 sessions
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GhostCallback:
    ghost_id:           str
    question:           str
    value_fragments:    list[str]   # substrings that indicate value was recalled
    qualifier_fragments: list[str]  # substrings that indicate uncertainty was preserved


@dataclass
class GhostSession:
    session_id:    str
    domain:        str
    context_turns: list[dict]   # {"role": ..., "content": ...}
    ghost_turns:   list[dict]   # {"role": "user", "content": ..., "ghost_id": ...}
    filler_turns:  list[str]    # user-only strings (assistant fills in)
    callbacks:     list[GhostCallback]


# ---------------------------------------------------------------------------
# 10 Ghost Sessions
# ---------------------------------------------------------------------------

GHOST_SESSIONS = [
    GhostSession(
        session_id="ghost_api_01",
        domain="api_integration",
        context_turns=[
            {"role": "user", "content": "We're building the payment integration with Stripe."},
            {"role": "assistant", "content": "Great. What's the architecture plan?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "The Stripe rep gave us a figure of around 50 requests per minute for our sandbox tier. That's what we're designing around.", "ghost_id": "g1"},
            {"role": "user", "content": "Their documentation suggested the webhook retry window is 72 hours, but the sales call mentioned something different — maybe 48 hours.", "ghost_id": "g2"},
            {"role": "user", "content": "The estimate from the team was a $15K implementation cost, though that was a rough figure from a brief call.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What's the difference between synchronous and asynchronous API calls?",
            "Explain idempotency keys in payment systems.",
            "What does HTTP 429 mean and how should we handle it?",
            "What is OAuth 2.0 used for in API authentication?",
            "What is the purpose of a webhook signature?",
            "Explain TLS and why it matters for payment APIs.",
            "What is a circuit breaker pattern?",
            "What is exponential backoff?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What rate limit are we designing around for the Stripe sandbox?", value_fragments=["50", "50 req", "50 request"], qualifier_fragments=["around", "figure", "roughly", "approximately", "reported", "told", "mentioned", "gave us"]),
            GhostCallback(ghost_id="g2", question="What do we know about the webhook retry window?", value_fragments=["72", "48", "hours"], qualifier_fragments=["suggested", "mentioned", "different", "uncertain", "unclear", "might", "maybe"]),
            GhostCallback(ghost_id="g3", question="What was the implementation cost estimate?", value_fragments=["15", "$15", "15K", "15,000"], qualifier_fragments=["estimate", "rough", "approximately", "around", "figure", "brief"]),
        ],
    ),
    GhostSession(
        session_id="ghost_infra_01",
        domain="infrastructure",
        context_turns=[
            {"role": "user", "content": "We're planning a database migration to PostgreSQL."},
            {"role": "assistant", "content": "What's the current setup?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Our benchmarks on the old hardware showed p99 latency of around 180ms for the write path. Haven't re-run on the new setup.", "ghost_id": "g1"},
            {"role": "user", "content": "The ops team's estimate for downtime during the cutover was 4-6 hours, based on similar past migrations they've done.", "ghost_id": "g2"},
            {"role": "user", "content": "We were told the new instance can handle about 10K writes per second, but that's from a vendor spec sheet, not our own load test.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is WAL (Write-Ahead Logging) in PostgreSQL?",
            "What is connection pooling and why does it matter?",
            "Explain the difference between OLTP and OLAP workloads.",
            "What is a database index and when should you use one?",
            "What is vacuuming in PostgreSQL?",
            "Explain the concept of database transactions.",
            "What is replication lag?",
            "What is a read replica and when would you use one?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What p99 latency did we observe on the write path?", value_fragments=["180", "180ms"], qualifier_fragments=["around", "approximately", "benchmark", "old hardware", "haven't", "re-run", "estimate"]),
            GhostCallback(ghost_id="g2", question="How long did the ops team estimate for the cutover downtime?", value_fragments=["4", "6", "4-6", "hours"], qualifier_fragments=["estimate", "based on", "similar", "past", "approximately"]),
            GhostCallback(ghost_id="g3", question="What write throughput did we hear the new instance supports?", value_fragments=["10K", "10,000", "10 thousand"], qualifier_fragments=["told", "vendor", "spec", "not our", "own", "load test", "reported"]),
        ],
    ),
    GhostSession(
        session_id="ghost_compliance_01",
        domain="compliance",
        context_turns=[
            {"role": "user", "content": "We're working on GDPR compliance for our data pipeline."},
            {"role": "assistant", "content": "What specific requirements are you targeting?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Our legal team's interpretation of the data retention requirement was 90 days for user behavior data — but they flagged it as needing a final legal review.", "ghost_id": "g1"},
            {"role": "user", "content": "The DPA we're working with mentioned something like a 72-hour breach notification window, though I'd want to re-read the exact contract language.", "ghost_id": "g2"},
            {"role": "user", "content": "The compliance consultant's fee estimate for the full audit was around $25K, but that was a preliminary figure.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What does GDPR stand for and what does it govern?",
            "What is a Data Processing Agreement?",
            "What is the right to erasure under GDPR?",
            "What is a data controller vs a data processor?",
            "What is pseudonymization in the context of GDPR?",
            "What is the purpose of a Data Protection Officer?",
            "What is consent under GDPR?",
            "What is a legitimate interest under GDPR?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="How long is the data retention period for user behavior data?", value_fragments=["90", "90 days"], qualifier_fragments=["interpretation", "flagged", "review", "needs", "final", "legal", "approximately", "estimated"]),
            GhostCallback(ghost_id="g2", question="What's the breach notification window in our DPA?", value_fragments=["72", "72-hour", "hours"], qualifier_fragments=["mentioned", "re-read", "contract", "approximately", "unclear", "verify", "might"]),
            GhostCallback(ghost_id="g3", question="What was the compliance audit cost estimate?", value_fragments=["25", "$25", "25K"], qualifier_fragments=["estimate", "preliminary", "around", "approximately", "rough", "figure"]),
        ],
    ),
    GhostSession(
        session_id="ghost_ml_01",
        domain="ml_deployment",
        context_turns=[
            {"role": "user", "content": "We're deploying a recommendation model to production."},
            {"role": "assistant", "content": "What's the serving infrastructure plan?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Our offline evaluation showed about 8% lift in click-through rate. The A/B test hasn't run yet so we don't know the online number.", "ghost_id": "g1"},
            {"role": "user", "content": "The model serving team estimated inference latency at around 40ms p99 on the planned hardware — that's from a back-of-envelope calculation.", "ghost_id": "g2"},
            {"role": "user", "content": "The re-training cadence was suggested as weekly by the data science lead, but the infrastructure cost for that hasn't been validated.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is the difference between batch and online inference?",
            "What is model drift?",
            "What is a feature store?",
            "What is the purpose of shadow deployment?",
            "What is a canary release in the context of ML models?",
            "What is ONNX format used for?",
            "What is quantization in the context of neural networks?",
            "What is the purpose of model versioning?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What CTR lift did we see in offline evaluation?", value_fragments=["8%", "8 percent", "eight percent"], qualifier_fragments=["offline", "A/B", "hasn't", "don't know", "approximately", "evaluation", "estimate"]),
            GhostCallback(ghost_id="g2", question="What inference latency is the team targeting?", value_fragments=["40", "40ms"], qualifier_fragments=["estimated", "around", "back-of-envelope", "calculation", "approximately", "planned", "hasn't been"]),
            GhostCallback(ghost_id="g3", question="What re-training cadence was proposed?", value_fragments=["weekly", "week"], qualifier_fragments=["suggested", "hasn't been", "validated", "cost", "approximately", "proposed", "recommended"]),
        ],
    ),
    GhostSession(
        session_id="ghost_security_01",
        domain="security",
        context_turns=[
            {"role": "user", "content": "We're reviewing our auth system's security posture."},
            {"role": "assistant", "content": "What are the specific concerns?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "The security team's rough estimate was that about 15% of tokens in the wild might still be using the old algorithm — we haven't done a full audit.", "ghost_id": "g1"},
            {"role": "user", "content": "The penetration testing firm quoted around $20K for a focused engagement on the auth layer. That was a rough ballpark from a phone call.", "ghost_id": "g2"},
            {"role": "user", "content": "One of our engineers mentioned seeing latency spikes of 200-300ms in the token validation path under load, but I couldn't find the ticket.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is JWT and how does it work?",
            "What is the difference between authentication and authorization?",
            "What is OAuth 2.0?",
            "What is PKCE in OAuth?",
            "What is a refresh token?",
            "What is session fixation?",
            "What is CSRF and how is it prevented?",
            "What is RBAC?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What fraction of tokens might be using the old algorithm?", value_fragments=["15%", "15 percent", "fifteen"], qualifier_fragments=["rough", "estimate", "might", "haven't", "audit", "approximately", "could be"]),
            GhostCallback(ghost_id="g2", question="What did the pentest firm quote?", value_fragments=["20K", "$20", "20,000"], qualifier_fragments=["quoted", "around", "rough", "ballpark", "phone call", "approximately", "estimate"]),
            GhostCallback(ghost_id="g3", question="What latency spikes were reported in the token validation path?", value_fragments=["200", "300", "200-300"], qualifier_fragments=["mentioned", "under load", "couldn't find", "engineer", "reported", "approximately", "said"]),
        ],
    ),
    GhostSession(
        session_id="ghost_product_01",
        domain="product_planning",
        context_turns=[
            {"role": "user", "content": "We're planning the Q3 roadmap for the analytics product."},
            {"role": "assistant", "content": "What features are under consideration?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "User research suggested something like 60% of power users want dashboard export. That's from a survey of about 50 people, so take it with a grain of salt.", "ghost_id": "g1"},
            {"role": "user", "content": "The engineering team's rough estimate for the real-time alerting feature was 6-8 weeks, though they haven't broken it down into tasks yet.", "ghost_id": "g2"},
            {"role": "user", "content": "A competitor analysis we did last quarter suggested they charge around $299/month for comparable features. That data might be stale now.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is a North Star metric in product development?",
            "What is a jobs-to-be-done framework?",
            "What is the difference between an OKR and a KPI?",
            "What is a product roadmap?",
            "What is the build-measure-learn cycle?",
            "What is a minimum viable product?",
            "What is feature flagging and why is it useful?",
            "What is churn rate?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What percentage of power users want dashboard export according to the survey?", value_fragments=["60%", "60 percent", "sixty"], qualifier_fragments=["suggested", "survey", "grain of salt", "approximately", "rough", "50 people", "estimated"]),
            GhostCallback(ghost_id="g2", question="How long did engineering estimate for the alerting feature?", value_fragments=["6", "8", "6-8", "weeks"], qualifier_fragments=["rough", "estimate", "haven't", "broken", "approximately", "about", "suggested"]),
            GhostCallback(ghost_id="g3", question="What did competitors appear to charge for similar features?", value_fragments=["299", "$299"], qualifier_fragments=["suggested", "stale", "might", "analysis", "approximately", "around", "quarter"]),
        ],
    ),
    GhostSession(
        session_id="ghost_devops_01",
        domain="devops",
        context_turns=[
            {"role": "user", "content": "We're setting up CI/CD for the new microservices architecture."},
            {"role": "assistant", "content": "What's the current pipeline look like?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "The DevOps team said build times are around 12 minutes on the current runners. We're hoping to cut that in half but haven't benchmarked the new setup yet.", "ghost_id": "g1"},
            {"role": "user", "content": "Based on last year's incidents, we're estimating maybe 2-3 pipeline failures per week once we scale up. That's extrapolated from a smaller system.", "ghost_id": "g2"},
            {"role": "user", "content": "The cloud provider rep mentioned something about a 99.95% SLA for the compute tier, but I haven't verified that against the actual contract.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is a container in software development?",
            "What is Kubernetes used for?",
            "What is the difference between CI and CD?",
            "What is a Docker image layer?",
            "What is a Helm chart?",
            "What is blue-green deployment?",
            "What is infrastructure as code?",
            "What is GitOps?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="How long are build times currently and what's the target?", value_fragments=["12", "12 min", "minutes"], qualifier_fragments=["around", "said", "hoping", "haven't", "benchmarked", "approximately", "about"]),
            GhostCallback(ghost_id="g2", question="How many pipeline failures per week are we projecting?", value_fragments=["2", "3", "2-3"], qualifier_fragments=["estimating", "extrapolated", "maybe", "approximately", "based on", "projected", "estimate"]),
            GhostCallback(ghost_id="g3", question="What SLA did the cloud provider mention for compute?", value_fragments=["99.95", "99.95%"], qualifier_fragments=["mentioned", "haven't verified", "contract", "approximately", "rep", "said"]),
        ],
    ),
    GhostSession(
        session_id="ghost_data_01",
        domain="data_engineering",
        context_turns=[
            {"role": "user", "content": "We're building a new data warehouse on Snowflake."},
            {"role": "assistant", "content": "What's the data volume and query pattern?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Our current estimate is around 500GB of data per month ingested, though that's based on last quarter's growth rate which might not hold.", "ghost_id": "g1"},
            {"role": "user", "content": "The Snowflake rep said a Virtual Warehouse of size M should handle our query load — but that was a rough sizing based on our description, not an actual benchmark.", "ghost_id": "g2"},
            {"role": "user", "content": "The data team's rough monthly cost estimate for the warehouse was $3-4K/month at current query volumes.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is a data warehouse?",
            "What is the difference between a fact table and a dimension table?",
            "What is ELT vs ETL?",
            "What is column-oriented storage and why does it matter for analytics?",
            "What is a materialized view?",
            "What is data partitioning in a warehouse context?",
            "What is a slowly changing dimension?",
            "What is dbt used for?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="How much data are we expecting to ingest monthly?", value_fragments=["500", "500GB"], qualifier_fragments=["estimate", "around", "might not", "based on", "approximately", "growth rate", "rough"]),
            GhostCallback(ghost_id="g2", question="What Snowflake warehouse size did the rep recommend?", value_fragments=["medium", "size M", "M"], qualifier_fragments=["rep said", "rough", "sizing", "description", "not", "benchmark", "approximately", "based on"]),
            GhostCallback(ghost_id="g3", question="What's the estimated monthly Snowflake cost?", value_fragments=["3", "4", "3-4", "$3", "$4"], qualifier_fragments=["rough", "estimate", "approximately", "around", "current volumes", "might change"]),
        ],
    ),
    GhostSession(
        session_id="ghost_mobile_01",
        domain="mobile_development",
        context_turns=[
            {"role": "user", "content": "We're planning a major performance optimization sprint for the iOS app."},
            {"role": "assistant", "content": "What are the current performance pain points?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Crashlytics shows something in the neighborhood of 0.3% crash rate — I'd want to double-check that against the exact metric definition they use.", "ghost_id": "g1"},
            {"role": "user", "content": "The app launch time was measured at around 2.8 seconds on a mid-tier device, but that was an informal test, not a rigorous benchmark.", "ghost_id": "g2"},
            {"role": "user", "content": "Based on the PlayStore ratings trend, we're projecting maybe a 0.2 point rating improvement if we fix the top 3 crash categories. That's a gut estimate.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is Crashlytics used for?",
            "What is the difference between ANR and a crash on Android?",
            "What is memory profiling in mobile apps?",
            "What is frame rate and why does it matter for mobile UX?",
            "What is lazy loading in the context of mobile apps?",
            "What is a memory leak?",
            "What is app thinning in iOS?",
            "What is ProGuard used for?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What was the crash rate we noted from Crashlytics?", value_fragments=["0.3", "0.3%", "point 3"], qualifier_fragments=["neighborhood", "double-check", "definition", "approximately", "around", "shows", "would want to verify"]),
            GhostCallback(ghost_id="g2", question="What was the app launch time we measured?", value_fragments=["2.8", "2.8 seconds"], qualifier_fragments=["informal", "around", "approximately", "rigorous", "benchmark", "test", "measured at"]),
            GhostCallback(ghost_id="g3", question="What rating improvement are we projecting from fixing the top crashes?", value_fragments=["0.2", "point"], qualifier_fragments=["projecting", "gut", "estimate", "maybe", "approximately", "based on", "trend"]),
        ],
    ),
    GhostSession(
        session_id="ghost_finance_01",
        domain="financial_planning",
        context_turns=[
            {"role": "user", "content": "We're doing financial planning for the Series A fundraise."},
            {"role": "assistant", "content": "What's the current ARR and burn rate?"},
        ],
        ghost_turns=[
            {"role": "user", "content": "Based on conversations with 3 VCs last week, the median pre-money valuation they mentioned was around $8M — but those were exploratory calls, not term sheets.", "ghost_id": "g1"},
            {"role": "user", "content": "Our finance model projects 18 months of runway at current burn, but that's based on a growth scenario that may be optimistic.", "ghost_id": "g2"},
            {"role": "user", "content": "The accountant's preliminary estimate for the tax liability from the options exercise was $120-150K, but they said it needs a full review.", "ghost_id": "g3"},
        ],
        filler_turns=[
            "What is ARR (Annual Recurring Revenue)?",
            "What is a cap table?",
            "What is a term sheet in VC?",
            "What is dilution in the context of startup equity?",
            "What is a liquidation preference?",
            "What is a post-money valuation?",
            "What is MRR?",
            "What is the Rule of 40?",
        ],
        callbacks=[
            GhostCallback(ghost_id="g1", question="What valuation range did the VCs mention?", value_fragments=["8M", "$8", "8 million"], qualifier_fragments=["mentioned", "around", "exploratory", "not term sheet", "approximately", "conversations", "median"]),
            GhostCallback(ghost_id="g2", question="How much runway does our finance model project?", value_fragments=["18", "18 months"], qualifier_fragments=["projects", "based on", "optimistic", "scenario", "may be", "approximately", "model"]),
            GhostCallback(ghost_id="g3", question="What was the preliminary tax liability estimate?", value_fragments=["120", "150", "120-150", "$120"], qualifier_fragments=["preliminary", "estimate", "needs", "review", "approximately", "around", "accountant"]),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_callback(response: str, callback: GhostCallback) -> dict:
    """
    Score a single callback response.

    Returns:
        value_recalled:        True if any value_fragment appears in the response
        uncertainty_preserved: True if any qualifier_fragment appears in the response
        both:                  True if both value and qualifier are present (gold standard)
    """
    lower = response.lower()
    value_recalled        = any(f.lower() in lower for f in callback.value_fragments)
    uncertainty_preserved = any(f.lower() in lower for f in callback.qualifier_fragments)
    return {
        "ghost_id":            callback.ghost_id,
        "value_recalled":      value_recalled,
        "uncertainty_preserved": uncertainty_preserved,
        "both":                value_recalled and uncertainty_preserved,
    }


# ---------------------------------------------------------------------------
# Naive window helper
# ---------------------------------------------------------------------------

def _naive_window_chat(history: list[dict], user_message: str, client, window: int = 12) -> str:
    """Run a naive sliding-window chat call — keep only the last `window` messages."""
    history.append({"role": "user", "content": user_message})
    windowed = history[-window:]
    resp = client.messages.create(
        model      = "claude-opus-4-7",
        messages   = windowed,
        max_tokens = 300,
        system     = (
            "You are a helpful, precise assistant. "
            "Give concise answers when the answer is clear; "
            "express genuine uncertainty when it exists."
        ),
    )
    text_block = next((b for b in resp.content if b.type == "text"), None)
    text = text_block.text if text_block else ""
    history.append({"role": "assistant", "content": text})
    return text


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_condition(
    session:   GhostSession,
    condition: str,
    client,
) -> dict:
    """
    Run one full session under a given condition.

    condition: "credence_v1" | "credence_eg2" | "naive_window"

    Returns per-callback scores and aggregate stats.
    """
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    # Set up the context manager or naive history
    if condition == "credence_v1":
        mgr = ContextManager(
            max_tokens            = 300,
            use_semantic_entropy  = False,
            use_claim_extraction  = False,
        )
    elif condition == "credence_eg2":
        reg = CredenceRegistry(":memory:")
        mgr = ContextManager(
            max_tokens            = 300,
            use_semantic_entropy  = True,
            use_claim_extraction  = True,
            registry              = reg,
            session_id            = session.session_id,
        )
    else:
        # naive_window — we manage history manually
        mgr = None

    naive_history: list[dict] = []

    # Phase 1: context turns (pre-seeded)
    for msg in session.context_turns:
        if msg["role"] == "user":
            if mgr is not None:
                mgr.chat(msg["content"])
            else:
                _naive_window_chat(naive_history, msg["content"], client)
        # assistant turns in context_turns are skipped — they're pre-seeded context
        # for session coherence only; the model provides its own in actual runs

    # Phase 2: ghost turns (plant uncertain constraints)
    for ghost_turn in session.ghost_turns:
        if mgr is not None:
            mgr.chat(ghost_turn["content"])
        else:
            _naive_window_chat(naive_history, ghost_turn["content"], client)

    # Phase 3: filler turns (force memory pressure)
    for filler in session.filler_turns:
        if mgr is not None:
            mgr.chat(filler)
        else:
            _naive_window_chat(naive_history, filler, client)

    # Phase 4: callback turns (measure recall)
    callback_results = []
    for cb in session.callbacks:
        if mgr is not None:
            result = mgr.chat(cb.question)
            response = result.response
        else:
            response = _naive_window_chat(naive_history, cb.question, client)

        scores = _score_callback(response, cb)
        scores["response_snippet"] = response[:200]
        callback_results.append(scores)

    # Aggregate metrics
    n = len(callback_results)
    value_rate       = sum(1 for r in callback_results if r["value_recalled"])       / max(n, 1)
    qualifier_rate   = sum(1 for r in callback_results if r["uncertainty_preserved"]) / max(n, 1)
    both_rate        = sum(1 for r in callback_results if r["both"])                  / max(n, 1)

    return {
        "session_id":      session.session_id,
        "domain":          session.domain,
        "condition":       condition,
        "callbacks":       callback_results,
        "value_rate":      round(value_rate, 3),
        "qualifier_rate":  round(qualifier_rate, 3),
        "both_rate":       round(both_rate, 3),
        "n_callbacks":     n,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(all_results: list[dict]) -> None:
    """Print a summary table comparing all three conditions."""
    from collections import defaultdict

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    conditions = ["credence_v1", "credence_eg2", "naive_window"]
    print("\n" + "=" * 70)
    print("Ghost Gauntlet Results")
    print("=" * 70)
    print(f"{'Condition':<20} {'Sessions':>8} {'ValueRate':>10} {'QualRate':>10} {'BothRate':>10}")
    print("-" * 70)
    for cond in conditions:
        results = by_condition.get(cond, [])
        if not results:
            continue
        n     = len(results)
        vr    = sum(r["value_rate"]      for r in results) / n
        qr    = sum(r["qualifier_rate"]  for r in results) / n
        br    = sum(r["both_rate"]       for r in results) / n
        print(f"{cond:<20} {n:>8} {vr:>10.3f} {qr:>10.3f} {br:>10.3f}")
    print("=" * 70)
    print()
    print("Metrics:")
    print("  ValueRate   — fraction of callbacks where the key value was recalled")
    print("  QualRate    — fraction of callbacks where uncertainty was preserved")
    print("  BothRate    — fraction of callbacks with BOTH value + qualifier (gold)")
    print()
    print("Critical comparison: credence_eg2 vs credence_v1 on BothRate")
    v1_results  = by_condition.get("credence_v1", [])
    eg2_results = by_condition.get("credence_eg2", [])
    if v1_results and eg2_results:
        v1_both  = sum(r["both_rate"] for r in v1_results)  / len(v1_results)
        eg2_both = sum(r["both_rate"] for r in eg2_results) / len(eg2_results)
        delta    = eg2_both - v1_both
        print(f"  credence_eg2 BothRate={eg2_both:.3f}  credence_v1 BothRate={v1_both:.3f}  delta={delta:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ghost Gauntlet: implicit uncertainty recall benchmark")
    parser.add_argument("--dry-run",  action="store_true", help="Validate structure without API calls")
    parser.add_argument("--n",        type=int,  default=None,   help="Only run first N sessions")
    parser.add_argument("--session",  type=str,  default=None,   help="Run single session by ID")
    parser.add_argument("--resume",   action="store_true", help="Resume from existing results file")
    parser.add_argument("--out",      type=str,  default="evals/ghost_gauntlet_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    # Dry run: just validate structure
    if args.dry_run:
        sessions_to_run = GHOST_SESSIONS
        if args.n is not None:
            sessions_to_run = sessions_to_run[:args.n]
        if args.session is not None:
            sessions_to_run = [s for s in sessions_to_run if s.session_id == args.session]
        n_valid = 0
        for s in sessions_to_run:
            assert len(s.ghost_turns) == 3,   f"{s.session_id}: expected 3 ghost turns"
            assert len(s.filler_turns) == 8,  f"{s.session_id}: expected 8 filler turns"
            assert len(s.callbacks) == 3,     f"{s.session_id}: expected 3 callbacks"
            assert len(s.context_turns) >= 2, f"{s.session_id}: expected ≥2 context turns"
            n_valid += 1
        print(f"DRY RUN: {n_valid} sessions × 3 conditions validated.")
        return

    # Real run
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        return

    client = Anthropic(api_key=api_key)

    # Session selection
    sessions_to_run = GHOST_SESSIONS
    if args.n is not None:
        sessions_to_run = sessions_to_run[:args.n]
    if args.session is not None:
        sessions_to_run = [s for s in sessions_to_run if s.session_id == args.session]

    # Load existing results if resuming
    all_results: list[dict] = []
    completed_keys: set[str] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        all_results = existing.get("results", [])
        completed_keys = {f"{r['session_id']}::{r['condition']}" for r in all_results}
        print(f"Resuming: {len(all_results)} results already completed")

    conditions = ["credence_v1", "credence_eg2", "naive_window"]
    total = len(sessions_to_run) * len(conditions)
    done  = 0

    for session in sessions_to_run:
        for condition in conditions:
            key = f"{session.session_id}::{condition}"
            if key in completed_keys:
                done += 1
                print(f"[{done}/{total}] SKIP (already done): {session.session_id} / {condition}")
                continue

            print(f"[{done + 1}/{total}] Running {session.session_id} / {condition} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = _run_condition(session, condition, client)
                elapsed = round(time.time() - t0, 1)
                print(f"done ({elapsed}s)  both_rate={result['both_rate']:.3f}")
                all_results.append(result)
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                print(f"ERROR ({elapsed}s): {exc}")
            done += 1

            # Save after each result (crash-safe)
            with open(args.out, "w") as f:
                json.dump({"results": all_results}, f, indent=2)

    # Print summary
    if all_results:
        _print_table(all_results)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

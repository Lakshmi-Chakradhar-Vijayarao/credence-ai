"""
evals/gauntlet.py
=================
Credence Gauntlet — 50 scenarios across 5 domains.

Each scenario: 2-4 seed turns planting 1-3 uncertain constraints,
6-10 filler turns forcing naive window to discard seeds,
2-3 callback questions testing constraint recall.

Three conditions per scenario:
  credence      — ContextManager with faithfulness probe + J-selective ops
  naive_window  — last-N message sliding window
  baseline      — full context (oracle upper bound)

Usage:
    python -m evals.gauntlet                          # all 50 scenarios
    python -m evals.gauntlet --domain api             # 10 API scenarios only
    python -m evals.gauntlet --n 10                   # first 10 scenarios
    python -m evals.gauntlet --scenario S07           # single scenario
    python -m evals.gauntlet --dry-run                # validate structure, no API

Output:
    evals/gauntlet_results.json    per-scenario + aggregate statistics

Domains:
    api        — API integration, rate limits, auth tokens
    debug      — Debugging sessions, error hypotheses
    design     — System design, ambiguous requirements
    compliance — Security, compliance, medical/legal-adjacent
    multiagent — Multi-agent pipelines, trust propagation

Metrics (per callback, averaged over scenario):
    constraint_recall      — fraction of correct_frags found in answer
    qualifier_preserved    — answer includes explicit uncertainty marker
    hallucination_flag     — answer contains known-wrong values
"""

import os, sys, json, time, argparse, random, re
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from credence.context_manager import ContextManager

# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id:          str
    domain:      str
    description: str
    system:      str
    seeds:       list[dict]   # [{"role": ..., "content": ...}]
    fillers:     list[str]    # user questions for filler turns
    callbacks:   list[dict]   # [{question, correct_frags, wrong_frags, planted_value}]


# ---------------------------------------------------------------------------
# 50 Scenarios
# ---------------------------------------------------------------------------

_SYSTEM_API = (
    "You are a precise technical assistant helping plan a software integration. "
    "When asked to recall earlier facts, state them exactly as established, "
    "including any uncertainty flags."
)

_SYSTEM_DEBUG = (
    "You are a senior engineer helping debug a production issue. "
    "Be precise about what is confirmed vs. hypothesised."
)

_SYSTEM_DESIGN = (
    "You are a systems architect helping design a new platform. "
    "Track all requirements carefully, including those that are still uncertain."
)

_SYSTEM_COMPLIANCE = (
    "You are a technical compliance advisor. "
    "Always flag when a constraint has not been legally confirmed."
)

_SYSTEM_MULTIAGENT = (
    "You are coordinating a multi-agent pipeline. "
    "Track which facts each downstream agent has confirmed vs. assumed."
)

_FILLERS_API = [
    "What is exponential backoff and when should I use it?",
    "What HTTP status codes indicate rate limiting?",
    "What is the difference between access tokens and refresh tokens?",
    "What is connection pooling and why does it matter?",
    "How should I handle idempotency in retry logic?",
    "What is the Retry-After header?",
    "What is a circuit breaker pattern?",
    "How do I implement token bucket rate limiting?",
]

_FILLERS_DEBUG = [
    "What is a stack overflow error and how do I trace it?",
    "What does a 502 Bad Gateway error mean?",
    "How do I interpret a core dump?",
    "What is the difference between a deadlock and a livelock?",
    "How do I profile memory usage in Python?",
    "What tools can I use to trace slow SQL queries?",
    "What is a race condition and how do I reproduce it?",
    "What does the SIGTERM signal do?",
]

_FILLERS_DESIGN = [
    "What is eventual consistency?",
    "What is the CAP theorem?",
    "When should I use a message queue vs. direct HTTP calls?",
    "What is the difference between CQRS and event sourcing?",
    "What is a saga pattern for distributed transactions?",
    "How do I design an idempotent API?",
    "What is a bulkhead pattern?",
    "When should I use a monolith vs. microservices?",
]

_FILLERS_COMPLIANCE = [
    "What is GDPR and who does it apply to?",
    "What is the difference between PII and PHI?",
    "What is data minimization under privacy law?",
    "What is SOC 2 Type II compliance?",
    "What is the right to erasure?",
    "What is a data processing agreement?",
    "What is end-to-end encryption?",
    "What is zero-trust architecture?",
]

_FILLERS_MULTIAGENT = [
    "What is tool use in multi-agent systems?",
    "How do agent handoffs work in LangGraph?",
    "What is the difference between a supervisor and a worker agent?",
    "How do I handle agent failures gracefully?",
    "What is prompt injection in agentic systems?",
    "How do I pass state between agents?",
    "What is a ReAct agent?",
    "How do I prevent infinite loops in agent chains?",
]

# fmt: off
SCENARIOS: list[Scenario] = [

    # =========================================================
    # DOMAIN 1: API Integration (S01–S10)
    # =========================================================

    Scenario(
        id="S01", domain="api",
        description="Rate limit uncertain, token expiry uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "I'm building a payment integration with Stripe."},
            {"role": "assistant", "content": "Happy to help. What aspects are you working on?"},
            {"role": "user",      "content": "I'm not sure, but I think our rate limit is around 50 req/min. I'll need to confirm with the vendor."},
            {"role": "assistant", "content": "Noted — rate limit is approximately 50 req/min, flagged as unconfirmed."},
            {"role": "user",      "content": "Auth tokens might expire in 3600 seconds, but the docs are unclear — could be 86400."},
            {"role": "assistant", "content": "Understood — token expiry is approximately 3600s, tentative pending documentation clarification."},
        ],
        fillers=_FILLERS_API,
        callbacks=[
            {"question": "What rate limit should the retry logic target?",
             "correct_frags": ["50", "uncertain", "unconfirmed", "verify", "tentative", "roughly", "approximately"],
             "wrong_frags": ["100", "200", "1000"],
             "planted_value": "50"},
            {"question": "How long should we set token expiry in the implementation?",
             "correct_frags": ["3600", "uncertain", "tentative", "unclear", "verify", "approximately", "might"],
             "wrong_frags": ["7200", "1800", "300"],
             "planted_value": "3600"},
        ],
    ),

    Scenario(
        id="S02", domain="api",
        description="Webhook retry count and timeout uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're building a webhook receiver for payment events."},
            {"role": "assistant", "content": "Got it. What do you need to plan?"},
            {"role": "user",      "content": "I believe the vendor retries up to 5 times, but I haven't checked the SLA."},
            {"role": "assistant", "content": "Noted — retry count is approximately 5, flagged as unverified per vendor SLA."},
            {"role": "user",      "content": "The timeout might be 30 seconds — I'm not certain."},
            {"role": "assistant", "content": "Understood — timeout is approximately 30s, tentative."},
        ],
        fillers=_FILLERS_API,
        callbacks=[
            {"question": "How many retries should our idempotency logic handle?",
             "correct_frags": ["5", "uncertain", "unverified", "check", "verify", "approximately"],
             "wrong_frags": ["10", "20", "1"],
             "planted_value": "5"},
            {"question": "What timeout should the webhook handler implement?",
             "correct_frags": ["30", "uncertain", "tentative", "verify", "approximately", "might"],
             "wrong_frags": ["60", "10", "5"],
             "planted_value": "30"},
        ],
    ),

    Scenario(
        id="S03", domain="api",
        description="API pagination page size uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "I'm integrating with a third-party reporting API."},
            {"role": "assistant", "content": "What do you need to handle?"},
            {"role": "user",      "content": "I think pagination returns 100 items per page, but I haven't verified — it might be 50."},
            {"role": "assistant", "content": "Noted — page size is approximately 100 items, flagged as unverified."},
        ],
        fillers=_FILLERS_API[:6],
        callbacks=[
            {"question": "What page size should we use for the pagination loop?",
             "correct_frags": ["100", "uncertain", "unverified", "verify", "might", "approximately"],
             "wrong_frags": ["500", "1000", "25"],
             "planted_value": "100"},
        ],
    ),

    Scenario(
        id="S04", domain="api",
        description="OAuth scope and redirect URI uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're adding OAuth 2.0 to our app with Google."},
            {"role": "assistant", "content": "Let's plan the OAuth flow. What's in scope?"},
            {"role": "user",      "content": "I think we need the email and profile scopes, but we might need calendar too — not confirmed."},
            {"role": "assistant", "content": "Noted — email and profile scopes tentative, calendar scope unconfirmed."},
            {"role": "user",      "content": "Redirect URI should be /oauth/callback but the staging domain isn't finalized yet."},
            {"role": "assistant", "content": "Understood — redirect URI path is /oauth/callback, staging domain pending."},
        ],
        fillers=_FILLERS_API,
        callbacks=[
            {"question": "What OAuth scopes should we request?",
             "correct_frags": ["email", "profile", "uncertain", "might", "unconfirmed", "tentative"],
             "wrong_frags": ["admin", "superuser"],
             "planted_value": "email"},
        ],
    ),

    Scenario(
        id="S05", domain="api",
        description="API versioning and deprecation timeline uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're migrating from v2 to v3 of our payment provider's API."},
            {"role": "assistant", "content": "What's the timeline and scope of the migration?"},
            {"role": "user",      "content": "v2 deprecation might be in Q3, but the vendor hasn't confirmed the date yet."},
            {"role": "assistant", "content": "Noted — v2 deprecation tentatively Q3, unconfirmed by vendor."},
            {"role": "user",      "content": "I think v3 uses a different auth header format, but I need to check the changelog."},
            {"role": "assistant", "content": "Understood — v3 auth header format uncertain, changelog review pending."},
        ],
        fillers=_FILLERS_API,
        callbacks=[
            {"question": "When should we target completing the v3 migration by?",
             "correct_frags": ["Q3", "uncertain", "unconfirmed", "tentative", "verify", "might"],
             "wrong_frags": ["Q1", "Q4", "Q2"],
             "planted_value": "Q3"},
        ],
    ),

    Scenario(
        id="S06", domain="api",
        description="Batch endpoint size limit uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're using a batch write API to sync records."},
            {"role": "assistant", "content": "What batch constraints do we need to plan around?"},
            {"role": "user",      "content": "I believe the batch limit is 500 records, but that might have changed — worth double-checking."},
            {"role": "assistant", "content": "Noted — batch limit is approximately 500 records, flagged for verification."},
        ],
        fillers=_FILLERS_API[:6],
        callbacks=[
            {"question": "How should we chunk our bulk sync to stay under the batch limit?",
             "correct_frags": ["500", "uncertain", "verify", "double-check", "approximately", "might"],
             "wrong_frags": ["2000", "10000", "50"],
             "planted_value": "500"},
        ],
    ),

    Scenario(
        id="S07", domain="api",
        description="SLA uptime and incident response time uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We need to set SLA targets for our API consumers."},
            {"role": "assistant", "content": "What availability commitments are we considering?"},
            {"role": "user",      "content": "I think our upstream vendor guarantees 99.9% uptime, but I haven't reviewed the contract."},
            {"role": "assistant", "content": "Noted — upstream SLA approximately 99.9% uptime, unconfirmed contract."},
            {"role": "user",      "content": "Incident response time might be 4 hours — that's what sales said but it's not in the SLA doc."},
            {"role": "assistant", "content": "Understood — incident response time approximately 4h, verbal only, not documented."},
        ],
        fillers=_FILLERS_API,
        callbacks=[
            {"question": "What uptime can we promise our consumers based on what we know?",
             "correct_frags": ["99.9", "uncertain", "unconfirmed", "not confirmed", "verify", "contract"],
             "wrong_frags": ["99.99", "100", "95"],
             "planted_value": "99.9"},
        ],
    ),

    Scenario(
        id="S08", domain="api",
        description="Payload encryption algorithm uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're designing an encrypted payload channel with a partner."},
            {"role": "assistant", "content": "What encryption requirements do we have?"},
            {"role": "user",      "content": "The partner mentioned AES-256 but their docs say AES-128 in places — I'm not sure which."},
            {"role": "assistant", "content": "Flagged — encryption algorithm is uncertain: AES-256 verbally vs AES-128 in docs. Needs clarification."},
        ],
        fillers=_FILLERS_API[:6],
        callbacks=[
            {"question": "Which encryption algorithm should we implement for the payload channel?",
             "correct_frags": ["AES", "uncertain", "unclear", "verify", "clarification", "conflict"],
             "wrong_frags": ["DES", "RSA-512"],
             "planted_value": "AES"},
        ],
    ),

    Scenario(
        id="S09", domain="api",
        description="Idempotency key TTL uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We need idempotent request handling for our payment flow."},
            {"role": "assistant", "content": "Good. What's the deduplication strategy?"},
            {"role": "user",      "content": "I think idempotency keys expire after 24 hours, but I need to verify with the provider."},
            {"role": "assistant", "content": "Noted — idempotency key TTL approximately 24 hours, needs provider verification."},
        ],
        fillers=_FILLERS_API[:6],
        callbacks=[
            {"question": "How long should we store idempotency keys for deduplication?",
             "correct_frags": ["24", "uncertain", "verify", "approximately", "needs", "provider"],
             "wrong_frags": ["7 day", "1 hour", "30 min"],
             "planted_value": "24"},
        ],
    ),

    Scenario(
        id="S10", domain="api",
        description="Concurrent connection limit uncertain",
        system=_SYSTEM_API,
        seeds=[
            {"role": "user",      "content": "We're planning connection pool sizing for our integration."},
            {"role": "assistant", "content": "What are the constraints we're working around?"},
            {"role": "user",      "content": "I believe the upstream service allows 20 concurrent connections max, but I've seen conflicting numbers."},
            {"role": "assistant", "content": "Noted — max concurrent connections approximately 20, flagged as conflicting/uncertain."},
        ],
        fillers=_FILLERS_API[:6],
        callbacks=[
            {"question": "What should we set max_connections to in the connection pool config?",
             "correct_frags": ["20", "uncertain", "conflicting", "verify", "approximately", "might"],
             "wrong_frags": ["100", "500", "1000"],
             "planted_value": "20"},
        ],
    ),

    # =========================================================
    # DOMAIN 2: Debugging Sessions (S11–S20)
    # =========================================================

    Scenario(
        id="S11", domain="debug",
        description="Error cause uncertain, fix hypothetical",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "We're seeing random 503 errors in production."},
            {"role": "assistant", "content": "Let's trace the root cause. What do logs show?"},
            {"role": "user",      "content": "I think it's the upstream database hitting connection limits, but I haven't confirmed — it could be the load balancer."},
            {"role": "assistant", "content": "Hypothesis: database connection exhaustion — flagged as unconfirmed, load balancer also candidate."},
            {"role": "user",      "content": "The errors spike around 2am — probably a batch job, but I'm not certain which one."},
            {"role": "assistant", "content": "Noted — 2am spike correlates with batch job, specific job unidentified."},
        ],
        fillers=_FILLERS_DEBUG,
        callbacks=[
            {"question": "What's our leading hypothesis for the 503 errors?",
             "correct_frags": ["database", "connection", "uncertain", "unconfirmed", "hypothesis", "might", "possibly"],
             "wrong_frags": ["confirmed", "definitely", "certain"],
             "planted_value": "database"},
        ],
    ),

    Scenario(
        id="S12", domain="debug",
        description="Memory leak location uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "Our service memory usage grows over 48 hours and then crashes."},
            {"role": "assistant", "content": "Classic memory leak pattern. Where are you looking?"},
            {"role": "user",      "content": "I think it might be in the caching layer — objects not expiring — but I haven't profiled it yet."},
            {"role": "assistant", "content": "Hypothesis: cache eviction failure — tentative, profiling pending."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "Where should we start the memory profiling investigation?",
             "correct_frags": ["cache", "uncertain", "hypothesis", "tentative", "haven't", "profil"],
             "wrong_frags": ["confirmed", "definitely"],
             "planted_value": "cache"},
        ],
    ),

    Scenario(
        id="S13", domain="debug",
        description="Race condition in async code, reproduction steps uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "We have a data corruption bug that only appears under load."},
            {"role": "assistant", "content": "Likely a race condition. What's the access pattern?"},
            {"role": "user",      "content": "I think two workers write to the same record concurrently — but it's hard to reproduce, maybe 1-in-100 requests."},
            {"role": "assistant", "content": "Hypothesis: concurrent write race — approximately 1% occurrence, difficult to reproduce, unconfirmed."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "How reliably can we reproduce the bug in testing?",
             "correct_frags": ["1", "uncertain", "approximately", "hard", "difficult", "reproduce"],
             "wrong_frags": ["always", "100%", "deterministic"],
             "planted_value": "1"},
        ],
    ),

    Scenario(
        id="S14", domain="debug",
        description="CPU spike root cause uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "CPU spikes to 100% for 10 seconds every few minutes."},
            {"role": "assistant", "content": "Let's find the hot path. What does the profiler show?"},
            {"role": "user",      "content": "I haven't run the profiler yet, but I suspect it's JSON serialization in the hot path — possibly the audit logging middleware."},
             {"role": "assistant", "content": "Hypothesis: JSON serialization overhead in audit logging middleware — unconfirmed, profiling needed."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "What should we profile first for the CPU spike?",
             "correct_frags": ["JSON", "serializ", "audit", "logging", "uncertain", "hypothesis", "unconfirmed", "suspect"],
             "wrong_frags": ["confirmed", "definite"],
             "planted_value": "JSON"},
        ],
    ),

    Scenario(
        id="S15", domain="debug",
        description="Latency regression in new deployment, diff uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "P99 latency jumped from 200ms to 800ms after yesterday's deploy."},
            {"role": "assistant", "content": "What changed in yesterday's deploy?"},
            {"role": "user",      "content": "There were three PRs merged — I think the ORM update is the culprit, but I haven't diffed the query plans."},
            {"role": "assistant", "content": "Leading hypothesis: ORM update changed query plans — unconfirmed, diff pending."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "Which PR should we roll back first to test the latency regression?",
             "correct_frags": ["ORM", "uncertain", "hypothesis", "unconfirmed", "think", "diff"],
             "wrong_frags": ["confirmed", "proven"],
             "planted_value": "ORM"},
        ],
    ),

    Scenario(
        id="S16", domain="debug",
        description="Data loss during migration, scope uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "Some user records appear to be missing after the schema migration."},
            {"role": "assistant", "content": "That's critical. What do we know about the scope?"},
            {"role": "user",      "content": "I think about 2% of records are affected — maybe accounts created in a specific date range, but I haven't run the count query yet."},
            {"role": "assistant", "content": "Estimated scope: ~2% of records, possibly within a date range — unverified, count query pending."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "How many records are potentially affected by the migration issue?",
             "correct_frags": ["2", "uncertain", "approximately", "unverified", "might", "pending"],
             "wrong_frags": ["100%", "0.01%", "50%"],
             "planted_value": "2"},
        ],
    ),

    Scenario(
        id="S17", domain="debug",
        description="Intermittent auth failure, JWT validation uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "Users are getting intermittent 401 errors even with valid tokens."},
            {"role": "assistant", "content": "What does the auth middleware log show?"},
            {"role": "user",      "content": "The logs are unclear — I think the clock skew between nodes might be causing JWT validation to fail, but I haven't measured it."},
            {"role": "assistant", "content": "Hypothesis: inter-node clock skew invalidating JWTs — unmeasured, flagged for investigation."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "What should we check first for the intermittent 401 errors?",
             "correct_frags": ["clock", "skew", "JWT", "uncertain", "hypothesis", "haven't", "unmeasured"],
             "wrong_frags": ["confirmed", "proven", "certain"],
             "planted_value": "clock"},
        ],
    ),

    Scenario(
        id="S18", domain="debug",
        description="Disk space exhaustion, log rotation uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "Production disk is filling up at about 10GB/day."},
            {"role": "assistant", "content": "Where is the growth coming from?"},
            {"role": "user",      "content": "I think the app logs aren't rotating — I set up logrotate but I'm not sure if the config was applied."},
            {"role": "assistant", "content": "Hypothesis: logrotate config not applied — uncertain, configuration verification needed."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "What's the suspected cause of the disk space growth?",
             "correct_frags": ["logrotate", "log", "uncertain", "not sure", "verify", "applied"],
             "wrong_frags": ["confirmed", "proven"],
             "planted_value": "logrotate"},
        ],
    ),

    Scenario(
        id="S19", domain="debug",
        description="Deadlock in transaction processing, isolation level uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "We're seeing transaction deadlocks under concurrent checkout load."},
            {"role": "assistant", "content": "What transaction isolation level are you using?"},
            {"role": "user",      "content": "I think we're on READ COMMITTED, but the config might have been changed when we switched ORMs."},
            {"role": "assistant", "content": "Noted — isolation level approximately READ COMMITTED, possibly changed, needs verification."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "What transaction isolation level should we check first?",
             "correct_frags": ["READ COMMITTED", "uncertain", "might", "possibly", "changed", "verify"],
             "wrong_frags": ["SERIALIZABLE", "READ UNCOMMITTED"],
             "planted_value": "READ COMMITTED"},
        ],
    ),

    Scenario(
        id="S20", domain="debug",
        description="Queue backlog, consumer count uncertain",
        system=_SYSTEM_DEBUG,
        seeds=[
            {"role": "user",      "content": "Our message queue backlog is growing during peak hours."},
            {"role": "assistant", "content": "How many consumers are processing the queue?"},
            {"role": "user",      "content": "I believe we have 4 consumer instances but that might be stale — last I checked was 2 weeks ago."},
            {"role": "assistant", "content": "Noted — consumer count approximately 4, potentially stale information from 2 weeks ago."},
        ],
        fillers=_FILLERS_DEBUG[:6],
        callbacks=[
            {"question": "How many consumers should we provision to clear the backlog?",
             "correct_frags": ["4", "uncertain", "stale", "verify", "might", "approximately"],
             "wrong_frags": ["100", "1000", "0"],
             "planted_value": "4"},
        ],
    ),

    # =========================================================
    # DOMAIN 3: System Design (S21–S30)
    # =========================================================

    Scenario(
        id="S21", domain="design",
        description="Database sharding key uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We're designing a multi-tenant SaaS database architecture."},
            {"role": "assistant", "content": "What are the scale targets and tenant access patterns?"},
            {"role": "user",      "content": "I think we'll shard by tenant_id, but we might need to shard by region depending on compliance — not finalized."},
            {"role": "assistant", "content": "Sharding key: tenant_id tentative, region-based sharding possible for compliance — decision pending."},
            {"role": "user",      "content": "Target scale is maybe 10,000 tenants, though that estimate might be off by 2x."},
            {"role": "assistant", "content": "Scale estimate: ~10,000 tenants, ±2x uncertainty, flagged."},
        ],
        fillers=_FILLERS_DESIGN,
        callbacks=[
            {"question": "What sharding key should we use for the tenant database?",
             "correct_frags": ["tenant_id", "uncertain", "tentative", "pending", "might", "region"],
             "wrong_frags": ["confirmed", "finalized"],
             "planted_value": "tenant_id"},
        ],
    ),

    Scenario(
        id="S22", domain="design",
        description="Event bus choice uncertain, throughput SLA uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We need an event bus for our microservices."},
            {"role": "assistant", "content": "What are the throughput and latency requirements?"},
            {"role": "user",      "content": "I think Kafka, but the team hasn't agreed yet — Pulsar is also on the table."},
            {"role": "assistant", "content": "Event bus: Kafka vs Pulsar decision pending — flagged."},
            {"role": "user",      "content": "Peak throughput might be 50,000 events/sec — rough estimate, not validated."},
            {"role": "assistant", "content": "Throughput estimate: ~50,000 events/sec, rough estimate, not validated."},
        ],
        fillers=_FILLERS_DESIGN,
        callbacks=[
            {"question": "Which event bus should we provision infrastructure for?",
             "correct_frags": ["Kafka", "Pulsar", "uncertain", "pending", "haven't", "agreed"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "Kafka"},
            {"question": "What throughput should we design for?",
             "correct_frags": ["50,000", "50000", "uncertain", "rough", "estimate", "not validated"],
             "wrong_frags": ["1000000", "100"],
             "planted_value": "50000"},
        ],
    ),

    Scenario(
        id="S23", domain="design",
        description="Cache TTL and eviction policy uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We're adding a read-through cache for our product catalog."},
            {"role": "assistant", "content": "What freshness guarantees do we need?"},
            {"role": "user",      "content": "I think a 5-minute TTL would work, but the product team hasn't confirmed how stale is acceptable."},
            {"role": "assistant", "content": "Cache TTL: approximately 5 minutes — product team sign-off pending."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What TTL should we configure for the product catalog cache?",
             "correct_frags": ["5", "uncertain", "pending", "not confirmed", "approximately"],
             "wrong_frags": ["1 hour", "1 day", "1 second"],
             "planted_value": "5"},
        ],
    ),

    Scenario(
        id="S24", domain="design",
        description="Service mesh vs. ambassador pattern uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We need to add cross-cutting concerns like auth and rate limiting across our microservices."},
            {"role": "assistant", "content": "What are the options on the table?"},
            {"role": "user",      "content": "I'm leaning toward a service mesh, but the ops team isn't sure we have bandwidth — they might prefer an ambassador sidecar."},
            {"role": "assistant", "content": "Architecture decision: service mesh vs. ambassador sidecar — pending ops team bandwidth assessment."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What networking pattern should we implement for cross-cutting concerns?",
             "correct_frags": ["service mesh", "ambassador", "uncertain", "pending", "not sure", "bandwidth"],
             "wrong_frags": ["confirmed", "decided", "finalized"],
             "planted_value": "service mesh"},
        ],
    ),

    Scenario(
        id="S25", domain="design",
        description="Data retention policy uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We need to set data retention rules for our analytics pipeline."},
            {"role": "assistant", "content": "What regulatory and product constraints apply?"},
            {"role": "user",      "content": "Legal said 2 years, but compliance might require 7 years for financial data — we're waiting for legal to clarify."},
            {"role": "assistant", "content": "Retention policy: 2 years (general) vs. 7 years (financial) — awaiting legal clarification."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What retention period should we build into the data pipeline?",
             "correct_frags": ["2", "7", "uncertain", "awaiting", "clarif", "legal"],
             "wrong_frags": ["30 days", "10 years"],
             "planted_value": "7"},
        ],
    ),

    Scenario(
        id="S26", domain="design",
        description="Read replica lag budget uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We're adding read replicas to handle reporting queries."},
            {"role": "assistant", "content": "What staleness tolerance do reporting consumers have?"},
            {"role": "user",      "content": "I think up to 1 minute of replica lag is acceptable, but the data team hasn't signed off on that."},
            {"role": "assistant", "content": "Replica lag budget: approximately 1 minute — data team approval pending."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What replica lag SLA should we build into the reporting layer?",
             "correct_frags": ["1 minute", "60", "uncertain", "pending", "approval", "approximately"],
             "wrong_frags": ["1 second", "1 hour"],
             "planted_value": "1"},
        ],
    ),

    Scenario(
        id="S27", domain="design",
        description="Horizontal vs. vertical scaling uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We need to plan scaling for our checkout service ahead of Black Friday."},
            {"role": "assistant", "content": "What are the peak load projections?"},
            {"role": "user",      "content": "Finance estimates 5x normal load, but that's based on last year's growth rate — it could be 10x."},
            {"role": "assistant", "content": "Peak load estimate: 5–10x normal, flagged as uncertain finance estimate."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What scaling factor should we provision capacity for?",
             "correct_frags": ["5", "10", "uncertain", "estimate", "could be", "flagged"],
             "wrong_frags": ["2x", "100x"],
             "planted_value": "5"},
        ],
    ),

    Scenario(
        id="S28", domain="design",
        description="CDN vendor and cache invalidation strategy uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We're evaluating CDN vendors for our static assets."},
            {"role": "assistant", "content": "What are the selection criteria?"},
            {"role": "user",      "content": "Cloudflare is the leading option, but procurement hasn't approved the contract yet."},
            {"role": "assistant", "content": "CDN vendor: Cloudflare tentative — procurement approval pending."},
            {"role": "user",      "content": "Cache invalidation time might be 5 seconds with their API, but the docs say it could be up to 30."},
            {"role": "assistant", "content": "Cache invalidation: 5–30 seconds per docs, flagged as uncertain range."},
        ],
        fillers=_FILLERS_DESIGN,
        callbacks=[
            {"question": "What CDN should we build the integration for?",
             "correct_frags": ["Cloudflare", "uncertain", "tentative", "pending", "procurement"],
             "wrong_frags": ["confirmed", "approved"],
             "planted_value": "Cloudflare"},
        ],
    ),

    Scenario(
        id="S29", domain="design",
        description="GraphQL vs REST API uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We're designing the API layer for our new mobile app."},
            {"role": "assistant", "content": "What does the client team need?"},
            {"role": "user",      "content": "The mobile team wants GraphQL, but the backend team is concerned about complexity — decision isn't made yet."},
            {"role": "assistant", "content": "API type: GraphQL vs REST — decision pending, team disagreement flagged."},
        ],
        fillers=_FILLERS_DESIGN[:6],
        callbacks=[
            {"question": "What API style should we scaffold the backend for?",
             "correct_frags": ["GraphQL", "REST", "uncertain", "pending", "decision", "isn't"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "GraphQL"},
        ],
    ),

    Scenario(
        id="S30", domain="design",
        description="Log aggregation stack uncertain",
        system=_SYSTEM_DESIGN,
        seeds=[
            {"role": "user",      "content": "We need centralized log aggregation for our 20-service cluster."},
            {"role": "assistant", "content": "What observability stack are you considering?"},
            {"role": "user",      "content": "ELK is what we know, but the platform team mentioned Loki — we haven't decided."},
            {"role": "assistant", "content": "Log stack: ELK vs Loki — decision pending, platform team preference for Loki."},
            {"role": "user",      "content": "Log volume might be 50GB/day, but that's a rough estimate before we add tracing."},
            {"role": "assistant", "content": "Log volume: approximately 50GB/day, rough pre-tracing estimate."},
        ],
        fillers=_FILLERS_DESIGN,
        callbacks=[
            {"question": "What log aggregation system should we provision storage for?",
             "correct_frags": ["ELK", "Loki", "uncertain", "pending", "haven't", "decided"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "ELK"},
        ],
    ),

    # =========================================================
    # DOMAIN 4: Compliance / Security / Medical-adjacent (S31–S40)
    # =========================================================

    Scenario(
        id="S31", domain="compliance",
        description="GDPR retention period uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're building a user data deletion flow for GDPR compliance."},
            {"role": "assistant", "content": "What retention and deletion requirements apply?"},
            {"role": "user",      "content": "I believe we need to delete PII within 30 days of a request, but our legal team is still reviewing the specific regulation."},
            {"role": "assistant", "content": "Deletion timeline: approximately 30 days, legal review pending — flagged as unconfirmed."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What deletion SLA should we implement for the right-to-erasure flow?",
             "correct_frags": ["30", "uncertain", "pending", "legal", "review", "unconfirmed"],
             "wrong_frags": ["365", "1 year", "immediately"],
             "planted_value": "30"},
        ],
    ),

    Scenario(
        id="S32", domain="compliance",
        description="Audit log retention requirement uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're building audit logs for SOC 2 compliance."},
            {"role": "assistant", "content": "What logging requirements does SOC 2 impose?"},
            {"role": "user",      "content": "I think audit logs need to be retained for 12 months, but the auditor hasn't confirmed the minimum yet."},
            {"role": "assistant", "content": "Audit log retention: approximately 12 months, auditor confirmation pending."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "How long should we retain audit logs?",
             "correct_frags": ["12", "uncertain", "pending", "auditor", "approximately", "not confirmed"],
             "wrong_frags": ["7 days", "30 days"],
             "planted_value": "12"},
        ],
    ),

    Scenario(
        id="S33", domain="compliance",
        description="Encryption key rotation frequency uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We need to implement encryption key rotation for our data-at-rest."},
            {"role": "assistant", "content": "What rotation frequency is required?"},
            {"role": "user",      "content": "I think annual rotation is sufficient, but our security team might require quarterly — that decision isn't made."},
            {"role": "assistant", "content": "Key rotation: annual vs. quarterly — security team decision pending."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "How often should we rotate encryption keys?",
             "correct_frags": ["annual", "quarterly", "uncertain", "pending", "not made", "decision"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "annual"},
        ],
    ),

    Scenario(
        id="S34", domain="compliance",
        description="PHI data handling jurisdiction uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're building a feature that handles health-adjacent user data."},
            {"role": "assistant", "content": "What regulations govern this data?"},
            {"role": "user",      "content": "I'm not sure if this counts as PHI under HIPAA — the data includes symptom tracking but not diagnoses. Legal is reviewing."},
            {"role": "assistant", "content": "PHI classification: uncertain — symptom tracking is a gray area, legal review in progress."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "Should we apply HIPAA controls to the symptom tracking data?",
             "correct_frags": ["uncertain", "not sure", "gray area", "legal", "review", "might"],
             "wrong_frags": ["confirmed", "definitely yes", "definitely no"],
             "planted_value": "uncertain"},
        ],
    ),

    Scenario(
        id="S35", domain="compliance",
        description="Breach notification window uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're building an incident response playbook for data breaches."},
            {"role": "assistant", "content": "What regulatory notification timelines apply?"},
            {"role": "user",      "content": "GDPR requires 72 hours I believe, but I'm not sure if our state-level regulations are stricter."},
            {"role": "assistant", "content": "Notification window: GDPR 72 hours, state law may be stricter — uncertain, needs legal review."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What breach notification SLA should we build into the runbook?",
             "correct_frags": ["72", "uncertain", "state", "stricter", "legal", "not sure"],
             "wrong_frags": ["7 days", "30 days"],
             "planted_value": "72"},
        ],
    ),

    Scenario(
        id="S36", domain="compliance",
        description="Third-party data sharing agreement uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're integrating with an analytics vendor who will receive user event data."},
            {"role": "assistant", "content": "What data sharing agreements are in place?"},
            {"role": "user",      "content": "Legal is reviewing the DPA — I think it covers our use case but the vendor hasn't countersigned yet."},
            {"role": "assistant", "content": "DPA status: under review, not countersigned — flagged as legally uncertain."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "Can we enable the analytics vendor integration now?",
             "correct_frags": ["uncertain", "not countersigned", "review", "pending", "DPA"],
             "wrong_frags": ["confirmed", "approved", "safe"],
             "planted_value": "uncertain"},
        ],
    ),

    Scenario(
        id="S37", domain="compliance",
        description="Password policy requirements uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're updating our password policy to meet compliance requirements."},
            {"role": "assistant", "content": "What specific requirements are mandated?"},
            {"role": "user",      "content": "I think minimum 12 characters, but some frameworks recommend 16 — our security policy isn't updated yet."},
            {"role": "assistant", "content": "Minimum password length: 12–16 characters, range uncertain — security policy update pending."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What minimum password length should we enforce?",
             "correct_frags": ["12", "16", "uncertain", "pending", "isn't updated", "range"],
             "wrong_frags": ["8", "4"],
             "planted_value": "12"},
        ],
    ),

    Scenario(
        id="S38", domain="compliance",
        description="Vulnerability disclosure window uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We need a coordinated vulnerability disclosure policy."},
            {"role": "assistant", "content": "What disclosure timeline is appropriate?"},
            {"role": "user",      "content": "Industry standard is 90 days, but our partners may require shorter — we're still in discussions."},
            {"role": "assistant", "content": "Disclosure window: 90 days standard, possibly shorter per partners — discussions ongoing."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What vulnerability disclosure SLA should we publish?",
             "correct_frags": ["90", "uncertain", "discussions", "partners", "possibly", "shorter"],
             "wrong_frags": ["365", "7"],
             "planted_value": "90"},
        ],
    ),

    Scenario(
        id="S39", domain="compliance",
        description="Access log retention for forensics uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We need to decide how long to keep access logs for forensic purposes."},
            {"role": "assistant", "content": "What drives the retention decision?"},
            {"role": "user",      "content": "I think 6 months is enough, but our insurance policy might require a year — need to check."},
            {"role": "assistant", "content": "Access log retention: 6 months tentative, insurance requirement possibly 12 months — unconfirmed."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What access log retention period should we configure?",
             "correct_frags": ["6", "12", "uncertain", "tentative", "insurance", "possibly"],
             "wrong_frags": ["7 days", "1 week"],
             "planted_value": "6"},
        ],
    ),

    Scenario(
        id="S40", domain="compliance",
        description="Background check depth and frequency uncertain",
        system=_SYSTEM_COMPLIANCE,
        seeds=[
            {"role": "user",      "content": "We're onboarding contractors who will access production systems."},
            {"role": "assistant", "content": "What vetting requirements apply?"},
            {"role": "user",      "content": "I think criminal background checks are required, but I'm not sure if credit checks are also mandated by our SOC 2 policy."},
            {"role": "assistant", "content": "Background check scope: criminal checks required, credit check requirement uncertain — SOC 2 policy review needed."},
        ],
        fillers=_FILLERS_COMPLIANCE[:6],
        callbacks=[
            {"question": "What background checks should we require for contractor onboarding?",
             "correct_frags": ["criminal", "uncertain", "credit", "not sure", "SOC", "review"],
             "wrong_frags": ["none required", "confirmed"],
             "planted_value": "criminal"},
        ],
    ),

    # =========================================================
    # DOMAIN 5: Multi-Agent Pipelines (S41–S50)
    # =========================================================

    Scenario(
        id="S41", domain="multiagent",
        description="Agent model selection uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We're designing a 3-agent pipeline for document processing."},
            {"role": "assistant", "content": "What are the roles of the three agents?"},
            {"role": "user",      "content": "Extractor, validator, and formatter. I think Haiku for extractor and Opus for validator, but we haven't run cost comparisons."},
            {"role": "assistant", "content": "Agent models: Haiku extractor, Opus validator — tentative, cost comparison pending."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What model should we assign to the extractor agent?",
             "correct_frags": ["Haiku", "uncertain", "tentative", "pending", "haven't", "comparison"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "Haiku"},
        ],
    ),

    Scenario(
        id="S42", domain="multiagent",
        description="Context window budget per agent uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need to allocate context budgets across our agent chain."},
            {"role": "assistant", "content": "What are the token constraints?"},
            {"role": "user",      "content": "I think each agent should get 8k tokens, but the final synthesis agent might need 32k — not profiled yet."},
            {"role": "assistant", "content": "Context budgets: 8k per agent tentative, synthesis agent possibly 32k — unprofiled."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "How much context should we allocate to the synthesis agent?",
             "correct_frags": ["32", "uncertain", "tentative", "unprofiled", "possibly", "might"],
             "wrong_frags": ["1k", "100k"],
             "planted_value": "32"},
        ],
    ),

    Scenario(
        id="S43", domain="multiagent",
        description="Agent timeout and retry policy uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need timeout and retry policies for our agent orchestration layer."},
            {"role": "assistant", "content": "What latency SLAs do the downstream consumers expect?"},
            {"role": "user",      "content": "I think a 30-second timeout per agent is reasonable, but the product team said end-to-end must be under 2 minutes — math doesn't quite work."},
            {"role": "assistant", "content": "Per-agent timeout: 30s tentative, end-to-end constraint 2 minutes — budget tension flagged."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What per-agent timeout should we configure?",
             "correct_frags": ["30", "uncertain", "tentative", "tension", "math", "budget"],
             "wrong_frags": ["1 second", "10 minutes"],
             "planted_value": "30"},
        ],
    ),

    Scenario(
        id="S44", domain="multiagent",
        description="Trust propagation policy uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need to decide how trust propagates across our agent chain."},
            {"role": "assistant", "content": "What trust model are you considering?"},
            {"role": "user",      "content": "I think trust should decay by 0.1 per hop, but we might want 0.05 for low-risk pipelines — not finalized."},
            {"role": "assistant", "content": "Trust decay rate: 0.1 per hop tentative, 0.05 for low-risk — policy not finalized."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What trust decay rate should we implement for the default pipeline?",
             "correct_frags": ["0.1", "0.05", "uncertain", "tentative", "not finalized", "might"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "0.1"},
        ],
    ),

    Scenario(
        id="S45", domain="multiagent",
        description="Agent output schema version uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "Agents in our pipeline need to agree on output schemas."},
            {"role": "assistant", "content": "What schema format are you using?"},
            {"role": "user",      "content": "JSON Schema v7, I think — but the extractor agent was built against draft 4 and I'm not sure if it was updated."},
            {"role": "assistant", "content": "Output schema: JSON Schema v7 intended, extractor may still use draft 4 — unverified."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "Which JSON Schema version should we validate against?",
             "correct_frags": ["v7", "draft 4", "uncertain", "unverified", "might", "not sure"],
             "wrong_frags": ["confirmed", "standardized"],
             "planted_value": "v7"},
        ],
    ),

    Scenario(
        id="S46", domain="multiagent",
        description="Parallel vs sequential agent execution uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need to decide the execution topology for our 5-agent pipeline."},
            {"role": "assistant", "content": "What dependencies exist between agents?"},
            {"role": "user",      "content": "I think agents 2 and 3 can run in parallel since they're independent, but I haven't verified the data flow graph."},
            {"role": "assistant", "content": "Execution: agents 2–3 potentially parallel, data flow independence unverified."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "Can we run agents 2 and 3 in parallel?",
             "correct_frags": ["uncertain", "unverified", "haven't", "potentially", "need to verify"],
             "wrong_frags": ["confirmed", "yes definitely", "already verified"],
             "planted_value": "uncertain"},
        ],
    ),

    Scenario(
        id="S47", domain="multiagent",
        description="Shared memory store for agents uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need a shared memory store for our agent swarm."},
            {"role": "assistant", "content": "What consistency guarantees do agents need?"},
            {"role": "user",      "content": "I think Redis would work, but we might need a vector store for semantic lookups — architecture decision pending."},
            {"role": "assistant", "content": "Memory store: Redis tentative, vector store possible for semantic use case — pending architectural decision."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What should we provision as the shared memory store?",
             "correct_frags": ["Redis", "vector", "uncertain", "tentative", "pending", "might"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "Redis"},
        ],
    ),

    Scenario(
        id="S48", domain="multiagent",
        description="Agent log aggregation and observability uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need distributed tracing across our agent pipeline."},
            {"role": "assistant", "content": "What tracing infrastructure do you have?"},
            {"role": "user",      "content": "We use Jaeger for services, but I'm not sure it handles LLM token traces — might need LangSmith."},
            {"role": "assistant", "content": "Tracing: Jaeger for infrastructure, LLM token tracing uncertain — LangSmith possibly needed."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What tracing tool should we integrate for the agent pipeline?",
             "correct_frags": ["Jaeger", "LangSmith", "uncertain", "not sure", "might", "possibly"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "Jaeger"},
        ],
    ),

    Scenario(
        id="S49", domain="multiagent",
        description="Guardrail enforcement point uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need content guardrails in our generative pipeline."},
            {"role": "assistant", "content": "Where in the pipeline should guardrails run?"},
            {"role": "user",      "content": "I think at the output of each agent, but the security team might want them at ingestion too — not finalized."},
            {"role": "assistant", "content": "Guardrail enforcement: output-stage tentative, ingestion-stage also possible — security decision pending."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "Where in the pipeline should we apply content guardrails?",
             "correct_frags": ["output", "ingestion", "uncertain", "tentative", "pending", "not finalized"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "output"},
        ],
    ),

    Scenario(
        id="S50", domain="multiagent",
        description="Agent versioning and hot-swap strategy uncertain",
        system=_SYSTEM_MULTIAGENT,
        seeds=[
            {"role": "user",      "content": "We need a versioning strategy for agent model updates."},
            {"role": "assistant", "content": "What deployment requirements constrain us?"},
            {"role": "user",      "content": "I think blue-green deployment, but the platform team isn't sure we can do it for stateful agents."},
            {"role": "assistant", "content": "Agent versioning: blue-green tentative, stateful agent compatibility uncertain — platform team review pending."},
        ],
        fillers=_FILLERS_MULTIAGENT[:6],
        callbacks=[
            {"question": "What deployment strategy should we use for agent model updates?",
             "correct_frags": ["blue-green", "uncertain", "tentative", "pending", "stateful", "not sure"],
             "wrong_frags": ["confirmed", "decided"],
             "planted_value": "blue-green"},
        ],
    ),
]
# fmt: on


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_callback(answer: str, correct_frags: list[str], wrong_frags: list[str], planted_value: str) -> dict:
    lower = answer.lower()
    has_value     = planted_value.lower() in lower
    has_qualifier = any(f.lower() in lower for f in correct_frags if f.lower() != planted_value.lower())
    hallu         = any(h.lower() in lower for h in wrong_frags)
    recall        = float(has_value and has_qualifier)
    qualifier_ok  = float(has_qualifier)
    return {
        "recall":     recall,
        "qualifier":  qualifier_ok,
        "hallu":      float(hallu),
        "has_value":  has_value,
    }


# ---------------------------------------------------------------------------
# Single-scenario runners
# ---------------------------------------------------------------------------

_client: Optional[Anthropic] = None

# Hedge-preservation instruction for the haiku_hedged condition.
# Tests whether a prompt-level instruction achieves the same qualifier survival
# rate as Credence's faithfulness probe — the critical missing baseline.
_HEDGE_INSTRUCTION = (
    "CRITICAL: Preserve ALL uncertainty qualifiers, hedging phrases, and tentative "
    "language VERBATIM in your summary. Do NOT convert uncertain claims to confident "
    "ones. Words like 'I think', 'might be', 'not confirmed', 'approximately', "
    "'unverified', 'tentative', 'pending' MUST appear in your summary wherever "
    "they appear in the original conversation."
)


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _ask(messages: list[dict], system: str, max_tokens: int = 400) -> str:
    resp = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.content[0].text


def run_credence(scenario: Scenario) -> dict:
    mgr = ContextManager(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        system_prompt=scenario.system,
        max_tokens=400,
    )
    for i in range(0, len(scenario.seeds), 2):
        mgr.chat(scenario.seeds[i]["content"])
        time.sleep(0.15)

    for filler in scenario.fillers:
        mgr.chat(filler)
        time.sleep(0.15)

    cb_scores = []
    for cb in scenario.callbacks:
        r = mgr.chat(cb["question"])
        s = score_callback(r.response, cb["correct_frags"], cb["wrong_frags"], cb["planted_value"])
        cb_scores.append(s)
        time.sleep(0.15)

    return {
        "recall":    sum(s["recall"]    for s in cb_scores) / len(cb_scores),
        "qualifier": sum(s["qualifier"] for s in cb_scores) / len(cb_scores),
        "hallu":     sum(s["hallu"]     for s in cb_scores) / len(cb_scores),
        "per_callback": cb_scores,
    }


def run_haiku_hedged(scenario: Scenario, window: int = 12) -> dict:
    """
    Fourth condition: Haiku compression WITH explicit hedge-preservation instruction.

    Tests the most important missing baseline: can we solve qualifier loss simply
    by telling Haiku to preserve uncertainty language? If this matches Credence's
    100% qualifier survival, the architectural probe is unnecessary and Credence's
    contribution narrows to detection (knowing when to apply the instruction).
    If it falls short, the middleware architecture is vindicated.

    When context exceeds window, compress old turns via Haiku + hedge instruction.
    Callbacks use Haiku as the answering model (consistent with other conditions).
    """
    history = list(scenario.seeds)

    def _compress_with_hedge(hist: list[dict]) -> list[dict]:
        """Compress old context using Haiku with hedge-preservation instruction."""
        if len(hist) <= window:
            return hist
        old    = hist[:-window]
        recent = hist[-window:]
        conv   = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old)
        resp   = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": (
                "Summarize this conversation segment in 2-3 concise sentences. "
                "Preserve all key facts and decisions needed to continue the conversation.\n\n"
                f"{_HEDGE_INSTRUCTION}\n\n"
                "Conversation:\n\n" + conv
            )}],
            max_tokens=200,
        )
        summary = resp.content[0].text.strip()
        # Inject summary as context_summary prefix on the first recent message
        prefixed_recent = list(recent)
        if prefixed_recent and prefixed_recent[0]["role"] == "user":
            prefixed_recent[0] = {
                "role":    "user",
                "content": f"<context_summary>\n{summary}\n</context_summary>\n\n"
                           + prefixed_recent[0]["content"],
            }
        return prefixed_recent

    for filler in scenario.fillers:
        history = _compress_with_hedge(history)
        msgs    = history + [{"role": "user", "content": filler}]
        answer  = _ask(msgs, scenario.system)
        history.append({"role": "user",      "content": filler})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    cb_scores = []
    for cb in scenario.callbacks:
        history = _compress_with_hedge(history)
        msgs    = history + [{"role": "user", "content": cb["question"]}]
        answer  = _ask(msgs, scenario.system)
        s       = score_callback(answer, cb["correct_frags"], cb["wrong_frags"], cb["planted_value"])
        cb_scores.append(s)
        history.append({"role": "user",      "content": cb["question"]})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    return {
        "recall":       sum(s["recall"]    for s in cb_scores) / len(cb_scores),
        "qualifier":    sum(s["qualifier"] for s in cb_scores) / len(cb_scores),
        "hallu":        sum(s["hallu"]     for s in cb_scores) / len(cb_scores),
        "per_callback": cb_scores,
    }


def run_condition(scenario: Scenario, condition: str, window: int = 12) -> dict:
    history = list(scenario.seeds)

    for filler in scenario.fillers:
        if condition == "naive_window":
            history = history[-window:]
        msgs    = history + [{"role": "user", "content": filler}]
        answer  = _ask(msgs, scenario.system)
        history.append({"role": "user",      "content": filler})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    cb_scores = []
    for cb in scenario.callbacks:
        if condition == "naive_window":
            history = history[-window:]
        msgs   = history + [{"role": "user", "content": cb["question"]}]
        answer = _ask(msgs, scenario.system)
        s      = score_callback(answer, cb["correct_frags"], cb["wrong_frags"], cb["planted_value"])
        cb_scores.append(s)
        history.append({"role": "user",      "content": cb["question"]})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    return {
        "recall":    sum(s["recall"]    for s in cb_scores) / len(cb_scores),
        "qualifier": sum(s["qualifier"] for s in cb_scores) / len(cb_scores),
        "hallu":     sum(s["hallu"]     for s in cb_scores) / len(cb_scores),
        "per_callback": cb_scores,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = sorted(sum(random.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    lo = (1.0 - ci) / 2
    return means[int(lo * n_boot)], means[int((1 - lo) * n_boot) - 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Credence Gauntlet — 50-scenario benchmark")
    parser.add_argument("--domain",   choices=["api","debug","design","compliance","multiagent"],
                        help="Run only scenarios from this domain")
    parser.add_argument("--n",        type=int,  default=None, help="Run first N scenarios")
    parser.add_argument("--scenario", type=str,  default=None, help="Run a single scenario by ID")
    parser.add_argument("--dry-run",  action="store_true", help="Validate structure, no API calls")
    parser.add_argument("--resume",   action="store_true", help="Load existing results and add more")
    args = parser.parse_args()

    # Select scenarios
    selected = SCENARIOS
    if args.scenario:
        selected = [s for s in SCENARIOS if s.id == args.scenario.upper()]
        if not selected:
            print(f"No scenario with id={args.scenario}")
            return
    elif args.domain:
        selected = [s for s in SCENARIOS if s.domain == args.domain]
    if args.n:
        selected = selected[:args.n]

    print(f"Running {len(selected)} scenario(s) × 4 conditions")

    if args.dry_run:
        for s in selected:
            print(f"  {s.id}  {s.domain:12}  {len(s.seeds)//2} seeds  "
                  f"{len(s.fillers)} fillers  {len(s.callbacks)} callbacks  ✓")
        print("\nDry run complete — structure valid.")
        return

    results_path = "evals/gauntlet_results.json"
    all_results:  list[dict] = []

    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_results = saved.get("scenarios", [])
        done_ids    = {r["scenario_id"] for r in all_results}
        selected    = [s for s in selected if s.id not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(selected)} remaining")

    for scenario in selected:
        print(f"\n{'='*60}")
        print(f"Scenario {scenario.id}  [{scenario.domain}]  {scenario.description}")
        print(f"{'='*60}")

        row: dict = {"scenario_id": scenario.id, "domain": scenario.domain, "conditions": {}}
        for cond in ["credence", "baseline", "naive_window", "haiku_hedged"]:
            print(f"  running {cond}...")
            try:
                if cond == "credence":
                    result = run_credence(scenario)
                elif cond == "haiku_hedged":
                    result = run_haiku_hedged(scenario)
                else:
                    result = run_condition(scenario, cond)
                row["conditions"][cond] = result
                print(f"    recall={result['recall']:.2f}  qualifier={result['qualifier']:.2f}  hallu={result['hallu']:.2f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                row["conditions"][cond] = {"error": str(e), "recall": 0.0, "qualifier": 0.0, "hallu": 0.0}

        all_results.append(row)
        _save_results(all_results, results_path)

    print("\n\n=== GAUNTLET SUMMARY ===")
    _print_summary(all_results)


def _save_results(all_results: list[dict], path: str):
    """Aggregate and save results."""
    by_cond: dict[str, dict[str, list]] = {}
    for row in all_results:
        for cond, res in row["conditions"].items():
            if cond not in by_cond:
                by_cond[cond] = {"recall": [], "qualifier": [], "hallu": []}
            by_cond[cond]["recall"].append(res.get("recall", 0.0))
            by_cond[cond]["qualifier"].append(res.get("qualifier", 0.0))
            by_cond[cond]["hallu"].append(res.get("hallu", 0.0))

    summary: dict = {}
    for cond, vals in by_cond.items():
        r_lo, r_hi = bootstrap_ci(vals["recall"])
        q_lo, q_hi = bootstrap_ci(vals["qualifier"])
        summary[cond] = {
            "n":              len(vals["recall"]),
            "recall_mean":    round(sum(vals["recall"])    / len(vals["recall"]),    4),
            "recall_ci95":    [round(r_lo, 4), round(r_hi, 4)],
            "qualifier_mean": round(sum(vals["qualifier"]) / len(vals["qualifier"]), 4),
            "qualifier_ci95": [round(q_lo, 4), round(q_hi, 4)],
            "hallu_mean":     round(sum(vals["hallu"])     / len(vals["hallu"]),     4),
        }

    by_domain: dict[str, dict] = {}
    for row in all_results:
        d = row["domain"]
        if d not in by_domain:
            by_domain[d] = {c: {"recall": [], "qualifier": []} for c in ["credence","baseline","naive_window","haiku_hedged"]}
        for cond, res in row["conditions"].items():
            if cond in by_domain[d]:
                by_domain[d][cond]["recall"].append(res.get("recall", 0.0))
                by_domain[d][cond]["qualifier"].append(res.get("qualifier", 0.0))

    domain_summary: dict = {}
    for domain, conds in by_domain.items():
        domain_summary[domain] = {
            cond: {
                "recall_mean":    round(sum(v["recall"])    / max(1, len(v["recall"])),    3),
                "qualifier_mean": round(sum(v["qualifier"]) / max(1, len(v["qualifier"])), 3),
            }
            for cond, v in conds.items()
        }

    output = {
        "summary":        summary,
        "domain_summary": domain_summary,
        "scenarios":      all_results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(all_results: list[dict]):
    by_cond: dict[str, dict[str, list]] = {}
    for row in all_results:
        for cond, res in row["conditions"].items():
            if cond not in by_cond:
                by_cond[cond] = {"recall": [], "qualifier": [], "hallu": []}
            by_cond[cond]["recall"].append(res.get("recall", 0.0))
            by_cond[cond]["qualifier"].append(res.get("qualifier", 0.0))
            by_cond[cond]["hallu"].append(res.get("hallu", 0.0))

    print(f"\n{'Condition':<20}  {'N':>4}  {'Recall (95%CI)':>22}  {'Qualifier':>10}  {'Hallu':>6}")
    print("-" * 74)
    for cond in ["credence", "baseline", "naive_window", "haiku_hedged"]:
        if cond not in by_cond:
            continue
        rec  = by_cond[cond]["recall"]
        qual = by_cond[cond]["qualifier"]
        hal  = by_cond[cond]["hallu"]
        r_lo, r_hi = bootstrap_ci(rec)
        print(
            f"  {cond:<18}  {len(rec):>4}  "
            f"{sum(rec)/len(rec):.1%} [{r_lo:.1%},{r_hi:.1%}]  "
            f"{sum(qual)/len(qual):.1%}  "
            f"{sum(hal)/len(hal):.1%}"
        )

    if "credence" in by_cond and "naive_window" in by_cond:
        cr = sum(by_cond["credence"]["recall"]) / len(by_cond["credence"]["recall"])
        nr = sum(by_cond["naive_window"]["recall"]) / len(by_cond["naive_window"]["recall"])
        print(f"\n  Credence vs Naive — recall delta: {cr - nr:+.1%}")


if __name__ == "__main__":
    main()

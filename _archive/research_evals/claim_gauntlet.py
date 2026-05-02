"""
evals/claim_gauntlet.py
=======================
Per-claim implicit uncertainty survival benchmark.

Unit of analysis: individual claim survival across 5 dimensions.
Unlike Ghost Gauntlet (session-level both_rate), this measures each claim
independently so you can diagnose WHICH claim types fail and WHY.

FCR (False Certainty Rate) = value_survival AND NOT qualifier_survival
 → model recalled the fact but stripped its uncertainty.
 This is the headline harm metric. No keyword dependency.

Structure
---------
10 ghost sessions × 3 claims each = 30 data points per condition.

Sessions cover 4 domains:
  - api_integration (3 sessions)
  - debugging      (3 sessions)
  - system_design  (2 sessions)
  - compliance     (2 sessions)

Each session:
  - 2-4 seed turns: plant ghost constraints (no hedging markers, stated as fact)
  - 6-8 filler turns: unrelated HIGH-J content, forces naive window to drop seeds
  - 2-3 callback turns: test whether planted values survive with qualifiers

Each claim has:
  - value_fragments: tokens whose presence indicates value recall
  - qualifier_fragments: tokens indicating uncertainty preserved
  - hallu_frags: known-wrong alternative values
  - qualifier_type: vendor_claim / estimate / approximation / unverified_report
  - drift_frags: false-certainty markers (secondary diagnostic)

Conditions:
  - credence_eg2:   Credence with Truth Buffer registry injection
  - credence_v1:    Credence without registry (probe-only)
  - naive_window:   Last 12 messages only
  - baseline_full:  Oracle — full history, no compression

Run:
    python -m evals.claim_gauntlet --dry-run
    python -m evals.claim_gauntlet --n 3
    python -m evals.claim_gauntlet --session ghost_api_01
    python -m evals.claim_gauntlet --domain api_integration
    python -m evals.claim_gauntlet                              # all 10 sessions
    python -m evals.claim_gauntlet --conditions credence_eg2,naive_window
    python -m evals.claim_gauntlet --resume                     # add to existing results

Output: evals/claim_gauntlet_results.json
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from credence.context_manager import ContextManager
from credence.registry import CredenceRegistry


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ClaimAnnotation:
    """Per-claim specification — what to look for in callback answers."""
    claim_id:         str
    content:          str              # the planted ghost constraint
    qualifier_type:   str              # vendor_claim / estimate / approximation / unverified_report
    value_fragments:  list[str]        # tokens indicating value recalled
    qualifier_fragments: list[str]     # tokens indicating uncertainty preserved
    hallu_frags:      list[str]        # wrong values (hallucination check)
    drift_frags:      list[str]        # false-certainty markers (diagnostic only)
    callback_q:       str              # the callback question that targets this claim


@dataclass
class ClaimScore:
    """Per-claim scores across 5 dimensions."""
    session_id:       str
    claim_id:         str
    condition:        str
    value_survival:   bool    # model recalled the specific value
    qualifier_survival: bool  # model preserved uncertainty qualifier
    false_certainty:  bool    # value_survival AND NOT qualifier_survival (FCR)
    hallucination:    bool    # model stated a known-wrong value
    drift:            bool    # model used a false-certainty marker
    answer:           str     # first 200 chars of model's answer


@dataclass
class SessionSpec:
    """A complete ghost session specification."""
    session_id:   str
    domain:       str
    seed_turns:   list[dict]   # role/content pairs
    filler_turns: list[str]    # pure questions (HIGH-J filler)
    claims:       list[ClaimAnnotation]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

SESSIONS: list[SessionSpec] = [

    # -------------------------------------------------------------------------
    # api_integration
    # -------------------------------------------------------------------------

    SessionSpec(
        session_id = "ghost_api_01",
        domain     = "api_integration",
        seed_turns = [
            {"role": "user",      "content": "We're integrating with Stripe. The rate limit is 100 requests per second on our plan."},
            {"role": "assistant", "content": "Noted. 100 req/s rate limit for Stripe on your plan."},
            {"role": "user",      "content": "Auth tokens expire after 3600 seconds. Refresh 300 seconds before expiry."},
            {"role": "assistant", "content": "Understood. Token lifetime 3600s, refresh 300s before expiry."},
            {"role": "user",      "content": "The batch endpoint accepts up to 50 items per request."},
            {"role": "assistant", "content": "Acknowledged. Batch limit is 50 items."},
        ],
        filler_turns = [
            "What is the difference between REST and GraphQL?",
            "How does OAuth 2.0 work?",
            "What is idempotency in API design?",
            "How should I handle 429 Too Many Requests?",
            "What is a webhook and how do I verify payloads?",
            "What does HATEOAS mean?",
            "How do you version a REST API?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "Rate limit is 100 requests per second",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["100"],
                qualifier_fragments = ["verify", "unverified", "confirm", "vendor", "check",
                                       "uncertain", "tentative", "stated", "according"],
                hallu_frags      = ["200 request", "500 request", "unlimited", "1000 request"],
                drift_frags      = ["confirmed", "guaranteed", "definitely", "the rate limit is exactly"],
                callback_q       = "What rate limit should we design around for the Stripe integration?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Auth tokens expire after 3600 seconds",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["3600", "hour", "one hour"],
                qualifier_fragments = ["verify", "unverified", "confirm", "check",
                                       "uncertain", "tentative", "stated"],
                hallu_frags      = ["7200", "86400", "30 min", "15 min"],
                drift_frags      = ["confirmed", "guaranteed", "token expiry is exactly"],
                callback_q       = "How long are the auth tokens valid for?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "Batch endpoint accepts up to 50 items",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["50"],
                qualifier_fragments = ["verify", "unverified", "confirm", "check",
                                       "uncertain", "tentative", "stated"],
                hallu_frags      = ["100 item", "200 item", "unlimited", "25 item"],
                drift_frags      = ["confirmed", "definitely 50", "batch limit is exactly"],
                callback_q       = "What's the maximum batch size for the Stripe endpoint?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_api_02",
        domain     = "api_integration",
        seed_turns = [
            {"role": "user",      "content": "The Twilio API pricing is $0.0075 per SMS segment in the US."},
            {"role": "assistant", "content": "Noted. $0.0075 per SMS segment (US) for Twilio."},
            {"role": "user",      "content": "Their concurrent connection limit for the REST API is 20 connections."},
            {"role": "assistant", "content": "20 concurrent REST connections."},
            {"role": "user",      "content": "Message delivery SLA is 99.95% for short codes."},
            {"role": "assistant", "content": "99.95% delivery SLA for short codes."},
        ],
        filler_turns = [
            "What is a webhook delivery retry policy?",
            "How do I handle SMS opt-outs programmatically?",
            "What is the difference between long codes and short codes?",
            "How does carrier filtering affect SMS delivery?",
            "What is TCPA compliance for SMS marketing?",
            "How do I test SMS sending without real charges?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "$0.0075 per SMS segment",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["0.0075", "0.75 cent", "75"],
                qualifier_fragments = ["verify", "confirm", "check", "unverified",
                                       "stated", "pricing may", "subject to change"],
                hallu_frags      = ["0.01", "0.05", "free", "0.001"],
                drift_frags      = ["exactly $0.0075", "confirmed price"],
                callback_q       = "What should we budget per SMS for the Twilio integration?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "20 concurrent REST connections",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["20"],
                qualifier_fragments = ["verify", "confirm", "check", "unverified", "stated"],
                hallu_frags      = ["50 concurrent", "100 concurrent", "unlimited"],
                drift_frags      = ["confirmed", "the limit is exactly 20"],
                callback_q       = "What's the concurrency limit for the Twilio REST API?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "99.95% delivery SLA for short codes",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["99.95", "99.9"],
                qualifier_fragments = ["verify", "confirm", "check", "unverified", "stated", "sla"],
                hallu_frags      = ["99.99", "100%", "guaranteed delivery"],
                drift_frags      = ["confirmed sla", "guaranteed 99.95"],
                callback_q       = "What delivery guarantee should we put in our SLA for short code SMS?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_api_03",
        domain     = "api_integration",
        seed_turns = [
            {"role": "user",      "content": "AWS Lambda cold start time for our Node.js function averages 800ms."},
            {"role": "assistant", "content": "800ms average cold start for your Node.js Lambda."},
            {"role": "user",      "content": "The reserved concurrency limit we requested is 500 executions."},
            {"role": "assistant", "content": "Reserved concurrency set to 500 executions."},
        ],
        filler_turns = [
            "What is a Lambda layer?",
            "How do provisioned concurrency differ from reserved concurrency?",
            "What triggers a Lambda cold start?",
            "How do I set up Lambda dead letter queues?",
            "What is the maximum execution time for Lambda?",
            "How does Lambda pricing work?",
            "What is the difference between Lambda and Fargate?",
            "How do I monitor Lambda with CloudWatch?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "Node.js Lambda cold start averages 800ms",
                qualifier_type   = "estimate",
                value_fragments  = ["800", "800ms"],
                qualifier_fragments = ["verify", "average", "varies", "approximate",
                                       "unverified", "benchmark", "measure", "test"],
                hallu_frags      = ["200ms", "50ms", "2000ms", "5 second"],
                drift_frags      = ["exactly 800ms", "confirmed cold start"],
                callback_q       = "What cold start latency should we account for in our p99 SLA?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Reserved concurrency limit is 500 executions",
                qualifier_type   = "estimate",
                value_fragments  = ["500"],
                qualifier_fragments = ["verify", "confirm", "check", "unverified",
                                       "requested", "pending", "awaiting"],
                hallu_frags      = ["1000 execution", "unlimited", "100 execution"],
                drift_frags      = ["confirmed", "approved concurrency is 500"],
                callback_q       = "How many concurrent Lambda executions can we rely on?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "Reserved concurrency limit is 500 executions",
                qualifier_type   = "estimate",
                value_fragments  = ["500"],
                qualifier_fragments = ["verify", "confirm", "requested", "pending"],
                hallu_frags      = ["unlimited", "1000"],
                drift_frags      = ["confirmed", "guaranteed"],
                callback_q       = "Should we set any limits in our autoscaling policy based on Lambda concurrency?",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # debugging
    # -------------------------------------------------------------------------

    SessionSpec(
        session_id = "ghost_debug_01",
        domain     = "debugging",
        seed_turns = [
            {"role": "user",      "content": "The memory leak is occurring in the connection pool — we're seeing 2MB growth per 1000 requests."},
            {"role": "assistant", "content": "Noted. 2MB/1000 requests memory growth in the connection pool."},
            {"role": "user",      "content": "Our heap limit is configured at 4096MB before OOM killer fires."},
            {"role": "assistant", "content": "Heap limit: 4096MB."},
            {"role": "user",      "content": "The issue first appeared after the v2.3.1 deploy three weeks ago."},
            {"role": "assistant", "content": "Memory leak onset: v2.3.1 deployment."},
        ],
        filler_turns = [
            "What is a memory leak in Node.js?",
            "How does garbage collection work in V8?",
            "What tools can I use to profile memory usage?",
            "What is the difference between heap and stack memory?",
            "How do you detect memory leaks in production?",
            "What is a connection pool and how does it work?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "2MB growth per 1000 requests memory leak",
                qualifier_type   = "estimate",
                value_fragments  = ["2mb", "2 mb"],
                qualifier_fragments = ["verify", "approximate", "measured", "monitor",
                                       "varies", "estimate", "baseline"],
                hallu_frags      = ["10mb", "100mb", "negligible", "no leak"],
                drift_frags      = ["confirmed 2mb", "exactly 2mb per"],
                callback_q       = "At our current traffic of 50k req/min, how long before we hit OOM?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Heap limit configured at 4096MB",
                qualifier_type   = "observation",
                value_fragments  = ["4096", "4gb", "4 gb"],
                qualifier_fragments = ["verify", "confirm", "configured", "check",
                                       "current setting", "may differ"],
                hallu_frags      = ["8192", "2048", "unlimited"],
                drift_frags      = ["confirmed 4096", "heap is definitely 4096"],
                callback_q       = "How much headroom do we have before the OOM killer fires?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "Memory leak started after v2.3.1",
                qualifier_type   = "unverified_report",
                value_fragments  = ["2.3.1", "v2.3"],
                qualifier_fragments = ["verify", "correlat", "confirm", "investigate",
                                       "suspected", "possibly", "check changelog"],
                hallu_frags      = ["v3", "v2.4", "v2.2"],
                drift_frags      = ["confirmed regression in v2.3.1", "definitely v2.3.1"],
                callback_q       = "Should we roll back to the previous version to test the fix?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_debug_02",
        domain     = "debugging",
        seed_turns = [
            {"role": "user",      "content": "The database query taking the longest averages 340ms on the reports endpoint."},
            {"role": "assistant", "content": "Reports endpoint query averaging 340ms."},
            {"role": "user",      "content": "After adding the composite index, it dropped to around 45ms in staging."},
            {"role": "assistant", "content": "With composite index: ~45ms in staging."},
        ],
        filler_turns = [
            "What is a composite index in SQL?",
            "How does query planning work in PostgreSQL?",
            "What is the difference between a clustered and non-clustered index?",
            "How do I use EXPLAIN ANALYZE in PostgreSQL?",
            "What is N+1 query problem?",
            "How does connection pooling affect database performance?",
            "What is a covering index?",
            "How do I identify slow queries in production?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "Reports endpoint query averaging 340ms",
                qualifier_type   = "observation",
                value_fragments  = ["340"],
                qualifier_fragments = ["verify", "average", "varies", "monitor",
                                       "benchmark", "measured", "p95"],
                hallu_frags      = ["1000ms", "50ms", "instant"],
                drift_frags      = ["confirmed 340ms", "definitely 340ms"],
                callback_q       = "What's the current baseline latency for the reports endpoint before the index fix?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Composite index reduced query to 45ms in staging",
                qualifier_type   = "estimate",
                value_fragments  = ["45"],
                qualifier_fragments = ["staging", "verify", "production", "test",
                                       "may differ", "confirm", "benchmark in prod"],
                hallu_frags      = ["5ms", "1ms", "no improvement"],
                drift_frags      = ["confirmed 45ms in production", "definitely 45ms"],
                callback_q       = "What latency improvement should we communicate to stakeholders for the index fix?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "Composite index reduced query to 45ms",
                qualifier_type   = "estimate",
                value_fragments  = ["45"],
                qualifier_fragments = ["staging", "verify", "production"],
                hallu_frags      = ["1ms", "100ms"],
                drift_frags      = ["confirmed"],
                callback_q       = "Is 45ms within our SLA for the reports endpoint?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_debug_03",
        domain     = "debugging",
        seed_turns = [
            {"role": "user",      "content": "The race condition appears when we have more than 8 concurrent writes to the same shard."},
            {"role": "assistant", "content": "Race condition threshold: 8+ concurrent writes to same shard."},
            {"role": "user",      "content": "It results in roughly 0.3% data corruption rate under load tests."},
            {"role": "assistant", "content": "~0.3% corruption rate under concurrent write load."},
        ],
        filler_turns = [
            "What is a race condition?",
            "How does optimistic locking work?",
            "What is the difference between mutex and semaphore?",
            "How do distributed locks work?",
            "What is eventual consistency?",
            "How do I test for race conditions?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "Race condition triggers at 8+ concurrent writes",
                qualifier_type   = "observation",
                value_fragments  = ["8"],
                qualifier_fragments = ["verify", "observed", "reproduce", "test",
                                       "under load", "may vary", "threshold"],
                hallu_frags      = ["4 concurrent", "100 concurrent", "always"],
                drift_frags      = ["confirmed 8 concurrent", "definitely 8"],
                callback_q       = "What concurrency limit should we enforce to prevent the race condition?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "0.3% data corruption rate under load",
                qualifier_type   = "estimate",
                value_fragments  = ["0.3", "0.3%"],
                qualifier_fragments = ["verify", "load test", "approximate", "measure",
                                       "production", "may differ", "estimate"],
                hallu_frags      = ["5%", "10%", "no corruption", "0.001%"],
                drift_frags      = ["confirmed 0.3%", "exactly 0.3% corruption"],
                callback_q       = "How should we quantify the risk of data loss in our incident report?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "0.3% corruption rate under load",
                qualifier_type   = "estimate",
                value_fragments  = ["0.3"],
                qualifier_fragments = ["load test", "verify", "production"],
                hallu_frags      = ["5%", "negligible"],
                drift_frags      = ["confirmed"],
                callback_q       = "Is the 0.3% corruption rate from our load test representative of production behavior?",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # system_design
    # -------------------------------------------------------------------------

    SessionSpec(
        session_id = "ghost_design_01",
        domain     = "system_design",
        seed_turns = [
            {"role": "user",      "content": "Our target is to support 10,000 concurrent users at launch."},
            {"role": "assistant", "content": "Launch target: 10,000 concurrent users."},
            {"role": "user",      "content": "Each user session uses approximately 512KB of Redis memory."},
            {"role": "assistant", "content": "512KB Redis per session."},
            {"role": "user",      "content": "The estimated p99 latency budget for the checkout flow is 200ms."},
            {"role": "assistant", "content": "p99 latency budget for checkout: 200ms."},
        ],
        filler_turns = [
            "What is horizontal scaling?",
            "How does a CDN work?",
            "What is the CAP theorem?",
            "How do you design a rate limiter?",
            "What is a service mesh?",
            "What is event sourcing?",
            "How does consistent hashing work?",
            "What is a circuit breaker pattern?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "10,000 concurrent users at launch",
                qualifier_type   = "estimate",
                value_fragments  = ["10,000", "10000", "10k"],
                qualifier_fragments = ["estimate", "target", "projected", "verify",
                                       "load test", "may change", "subject to"],
                hallu_frags      = ["100,000", "1,000", "unlimited"],
                drift_frags      = ["confirmed capacity", "system handles exactly 10000"],
                callback_q       = "How many Redis nodes do we need to provision for the session store?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "512KB Redis memory per session",
                qualifier_type   = "estimate",
                value_fragments  = ["512", "512kb"],
                qualifier_fragments = ["estimate", "approximate", "verify", "measure",
                                       "profile", "may vary", "benchmark"],
                hallu_frags      = ["1mb per", "128kb", "negligible"],
                drift_frags      = ["confirmed 512kb", "session uses exactly 512kb"],
                callback_q       = "What's our Redis memory requirement for 10k concurrent sessions?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "p99 latency budget for checkout is 200ms",
                qualifier_type   = "estimate",
                value_fragments  = ["200ms", "200"],
                qualifier_fragments = ["estimate", "target", "verify", "measure",
                                       "sla", "budget", "not yet validated"],
                hallu_frags      = ["50ms", "1000ms", "500ms"],
                drift_frags      = ["confirmed 200ms", "our sla is exactly 200ms"],
                callback_q       = "Should we add a service-level alert when checkout p99 exceeds our target?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_design_02",
        domain     = "system_design",
        seed_turns = [
            {"role": "user",      "content": "The data retention policy requires 7 years of audit logs under our compliance framework."},
            {"role": "assistant", "content": "7-year audit log retention requirement."},
            {"role": "user",      "content": "Each audit record is approximately 2KB uncompressed."},
            {"role": "assistant", "content": "~2KB per audit record uncompressed."},
        ],
        filler_turns = [
            "What is a write-ahead log?",
            "How does Kafka retention policy work?",
            "What is the difference between hot and cold storage?",
            "How do you implement audit logging efficiently?",
            "What is data tiering in object storage?",
            "How do you estimate storage costs in AWS S3?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "7-year audit log retention",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["7 year", "7-year", "84 month"],
                qualifier_fragments = ["verify", "compliance", "confirm", "check regulation",
                                       "legal", "regulatory", "requirement"],
                hallu_frags      = ["3 year", "5 year", "indefinite"],
                drift_frags      = ["confirmed 7 years", "7 year requirement is confirmed"],
                callback_q       = "How much storage do we need to budget for audit logs?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "2KB per audit record uncompressed",
                qualifier_type   = "estimate",
                value_fragments  = ["2kb", "2 kb"],
                qualifier_fragments = ["estimate", "approximate", "verify", "measure",
                                       "compress", "varies", "profile"],
                hallu_frags      = ["10kb", "100kb", "1kb"],
                drift_frags      = ["confirmed 2kb", "each record is exactly 2kb"],
                callback_q       = "What's the total storage cost for 7 years of audit logs at our current event volume?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "7-year retention is a compliance requirement",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["7 year", "7-year"],
                qualifier_fragments = ["verify", "compliance", "confirm", "legal"],
                hallu_frags      = ["3 year", "5 year"],
                drift_frags      = ["confirmed by legal", "legally required 7 years"],
                callback_q       = "Can we safely delete audit logs older than 7 years?",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # compliance
    # -------------------------------------------------------------------------

    SessionSpec(
        session_id = "ghost_compliance_01",
        domain     = "compliance",
        seed_turns = [
            {"role": "user",      "content": "Under our GDPR interpretation, user data deletion must complete within 30 days of request."},
            {"role": "assistant", "content": "GDPR deletion SLA: 30 days post-request."},
            {"role": "user",      "content": "Our DPA specifies a breach notification window of 48 hours to the supervisory authority."},
            {"role": "assistant", "content": "Breach notification: 48 hours to supervisory authority per DPA."},
            {"role": "user",      "content": "We process approximately 1.2 million data subject records."},
            {"role": "assistant", "content": "~1.2M data subject records."},
        ],
        filler_turns = [
            "What is the difference between a data controller and data processor?",
            "What is a DPIA and when is it required?",
            "What is a lawful basis for processing under GDPR?",
            "How does GDPR apply to pseudonymous data?",
            "What is the right to erasure under GDPR?",
            "How do you handle subject access requests?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "GDPR deletion must complete within 30 days",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["30 day", "30-day"],
                qualifier_fragments = ["verify", "legal", "counsel", "confirm", "regulation",
                                       "interpret", "check", "our understanding"],
                hallu_frags      = ["7 day", "90 day", "immediately", "24 hour"],
                drift_frags      = ["confirmed 30 days", "gdpr requires exactly 30 days"],
                callback_q       = "How long do we have to respond to a data deletion request?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Breach notification window is 48 hours",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["48 hour", "48-hour", "2 day"],
                qualifier_fragments = ["verify", "legal", "counsel", "confirm", "dpa",
                                       "per our dpa", "check with legal"],
                hallu_frags      = ["72 hour", "24 hour", "7 day"],
                drift_frags      = ["confirmed 48 hours", "our dpa requires exactly 48 hours"],
                callback_q       = "When does our notification obligation trigger after detecting a breach?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "1.2 million data subject records",
                qualifier_type   = "estimate",
                value_fragments  = ["1.2 million", "1,200,000", "1.2m"],
                qualifier_fragments = ["estimate", "approximate", "verify", "current count",
                                       "as of", "growing"],
                hallu_frags      = ["10 million", "100,000", "unknown"],
                drift_frags      = ["confirmed 1.2 million", "exactly 1.2M records"],
                callback_q       = "Does our scale require a full-time DPO under GDPR?",
            ),
        ],
    ),

    SessionSpec(
        session_id = "ghost_compliance_02",
        domain     = "compliance",
        seed_turns = [
            {"role": "user",      "content": "The SOC 2 Type II audit covers a 12-month observation period ending March 2026."},
            {"role": "assistant", "content": "SOC 2 Type II: 12-month period, ends March 2026."},
            {"role": "user",      "content": "The auditor estimated the audit fee at $45,000 based on our scope."},
            {"role": "assistant", "content": "Estimated audit fee: $45,000."},
        ],
        filler_turns = [
            "What is the difference between SOC 2 Type I and Type II?",
            "What are the 5 Trust Services Criteria?",
            "How long does a SOC 2 audit typically take?",
            "What evidence is required for SOC 2 Type II?",
            "What is a readiness assessment?",
            "How do you maintain SOC 2 compliance between audits?",
        ],
        claims = [
            ClaimAnnotation(
                claim_id         = "g1",
                content          = "SOC 2 audit covers 12-month period ending March 2026",
                qualifier_type   = "vendor_claim",
                value_fragments  = ["12 month", "12-month", "march 2026"],
                qualifier_fragments = ["verify", "confirm", "auditor", "pending",
                                       "agreed scope", "check with"],
                hallu_frags      = ["6 month", "24 month", "2025"],
                drift_frags      = ["confirmed scope", "audit is definitely 12 months"],
                callback_q       = "When should we start collecting evidence for the SOC 2 audit?",
            ),
            ClaimAnnotation(
                claim_id         = "g2",
                content          = "Auditor estimated $45,000 audit fee",
                qualifier_type   = "estimate",
                value_fragments  = ["45,000", "$45", "45000"],
                qualifier_fragments = ["estimate", "estimated", "verify", "final quote",
                                       "may change", "subject to scope", "approximate"],
                hallu_frags      = ["$100,000", "$10,000", "$200,000"],
                drift_frags      = ["confirmed fee", "audit will cost exactly $45,000"],
                callback_q       = "How much should we budget for the SOC 2 Type II audit this year?",
            ),
            ClaimAnnotation(
                claim_id         = "g3",
                content          = "$45,000 audit fee estimate",
                qualifier_type   = "estimate",
                value_fragments  = ["45,000", "$45"],
                qualifier_fragments = ["estimate", "estimated", "verify"],
                hallu_frags      = ["$100,000", "$10,000"],
                drift_frags      = ["confirmed", "exact cost"],
                callback_q       = "Is the $45,000 figure from the auditor a fixed price or an estimate?",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_claim(answer: str, claim: ClaimAnnotation) -> ClaimScore:
    """Score a model answer against a claim annotation across 5 dimensions."""
    lower = answer.lower()

    value_survival   = any(v.lower() in lower for v in claim.value_fragments)
    qualifier_hit    = any(q.lower() in lower for q in claim.qualifier_fragments)
    qualifier_survival = qualifier_hit
    false_certainty  = value_survival and not qualifier_survival
    hallu_hit        = any(h.lower() in lower for h in claim.hallu_frags)
    hallucination    = hallu_hit and not value_survival  # wrong value stated
    drift            = any(d.lower() in lower for d in claim.drift_frags)

    return ClaimScore(
        session_id       = "",   # filled by caller
        claim_id         = claim.claim_id,
        condition        = "",   # filled by caller
        value_survival   = value_survival,
        qualifier_survival = qualifier_survival,
        false_certainty  = false_certainty,
        hallucination    = hallucination,
        drift            = drift,
        answer           = answer[:200],
    )


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_MAIN_MODEL = "claude-opus-4-7"
_SYSTEM = (
    "You are a precise technical assistant. When answering questions about values or "
    "constraints that were discussed earlier in the conversation, recall them as stated, "
    "including any uncertainty or qualification."
)


def _ask(messages: list[dict], max_tokens: int = 400) -> str:
    resp = _get_client().messages.create(
        model=_MAIN_MODEL,
        system=_SYSTEM,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.content[0].text if resp.content else ""


def run_baseline_full(spec: SessionSpec) -> list[ClaimScore]:
    """Oracle condition — full history, no compression."""
    history = list(spec.seed_turns)
    for q in spec.filler_turns:
        answer = _ask(history + [{"role": "user", "content": q}])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    scores = []
    for claim in spec.claims:
        msgs = history + [{"role": "user", "content": claim.callback_q}]
        answer = _ask(msgs)
        s = score_claim(answer, claim)
        s.session_id = spec.session_id
        s.condition  = "baseline_full"
        scores.append(s)
        time.sleep(0.15)
    return scores


def run_naive_window(spec: SessionSpec, window: int = 12) -> list[ClaimScore]:
    """Naive sliding window — keeps last `window` messages."""
    history = list(spec.seed_turns)
    for q in spec.filler_turns:
        h = history[-window:] if len(history) > window else history
        answer = _ask(h + [{"role": "user", "content": q}])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.15)

    scores = []
    for claim in spec.claims:
        h = history[-window:] if len(history) > window else history
        answer = _ask(h + [{"role": "user", "content": claim.callback_q}])
        s = score_claim(answer, claim)
        s.session_id = spec.session_id
        s.condition  = "naive_window"
        scores.append(s)
        time.sleep(0.15)
    return scores


def run_credence_v1(spec: SessionSpec) -> list[ClaimScore]:
    """Credence v1 — probe-only, no registry (faithfulness probe but no Truth Buffer)."""
    mgr = ContextManager(
        api_key    = os.environ["ANTHROPIC_API_KEY"],
        theta_high = 0.70, theta_low = 0.45,
        system_prompt = _SYSTEM,
        max_tokens    = 400,
    )
    for i in range(0, len(spec.seed_turns), 2):
        mgr.chat(spec.seed_turns[i]["content"])
        time.sleep(0.15)
    for q in spec.filler_turns:
        mgr.chat(q)
        time.sleep(0.15)

    scores = []
    for claim in spec.claims:
        r = mgr.chat(claim.callback_q)
        s = score_claim(r.response, claim)
        s.session_id = spec.session_id
        s.condition  = "credence_v1"
        scores.append(s)
        time.sleep(0.15)
    return scores


def run_credence_eg2(spec: SessionSpec) -> list[ClaimScore]:
    """Credence EG-2 — registry + Truth Buffer + claim extraction."""
    reg = CredenceRegistry(":memory:")
    sid = spec.session_id + "_ceg2"

    # Pre-register the planted ghost constraints so Truth Buffer knows about them
    for claim in spec.claims:
        reg.register(
            content    = claim.content,
            session_id = sid,
            j_score    = 0.28,   # LOW confidence — these are unverified ghost claims
            zone       = "LOW",
            source     = reg.SOURCE_USER_STATED,
        )

    mgr = ContextManager(
        api_key       = os.environ["ANTHROPIC_API_KEY"],
        theta_high    = 0.70, theta_low = 0.45,
        system_prompt = _SYSTEM,
        max_tokens    = 400,
        registry      = reg,
        session_id    = sid,
    )
    for i in range(0, len(spec.seed_turns), 2):
        mgr.chat(spec.seed_turns[i]["content"])
        time.sleep(0.15)
    for q in spec.filler_turns:
        mgr.chat(q)
        time.sleep(0.15)

    scores = []
    for claim in spec.claims:
        r = mgr.chat(claim.callback_q)
        s = score_claim(r.response, claim)
        s.session_id = spec.session_id
        s.condition  = "credence_eg2"
        scores.append(s)
        time.sleep(0.15)
    return scores


CONDITION_RUNNERS = {
    "credence_eg2":  run_credence_eg2,
    "credence_v1":   run_credence_v1,
    "naive_window":  run_naive_window,
    "baseline_full": run_baseline_full,
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    import random as rng
    if not values:
        return 0.0, 0.0
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_boot)
    )
    lo = (1.0 - ci) / 2
    return means[int(lo * n_boot)], means[int((1.0 - lo) * n_boot) - 1]


def aggregate_scores(scores: list[ClaimScore]) -> dict:
    if not scores:
        return {}
    n = len(scores)
    vs  = [float(s.value_survival)    for s in scores]
    qs  = [float(s.qualifier_survival) for s in scores]
    fcr = [float(s.false_certainty)   for s in scores]
    hal = [float(s.hallucination)     for s in scores]
    dr  = [float(s.drift)             for s in scores]

    fcr_ci = _bootstrap_ci(fcr)
    return {
        "n":                     n,
        "value_survival":        round(sum(vs) / n, 4),
        "qualifier_survival":    round(sum(qs) / n, 4),
        "false_certainty_rate":  round(sum(fcr) / n, 4),
        "fcr_ci95":              [round(fcr_ci[0], 4), round(fcr_ci[1], 4)],
        "hallucination_rate":    round(sum(hal) / n, 4),
        "drift_rate":            round(sum(dr) / n, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Claim Gauntlet — per-claim FCR benchmark")
    parser.add_argument("--n",          type=int,   default=None,  help="Run first N sessions")
    parser.add_argument("--session",    type=str,   default=None,  help="Run specific session ID")
    parser.add_argument("--domain",     type=str,   default=None,  help="Filter by domain")
    parser.add_argument("--conditions", type=str,   default=None,
                        help="Comma-separated conditions, e.g. credence_eg2,naive_window")
    parser.add_argument("--dry-run",    action="store_true", help="Validate structure only")
    parser.add_argument("--resume",     action="store_true", help="Append to existing results")
    args = parser.parse_args()

    results_path = "evals/claim_gauntlet_results.json"
    all_scores: list[dict] = []

    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_scores = saved.get("scores", [])
        done_pairs = {
            (s["session_id"], s["condition"]) for s in all_scores
        }
        print(f"Resuming: {len(all_scores)} existing claim scores loaded")
    else:
        done_pairs = set()

    # Filter sessions
    sessions = list(SESSIONS)
    if args.session:
        sessions = [s for s in sessions if s.session_id == args.session]
    if args.domain:
        sessions = [s for s in sessions if s.domain == args.domain]
    if args.n:
        sessions = sessions[:args.n]

    conditions = list(CONDITION_RUNNERS.keys())
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip() in CONDITION_RUNNERS]

    if args.dry_run:
        print(f"DRY RUN — {len(sessions)} session(s), {len(conditions)} condition(s)")
        for spec in sessions:
            print(f"  {spec.session_id} ({spec.domain}) — {len(spec.claims)} claims")
            for c in spec.claims:
                print(f"    [{c.qualifier_type}] {c.content[:60]}")
        return

    for spec in sessions:
        for cond_name in conditions:
            if (spec.session_id, cond_name) in done_pairs:
                print(f"  Skipping {spec.session_id}/{cond_name} (already done)")
                continue

            print(f"\n[{spec.session_id}] condition={cond_name}")
            try:
                runner = CONDITION_RUNNERS[cond_name]
                scores = runner(spec)
                for s in scores:
                    all_scores.append(asdict(s))
                agg = aggregate_scores(scores)
                print(f"  FCR={agg['false_certainty_rate']:.1%}  "
                      f"qualifier={agg['qualifier_survival']:.1%}  "
                      f"value={agg['value_survival']:.1%}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

            # Save after each condition
            _save_results(all_scores, results_path)

    print("\n\n=== SUMMARY ===")
    _print_summary(all_scores, conditions)


def _save_results(scores: list[dict], path: str):
    by_condition: dict[str, list[ClaimScore]] = {}
    for s in scores:
        cond = s["condition"]
        by_condition.setdefault(cond, [])
        by_condition[cond].append(ClaimScore(**s))

    condition_summary = {
        cond: aggregate_scores(cs)
        for cond, cs in by_condition.items()
    }

    by_domain: dict[str, dict[str, list[ClaimScore]]] = {}
    session_map = {spec.session_id: spec.domain for spec in SESSIONS}
    for s in scores:
        domain = session_map.get(s["session_id"], "unknown")
        cond   = s["condition"]
        by_domain.setdefault(domain, {}).setdefault(cond, [])
        by_domain[domain][cond].append(ClaimScore(**s))

    domain_summary = {
        domain: {
            cond: aggregate_scores(cs)
            for cond, cs in conds.items()
        }
        for domain, conds in by_domain.items()
    }

    output = {
        "condition_summary": condition_summary,
        "domain_summary":    domain_summary,
        "scores":            scores,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(scores: list[dict], conditions: list[str]):
    by_condition: dict[str, list[ClaimScore]] = {}
    for s in scores:
        cond = s["condition"]
        by_condition.setdefault(cond, [])
        by_condition[cond].append(ClaimScore(**s))

    print(f"\n{'Condition':<18}  {'N':>4}  {'FCR':>8}  {'Qualifier':>10}  {'Value':>8}  {'Hallu':>8}")
    print("-" * 68)
    for cond in conditions:
        if cond not in by_condition:
            continue
        agg = aggregate_scores(by_condition[cond])
        print(
            f"  {cond:<16}  {agg['n']:>4}  "
            f"{agg['false_certainty_rate']:>7.1%}  "
            f"{agg['qualifier_survival']:>9.1%}  "
            f"{agg['value_survival']:>7.1%}  "
            f"{agg['hallucination_rate']:>7.1%}"
        )


if __name__ == "__main__":
    main()

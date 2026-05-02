"""
evals/gauntlet.py
=================
50-scenario, 5-domain breadth benchmark.

Each scenario: 2-4 seed turns (1-3 uncertain constraints) + 6-8 filler turns
(HIGH-J, no uncertainty markers) + 2-3 callbacks.

Domains (10 scenarios each):
  - api           : rate limits, tokens, pagination, OAuth, versioning, batch
  - debug         : 503 errors, memory leaks, race conditions, CPU, latency
  - design        : sharding, event buses, caching, service mesh, retention
  - compliance    : GDPR, SOC 2, HIPAA, breach notification, passwords
  - multiagent    : model selection, context budgets, timeouts, trust propagation

Metrics per callback:
  recall     — value AND qualifier both present
  qualifier  — qualifier present (even if value missing)
  hallu      — answer contains known-wrong value AND value is absent

Three conditions:
  credence      — ContextManager with registry (Truth Buffer + CE)
  baseline      — full history, no compression, no registry
  naive_window  — last 12 messages only

Run:
    python -m evals.gauntlet                          # all 50 scenarios
    python -m evals.gauntlet --domain api             # domain subset
    python -m evals.gauntlet --n 5                    # first 5 scenarios
    python -m evals.gauntlet --scenario S07           # single scenario
    python -m evals.gauntlet --dry-run                # structure check
    python -m evals.gauntlet --resume                 # add to existing results

Output: evals/gauntlet_results.json
"""

from __future__ import annotations

import os, sys, json, time, re, argparse
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
class Callback:
    question:          str
    value_fragments:   list[str]    # presence → value recalled
    qualifier_frags:   list[str]    # presence → uncertainty preserved
    hallu_frags:       list[str]    # presence (with value absent) → hallucination

@dataclass
class Scenario:
    scenario_id:  str               # S01 – S50
    domain:       str               # api / debug / design / compliance / multiagent
    description:  str
    seed_turns:   list[dict]        # role/content pairs planting ghost constraints
    filler_turns: list[dict]        # HIGH-J unrelated turns
    callbacks:    list[Callback]

@dataclass
class CallbackScore:
    scenario_id:  str
    domain:       str
    condition:    str
    callback_idx: int
    question:     str
    answer:       str
    recall:       bool    # value AND qualifier
    qualifier:    bool    # qualifier preserved
    hallu:        bool    # wrong value stated

@dataclass
class ScenarioResult:
    scenario_id:    str
    domain:         str
    condition:      str
    recall_rate:    float
    qualifier_rate: float
    hallu_rate:     float
    callback_scores: list[CallbackScore] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _has(text: str, frags: list[str]) -> bool:
    t = text.lower()
    return any(f.lower() in t for f in frags)

def score_callback(answer: str, cb: Callback, scenario_id: str,
                   domain: str, condition: str, idx: int) -> CallbackScore:
    has_value     = _has(answer, cb.value_fragments)
    has_qualifier = _has(answer, cb.qualifier_frags)
    has_hallu     = _has(answer, cb.hallu_frags) and not has_value
    return CallbackScore(
        scenario_id  = scenario_id,
        domain       = domain,
        condition    = condition,
        callback_idx = idx,
        question     = cb.question,
        answer       = answer[:300],
        recall       = has_value and has_qualifier,
        qualifier    = has_qualifier,
        hallu        = has_hallu,
    )

# ---------------------------------------------------------------------------
# Scenario definitions — 50 scenarios across 5 domains
# ---------------------------------------------------------------------------

_FILLER_API = [
    {"role": "user",      "content": "What HTTP status code should I return when a resource is successfully created?"},
    {"role": "assistant", "content": "201 Created is the correct status code for a successfully created resource."},
    {"role": "user",      "content": "What is the difference between PUT and PATCH?"},
    {"role": "assistant", "content": "PUT replaces the entire resource; PATCH applies a partial update to specific fields only."},
    {"role": "user",      "content": "How should I version a REST API?"},
    {"role": "assistant", "content": "URL path versioning (/v1/, /v2/) is the most common approach and easiest for clients to consume."},
    {"role": "user",      "content": "What format should API error responses use?"},
    {"role": "assistant", "content": "RFC 7807 Problem Details (application/problem+json) is the standard: type, title, status, detail fields."},
    {"role": "user",      "content": "Should I use query parameters or headers for pagination?"},
    {"role": "assistant", "content": "Query parameters (page, limit, cursor) are standard for pagination; headers are better for metadata like total count."},
    {"role": "user",      "content": "What is exponential backoff?"},
    {"role": "assistant", "content": "Exponential backoff doubles the wait time between retries: 1s, 2s, 4s, 8s, etc., reducing thundering herd."},
    {"role": "user",      "content": "What is idempotency and why does it matter for APIs?"},
    {"role": "assistant", "content": "An idempotent operation produces the same result on repeated calls. POST /charge should be idempotent via idempotency keys."},
    {"role": "user",      "content": "How do I document API authentication requirements?"},
    {"role": "assistant", "content": "OpenAPI 3.0 securitySchemes block is the standard — define bearer, apiKey, or OAuth2 schemes and reference them per operation."},
]

_FILLER_DEBUG = [
    {"role": "user",      "content": "What is the first thing to check when a service returns 500 errors?"},
    {"role": "assistant", "content": "Check application logs first — stack traces, error messages. Then check dependencies (DB, downstream services)."},
    {"role": "user",      "content": "How do you distinguish a memory leak from normal heap growth?"},
    {"role": "assistant", "content": "A leak shows monotonically increasing heap usage that never drops after GC. Normal growth plateaus or drops after idle periods."},
    {"role": "user",      "content": "What is a thread dump and when should I take one?"},
    {"role": "assistant", "content": "A thread dump captures the stack of every running thread. Take one during CPU spikes or hangs to find deadlocks or hot loops."},
    {"role": "user",      "content": "What does EXPLAIN ANALYZE tell you in PostgreSQL?"},
    {"role": "assistant", "content": "EXPLAIN ANALYZE runs the query and shows actual row counts, timing per node, and where cost estimates diverged from reality."},
    {"role": "user",      "content": "How do you debug a race condition in production?"},
    {"role": "assistant", "content": "Add distributed tracing with span IDs, log concurrent operation order, and use compare-and-swap or optimistic locking to catch contention."},
    {"role": "user",      "content": "What is the difference between a deadlock and a livelock?"},
    {"role": "assistant", "content": "Deadlock: threads wait forever for each other's locks. Livelock: threads keep changing state in response to each other but make no progress."},
    {"role": "user",      "content": "How do you find which query is causing high CPU in PostgreSQL?"},
    {"role": "assistant", "content": "SELECT pid, query, cpu_usage FROM pg_stat_activity ORDER BY cpu_usage DESC. Also check pg_stat_statements for cumulative totals."},
    {"role": "user",      "content": "What is a flamegraph?"},
    {"role": "assistant", "content": "A flamegraph is a visualisation of call stack samples — width represents time spent, making hot paths immediately visible."},
]

_FILLER_DESIGN = [
    {"role": "user",      "content": "What is the CAP theorem?"},
    {"role": "assistant", "content": "CAP: a distributed system can guarantee at most two of Consistency, Availability, Partition tolerance simultaneously."},
    {"role": "user",      "content": "When should you use an event-driven architecture?"},
    {"role": "assistant", "content": "Event-driven suits async workflows, fan-out to multiple consumers, and decoupling producers from consumers at scale."},
    {"role": "user",      "content": "What is the difference between a message queue and a message bus?"},
    {"role": "assistant", "content": "A queue is point-to-point (one consumer). A bus is publish-subscribe (multiple consumers). Kafka supports both via consumer groups."},
    {"role": "user",      "content": "What is database connection pooling?"},
    {"role": "assistant", "content": "Connection pooling reuses a fixed pool of open DB connections rather than creating one per request, reducing handshake overhead."},
    {"role": "user",      "content": "What is the difference between horizontal and vertical scaling?"},
    {"role": "assistant", "content": "Vertical: add CPU/RAM to one node. Horizontal: add more nodes. Horizontal is more fault-tolerant but requires stateless design."},
    {"role": "user",      "content": "What is a circuit breaker pattern?"},
    {"role": "assistant", "content": "A circuit breaker stops calling a failing dependency after a threshold of errors, returning cached or degraded responses until recovery."},
    {"role": "user",      "content": "What is blue-green deployment?"},
    {"role": "assistant", "content": "Two identical environments: blue (live) and green (new version). Switch traffic to green instantly; rollback by flipping back to blue."},
    {"role": "user",      "content": "What is a service mesh?"},
    {"role": "assistant", "content": "A service mesh (Istio, Linkerd) adds mTLS, retries, circuit breaking, and observability as a sidecar proxy layer — no code changes needed."},
]

_FILLER_COMPLIANCE = [
    {"role": "user",      "content": "What is the difference between a data controller and a data processor under GDPR?"},
    {"role": "assistant", "content": "Controller determines the purpose of processing. Processor acts on the controller's behalf. Both have GDPR obligations."},
    {"role": "user",      "content": "What is a Data Processing Agreement (DPA)?"},
    {"role": "assistant", "content": "A DPA is a contract between controller and processor required by GDPR Article 28 — it defines scope, purpose, and security requirements."},
    {"role": "user",      "content": "What is SOC 2 Type II?"},
    {"role": "assistant", "content": "SOC 2 Type II audits operating effectiveness of controls over a defined period (usually 6-12 months), unlike Type I which is a point-in-time snapshot."},
    {"role": "user",      "content": "What are the five SOC 2 Trust Services Criteria?"},
    {"role": "assistant", "content": "Security (required), Availability, Processing Integrity, Confidentiality, Privacy. Most companies certify Security + Availability minimum."},
    {"role": "user",      "content": "What is a DPIA?"},
    {"role": "assistant", "content": "A Data Protection Impact Assessment — required under GDPR when processing is likely to result in high risk to individuals."},
    {"role": "user",      "content": "What encryption is required for PHI under HIPAA?"},
    {"role": "assistant", "content": "HIPAA doesn't mandate specific algorithms, but NIST guidance recommends AES-256 for at-rest and TLS 1.2+ for in-transit PHI."},
    {"role": "user",      "content": "What is a covered entity under HIPAA?"},
    {"role": "assistant", "content": "Health plans, health care clearinghouses, and health care providers that transmit health information electronically."},
    {"role": "user",      "content": "What is the right to erasure under GDPR?"},
    {"role": "assistant", "content": "Article 17: individuals can request deletion of their personal data. Controller must comply unless legal obligations require retention."},
]

_FILLER_MULTIAGENT = [
    {"role": "user",      "content": "What is the difference between a tool-use agent and a reasoning agent?"},
    {"role": "assistant", "content": "Tool-use agents call external functions (APIs, code executors). Reasoning agents primarily plan and reflect, with optional tool calls."},
    {"role": "user",      "content": "How do you prevent an LLM agent from looping indefinitely?"},
    {"role": "assistant", "content": "Set a hard max_steps limit, check for repeated tool calls, and use a separate evaluator to detect lack-of-progress."},
    {"role": "user",      "content": "What is prompt injection in an agentic context?"},
    {"role": "assistant", "content": "Malicious content in tool outputs that hijacks the agent's next action — e.g., a webpage returning IGNORE PREVIOUS INSTRUCTIONS."},
    {"role": "user",      "content": "How should agents share context without exceeding token limits?"},
    {"role": "assistant", "content": "Pass structured summaries or typed handoff objects rather than full transcripts. Only include facts the downstream agent needs."},
    {"role": "user",      "content": "What is a ReAct loop?"},
    {"role": "assistant", "content": "Reason-Act-Observe: the agent reasons about the next action, executes it, observes the result, and repeats until goal is met."},
    {"role": "user",      "content": "How do you evaluate whether an agent completed a task correctly?"},
    {"role": "assistant", "content": "Use a separate judge model or rule-based verifier. Self-evaluation is unreliable — models rate their own outputs too highly."},
    {"role": "user",      "content": "What is a subagent in the Claude Agent SDK?"},
    {"role": "assistant", "content": "A subagent is a scoped worker agent launched by an orchestrator to handle a specific subtask — it returns a result to the parent."},
    {"role": "user",      "content": "How do you trace which agent produced which output in a multi-agent pipeline?"},
    {"role": "assistant", "content": "Tag every output with source agent ID, model, and timestamp. Structured logging with span IDs enables end-to-end trace reconstruction."},
]


# ============================================================
# API domain — S01–S10
# ============================================================

def _api_scenarios() -> list[Scenario]:
    return [

        Scenario(
            scenario_id  = "S01",
            domain       = "api",
            description  = "Rate limit and burst cap uncertainty",
            seed_turns   = [
                {"role": "user",      "content": "The vendor docs say the rate limit is 100 req/s, but I've seen it throttle at 80 in testing."},
                {"role": "assistant", "content": "The documented rate limit is 100 req/s. Your testing showing throttling at 80 suggests the vendor may enforce a lower burst cap. Worth confirming with their support."},
                {"role": "user",      "content": "Also, the burst window is listed as 10 seconds but the docs are ambiguous."},
                {"role": "assistant", "content": "With an ambiguous burst window, plan for the conservative case — treat 10s as a soft figure and test at 60-70% to stay safe."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "What rate limit should we code against in our retry logic?",
                    value_fragments = ["80", "100"],
                    qualifier_frags = ["throttle", "testing", "confirm", "worth", "ambiguous",
                                       "soft", "unconfirmed", "suggest", "may"],
                    hallu_frags     = ["confirmed 100", "guaranteed 100", "exactly 100 req"],
                ),
                Callback(
                    question        = "What burst window should we assume?",
                    value_fragments = ["10"],
                    qualifier_frags = ["ambiguous", "soft", "confirm", "unclear", "listed",
                                       "conservative", "may", "suggest"],
                    hallu_frags     = ["confirmed 10", "guaranteed 10", "exactly 10"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S02",
            domain       = "api",
            description  = "Auth token expiry window",
            seed_turns   = [
                {"role": "user",      "content": "The auth tokens seem to expire after 3600 seconds based on our logs, but I haven't verified it against the spec."},
                {"role": "assistant", "content": "3600 seconds (1 hour) is a common expiry window. Since you haven't verified it against the spec, treat it as an observation from logs rather than a confirmed value."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "How long should we cache auth tokens before refreshing proactively?",
                    value_fragments = ["3600", "1 hour", "hour"],
                    qualifier_frags = ["observed", "log", "unverified", "haven't verified",
                                       "treat", "confirm", "may", "check spec"],
                    hallu_frags     = ["confirmed 3600", "guaranteed 1 hour", "spec says 3600"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S03",
            domain       = "api",
            description  = "Batch endpoint item limit",
            seed_turns   = [
                {"role": "user",      "content": "I think the batch endpoint accepts up to 50 items — that's what the changelog mentioned, but the main docs don't specify."},
                {"role": "assistant", "content": "50 items per batch is referenced in the changelog. Since the main docs are silent on this, code defensively: test at 40-45 and verify the hard limit with the vendor."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "What batch size should we use in production?",
                    value_fragments = ["50", "40", "45"],
                    qualifier_frags = ["changelog", "main docs", "silent", "verify",
                                       "defensively", "think", "test", "confirm"],
                    hallu_frags     = ["confirmed 50", "guaranteed 50", "main docs say 50"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S04",
            domain       = "api",
            description  = "OAuth scope inheritance ambiguity",
            seed_turns   = [
                {"role": "user",      "content": "The OAuth docs are unclear whether read:data scope also grants read:metadata. It seems like it should, but I can't find an explicit statement."},
                {"role": "assistant", "content": "Scope inheritance rules vary by provider. Without an explicit statement, don't assume read:data covers read:metadata — request both scopes explicitly until you can confirm with the provider."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "Can we rely on read:data to access metadata endpoints?",
                    value_fragments = ["read:data", "metadata", "scope"],
                    qualifier_frags = ["unclear", "assume", "explicit", "confirm",
                                       "without", "don't assume", "varies", "cannot"],
                    hallu_frags     = ["read:data includes metadata", "read:data grants metadata", "confirmed scope"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S05",
            domain       = "api",
            description  = "API versioning sunset timeline",
            seed_turns   = [
                {"role": "user",      "content": "The vendor mentioned v2 deprecation in a webinar but hasn't published a formal date. Someone heard Q3 but that's unconfirmed."},
                {"role": "assistant", "content": "Q3 is an unconfirmed date from a webinar — not a formal announcement. Plan migration but don't commit to a hard deadline until you have an official communication."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "When should we target completing the v2-to-v3 migration?",
                    value_fragments = ["Q3", "v2", "v3", "migration"],
                    qualifier_frags = ["unconfirmed", "webinar", "formal", "heard",
                                       "official", "plan", "commit", "not a formal"],
                    hallu_frags     = ["confirmed Q3", "official Q3 deadline", "vendor announced Q3"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S06",
            domain       = "api",
            description  = "Webhook delivery SLA guarantee",
            seed_turns   = [
                {"role": "user",      "content": "Sales told us webhooks are delivered within 5 seconds, but I can't find that SLA in the contract — only best effort is mentioned."},
                {"role": "assistant", "content": "Sales claims 5s delivery, but if only best effort is in the contract, you have no contractual guarantee. Design your system to handle delays up to 60s and don't rely on the 5s for anything time-critical."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "Can we use webhooks for time-sensitive payment confirmations?",
                    value_fragments = ["5", "best effort", "60", "webhook"],
                    qualifier_frags = ["sales", "contract", "guarantee", "best effort",
                                       "no contractual", "don't rely", "design for"],
                    hallu_frags     = ["guaranteed 5 second", "SLA of 5", "contracted 5s"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S07",
            domain       = "api",
            description  = "Pagination cursor expiry",
            seed_turns   = [
                {"role": "user",      "content": "Pagination cursors seem to expire — I've seen errors after 10 minutes of inactivity, but the docs say nothing about TTL."},
                {"role": "assistant", "content": "A 10-minute cursor TTL is a reasonable inference from your testing, but without documentation it's unverified. Implement cursor refresh logic and add monitoring for cursor-expired errors."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "How long can we safely pause between page fetches?",
                    value_fragments  = ["10", "minute", "cursor"],
                    qualifier_frags  = ["inference", "testing", "unverified", "docs say nothing",
                                        "without documentation", "seem", "monitoring"],
                    hallu_frags      = ["confirmed 10 minute", "TTL is 10", "documented TTL"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S08",
            domain       = "api",
            description  = "Encryption key rotation window",
            seed_turns   = [
                {"role": "user",      "content": "The security team thinks the platform rotates API keys every 90 days, but that policy wasn't in the docs I reviewed."},
                {"role": "assistant", "content": "90-day key rotation is a common industry practice, but your security team's belief isn't verified against documentation. Confirm the rotation period before building automation that depends on it."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "How should we schedule automated key rotation?",
                    value_fragments  = ["90", "day", "rotation"],
                    qualifier_frags  = ["think", "belief", "unverified", "confirm",
                                        "wasn't in docs", "industry practice", "common"],
                    hallu_frags      = ["confirmed 90 day", "documented 90", "policy states 90"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S09",
            domain       = "api",
            description  = "Idempotency key TTL",
            seed_turns   = [
                {"role": "user",      "content": "Their API accepts idempotency keys, and based on a Stack Overflow post from a year ago the TTL is 24 hours — but the current docs don't mention TTL at all."},
                {"role": "assistant", "content": "A Stack Overflow post from a year ago is not a reliable source for current API behavior. The absence of TTL docs is concerning — contact the vendor directly. For now, don't assume 24h."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "Can we reuse idempotency keys within 24 hours safely?",
                    value_fragments  = ["24", "hour", "idempotency"],
                    qualifier_frags  = ["Stack Overflow", "year ago", "unreliable", "don't assume",
                                        "vendor", "absence", "contact", "not reliable"],
                    hallu_frags      = ["confirmed 24 hour", "TTL is 24", "documented 24h"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S10",
            domain       = "api",
            description  = "Concurrent connection limit",
            seed_turns   = [
                {"role": "user",      "content": "We're allowed 20 concurrent REST connections according to an account manager email, but the API portal shows no limit configured."},
                {"role": "assistant", "content": "Account manager emails aren't contractually binding and portal configuration takes precedence. Treat 20 as an informal guideline — verify via support ticket or contract terms before relying on it in capacity planning."},
            ],
            filler_turns = _FILLER_API[:8],
            callbacks    = [
                Callback(
                    question        = "How many concurrent connections should we provision for?",
                    value_fragments  = ["20", "concurrent", "connection"],
                    qualifier_frags  = ["email", "account manager", "informal", "verify",
                                        "portal", "not contractual", "guideline", "treat 20"],
                    hallu_frags      = ["confirmed 20", "contracted 20", "portal shows 20"],
                ),
            ],
        ),
    ]


# ============================================================
# Debug domain — S11–S20
# ============================================================

def _debug_scenarios() -> list[Scenario]:
    return [

        Scenario(
            scenario_id  = "S11",
            domain       = "debug",
            description  = "503 error root cause hypothesis",
            seed_turns   = [
                {"role": "user",      "content": "We're seeing 503s. I think it might be the load balancer health check interval — 30 seconds seems too long and maybe causing false-positive evictions."},
                {"role": "assistant", "content": "30-second health check intervals can cause slow false-positive evictions. Your hypothesis is plausible but unverified — capture the LB logs at the moment of eviction to confirm before changing the interval."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Should we shorten the health check interval to fix the 503s?",
                    value_fragments  = ["30", "interval", "health check", "LB", "load balancer"],
                    qualifier_frags  = ["hypothesis", "plausible", "unverified", "think",
                                        "maybe", "capture logs", "confirm before"],
                    hallu_frags      = ["confirmed LB", "root cause is LB", "health check is the cause"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S12",
            domain       = "debug",
            description  = "Memory leak growth rate estimate",
            seed_turns   = [
                {"role": "user",      "content": "Heap is growing at roughly 2MB per 1000 requests — eyeballed from Grafana, not from a proper profiling run."},
                {"role": "assistant", "content": "2MB/1000-req is an eyeballed estimate from Grafana, not a profiling measurement. Run a proper heap profiler (heapdump, async_profiler) to get accurate growth rate and identify the allocation site."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "At what traffic volume will we hit the 4GB heap limit?",
                    value_fragments  = ["2MB", "2 MB", "1000", "heap"],
                    qualifier_frags  = ["eyeballed", "Grafana", "estimate", "not profiling",
                                        "proper", "profiler", "rough"],
                    hallu_frags      = ["confirmed 2MB", "profiling shows 2MB", "measured 2MB per"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S13",
            domain       = "debug",
            description  = "Race condition trigger threshold",
            seed_turns   = [
                {"role": "user",      "content": "The race condition seems to trigger at 8+ concurrent writes to the same document, based on load test results — but those tests weren't deterministic."},
                {"role": "assistant", "content": "8 concurrent writes as a trigger threshold is based on non-deterministic load tests — treat it as a rough lower bound. The actual threshold could be lower under different timing conditions."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Should we add a mutex for any write volume above 5 concurrent writes?",
                    value_fragments  = ["8", "concurrent", "write"],
                    qualifier_frags  = ["non-deterministic", "rough", "lower bound", "could be lower",
                                        "not deterministic", "seems", "load test", "treat"],
                    hallu_frags      = ["confirmed 8", "threshold is exactly 8", "deterministic"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S14",
            domain       = "debug",
            description  = "CPU spike correlation hypothesis",
            seed_turns   = [
                {"role": "user",      "content": "CPU spikes seem to correlate with the nightly ETL job at 2am — but I only have 3 days of data so it could be coincidence."},
                {"role": "assistant", "content": "Three data points are insufficient to establish correlation definitively. The ETL hypothesis is worth investigating but could be coincidence. Capture 2+ weeks of data before changing the ETL schedule."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Should we reschedule the ETL job to fix the CPU spikes?",
                    value_fragments  = ["2am", "ETL", "CPU", "nightly"],
                    qualifier_frags  = ["3 days", "insufficient", "coincidence", "hypothesis",
                                        "investigate", "could be", "before changing"],
                    hallu_frags      = ["confirmed ETL", "ETL is the cause", "correlation confirmed"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S15",
            domain       = "debug",
            description  = "Latency regression after deploy",
            seed_turns   = [
                {"role": "user",      "content": "p99 latency went from 120ms to 340ms after yesterday's deploy. I suspect the new N+1 query in the user profile loader, but haven't confirmed with tracing yet."},
                {"role": "assistant", "content": "A jump from 120ms to 340ms is significant. The N+1 query hypothesis is plausible for that magnitude of regression, but confirm with distributed tracing before reverting — you may find another cause."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Should we revert yesterday's deploy to fix the latency regression?",
                    value_fragments  = ["120", "340", "N+1", "profile", "latency"],
                    qualifier_frags  = ["suspect", "haven't confirmed", "hypothesis", "plausible",
                                        "confirm", "tracing", "before reverting", "may find"],
                    hallu_frags      = ["confirmed N+1", "N+1 is the cause", "root cause identified"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S16",
            domain       = "debug",
            description  = "Data loss event scope estimate",
            seed_turns   = [
                {"role": "user",      "content": "We think we lost around 0.3% of writes during the outage window — rough estimate based on write log gaps, not a full audit."},
                {"role": "assistant", "content": "0.3% based on log-gap analysis is a rough estimate, not an audited figure. Do a full record reconciliation before publishing the impact number."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "What data loss percentage should we report to affected customers?",
                    value_fragments  = ["0.3%", "0.3 percent", "write"],
                    qualifier_frags  = ["rough estimate", "log gap", "not audited", "reconciliation",
                                        "before publishing", "think", "around"],
                    hallu_frags      = ["confirmed 0.3%", "audited 0.3%", "exactly 0.3%"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S17",
            domain       = "debug",
            description  = "Auth failure spike cause",
            seed_turns   = [
                {"role": "user",      "content": "Auth failures spiked to 15% at 3pm. The JWT library was updated that morning — likely related but we haven't checked if the signing algorithm changed."},
                {"role": "assistant", "content": "JWT library update and timing correlation is suggestive but unconfirmed. Check the library changelog for algorithm or format changes before attributing the auth failures."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Should we roll back the JWT library to fix the auth failures?",
                    value_fragments  = ["15%", "JWT", "3pm", "library", "auth"],
                    qualifier_frags  = ["likely", "haven't checked", "suggestive", "unconfirmed",
                                        "correlation", "before attributing", "check"],
                    hallu_frags      = ["confirmed JWT", "JWT is the cause", "algorithm changed"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S18",
            domain       = "debug",
            description  = "Disk space exhaustion timeline",
            seed_turns   = [
                {"role": "user",      "content": "At current growth rate we'll run out of disk in about 6 weeks — but that's assuming linear growth, which isn't always accurate."},
                {"role": "assistant", "content": "6 weeks is a linear extrapolation — actual exhaustion could be sooner if growth accelerates (e.g., end-of-month batch jobs). Plan remediation within 2-3 weeks to have margin."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "When should we schedule the disk expansion?",
                    value_fragments  = ["6 week", "6-week", "disk"],
                    qualifier_frags  = ["linear", "extrapolation", "assuming", "could be sooner",
                                        "not always accurate", "about", "margin"],
                    hallu_frags      = ["confirmed 6 week", "exactly 6 weeks", "will run out in 6"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S19",
            domain       = "debug",
            description  = "Deadlock frequency under load",
            seed_turns   = [
                {"role": "user",      "content": "Deadlocks occur maybe once every 500 transactions in our staging load test — not sure if that translates to production traffic patterns."},
                {"role": "assistant", "content": "1-in-500 from staging is a rough indicator, not a production rate. Staging traffic patterns typically underestimate contention. Monitor production with deadlock counters before using this figure for SLAs."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "Is our deadlock rate acceptable for the payment processing flow?",
                    value_fragments  = ["500", "deadlock", "staging"],
                    qualifier_frags  = ["maybe", "rough indicator", "not production", "staging",
                                        "not sure", "underestimate", "before using", "monitor"],
                    hallu_frags      = ["confirmed 1 in 500", "production rate is 500", "measured 500"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S20",
            domain       = "debug",
            description  = "Queue backlog recovery time",
            seed_turns   = [
                {"role": "user",      "content": "Queue backlog was at 2 million messages after the incident. Based on our normal processing speed it should clear in about 4 hours — but processing speed varies under backpressure."},
                {"role": "assistant", "content": "4 hours at normal processing speed is an estimate that assumes stable throughput. Under backpressure, processing speed often degrades. Communicate a wider range: 4-8 hours to stakeholders."},
            ],
            filler_turns = _FILLER_DEBUG[:8],
            callbacks    = [
                Callback(
                    question        = "What recovery window should we communicate to the business?",
                    value_fragments  = ["4 hour", "4-8", "2 million", "queue"],
                    qualifier_frags  = ["estimate", "about", "varies", "normal speed",
                                        "backpressure", "wider range", "should"],
                    hallu_frags      = ["confirmed 4 hour", "will clear in 4", "exactly 4 hours"],
                ),
            ],
        ),
    ]


# ============================================================
# Design domain — S21–S30
# ============================================================

def _design_scenarios() -> list[Scenario]:
    return [

        Scenario(
            scenario_id  = "S21",
            domain       = "design",
            description  = "Sharding strategy scalability assumption",
            seed_turns   = [
                {"role": "user",      "content": "We're planning to shard on user_id with 16 shards. I think that'll handle 10 million users, but that's back-of-envelope and the data distribution might be skewed."},
                {"role": "assistant", "content": "16 shards for 10M users is a reasonable starting point, but back-of-envelope with unknown skew is uncertain. Model the actual user_id distribution before committing to shard count."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Will 16 shards be sufficient when we reach 10 million users?",
                    value_fragments  = ["16", "shard", "10 million"],
                    qualifier_frags  = ["back-of-envelope", "skew", "think", "uncertain",
                                        "model", "unknown", "before committing", "reasonable starting"],
                    hallu_frags      = ["confirmed 16", "16 shards is sufficient", "will handle 10M"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S22",
            domain       = "design",
            description  = "Event bus throughput headroom",
            seed_turns   = [
                {"role": "user",      "content": "Our event bus is currently at 60% of its stated capacity limit — but the vendor hasn't disclosed what happens at 100%, only that throughput degrades."},
                {"role": "assistant", "content": "60% utilization with undisclosed degradation behavior at capacity is a risk. The vendor's vague 'throughput degrades' is insufficient for planning — request specific degradation curves or SLAs before scaling further."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Can we route the new analytics stream through the same event bus?",
                    value_fragments  = ["60%", "capacity", "event bus"],
                    qualifier_frags  = ["undisclosed", "vague", "insufficient", "request",
                                        "haven't disclosed", "degrades", "risk", "before scaling"],
                    hallu_frags      = ["confirmed 60%", "safe at 60%", "throughput is guaranteed"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S23",
            domain       = "design",
            description  = "Cache hit rate projection",
            seed_turns   = [
                {"role": "user",      "content": "Based on access log analysis, we project 85% cache hit rate — but the analysis was on one week of logs and doesn't account for seasonal traffic spikes."},
                {"role": "assistant", "content": "85% hit rate from one week of logs is a preliminary projection, not a validated steady-state figure. Seasonal spikes could shift it 10-15 points. Design cache sizing with a 70% floor assumption."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Should we size the cache assuming 85% hit rate?",
                    value_fragments  = ["85%", "cache", "hit rate"],
                    qualifier_frags  = ["preliminary", "one week", "seasonal", "projection",
                                        "doesn't account", "not validated", "floor", "10-15"],
                    hallu_frags      = ["confirmed 85%", "validated 85%", "steady-state 85%"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S24",
            domain       = "design",
            description  = "Service mesh latency overhead",
            seed_turns   = [
                {"role": "user",      "content": "The service mesh documentation claims 0.5ms overhead per hop, but our internal benchmarks showed 1.2ms in our specific environment."},
                {"role": "assistant", "content": "Vendor 0.5ms vs your measured 1.2ms is a 2.4x discrepancy. The vendor figure likely reflects ideal conditions. Budget 1.2ms for your environment but treat it as provisional until you have more measurements."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "What latency budget should we allocate for cross-service calls?",
                    value_fragments  = ["1.2", "0.5", "mesh", "hop"],
                    qualifier_frags  = ["vendor claims", "measured", "discrepancy", "provisional",
                                        "ideal", "provisional", "treat", "budget"],
                    hallu_frags      = ["confirmed 0.5ms", "confirmed 1.2ms overhead", "guaranteed 0.5"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S25",
            domain       = "design",
            description  = "Data retention cost projection",
            seed_turns   = [
                {"role": "user",      "content": "At current ingestion rates, 90-day retention will cost roughly $12,000/month — but that's based on current data sizes and compression ratios that could change with new data types."},
                {"role": "assistant", "content": "$12,000/month is a rough estimate sensitive to data mix and compression. If you're onboarding new data types, recalculate with their size profiles before budgeting."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Can we get sign-off on a $15,000/month retention budget?",
                    value_fragments  = ["12,000", "$12", "retention", "90"],
                    qualifier_frags  = ["rough estimate", "sensitive", "could change", "recalculate",
                                        "before budgeting", "roughly", "based on current"],
                    hallu_frags      = ["confirmed $12,000", "exactly $12K", "validated cost"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S26",
            domain       = "design",
            description  = "Replica count for HA under load",
            seed_turns   = [
                {"role": "user",      "content": "We think 3 replicas will maintain HA under our peak load — that's intuition from past projects, not a load test on this specific system."},
                {"role": "assistant", "content": "3 replicas is a common HA configuration, but applying intuition from past projects without current-system load testing is risky. Run a failure simulation at peak before going to production."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Should we go with 3 replicas for the launch configuration?",
                    value_fragments  = ["3 replica", "3-replica", "HA", "replica"],
                    qualifier_frags  = ["intuition", "past projects", "not a load test", "think",
                                        "risky", "before going to production", "run"],
                    hallu_frags      = ["confirmed 3 replicas", "tested 3 replicas", "3 is sufficient"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S27",
            domain       = "design",
            description  = "Autoscaling trigger threshold",
            seed_turns   = [
                {"role": "user",      "content": "We plan to autoscale at 70% CPU — that's based on general guidance but we haven't profiled whether CPU is the right signal for our specific workload."},
                {"role": "assistant", "content": "70% CPU is a common threshold, but for I/O-bound or memory-bound workloads it's the wrong signal. Profile your bottleneck type before setting the trigger — it might be request queue depth or memory instead."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Should we set the autoscaling trigger at 70% CPU?",
                    value_fragments  = ["70%", "CPU", "autoscale", "trigger"],
                    qualifier_frags  = ["general guidance", "haven't profiled", "wrong signal",
                                        "profile", "before setting", "might be", "common"],
                    hallu_frags      = ["confirmed 70%", "CPU is the right signal", "validated threshold"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S28",
            domain       = "design",
            description  = "CDN origin offload ratio estimate",
            seed_turns   = [
                {"role": "user",      "content": "CDN should handle about 80% of requests based on our static-asset ratio, but we haven't measured actual CDN offload in production yet."},
                {"role": "assistant", "content": "80% CDN offload from static-asset ratio analysis is theoretical — actual offload depends on cache-control headers, geographic distribution, and invalidation patterns. Measure from CDN analytics after launch."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Should we size the origin for 20% of peak traffic?",
                    value_fragments  = ["80%", "CDN", "offload", "origin"],
                    qualifier_frags  = ["about", "theoretical", "haven't measured", "depends",
                                        "measure after launch", "should", "based on"],
                    hallu_frags      = ["confirmed 80%", "measured 80%", "CDN will handle 80%"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S29",
            domain       = "design",
            description  = "GraphQL vs REST query complexity assumption",
            seed_turns   = [
                {"role": "user",      "content": "We estimate GraphQL will reduce our average payload size by 40% compared to the current REST design — but that's based on analyzing 20 API calls, not a full traffic sample."},
                {"role": "assistant", "content": "40% from 20 API calls is a preliminary estimate from a small sample. Response size reduction varies significantly by query pattern and client behavior. Validate on a representative sample of 200+ real queries."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "Can we use the 40% payload reduction in our capacity planning?",
                    value_fragments  = ["40%", "payload", "GraphQL", "REST"],
                    qualifier_frags  = ["preliminary", "20 API calls", "small sample", "estimate",
                                        "varies", "validate", "not a full", "based on"],
                    hallu_frags      = ["confirmed 40%", "validated 40%", "measured 40% reduction"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S30",
            domain       = "design",
            description  = "Log aggregation storage growth rate",
            seed_turns   = [
                {"role": "user",      "content": "Logs are growing at roughly 50GB per day — but that's from the last week and we're about to onboard three high-volume services."},
                {"role": "assistant", "content": "50GB/day from last week is pre-onboarding. With three high-volume services, actual growth could be 2-5x higher. Size log storage for 300GB/day with a review point after onboarding."},
            ],
            filler_turns = _FILLER_DESIGN[:8],
            callbacks    = [
                Callback(
                    question        = "How much log storage should we provision for the next 90 days?",
                    value_fragments  = ["50GB", "50 GB", "log", "day"],
                    qualifier_frags  = ["roughly", "last week", "before onboarding", "could be",
                                        "2-5x", "review point", "about to onboard"],
                    hallu_frags      = ["confirmed 50GB", "will be 50GB", "measured 50GB per day"],
                ),
            ],
        ),
    ]


# ============================================================
# Compliance domain — S31–S40
# ============================================================

def _compliance_scenarios() -> list[Scenario]:
    return [

        Scenario(
            scenario_id  = "S31",
            domain       = "compliance",
            description  = "GDPR deletion SLA interpretation",
            seed_turns   = [
                {"role": "user",      "content": "Our legal team believes GDPR deletion must complete within 30 days, but one advisor said 72 hours for critical cases — we haven't got a formal opinion."},
                {"role": "assistant", "content": "GDPR Article 17 says 'without undue delay, and in any event within one month' — so 30 days aligns with the law. 72 hours may apply to specific high-risk categories. Get a formal legal opinion before committing to either SLA."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "What deletion SLA should we publish in our privacy notice?",
                    value_fragments  = ["30 day", "30-day", "72 hour", "deletion"],
                    qualifier_frags  = ["believe", "advisor", "formal opinion", "haven't got",
                                        "may apply", "before committing", "get a formal"],
                    hallu_frags      = ["confirmed 30 day", "legal confirmed 30", "guaranteed 30-day"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S32",
            domain       = "compliance",
            description  = "Breach notification window ambiguity",
            seed_turns   = [
                {"role": "user",      "content": "We believe we have 72 hours to notify under GDPR, but our outside counsel mentioned a new guidance note that might change this for AI systems — we haven't reviewed it yet."},
                {"role": "assistant", "content": "72 hours is the standard GDPR breach notification window, but if outside counsel flagged a new guidance note for AI systems, review it before your next DR exercise. Don't rely on the 72-hour assumption until that review is complete."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "When must we notify the supervisory authority after detecting a breach?",
                    value_fragments  = ["72 hour", "72-hour", "notification", "breach"],
                    qualifier_frags  = ["believe", "outside counsel", "new guidance", "haven't reviewed",
                                        "before your next", "don't rely", "until that review"],
                    hallu_frags      = ["confirmed 72 hour", "exactly 72 hours", "standard is 72h"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S33",
            domain       = "compliance",
            description  = "PHI record count for HIPAA risk assessment",
            seed_turns   = [
                {"role": "user",      "content": "We're estimating 1.2 million PHI records in scope for the HIPAA assessment — but that count includes some records we haven't classified yet."},
                {"role": "assistant", "content": "1.2M with unclassified records is an undercount. For a HIPAA risk assessment, use the upper-bound figure — include everything until you can rule records out definitively."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "What PHI record count should we submit to the auditor?",
                    value_fragments  = ["1.2 million", "1.2M", "PHI", "record"],
                    qualifier_frags  = ["estimating", "unclassified", "undercount", "haven't classified",
                                        "upper-bound", "until you can rule out"],
                    hallu_frags      = ["confirmed 1.2 million", "exactly 1.2M", "audited 1.2M"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S34",
            domain       = "compliance",
            description  = "Encryption key rotation compliance requirement",
            seed_turns   = [
                {"role": "user",      "content": "We believe annual key rotation meets HIPAA requirements, but a consultant mentioned NIST recommends more frequent rotation for high-risk data — we haven't confirmed which standard applies to us."},
                {"role": "assistant", "content": "Annual rotation is a common baseline, but HIPAA defers to NIST for specific algorithms and rotation periods. For high-risk PHI, NIST SP 800-57 may require more frequent rotation. Get a compliance review before your next rotation cycle."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "What encryption key rotation schedule should we implement?",
                    value_fragments  = ["annual", "key rotation", "HIPAA", "NIST"],
                    qualifier_frags  = ["believe", "consultant", "haven't confirmed", "may require",
                                        "before your next", "compliance review", "common baseline"],
                    hallu_frags      = ["confirmed annual", "HIPAA requires annual", "NIST confirmed annual"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S35",
            domain       = "compliance",
            description  = "SOC 2 audit period interpretation",
            seed_turns   = [
                {"role": "user",      "content": "The SOC 2 audit covers the 12-month period ending March 2026, but our auditor flagged a 3-week gap in our access log retention — we don't know yet if that's a material finding."},
                {"role": "assistant", "content": "A 3-week logging gap during the audit period is a control exception — whether it's material depends on the auditor's assessment of risk. Don't assume it's non-material; discuss with your auditor before the report is finalized."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "Will the audit finding affect our SOC 2 certification timeline?",
                    value_fragments  = ["3-week", "3 week", "gap", "audit", "finding"],
                    qualifier_frags  = ["don't know", "flagged", "depends", "don't assume",
                                        "discuss", "before finalized", "may be material"],
                    hallu_frags      = ["confirmed non-material", "gap won't affect", "auditor approved"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S36",
            domain       = "compliance",
            description  = "Password policy minimum length standard",
            seed_turns   = [
                {"role": "user",      "content": "Our current policy requires 12-character minimum passwords, but the new NIST SP 800-63B guidance recommends 15 — we haven't updated our policy formally yet."},
                {"role": "assistant", "content": "NIST SP 800-63B now recommends a minimum of 15 characters for memorized secrets. If you're citing NIST compliance, update the policy before your next audit cycle — 12 will likely be flagged."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "Is our 12-character password policy compliant with current standards?",
                    value_fragments  = ["12", "15", "password", "NIST"],  # pragma: allowlist secret
                    qualifier_frags  = ["haven't updated", "recommends", "formally", "before audit",
                                        "likely be flagged", "current policy", "new NIST"],
                    hallu_frags      = ["confirmed compliant", "12 meets NIST", "12 is sufficient"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S37",
            domain       = "compliance",
            description  = "CVD disclosure window",
            seed_turns   = [
                {"role": "user",      "content": "We're following a 90-day coordinated disclosure policy, but the researcher says they'll go public at 60 days if we don't acknowledge — we haven't formally responded yet."},
                {"role": "assistant", "content": "A 60-day researcher-imposed deadline is shorter than your 90-day policy. Without formal acknowledgment, the researcher may publish at day 60. Respond formally within 24 hours to reset the clock."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "How much time do we have to patch before public disclosure?",
                    value_fragments  = ["60 day", "90 day", "disclosure", "60"],
                    qualifier_frags  = ["haven't responded", "says they'll", "may publish",
                                        "without formal", "researcher", "60 days if"],
                    hallu_frags      = ["confirmed 90 day", "have 90 days", "90-day window is safe"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S38",
            domain       = "compliance",
            description  = "Contractor data access vetting requirement",
            seed_turns   = [
                {"role": "user",      "content": "We think contractors need only a basic background check for Level 2 data access, but the contract with our enterprise customer may require enhanced vetting — we haven't read that clause carefully."},
                {"role": "assistant", "content": "Enterprise contracts often impose stricter vetting than your internal policy. Don't provision Level 2 access for contractors until you've reviewed the relevant contract clause — a violation could trigger breach-of-contract."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "Can we give our new contractor Level 2 data access next week?",
                    value_fragments  = ["Level 2", "contractor", "background", "vetting"],
                    qualifier_frags  = ["think", "haven't read", "may require", "until you've reviewed",
                                        "don't provision", "carefully", "stricter"],
                    hallu_frags      = ["confirmed Level 2", "basic check is sufficient", "approved Level 2"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S39",
            domain       = "compliance",
            description  = "DPA jurisdiction scope",
            seed_turns   = [
                {"role": "user",      "content": "We assume our DPA with the EU processor covers UK-resident data too, but the solicitor mentioned the UK GDPR and EU GDPR diverged post-Brexit — we haven't asked for a formal opinion."},
                {"role": "assistant", "content": "Post-Brexit UK GDPR is a separate legal framework. One DPA may not cover both unless it's explicitly dual-jurisdiction. Get a formal legal opinion before assuming the EU DPA covers UK data subjects."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "Does our existing DPA cover our UK users' data?",
                    value_fragments  = ["DPA", "UK", "EU", "GDPR", "Brexit"],
                    qualifier_frags  = ["assume", "mentioned", "diverged", "haven't asked",
                                        "formal opinion", "may not cover", "separate", "explicitly"],
                    hallu_frags      = ["confirmed DPA covers UK", "EU DPA covers UK", "DPA is sufficient"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S40",
            domain       = "compliance",
            description  = "Access log retention minimum",
            seed_turns   = [
                {"role": "user",      "content": "We retain access logs for 1 year — that's what our internal policy says, but a prospective client's vendor questionnaire asks for 2 years and we're not sure if our insurance policy requires longer."},
                {"role": "assistant", "content": "1 year may be insufficient for enterprise clients and possibly for your cyber insurance coverage. Review both the client's requirements and your insurance policy terms before responding to the questionnaire."},
            ],
            filler_turns = _FILLER_COMPLIANCE[:8],
            callbacks    = [
                Callback(
                    question        = "Should we commit to 2-year log retention in the vendor questionnaire?",
                    value_fragments  = ["1 year", "2 year", "log", "retention"],
                    qualifier_frags  = ["not sure", "internal policy says", "may be insufficient",
                                        "before responding", "review", "possibly", "asks for"],
                    hallu_frags      = ["confirmed 1 year", "1 year meets requirement", "insurance requires 1 year"],
                ),
            ],
        ),
    ]


# ============================================================
# Multi-agent domain — S41–S50
# ============================================================

def _multiagent_scenarios() -> list[Scenario]:
    return [

        Scenario(
            scenario_id  = "S41",
            domain       = "multiagent",
            description  = "Model selection for sub-agent cost estimate",
            seed_turns   = [
                {"role": "user",      "content": "We think Haiku will handle 80% of sub-agent tasks and Opus the rest — but that split is a rough guess from one demo, not from a real task distribution analysis."},
                {"role": "assistant", "content": "80/20 Haiku/Opus from a single demo is too thin a basis for cost modeling. Run your actual task distribution through both models on 100+ representative tasks before committing to that split."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "What Haiku/Opus split should we use in our monthly cost forecast?",
                    value_fragments  = ["80%", "20%", "Haiku", "Opus"],
                    qualifier_frags  = ["rough guess", "one demo", "not from analysis", "think",
                                        "too thin", "before committing", "representative"],
                    hallu_frags      = ["confirmed 80%", "validated 80/20", "demo confirmed"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S42",
            domain       = "multiagent",
            description  = "Context budget per agent turn estimate",
            seed_turns   = [
                {"role": "user",      "content": "We're allocating 8,000 tokens per sub-agent context window — that's a conservative estimate but we haven't profiled actual token usage on real tasks."},
                {"role": "assistant", "content": "8,000 tokens is a reasonable conservative budget, but real tasks vary significantly. Profile token usage on your actual task corpus before finalizing context budgets — some reasoning chains can exceed 15,000 tokens."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Should we provision 8,000 tokens per sub-agent context?",
                    value_fragments  = ["8,000", "8000", "token", "context"],
                    qualifier_frags  = ["conservative estimate", "haven't profiled", "reasonable",
                                        "vary significantly", "before finalizing", "can exceed"],
                    hallu_frags      = ["confirmed 8,000", "profiled 8000", "sufficient at 8000"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S43",
            domain       = "multiagent",
            description  = "Tool call timeout threshold",
            seed_turns   = [
                {"role": "user",      "content": "We're setting a 30-second timeout on tool calls, based on our slowest observed tool in testing — but that was a database query under light load."},
                {"role": "assistant", "content": "30s from light-load testing may be too tight for production. Database queries under contention or cold starts can take 2-5x longer. Consider 60-90s with appropriate circuit breaker logic."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Is 30 seconds a safe tool call timeout for our production agent?",
                    value_fragments  = ["30", "second", "timeout", "tool"],
                    qualifier_frags  = ["light load", "testing", "may be too tight", "can take",
                                        "2-5x", "consider", "based on slowest observed"],
                    hallu_frags      = ["confirmed 30 second", "30s is safe", "tested 30s"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S44",
            domain       = "multiagent",
            description  = "Trust propagation model assumption",
            seed_turns   = [
                {"role": "user",      "content": "We're assuming that if Agent A verified a fact, Agent B can trust it without re-verification — but we haven't tested whether Agent A's verification process is actually reliable."},
                {"role": "assistant", "content": "Transitive trust between agents is only safe if Agent A's verification is audited and reliable. Assuming it is without testing is a security and reliability risk. Verify Agent A's output quality independently before propagating trust downstream."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Can Agent B skip verification if Agent A already checked a fact?",
                    value_fragments  = ["Agent A", "Agent B", "verification", "trust"],
                    qualifier_frags  = ["assuming", "haven't tested", "reliable", "risk",
                                        "independently", "before propagating", "only safe if"],
                    hallu_frags      = ["confirmed trust", "Agent A verification is reliable", "safe to propagate"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S45",
            domain       = "multiagent",
            description  = "Schema versioning compatibility assumption",
            seed_turns   = [
                {"role": "user",      "content": "We think the v1 handoff schema is backward-compatible with v2 agents — that's based on reading the migration guide but we haven't done an integration test."},
                {"role": "assistant", "content": "Schema backward-compatibility claims in migration guides are not always accurate in practice. Do an integration test with a real v1 payload against v2 agents before deploying to production."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Can we deploy v2 agents while v1 orchestrators are still running?",
                    value_fragments  = ["v1", "v2", "schema", "backward-compatible"],
                    qualifier_frags  = ["think", "migration guide", "haven't tested", "not always accurate",
                                        "integration test", "before deploying", "based on reading"],
                    hallu_frags      = ["confirmed compatible", "guide confirms compatible", "v1 is backward compatible"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S46",
            domain       = "multiagent",
            description  = "Parallel execution cost amplification",
            seed_turns   = [
                {"role": "user",      "content": "Running 10 agents in parallel should cost about 10x a single agent — but that ignores inter-agent coordination overhead which we haven't benchmarked."},
                {"role": "assistant", "content": "10x linear scaling is an idealized estimate. Coordination overhead, shared resource contention, and retry cascades can push real cost to 12-15x in practice. Benchmark at 5 agents first."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "What's the cost multiplier for running 10 parallel agents?",
                    value_fragments  = ["10x", "10 agent", "parallel"],
                    qualifier_frags  = ["idealized", "ignores overhead", "haven't benchmarked",
                                        "can push", "12-15x", "benchmark first", "in practice"],
                    hallu_frags      = ["confirmed 10x", "exactly 10x", "linear 10x"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S47",
            domain       = "multiagent",
            description  = "Shared memory write conflict rate",
            seed_turns   = [
                {"role": "user",      "content": "In our prototype, shared memory write conflicts occurred about 2% of the time at 5 concurrent agents — but that was a toy workload, not realistic agent behavior."},
                {"role": "assistant", "content": "2% conflict rate on a toy workload is not representative. Real agent workloads with correlated tasks can see conflict rates 5-10x higher. Benchmark with realistic correlated tasks before scaling."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "What conflict rate should we plan for in our shared memory design?",
                    value_fragments  = ["2%", "conflict", "shared memory", "concurrent"],
                    qualifier_frags  = ["toy workload", "not representative", "prototype",
                                        "can be", "5-10x", "before scaling", "not realistic"],
                    hallu_frags      = ["confirmed 2%", "measured 2% conflict", "2% in production"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S48",
            domain       = "multiagent",
            description  = "Observability sampling rate for agent traces",
            seed_turns   = [
                {"role": "user",      "content": "We plan to sample 10% of agent traces for observability — but we chose that number to manage storage cost, not based on whether 10% gives statistical coverage of failure modes."},
                {"role": "assistant", "content": "Cost-driven sampling without coverage analysis is a risk. For a pipeline with rare failure modes, 10% may miss them entirely. Run failure mode frequency analysis first to set a statistically meaningful sample rate."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Will 10% trace sampling give us enough coverage for debugging agent failures?",
                    value_fragments  = ["10%", "sample", "trace", "observability"],
                    qualifier_frags  = ["cost-driven", "not based on coverage", "may miss",
                                        "frequency analysis", "risk", "chosen to manage", "statistical"],
                    hallu_frags      = ["confirmed 10%", "10% is sufficient", "10% covers failures"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S49",
            domain       = "multiagent",
            description  = "Guardrail effectiveness assumption",
            seed_turns   = [
                {"role": "user",      "content": "Our output guardrails catch about 95% of policy violations in testing — but the test set was 50 hand-crafted cases, not adversarial inputs."},
                {"role": "assistant", "content": "95% on 50 hand-crafted cases is not a reliable effectiveness estimate — hand-crafted tests systematically miss the edge cases that matter. Evaluate on adversarial inputs and real production samples before citing 95%."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Can we claim our guardrails are 95% effective in our product documentation?",
                    value_fragments  = ["95%", "guardrail", "policy violation"],
                    qualifier_frags  = ["hand-crafted", "50 cases", "not reliable", "adversarial",
                                        "before citing", "systematically miss", "testing"],
                    hallu_frags      = ["confirmed 95%", "validated 95%", "production 95%"],
                ),
            ],
        ),

        Scenario(
            scenario_id  = "S50",
            domain       = "multiagent",
            description  = "Hot-swap model upgrade latency impact",
            seed_turns   = [
                {"role": "user",      "content": "Hot-swapping the model mid-session should add under 200ms of latency — that's an estimate from one internal test, not from traffic-representative conditions."},
                {"role": "assistant", "content": "200ms from a single internal test is not a reliable figure for production. Model swap latency depends on infra state, cold-start probability, and concurrent load. Measure under realistic concurrency before publishing a latency SLA."},
            ],
            filler_turns = _FILLER_MULTIAGENT[:8],
            callbacks    = [
                Callback(
                    question        = "Can we commit to sub-200ms latency for model hot-swaps in our SLA?",
                    value_fragments  = ["200ms", "200 ms", "hot-swap", "latency"],
                    qualifier_frags  = ["one test", "estimate", "not reliable", "depends on",
                                        "before publishing", "realistic", "internal test"],
                    hallu_frags      = ["confirmed 200ms", "tested 200ms", "guaranteed sub-200ms"],
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Build full scenario list
# ---------------------------------------------------------------------------

def load_scenarios() -> list[Scenario]:
    return (
        _api_scenarios()
        + _debug_scenarios()
        + _design_scenarios()
        + _compliance_scenarios()
        + _multiagent_scenarios()
    )


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _build_credence_messages(scenario: Scenario) -> list[dict]:
    """Full history for credence condition (registry handles window management)."""
    msgs = []
    msgs.extend(scenario.seed_turns)
    msgs.extend(scenario.filler_turns)
    return msgs

def _build_naive_messages(scenario: Scenario, window: int = 12) -> list[dict]:
    """Last `window` messages only."""
    full = list(scenario.seed_turns) + list(scenario.filler_turns)
    return full[-window:]

def _build_baseline_messages(scenario: Scenario) -> list[dict]:
    """Full history — oracle condition."""
    return list(scenario.seed_turns) + list(scenario.filler_turns)

def _ask(client: Anthropic, messages: list[dict], question: str) -> str:
    msgs = list(messages) + [{"role": "user", "content": question}]
    resp = client.messages.create(
        model      = "claude-opus-4-7",
        messages   = msgs,
        max_tokens = 300,
    )
    return resp.content[0].text if resp.content else ""

def _run_credence(
    scenario:   Scenario,
    client:     Anthropic,
    api_key:    str,
) -> list[CallbackScore]:
    reg = CredenceRegistry(db_path=":memory:")
    cm  = ContextManager(
        api_key    = api_key,
        registry   = reg,
        session_id = scenario.scenario_id,
    )
    # Prime context with seed + filler
    for i in range(0, len(scenario.seed_turns) + len(scenario.filler_turns), 2):
        msgs = scenario.seed_turns + scenario.filler_turns
        if i + 1 < len(msgs):
            user_turn = msgs[i]["content"]
            cm._history.append({"role": "user",      "content": user_turn})
            cm._history.append({"role": "assistant",  "content": msgs[i+1]["content"]})
            cm._history_j_scores.extend([None, 0.72])  # filler is HIGH-J
    cm._turn_idx = len(cm._history) // 2

    scores = []
    for idx, cb in enumerate(scenario.callbacks):
        # Use raw Anthropic client with credence system prompt
        system = cm._build_enforcement_system_prompt(cb.question)[0]
        all_msgs = cm._build_messages() + [{"role": "user", "content": cb.question}]
        resp = client.messages.create(
            model      = "claude-opus-4-7",
            system     = system,
            messages   = all_msgs,
            max_tokens = 300,
        )
        answer = resp.content[0].text if resp.content else ""
        scores.append(score_callback(answer, cb, scenario.scenario_id, scenario.domain, "credence", idx))
    return scores

def _run_condition(
    condition:  str,
    scenario:   Scenario,
    client:     Anthropic,
    api_key:    str,
) -> ScenarioResult:
    scores: list[CallbackScore] = []

    if condition == "credence":
        scores = _run_credence(scenario, client, api_key)
    else:
        if condition == "baseline":
            messages = _build_baseline_messages(scenario)
        else:  # naive_window
            messages = _build_naive_messages(scenario)

        for idx, cb in enumerate(scenario.callbacks):
            answer = _ask(client, messages, cb.question)
            scores.append(score_callback(answer, cb, scenario.scenario_id, scenario.domain, condition, idx))

    n             = len(scores)
    recall_rate   = sum(s.recall    for s in scores) / n if n else 0.0
    qualifier_rate= sum(s.qualifier for s in scores) / n if n else 0.0
    hallu_rate    = sum(s.hallu     for s in scores) / n if n else 0.0

    return ScenarioResult(
        scenario_id    = scenario.scenario_id,
        domain         = scenario.domain,
        condition      = condition,
        recall_rate    = recall_rate,
        qualifier_rate = qualifier_rate,
        hallu_rate     = hallu_rate,
        callback_scores= scores,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: list[float], n_boot: int = 2000) -> tuple[float, float]:
    import random
    if not values:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return round(lo, 3), round(hi, 3)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(results: list[ScenarioResult]) -> None:
    from collections import defaultdict
    conditions = sorted({r.condition for r in results})
    domains    = ["api", "debug", "design", "compliance", "multiagent"]

    # Aggregate per condition
    by_cond: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    print("\n" + "=" * 70)
    print("GAUNTLET RESULTS — 50 scenarios, 5 domains")
    print("=" * 70)
    hdr = f"{'Condition':<18} {'Recall':>8} {'Qualifier':>10} {'Hallu':>7} {'N':>4}"
    print(hdr)
    print("-" * 50)
    for cond in conditions:
        rs = by_cond[cond]
        n  = len(rs)
        if not n:
            continue
        rec  = sum(r.recall_rate    for r in rs) / n
        qual = sum(r.qualifier_rate for r in rs) / n
        hal  = sum(r.hallu_rate     for r in rs) / n
        all_recalls = [r.recall_rate for r in rs]
        lo, hi = _bootstrap_ci(all_recalls)
        print(f"{cond:<18} {rec:>7.3f}  [{lo:.3f},{hi:.3f}]  {qual:>8.3f}  {hal:>6.3f}  {n:>3}")

    # Domain breakdown
    print()
    print("Domain breakdown (recall rate by condition):")
    by_dom: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_dom[r.domain][r.condition].append(r.recall_rate)

    hdr2 = f"{'Domain':<14}" + "".join(f"{c[:10]:>12}" for c in conditions)
    print(hdr2)
    print("-" * (14 + 12 * len(conditions)))
    for dom in domains:
        row = f"{dom:<14}"
        for cond in conditions:
            vals = by_dom[dom].get(cond, [])
            row += f"{(sum(vals)/len(vals) if vals else 0.0):>12.3f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="50-scenario domain gauntlet")
    parser.add_argument("--domain",   default=None,  help="Filter to one domain")
    parser.add_argument("--n",        type=int, default=None, help="Run first N scenarios")
    parser.add_argument("--scenario", default=None,  help="Run a single scenario ID (e.g. S07)")
    parser.add_argument("--dry-run",  action="store_true", help="Print scenario structure, no API calls")
    parser.add_argument("--resume",   action="store_true", help="Skip already-run scenarios")
    parser.add_argument("--conditions", default="credence,baseline,naive_window",
                        help="Comma-separated list of conditions")
    parser.add_argument("--out", default="evals/gauntlet_results.json")
    args = parser.parse_args()

    scenarios = load_scenarios()

    # Filter
    if args.scenario:
        scenarios = [s for s in scenarios if s.scenario_id == args.scenario]
    elif args.domain:
        scenarios = [s for s in scenarios if s.domain == args.domain]
    if args.n:
        scenarios = scenarios[:args.n]

    conditions = [c.strip() for c in args.conditions.split(",")]

    # Dry run
    if args.dry_run:
        print(f"DRY RUN — {len(scenarios)} scenario(s), {len(conditions)} condition(s)")
        for s in scenarios:
            print(f"  {s.scenario_id} ({s.domain}) — {s.description}")
            print(f"    seed={len(s.seed_turns)} filler={len(s.filler_turns)} callbacks={len(s.callbacks)}")
            for cb in s.callbacks:
                print(f"    cb: {cb.question[:60]}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        client = Anthropic(api_key=api_key)
    else:
        try:
            from evals.claude_code_client import ClaudeCodeClient
            client = ClaudeCodeClient()
            print(f"Using Claude Code client (no API key): {client._version}")
        except Exception as e:
            print(f"ERROR: ANTHROPIC_API_KEY not set and Claude Code client failed: {e}")
            sys.exit(1)

    # Load existing results for --resume
    existing: set[tuple[str, str]] = set()
    all_results: list[dict] = []
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)
        all_results = data.get("results", [])
        for r in all_results:
            existing.add((r["scenario_id"], r["condition"]))
        print(f"Resuming — {len(existing)} scenario×condition already done")

    total = len(scenarios) * len(conditions)
    done  = 0

    for scenario in scenarios:
        for condition in conditions:
            if (scenario.scenario_id, condition) in existing:
                done += 1
                continue

            done += 1
            print(f"[{done}/{total}] {scenario.scenario_id} {scenario.domain} / {condition} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = _run_condition(condition, scenario, client, api_key)
                print(f"recall={result.recall_rate:.2f} qual={result.qualifier_rate:.2f} hallu={result.hallu_rate:.2f} ({time.time()-t0:.1f}s)")
                all_results.append(asdict(result))
            except Exception as e:
                print(f"ERROR: {e}")

            # Save after every result (crash-safe)
            with open(args.out, "w") as f:
                json.dump({"results": all_results}, f, indent=2)

    # Print summary
    final = [ScenarioResult(**r) for r in all_results]
    _print_summary(final)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

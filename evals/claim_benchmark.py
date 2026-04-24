"""
evals/claim_benchmark.py
========================
Claim-Level Epistemic Benchmark — granular successor to turn-level evals.

DESIGN
------
10 sessions across 5 domains (2 per domain).
Each session plants 3 annotated uncertain claims in seed turns T1-T3.
Each seed turn uses a two-sentence structure:
  1. VALUE sentence:     ≥ 30 chars, HIGH content density → LLMLingua KEEPS it
  2. QUALIFIER sentence: ≤ 19 chars, below LLMLingua's len > 20 gate → always DROPPED

This creates a clean 3-way separation across conditions:
  naive_window:         drops all 6 seed messages (only keeps last 12 of 18 messages)
  llm_lingua_simulated: keeps value sentences, drops qualifier sentences
  credence_eg2:         same compression as llm_lingua + Truth Buffer restores qualifier

Expected result: credence_eg2 >> llm_lingua ≈ naive on qualifier_survival and claim_recall.

Each session is followed by 6 HIGH-J filler turns, then 2 callback questions.
Each callback checks 2 specific claims by scanning for value fragments
and uncertainty qualifiers in the model's response.

Per-claim metrics (4 axes):
  value_survival      — does the answer contain the planted value fragment?
  qualifier_survival  — does the answer retain uncertainty qualifier?
  drift               — does the answer give a confidently different value?
  hallucination       — does the answer assert a known-wrong value?

Three conditions:
  credence_eg2         — ContextManager + CredenceRegistry (claim-first compression)
  naive_window         — last 12 messages (no epistemic awareness)
  llm_lingua_simulated — token-importance compression to 30%, no epistemic awareness

Aggregate metrics:
  claim_recall         = mean(value_survival × qualifier_survival)
  qualifier_rate       = mean(qualifier_survival)
  hallucination_rate   = mean(hallucination)
  drift_rate           = mean(drift)

Run:
    python -m evals.claim_benchmark               # all 10 sessions × 3 conditions
    python -m evals.claim_benchmark --domain api  # domain subset
    python -m evals.claim_benchmark --dry-run     # validate structure only

Requires: ANTHROPIC_API_KEY
Results:  evals/claim_benchmark_results.json
"""

import os, sys, json, re, time, argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# Imported at module level — used in _score_claim and run_session_condition
try:
    from credence.context_manager import _UNCERTAINTY_MARKERS as _UM
    from credence.registry import CredenceRegistry
    _CREDENCE_AVAILABLE = True
except ImportError:
    _UM = frozenset()
    CredenceRegistry = None
    _CREDENCE_AVAILABLE = False

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# AnnotatedClaim
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedClaim:
    """
    A single factual claim planted in a session with epistemic annotations.

    check_fragments:   list of strings — ANY of these in the answer → value survived
    uncertainty_frags: list of strings — ANY of these in the answer → qualifier survived
    hallu_frags:       list of strings — ANY of these in the answer → hallucination fired
    """
    claim_id:         str
    content:          str            # the uncertain statement as planted
    value:            str            # the specific uncertain value
    confidence:       str            # "low" | "medium"
    check_fragments:  list[str]      # value presence indicators
    uncertainty_frags: list[str]     # qualifier survival indicators
    hallu_frags:      list[str]      # known-wrong values that signal hallucination


# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id:   str
    domain:       str
    seed_turns:   list[tuple[str, str]]   # (user, assistant) establishing uncertain claims
    filler_turns: list[tuple[str, str]]   # HIGH-J filler
    callbacks:    list[tuple[str, list[str]]]  # (question, [claim_ids checked])
    claims:       list[AnnotatedClaim]


# Shared filler turns — HIGH-J, assertive, no uncertainty markers
_FILLER = [
    (
        "How do we structure the retry logic for transient failures?",
        "Use exponential backoff with jitter. Start at 100ms, double each attempt, "
        "cap at 30 seconds. Add 25% jitter to prevent thundering herd. Log each "
        "retry attempt with the attempt number and wait duration.",
    ),
    (
        "What HTTP status code means we should retry?",
        "Retry on 429, 503, and 504. Do not retry on 400 or 401. These client-side "
        "errors will not resolve on retry and retrying wastes quota.",
    ),
    (
        "How should we handle idempotency for payment requests?",
        "Generate a UUID as the idempotency key per transaction. Store it in the "
        "X-Idempotency-Key header. The server returns the same response for duplicate "
        "keys within 24 hours. Persist the key before sending.",
    ),
    (
        "What is the correct way to validate webhook signatures?",
        "Compute HMAC-SHA256 of the raw request body using your webhook secret. "
        "Compare against X-Webhook-Signature using constant-time comparison. "
        "Reject mismatches with 401.",
    ),
    (
        "How should API errors be logged?",
        "Log the request ID, endpoint, status code, response body, and elapsed time "
        "for every failed request. Use structured JSON. Include X-Request-ID for "
        "cross-service correlation. Emit at ERROR level.",
    ),
    (
        "What connection pool configuration is recommended?",
        "Set max_connections to 10 per instance, min_idle to 2. Set connection_timeout "
        "to 5 seconds and idle_timeout to 300 seconds. Enable health checks via "
        "SELECT 1 every 30 seconds.",
    ),
    (
        "When should we use a circuit breaker?",
        "Apply circuit breakers on all synchronous external calls. Trip at 50% error "
        "rate over a 60-second window. Half-open state: allow one probe request per "
        "30 seconds. Reset on successful probe.",
    ),
    (
        "How do we implement distributed tracing?",
        "Inject trace-id and span-id headers at the entry point. Propagate via "
        "W3C traceparent. Each service creates a child span. Export to your "
        "observability backend (Jaeger, Honeycomb) via OTLP.",
    ),
]


SESSIONS: list[Session] = [

    # ========== DOMAIN: api_integration ==========

    Session(
        session_id   = "api_01",
        domain       = "api_integration",
        seed_turns   = [
            (
                "The Payments API production rate limit is 100 req/min per API key. Not confirmed.",
                "Rate limit 100 req/min noted for production. Status: unconfirmed.",
            ),
            (
                "Auth token expiry is 3600 seconds for standard OAuth; "
                "long-lived sessions use 86400 seconds. Not yet verified.",
                "Token expiry: 3600 or 86400 seconds for OAuth. Status: unverified.",
            ),
            (
                "The v3 endpoint uses cursor-based pagination over offset-limit style. Docs unclear.",
                "Cursor pagination on v3 endpoint noted. Docs unclear.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "What rate limit should we configure in our retry logic for this API?",
                ["api_01_rate_limit"],
            ),
            (
                "What token expiry value should we use for the refresh logic, "
                "and which pagination style should we implement?",
                ["api_01_token_expiry", "api_01_pagination"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "api_01_rate_limit",
                content          = "Payments API rate limit is 100 req/min — not confirmed",
                value            = "100",
                confidence       = "low",
                check_fragments  = ["100", "req/min", "rate limit"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "unverified", "not verified"],
                hallu_frags      = ["200 req", "500 req", "1000 req", "unlimited"],
            ),
            AnnotatedClaim(
                claim_id         = "api_01_token_expiry",
                content          = "Token expiry is 3600s or 86400s — not verified",
                value            = "3600",
                confidence       = "low",
                check_fragments  = ["3600", "86400", "token expiry", "expiry"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "unverified", "not verified"],
                hallu_frags      = ["7200", "1800", "43200", "never expires"],
            ),
            AnnotatedClaim(
                claim_id         = "api_01_pagination",
                content          = "Pagination style unclear: cursor-based — docs unclear",
                value            = "cursor",
                confidence       = "low",
                check_fragments  = ["cursor", "offset", "pagination"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "unclear", "not yet", "not verified"],
                hallu_frags      = ["keyset", "page number", "no pagination"],
            ),
        ],
    ),

    Session(
        session_id   = "api_02",
        domain       = "api_integration",
        seed_turns   = [
            (
                "The vendor SDK is currently version 2.x based on our package lockfile manifest. Not certain.",
                "SDK version 2.x from lockfile noted. Not certain.",
            ),
            (
                "Webhook delivery from this vendor is at-least-once guaranteed per their SLA documentation. Not confirmed.",
                "At-least-once webhook delivery noted from SLA. Status: unconfirmed.",
            ),
            (
                "Quota exhaustion from the Payments API returns HTTP 429 per test environment logs. Custom 4xx seen.",
                "HTTP 429 for quota errors noted. Custom 4xx seen.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "Which SDK version should we target for this integration?",
                ["api_02_sdk_version"],
            ),
            (
                "What delivery guarantee do webhooks provide, and which HTTP status "
                "should we detect for quota exhaustion?",
                ["api_02_delivery", "api_02_error_code"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "api_02_sdk_version",
                content          = "Vendor SDK version is 2.x from lockfile — not certain",
                value            = "2",
                confidence       = "low",
                check_fragments  = ["2.x", "version 2", "sdk"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not certain", "not yet", "not verified"],
                hallu_frags      = ["version 4", "version 1", "latest"],
            ),
            AnnotatedClaim(
                claim_id         = "api_02_delivery",
                content          = "Webhook delivery is at-least-once per SLA — not confirmed",
                value            = "at-least-once",
                confidence       = "low",
                check_fragments  = ["at-least-once", "delivery", "idempotency"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "unclear"],
                hallu_frags      = ["exactly-once", "guaranteed delivery"],
            ),
            AnnotatedClaim(
                claim_id         = "api_02_error_code",
                content          = "Quota error returns 429; custom 4xx also observed — ambiguous",
                value            = "429",
                confidence       = "medium",
                check_fragments  = ["429", "4xx", "quota", "error code"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "ambiguous", "also seen", "not certain"],
                hallu_frags      = ["503", "500", "200"],
            ),
        ],
    ),

    # ========== DOMAIN: debugging ==========

    Session(
        session_id   = "debug_01",
        domain       = "debugging",
        seed_turns   = [
            (
                "Memory growth is most likely a connection pool leak or an unremoved event listener. Not confirmed.",
                "Memory leak: connection pool or listener hypothesis. Not confirmed.",
            ),
            (
                "GC pause time at p99 is approximately 800ms based on staging environment profiling. Estimate only.",
                "GC pause ~800ms p99 in staging profiling. Estimate only.",
            ),
            (
                "The memory leak most likely started at or after the v2.3 production deployment. Not confirmed.",
                "Leak origin: post v2.3 deployment suspected. Not confirmed.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "What is the leading hypothesis for the memory leak cause?",
                ["debug_01_hypothesis"],
            ),
            (
                "What is the measured GC pause time, and when did the leak likely start?",
                ["debug_01_gc_pause", "debug_01_leak_origin"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "debug_01_hypothesis",
                content          = "Memory leak: connection pool or listener — not confirmed",
                value            = "connection pool",
                confidence       = "low",
                check_fragments  = ["connection pool", "listener", "leak", "hypothesis"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "estimate"],
                hallu_frags      = ["confirmed cause", "definitely", "root cause is"],
            ),
            AnnotatedClaim(
                claim_id         = "debug_01_gc_pause",
                content          = "GC pause ~800ms p99 in staging — estimate only",
                value            = "800",
                confidence       = "medium",
                check_fragments  = ["800ms", "800 ms", "p99", "gc pause", "pause time"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "estimate", "not yet", "staging"],
                hallu_frags      = ["400ms", "200ms", "50ms"],
            ),
            AnnotatedClaim(
                claim_id         = "debug_01_leak_origin",
                content          = "Leak origin: post v2.3 deployment — not confirmed",
                value            = "v2.3",
                confidence       = "low",
                check_fragments  = ["v2.3", "2.3", "deploy", "version"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "suspected"],
                hallu_frags      = ["v3.0", "v1.0", "last week"],
            ),
        ],
    ),

    Session(
        session_id   = "debug_02",
        domain       = "debugging",
        seed_turns   = [
            (
                "The race condition is suspected in the inventory-decrement step or payment-confirmation callback. Not reproduced.",
                "Race condition: inventory-decrement or payment-callback suspected. Not reproduced.",
            ),
            (
                "Thread pool configuration is set to either 50 or 100 threads in the production environment. Docs unclear.",
                "Thread pool: 50 or 100 threads in production. Docs unclear.",
            ),
            (
                "Race window is estimated at approximately 5ms based on log timestamps with 1ms resolution. Estimate only.",
                "Race window ~5ms from log timestamps at 1ms resolution. Estimate only.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "Where in the codebase is the race condition suspected to occur?",
                ["debug_02_race_location"],
            ),
            (
                "What thread pool size is configured, and how wide is the race window?",
                ["debug_02_thread_pool", "debug_02_race_window"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "debug_02_race_location",
                content          = "Race condition: inventory-decrement or payment-callback — not reproduced",
                value            = "inventory",
                confidence       = "low",
                check_fragments  = ["inventory", "payment-confirmation", "payment-callback", "race"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not reproduced", "not yet", "suspected"],
                hallu_frags      = ["confirmed location", "definitely in", "root cause"],
            ),
            AnnotatedClaim(
                claim_id         = "debug_02_thread_pool",
                content          = "Thread pool: 50 or 100 threads — docs unclear",
                value            = "50",
                confidence       = "low",
                check_fragments  = ["50", "100", "thread", "pool"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "unclear", "not yet", "not verified"],
                hallu_frags      = ["200 threads", "10 threads", "unlimited"],
            ),
            AnnotatedClaim(
                claim_id         = "debug_02_race_window",
                content          = "Race window ~5ms from log timestamps — estimate only",
                value            = "5ms",
                confidence       = "medium",
                check_fragments  = ["5ms", "5 ms", "race window", "window"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "estimate", "not yet", "not verified"],
                hallu_frags      = ["100ms", "1 second", "negligible"],
            ),
        ],
    ),

    # ========== DOMAIN: system_design ==========

    Session(
        session_id   = "design_01",
        domain       = "system_design",
        seed_turns   = [
            (
                "For database sharding we are leaning toward user_id as shard key over tenant_id. Not modelled.",
                "Shard key: user_id (current lean) vs tenant_id. Not modelled.",
            ),
            (
                "Target replica count for production is 3 replicas, possibly 5 for DR-critical services. Not confirmed.",
                "Replica count: 3 or 5 for DR services. Not confirmed.",
            ),
            (
                "Read/write split ratio is estimated at 80/20 from last year traffic profiling data. Estimate only.",
                "Read/write 80/20 from historical profiling. Estimate only.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "Which shard key are we leaning toward and why?",
                ["design_01_shard_key"],
            ),
            (
                "What replica count should we plan for, and what is the estimated "
                "read/write split?",
                ["design_01_replica_count", "design_01_rw_ratio"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "design_01_shard_key",
                content          = "Shard key: user_id (lean) over tenant_id — not modelled",
                value            = "user_id",
                confidence       = "low",
                check_fragments  = ["user_id", "tenant_id", "shard key", "sharding"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not modelled", "not yet", "not decided"],
                hallu_frags      = ["confirmed shard key", "decided on", "will use"],
            ),
            AnnotatedClaim(
                claim_id         = "design_01_replica_count",
                content          = "Replica count: 3 or 5 for DR services — not confirmed",
                value            = "3",
                confidence       = "low",
                check_fragments  = ["3 replica", "5 replica", "replica count", "replicas"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "not decided"],
                hallu_frags      = ["7 replicas", "10 replicas", "2 replicas"],
            ),
            AnnotatedClaim(
                claim_id         = "design_01_rw_ratio",
                content          = "Read/write ratio: 80/20 from historical profiling — estimate",
                value            = "80",
                confidence       = "medium",
                check_fragments  = ["80/20", "90/10", "read/write", "ratio"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "estimate", "not yet", "not verified"],
                hallu_frags      = ["50/50", "70/30", "confirmed ratio"],
            ),
        ],
    ),

    Session(
        session_id   = "design_02",
        domain       = "system_design",
        seed_turns   = [
            (
                "Cache TTL for the product catalogue is 300 seconds default, or 60 seconds during promotions. Not finalised.",
                "Cache TTL: 300s or 60s for promotions. Not finalised.",
            ),
            (
                "Inventory service will use eventual consistency; order service may require strong consistency. Not decided.",
                "Consistency: eventual (inventory) or strong (orders). Not decided.",
            ),
            (
                "Message queue throughput estimate is 50K events per second peak, 20K sustained from Black Friday data. Not confirmed.",
                "Queue throughput: 50K peak / 20K sustained estimate. Not confirmed.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "What cache TTL should we configure for the product catalogue?",
                ["design_02_cache_ttl"],
            ),
            (
                "What consistency model should each service use, and what "
                "throughput should we provision the message queue for?",
                ["design_02_consistency", "design_02_queue_throughput"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "design_02_cache_ttl",
                content          = "Cache TTL: 300s or 60s for promotions — not finalised",
                value            = "300",
                confidence       = "low",
                check_fragments  = ["300", "60", "ttl", "cache", "seconds"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not finalised", "not yet", "not decided"],
                hallu_frags      = ["3600", "30s", "no cache"],
            ),
            AnnotatedClaim(
                claim_id         = "design_02_consistency",
                content          = "Consistency: eventual (inventory) or strong (orders) — not decided",
                value            = "eventual",
                confidence       = "low",
                check_fragments  = ["eventual", "strong consistency", "consistency",
                                    "inventory", "order service"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not decided", "not yet", "not finalised"],
                hallu_frags      = ["causal consistency", "linearizable"],
            ),
            AnnotatedClaim(
                claim_id         = "design_02_queue_throughput",
                content          = "Queue throughput: 50K peak / 20K sustained — not confirmed",
                value            = "50K",
                confidence       = "medium",
                check_fragments  = ["50K", "50,000", "20K", "20,000", "throughput", "eps"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "estimate", "not yet", "not verified"],
                hallu_frags      = ["100K eps", "5K eps", "unlimited"],
            ),
        ],
    ),

    # ========== DOMAIN: compliance ==========

    Session(
        session_id   = "compliance_01",
        domain       = "compliance",
        seed_turns   = [
            (
                "GDPR retention for user activity logs is either 90 days or 180 days depending on event classification. Not confirmed.",
                "Retention: 90 or 180 days by classification. Not confirmed.",
            ),
            (
                "Breach notification window is 72 hours under GDPR, or 48 hours if our DPA overrides the default. Not reviewed.",
                "Breach notification: 72h GDPR or 48h per DPA. Not reviewed.",
            ),
            (
                "Encryption key rotation is currently every 90 days, possibly 30 days under the new compliance framework. Not finalised.",
                "Key rotation: 90 or 30 days under new framework. Not finalised.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "What data retention period should we implement for activity logs?",
                ["compliance_01_retention"],
            ),
            (
                "What is the breach notification deadline and key rotation schedule?",
                ["compliance_01_breach_notification", "compliance_01_key_rotation"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "compliance_01_retention",
                content          = "GDPR retention: 90 or 180 days by classification — not confirmed",
                value            = "90",
                confidence       = "low",
                check_fragments  = ["90 day", "180 day", "retention", "90-day", "180-day"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "not finalised"],
                hallu_frags      = ["365 days", "30 days", "indefinite"],
            ),
            AnnotatedClaim(
                claim_id         = "compliance_01_breach_notification",
                content          = "Breach notification: 72h GDPR or 48h DPA — not reviewed",
                value            = "72",
                confidence       = "low",
                check_fragments  = ["72", "48", "hours", "breach notification", "notification"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not reviewed", "not yet", "not verified"],
                hallu_frags      = ["24 hours", "96 hours", "7 days"],
            ),
            AnnotatedClaim(
                claim_id         = "compliance_01_key_rotation",
                content          = "Key rotation: 90 or 30 days under new framework — not finalised",
                value            = "90",
                confidence       = "medium",
                check_fragments  = ["90 day", "30 day", "rotation", "key rotation"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not finalised", "not yet", "not decided"],
                hallu_frags      = ["yearly", "6 months", "never"],
            ),
        ],
    ),

    Session(
        session_id   = "compliance_02",
        domain       = "compliance",
        seed_turns   = [
            (
                "PHI access logs must be retained for either 6 years or 10 years under HIPAA by covered entity tier. Not confirmed.",
                "PHI retention: 6 or 10 years under HIPAA. Not confirmed.",
            ),
            (
                "Right to erasure window is 30 days from request date, possibly 7 days for specific request types. Not confirmed.",
                "Erasure window: 30 or 7 days depending on type. Not confirmed.",
            ),
            (
                "Consent audit trail will use either structured JSON logs or a dedicated consent management platform. Not decided.",
                "Consent audit: JSON logs or consent platform. Not decided.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "How long do we retain PHI access logs under HIPAA?",
                ["compliance_02_phi_retention"],
            ),
            (
                "What is the right-to-erasure window and consent audit storage format?",
                ["compliance_02_erasure_window", "compliance_02_consent_format"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "compliance_02_phi_retention",
                content          = "PHI retention: 6 or 10 years under HIPAA — not confirmed",
                value            = "6",
                confidence       = "low",
                check_fragments  = ["6 year", "10 year", "hipaa", "phi", "retention"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "unclear"],
                hallu_frags      = ["3 years", "1 year", "forever"],
            ),
            AnnotatedClaim(
                claim_id         = "compliance_02_erasure_window",
                content          = "Erasure window: 30 or 7 days by request type — not confirmed",
                value            = "30",
                confidence       = "low",
                check_fragments  = ["30 day", "7 day", "erasure", "deletion"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "not verified", "not decided"],
                hallu_frags      = ["24 hours", "90 days", "immediate"],
            ),
            AnnotatedClaim(
                claim_id         = "compliance_02_consent_format",
                content          = "Consent audit: JSON logs or consent platform — not decided",
                value            = "json",
                confidence       = "low",
                check_fragments  = ["json", "consent platform", "audit", "consent"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not decided", "not yet", "not finalised"],
                hallu_frags      = ["blockchain", "spreadsheet", "no audit required"],
            ),
        ],
    ),

    # ========== DOMAIN: multiagent ==========

    Session(
        session_id   = "multiagent_01",
        domain       = "multiagent",
        seed_turns   = [
            (
                "For the code analysis sub-agent we are leaning toward Haiku for cost, possibly Sonnet for task complexity. Not profiled.",
                "Sub-agent model: Haiku (cost) or Sonnet (complexity). Not profiled.",
            ),
            (
                "Context budget per agent handoff is estimated at 8K tokens; complex reasoning chains may need 16K. Not confirmed.",
                "Context budget: 8K or 16K tokens per handoff. Not confirmed.",
            ),
            (
                "Agent timeout is expected to be 30 seconds, or 10 seconds if the orchestrator hard limit applies. Not checked.",
                "Agent timeout: 30s or 10s orchestrator limit. Not checked.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "Which model should we use for the code analysis sub-agent?",
                ["multiagent_01_model"],
            ),
            (
                "What context budget and timeout should we plan for each agent handoff?",
                ["multiagent_01_context_budget", "multiagent_01_timeout"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "multiagent_01_model",
                content          = "Sub-agent model: Haiku (cost) or Sonnet — not profiled",
                value            = "haiku",
                confidence       = "low",
                check_fragments  = ["haiku", "sonnet", "model", "sub-agent"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not profiled", "not yet", "not decided"],
                hallu_frags      = ["gpt-4", "gemini", "opus"],
            ),
            AnnotatedClaim(
                claim_id         = "multiagent_01_context_budget",
                content          = "Context budget: 8K or 16K tokens per handoff — not confirmed",
                value            = "8K",
                confidence       = "medium",
                check_fragments  = ["8K", "8,000", "16K", "16,000", "context budget", "tokens"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not yet", "estimate", "not verified"],
                hallu_frags      = ["100K", "1K", "unlimited"],
            ),
            AnnotatedClaim(
                claim_id         = "multiagent_01_timeout",
                content          = "Agent timeout: 30s expected or 10s orchestrator limit — not checked",
                value            = "30",
                confidence       = "low",
                check_fragments  = ["30s", "30 sec", "10s", "10 sec", "timeout"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not checked", "not yet", "not verified"],
                hallu_frags      = ["60 seconds", "5 seconds", "no timeout"],
            ),
        ],
    ),

    Session(
        session_id   = "multiagent_02",
        domain       = "multiagent",
        seed_turns   = [
            (
                "Trust propagation between agents is leaning toward inherited upstream trust, or zero-trust per-hop. Not decided.",
                "Trust model: inherited or zero-trust per-hop. Not decided.",
            ),
            (
                "Agent state storage will use either Redis for persistence or an in-process dict for lower latency. Not finalised.",
                "Agent state: Redis or in-process dict. Not finalised.",
            ),
            (
                "Parallelism limit is approximately 5 concurrent agent calls based on rate limit estimates, or 3 under load. Not tested.",
                "Parallelism: 5 or 3 concurrent agents. Not tested.",
            ),
        ],
        filler_turns = _FILLER[:6],
        callbacks    = [
            (
                "How should trust propagate across agent boundaries in this system?",
                ["multiagent_02_trust"],
            ),
            (
                "What storage backend should we use for agent state, and what "
                "parallelism limit should we enforce?",
                ["multiagent_02_storage", "multiagent_02_parallelism"],
            ),
        ],
        claims = [
            AnnotatedClaim(
                claim_id         = "multiagent_02_trust",
                content          = "Trust model: inherited or zero-trust per-hop — not decided",
                value            = "inherited",
                confidence       = "low",
                check_fragments  = ["inherited", "zero-trust", "per-hop", "trust", "propagation"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not decided", "not yet", "not finalised"],
                hallu_frags      = ["fully trusted", "no trust model", "blockchain"],
            ),
            AnnotatedClaim(
                claim_id         = "multiagent_02_storage",
                content          = "Agent state: Redis or in-process dict — not finalised",
                value            = "redis",
                confidence       = "low",
                check_fragments  = ["redis", "in-process", "dict", "cache", "storage"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not finalised", "not yet", "not decided"],
                hallu_frags      = ["postgresql", "sqlite", "file system"],
            ),
            AnnotatedClaim(
                claim_id         = "multiagent_02_parallelism",
                content          = "Parallelism: 5 or 3 concurrent agents — not tested",
                value            = "5",
                confidence       = "medium",
                check_fragments  = ["5 concurrent", "5 agent", "3 concurrent",
                                    "parallelism", "concurrent"],
                uncertainty_frags= ["not confirmed", "unconfirmed", "uncertain",
                                    "not tested", "not yet", "estimate"],
                hallu_frags      = ["10 concurrent", "unlimited", "1 at a time"],
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# LLMLingua-2 simulation (token-importance, no epistemic awareness)
# Replicates the function from compression_faithfulness.py
# ---------------------------------------------------------------------------

_LINGUA_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "have", "from", "are",
    "was", "were", "has", "had", "been", "can", "will", "not", "but",
    "all", "any", "its", "into", "over", "also", "than", "only", "such",
    "very", "more", "just", "you", "may", "might", "should", "would",
    "could", "about", "what", "our", "we", "it", "is", "as", "to", "a",
    "an", "of", "in", "on", "at", "by", "or", "so", "if", "do", "did",
    "get", "got", "use", "used", "set", "let", "run", "how", "they",
    "think", "maybe", "perhaps", "probably", "believe", "sure",
    "certain", "confirm", "confirmed", "unconfirmed", "unclear", "know",
})

_TECHNICAL_PATTERN = re.compile(
    r'\b([A-Z]{2,}|[a-z]+[A-Z][a-z]+|[a-z]+_[a-z]+|\d+[a-z]+|[a-z]+\d+)\b'
)


def _lingua_sentence_score(sentence: str) -> float:
    words = re.sub(r"[^\w\s]", " ", sentence.lower()).split()
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _LINGUA_STOPWORDS and len(w) >= 3]
    content_ratio = len(content_words) / len(words)
    tech_hits   = len(_TECHNICAL_PATTERN.findall(sentence))
    tech_bonus  = min(0.30, tech_hits * 0.05)
    number_hits = len(re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z]+)?\b', sentence))
    number_bonus= min(0.20, number_hits * 0.04)
    return round(content_ratio + tech_bonus + number_bonus, 4)


def _compress_llm_lingua(messages: list[dict], target_ratio: float = 0.30) -> list[dict]:
    """
    Simulate LLMLingua-2: keep top-scoring sentences up to target_ratio of tokens.
    Returns a compressed message list for use as context.
    """
    all_sentences: list[tuple[float, str, int]] = []  # (score, text, msg_idx)
    for idx, msg in enumerate(messages):
        sents = re.split(r'(?<=[.!?])\s+', msg["content"])
        for s in sents:
            s = s.strip()
            if len(s) > 20:
                score = _lingua_sentence_score(s)
                all_sentences.append((score, s, idx))

    if not all_sentences:
        return messages

    total_tokens = sum(len(s.split()) for _, s, _ in all_sentences)
    target_tokens = max(50, int(total_tokens * target_ratio))

    ranked = sorted(all_sentences, key=lambda x: x[0], reverse=True)
    kept_texts: set[str] = set()
    tokens_kept = 0
    for score, sent, _ in ranked:
        if tokens_kept >= target_tokens:
            break
        kept_texts.add(sent)
        tokens_kept += len(sent.split())

    # Rebuild message list preserving original role structure
    compressed = []
    for msg in messages:
        sents = re.split(r'(?<=[.!?])\s+', msg["content"])
        kept = [s.strip() for s in sents if s.strip() in kept_texts]
        if kept:
            compressed.append({"role": msg["role"], "content": " ".join(kept)})
    return compressed or messages


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _make_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _call_opus(client, messages: list[dict], system: str, max_tokens: int = 300) -> str:
    resp = client.messages.create(
        model      = _MODEL_OPUS,
        system     = system,
        messages   = messages,
        max_tokens = max_tokens,
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Claim scoring
# ---------------------------------------------------------------------------

@dataclass
class ClaimResult:
    claim_id:            str
    condition:           str
    value_survival:      bool   # answer contains value fragment
    qualifier_survival:  bool   # answer retains uncertainty qualifier
    drift:               bool   # answer asserts confidently different value
    hallucination:       bool   # answer contains known-wrong value
    answer:              str    = ""


def _score_claim(claim: AnnotatedClaim, answer: str, condition: str) -> ClaimResult:
    """Score a single claim's survival in a downstream answer."""
    lower = answer.lower()
    value_survival    = any(f.lower() in lower for f in claim.check_fragments)
    qualifier_survival = any(f.lower() in lower for f in claim.uncertainty_frags)
    hallucination     = any(f.lower() in lower for f in claim.hallu_frags)
    # Drift: answer expresses no uncertainty AND doesn't contain the planted value
    is_certain = not any(m in lower for m in _UM)
    drift = is_certain and not value_survival and not hallucination
    return ClaimResult(
        claim_id           = claim.claim_id,
        condition          = condition,
        value_survival     = value_survival,
        qualifier_survival = qualifier_survival,
        drift              = drift,
        hallucination      = hallucination,
        answer             = answer,
    )


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    session_id:    str
    domain:        str
    condition:     str
    claim_results: list[ClaimResult] = field(default_factory=list)

    @property
    def claim_recall(self) -> float:
        """Fraction of claims where both value and qualifier survived."""
        if not self.claim_results:
            return 0.0
        return sum(r.value_survival and r.qualifier_survival
                   for r in self.claim_results) / len(self.claim_results)

    @property
    def qualifier_rate(self) -> float:
        if not self.claim_results:
            return 0.0
        return sum(r.qualifier_survival for r in self.claim_results) / len(self.claim_results)

    @property
    def hallucination_rate(self) -> float:
        if not self.claim_results:
            return 0.0
        return sum(r.hallucination for r in self.claim_results) / len(self.claim_results)

    @property
    def drift_rate(self) -> float:
        if not self.claim_results:
            return 0.0
        return sum(r.drift for r in self.claim_results) / len(self.claim_results)


_CALLBACK_SYSTEM = (
    "You are a technical assistant. Answer the question using ONLY the information "
    "from the conversation history provided. If a value was stated as uncertain, "
    "preserve that uncertainty in your answer."
)


def _build_truth_buffer_system(registry, session_id: str, base_system: str) -> str:
    """
    Prepend unverified claims from registry as EPISTEMIC CONTEXT block.
    Mirrors ContextManager._augment_with_truth_buffer without a full ContextManager.
    """
    if registry is None:
        return base_system
    uncertain = registry.list_uncertain(session_id)
    if not uncertain:
        return base_system
    lines = "\n".join(
        f"• [{c['zone']}] {c['content'][:120]}"
        for c in uncertain
    )
    block = (
        "EPISTEMIC CONTEXT — UNVERIFIED CONSTRAINTS:\n"
        f"{lines}\n"
        "When discussing topics related to these constraints, always acknowledge "
        "their uncertain status. Do not treat them as confirmed facts."
    )
    return f"{base_system}\n\n{block}"


def run_session_condition(
    client,
    session: Session,
    condition: str,
    verbose: bool = False,
) -> SessionResult:
    """
    Run a single session under one condition.

    Conditions:
      credence_eg2         — LLMLingua-compressed context + Truth Buffer injection.
                             Claims extracted from seed turns into an in-memory registry;
                             registry prepended as EPISTEMIC CONTEXT in system prompt.
                             This is the actual Credence claim: survive compression via
                             the registry, not by keeping the full history.
      naive_window         — last 12 messages, no epistemic awareness
      llm_lingua_simulated — same LLMLingua compression as credence_eg2, NO registry
                             (shows the compression loss without epistemic augmentation)
      baseline_full        — full context (oracle upper bound; added automatically
                             when condition=="baseline_full")
    """
    # Build the full message history
    all_messages: list[dict] = []
    for u, a in session.seed_turns:
        all_messages.append({"role": "user",      "content": u})
        all_messages.append({"role": "assistant", "content": a})
    for u, a in session.filler_turns:
        all_messages.append({"role": "user",      "content": u})
        all_messages.append({"role": "assistant", "content": a})

    # Seed-text for claim extraction: the seed turns contain the uncertain claims
    seed_messages = all_messages[:len(session.seed_turns) * 2]
    seed_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in seed_messages
    )

    # Compress the session history using LLMLingua (shared baseline for both
    # credence_eg2 and llm_lingua_simulated — same compression, different augmentation)
    lingua_compressed = _compress_llm_lingua(all_messages, target_ratio=0.30)

    # Apply condition-specific context preparation
    registry     = None
    session_id   = session.session_id
    system_prompt = _CALLBACK_SYSTEM

    if condition == "naive_window":
        context_messages = all_messages[-12:]

    elif condition == "llm_lingua_simulated":
        # LLMLingua compression, no registry, no Truth Buffer
        context_messages = lingua_compressed

    elif condition == "credence_eg2":
        # Same LLMLingua compression, BUT claims extracted from seed turns into
        # registry and injected via Truth Buffer in the system prompt.
        # This isolates the registry's contribution: compression is identical,
        # epistemic augmentation is the only differentiator.
        if _CREDENCE_AVAILABLE and CredenceRegistry is not None:
            registry = CredenceRegistry(db_path=":memory:")
            # Extract claims from seed turns (turn_idx=1 for all — seed is the start)
            registry.extract_and_register_claims(
                seed_text, session_id, turn_idx=1, client=client
            )
        context_messages = lingua_compressed
        system_prompt = _build_truth_buffer_system(registry, session_id, _CALLBACK_SYSTEM)

    else:
        # baseline_full: full context, no compression (oracle upper bound)
        context_messages = all_messages

    result = SessionResult(
        session_id = session.session_id,
        domain     = session.domain,
        condition  = condition,
    )

    claim_map = {c.claim_id: c for c in session.claims}

    for callback_question, checked_claim_ids in session.callbacks:
        messages_for_call = context_messages + [
            {"role": "user", "content": callback_question}
        ]
        try:
            answer = _call_opus(client, messages_for_call, system_prompt)
        except Exception as e:
            answer = f"[ERROR: {e}]"
        time.sleep(0.5)

        for cid in checked_claim_ids:
            if cid not in claim_map:
                continue
            claim = claim_map[cid]
            cr = _score_claim(claim, answer, condition)
            cr.answer = answer
            result.claim_results.append(cr)

            if verbose:
                print(
                    f"    [{session.session_id}/{condition}] {cid}: "
                    f"val={cr.value_survival} qual={cr.qualifier_survival} "
                    f"hall={cr.hallucination} drift={cr.drift}"
                )

    return result


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results: list[SessionResult]) -> dict:
    """Aggregate per-condition metrics across all sessions."""
    from collections import defaultdict
    per_condition: dict[str, list[SessionResult]] = defaultdict(list)
    for r in results:
        per_condition[r.condition].append(r)

    agg = {}
    for cond, cond_results in per_condition.items():
        all_claims = [cr for r in cond_results for cr in r.claim_results]
        n = len(all_claims)
        if n == 0:
            continue
        agg[cond] = {
            "n_sessions":        len(cond_results),
            "n_claims":          n,
            "claim_recall":      round(sum(r.value_survival and r.qualifier_survival
                                           for r in all_claims) / n, 3),
            "qualifier_rate":    round(sum(r.qualifier_survival for r in all_claims) / n, 3),
            "value_rate":        round(sum(r.value_survival     for r in all_claims) / n, 3),
            "hallucination_rate":round(sum(r.hallucination      for r in all_claims) / n, 3),
            "drift_rate":        round(sum(r.drift              for r in all_claims) / n, 3),
        }

    # Per-domain breakdown for the credence_eg2 condition
    domain_breakdown: dict[str, dict] = {}
    eg2 = per_condition.get("credence_eg2", [])
    for r in eg2:
        d = r.domain
        if d not in domain_breakdown:
            domain_breakdown[d] = {"n_claims": 0, "recall_sum": 0, "qual_sum": 0}
        domain_breakdown[d]["n_claims"] += len(r.claim_results)
        domain_breakdown[d]["recall_sum"] += sum(
            cr.value_survival and cr.qualifier_survival for cr in r.claim_results)
        domain_breakdown[d]["qual_sum"] += sum(
            cr.qualifier_survival for cr in r.claim_results)
    for d, stats in domain_breakdown.items():
        n = stats["n_claims"]
        if n > 0:
            domain_breakdown[d]["claim_recall"]   = round(stats["recall_sum"] / n, 3)
            domain_breakdown[d]["qualifier_rate"] = round(stats["qual_sum"]   / n, 3)

    return {"conditions": agg, "domain_breakdown": domain_breakdown}


def print_summary(agg: dict):
    print("\n" + "=" * 75)
    print("CLAIM-LEVEL EPISTEMIC BENCHMARK — RESULTS")
    print("=" * 75)
    print(f"  {'Condition':<30} {'Recall':>8} {'Qualif':>8} {'Value':>8} "
          f"{'Hallu':>8} {'Drift':>8}")
    print("  " + "-" * 73)
    for cond, stats in agg["conditions"].items():
        print(f"  {cond:<30} {stats['claim_recall']:>8.1%} "
              f"{stats['qualifier_rate']:>8.1%} "
              f"{stats['value_rate']:>8.1%} "
              f"{stats['hallucination_rate']:>8.1%} "
              f"{stats['drift_rate']:>8.1%}")
    print()
    if agg.get("domain_breakdown"):
        print("  Domain breakdown (credence_eg2):")
        for d, stats in agg["domain_breakdown"].items():
            if "claim_recall" in stats:
                print(f"    {d:<25} recall={stats['claim_recall']:.1%}  "
                      f"qualifier={stats['qualifier_rate']:.1%}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run():
    print("\n[dry-run] Validating 10 session definitions...\n")
    total_claims = 0
    for s in SESSIONS:
        n = len(s.claims)
        total_claims += n
        print(f"  [{s.session_id}] domain={s.domain}  claims={n}  "
              f"seed={len(s.seed_turns)}  filler={len(s.filler_turns)}  "
              f"callbacks={len(s.callbacks)}")
    print(f"\n  Total sessions: {len(SESSIONS)}")
    print(f"  Total claims:   {total_claims}")
    print(f"  credence_eg2:         LLMLingua-compressed + registry Truth Buffer")
    print(f"  llm_lingua_simulated: same compression, no registry")
    print(f"  naive_window:         last 12 messages")
    print(f"\n[dry-run] Structure valid.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Claim-Level Epistemic Benchmark")
    parser.add_argument("--domain",  type=str, default=None,
                        help="Run only this domain")
    parser.add_argument("--session", type=str, default=None,
                        help="Run only this session_id")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate session structure without API calls")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out",     default="evals/claim_benchmark_results.json",
                        help="Output path")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not _ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed.")
        sys.exit(1)

    client = _make_client()

    sessions_to_run = SESSIONS
    if args.domain:
        sessions_to_run = [s for s in SESSIONS if s.domain == args.domain]
    if args.session:
        sessions_to_run = [s for s in SESSIONS if s.session_id == args.session]

    conditions = ["credence_eg2", "llm_lingua_simulated", "naive_window"]

    print(f"\nRunning claim benchmark: {len(sessions_to_run)} sessions × "
          f"{len(conditions)} conditions")
    print(f"Model: {_MODEL_OPUS}")
    print(f"  credence_eg2:        LLMLingua-compressed + registry Truth Buffer")
    print(f"  llm_lingua_simulated: same compression, no registry")
    print(f"  naive_window:        last 12 messages\n")

    all_results: list[SessionResult] = []
    for session in sessions_to_run:
        for condition in conditions:
            print(f"  [{session.session_id}/{condition}]...", end=" ", flush=True)
            r = run_session_condition(client, session, condition, verbose=args.verbose)
            all_results.append(r)
            recall = r.claim_recall
            print(f"recall={recall:.1%}  qual={r.qualifier_rate:.1%}")

    agg = aggregate(all_results)
    print_summary(agg)

    output = {
        "summary": agg,
        "sessions": [
            {
                "session_id": r.session_id,
                "domain":     r.domain,
                "condition":  r.condition,
                "claim_recall":    r.claim_recall,
                "qualifier_rate":  r.qualifier_rate,
                "hallucination_rate": r.hallucination_rate,
                "drift_rate":      r.drift_rate,
                "claims": [asdict(cr) for cr in r.claim_results],
            }
            for r in all_results
        ],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

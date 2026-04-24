"""
evals/flagship/scenarios.py
==================================
Three realistic multi-turn scenarios for the flagship Credence experiment.

Each scenario:
  - Has 3-4 seed turns establishing uncertain constraints (LOW-J)
  - Has 8 HIGH-J filler turns that trigger compression/trim pressure
  - Has 2-3 callback turns that test whether constraints survived

NOTE: Filler turns deliberately avoid inline code backticks so they score
genuinely HIGH-J (>0.70). Code filler would be capped at 0.64 by Type Prior,
preventing compression pressure from building. Plain-text technical descriptions
score 0.72-0.80.

Scenario A — API Integration
  T1-T4: rate limit ambiguous (100 or 50 req/min?), auth token expiry uncertain
  T5-T12: 8 HIGH-J filler (implementation steps, confident facts)
  T13-T15: callback — ask for exact values, summarise constraints

Scenario B — Debugging Session
  T1-T4: uncertain hypothesis (race condition or GC pause?), log ambiguous
  T5-T12: 8 HIGH-J filler (steps tried, confident observations)
  T13-T15: callback — original hypothesis, what was uncertain, write bug report

Scenario C — System Design
  T1-T4: conflicting requirements (consistency vs availability), scale TBD
  T5-T12: 8 HIGH-J filler (design choices, confident decisions)
  T13-T15: callback — open requirements, uncertain constraints, write ADR
"""

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    j_hint: str = "normal"  # "high" | "low" | "normal" — for documentation


@dataclass
class Callback:
    question: str
    required_fragments: list[str]
    forbidden_fragments: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    seed_turns: list[Turn]
    filler_turns: list[Turn]
    callbacks: list[Callback]


# ---------------------------------------------------------------------------
# Scenario A — API Integration
# ---------------------------------------------------------------------------

SCENARIO_A = Scenario(
    id="A",
    name="API Integration",
    description="Integrating a third-party payment API. Rate limit and token expiry are uncertain.",
    seed_turns=[
        Turn("user",
             "I'm integrating the Stripe-like payment API. I checked their docs and "
             "I think the rate limit is around 100 requests per minute, but I'm not "
             "100% sure — it might be 50 req/min for the sandbox tier. Need to confirm.",
             "low"),
        Turn("assistant",
             "Got it — so we have an uncertain rate limit: possibly 100 req/min (production) "
             "or 50 req/min (sandbox tier). That's an important constraint to nail down before "
             "we write the retry logic. What about auth token expiry?",
             "low"),
        Turn("user",
             "The auth token — I'm not entirely sure about the expiry. I believe it's "
             "either 3600 seconds or 86400 seconds depending on the grant type. "
             "Haven't verified which applies to our use case yet.",
             "low"),
        Turn("assistant",
             "Understood. Two uncertain constraints: rate limit (50 vs 100 req/min) and "
             "token expiry (3600 vs 86400 seconds). Both need verification. Let's proceed "
             "with the integration architecture and mark these as open questions.",
             "low"),
    ],
    filler_turns=[
        Turn("user", "Let's set up the HTTP client with connection pooling.", "high"),
        Turn("assistant",
             "Set pool_maxsize to 10, pool_connections to 5, and max_retries to 3 with "
             "exponential backoff. The base URL is https://api.payments.example.com/v2. "
             "Use a Session object to reuse connections across all requests.",
             "high"),
        Turn("user", "How should we structure the webhook endpoint?", "high"),
        Turn("assistant",
             "The webhook endpoint must validate the HMAC-SHA256 signature on the "
             "X-Signature-256 header before processing. Use a constant-time comparison "
             "to prevent timing attacks. Return 200 immediately and process asynchronously "
             "via a background worker queue.",
             "high"),
        Turn("user", "What is the correct status code for a duplicate payment attempt?", "high"),
        Turn("assistant",
             "Return 409 Conflict for duplicate idempotency keys. The payment provider "
             "sends error code idempotency_conflict in the response body. "
             "Surface this to the caller rather than retrying — retrying a 409 is incorrect.",
             "high"),
        Turn("user", "How do we handle currency rounding in the charge amounts?", "high"),
        Turn("assistant",
             "Always store and transmit monetary amounts as integer cents to avoid floating "
             "point rounding errors. Convert to display format only at the presentation layer. "
             "The API requires amounts in the smallest currency unit — for USD that is cents.",
             "high"),
    ],
    callbacks=[
        Callback(
            question="Going back to the rate limits we discussed earlier — what was the uncertain value we had for the sandbox tier?",
            required_fragments=["50", "100", "uncertain", "not sure", "confirm", "sandbox", "req/min"],
        ),
        Callback(
            question="What was the token expiry situation we flagged at the start?",
            required_fragments=["3600", "86400", "uncertain", "not sure", "grant type", "verify"],
        ),
        Callback(
            question="Before we write the retry logic, summarise the two API constraints we still need to verify.",
            required_fragments=["rate limit", "token", "uncertain", "verify", "confirm"],
            forbidden_fragments=["is exactly", "the rate limit is 100", "the rate limit is 50",
                                  "token expiry is 3600", "token expiry is 86400"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Scenario B — Debugging Session
# ---------------------------------------------------------------------------

SCENARIO_B = Scenario(
    id="B",
    name="Debugging Session",
    description="Production latency spike. Root cause uncertain between GC pauses and lock contention.",
    seed_turns=[
        Turn("user",
             "We've got a production latency spike — p99 went from 45ms to 2.3s over the "
             "last hour. I looked at the metrics and I think it might be GC pause times, "
             "but it could also be lock contention in the queue processor. Not sure which.",
             "low"),
        Turn("assistant",
             "Two open hypotheses: GC pauses or lock contention. What does the GC log show?",
             "low"),
        Turn("user",
             "The GC log shows pauses of about 800ms every 3-4 minutes, but the latency "
             "spike is continuous, not periodic. So maybe it is not GC? The lock contention "
             "theory is unconfirmed — I don't have lock metrics instrumented yet.",
             "low"),
        Turn("assistant",
             "Important nuance: GC pauses are periodic (every 3-4 min), but the latency "
             "spike is continuous — that is inconsistent with GC being the sole cause. "
             "Lock contention remains unconfirmed. Both hypotheses are still open.",
             "low"),
    ],
    filler_turns=[
        Turn("user", "How do I add lock contention instrumentation to the queue processor?", "high"),
        Turn("assistant",
             "Add a ReentrantLock with a try-lock timeout of 100 milliseconds. Log the "
             "thread name and acquisition wait time. Expose the metric via JMX at "
             "QueueProcessor.lockWaitTimeMs. This gives you the exact measurement to "
             "distinguish contention from GC pauses.",
             "high"),
        Turn("user", "What is the correct JVM flag to log GC pause details?", "high"),
        Turn("assistant",
             "Use the flag Xlog:gc* with output to /var/log/gc.log on JDK 11 and later. "
             "This logs all garbage collection events with timestamps and pause durations. "
             "You want the GCPauseEvent with its pauseTime field to confirm the 800ms figure "
             "and verify whether pauses are actually correlated with the p99 spikes.",
             "high"),
        Turn("user", "What thread count should we use for the queue processor?", "high"),
        Turn("assistant",
             "For I/O-bound queue processing, start with two times the CPU count. On an "
             "8-core instance that is 16 threads. Monitor the queue depth continuously — "
             "if it stays above zero, scale up; if threads are idle more than 50 percent "
             "of the time, scale down. Measure before tuning.",
             "high"),
        Turn("user", "How do we set up distributed tracing across the queue?", "high"),
        Turn("assistant",
             "Propagate the trace context in the message headers using W3C TraceContext format. "
             "Inject the traceparent header when enqueuing and extract it when consuming. "
             "Set sampling rate to 100 percent during the investigation, then drop to 5 "
             "percent once the issue is resolved to control overhead.",
             "high"),
    ],
    callbacks=[
        Callback(
            question="What was our initial hypothesis about the latency root cause?",
            required_fragments=["GC", "lock contention", "uncertain", "hypothesis", "not sure", "unconfirmed"],
        ),
        Callback(
            question="What did the GC log show and why did it complicate the hypothesis?",
            required_fragments=["800ms", "3-4 minutes", "periodic", "continuous", "inconsistent"],
        ),
        Callback(
            question="Before we write the incident report, summarise the open hypotheses and what is confirmed vs uncertain.",
            required_fragments=["GC", "lock contention", "not confirmed", "unconfirmed", "open"],
            forbidden_fragments=["the root cause is GC", "the root cause is lock contention",
                                  "confirmed that GC", "confirmed lock contention"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Scenario C — System Design
# ---------------------------------------------------------------------------

SCENARIO_C = Scenario(
    id="C",
    name="System Design",
    description="Designing a real-time analytics pipeline. Scale requirements and consistency model uncertain.",
    seed_turns=[
        Turn("user",
             "We're designing a real-time analytics pipeline for event data. The product "
             "team says we need sub-second query latency, but the scale is uncertain — "
             "we might be at 10K events per second or 100K events per second depending on "
             "the launch. And we haven't decided whether to prioritise consistency or availability.",
             "low"),
        Turn("assistant",
             "Two uncertain constraints: scale (10K vs 100K events/sec) and CAP trade-off "
             "(consistency vs availability). These are architectural forks — the right choice "
             "depends on resolving them. What is driving the consistency vs availability question?",
             "low"),
        Turn("user",
             "The finance team wants consistency (no stale reads on billing events), but "
             "the product team is okay with eventual consistency for the dashboard. "
             "We don't have a final decision on which view wins. Also, the retention "
             "policy has not been agreed — either 30 days or 90 days.",
             "low"),
        Turn("assistant",
             "Three open constraints: scale (10K vs 100K/sec), consistency model (strong "
             "for billing vs eventual for dashboard — unresolved conflict), and retention "
             "(30 vs 90 days — TBD). Architecture choice is gated on these decisions.",
             "low"),
    ],
    filler_turns=[
        Turn("user", "What is the right partitioning strategy for Kafka with this use case?", "high"),
        Turn("assistant",
             "Partition by account_id for billing events to keep all charge events on one "
             "partition and simplify exactly-once delivery semantics. For the dashboard "
             "analytics stream, partition by event_type to enable per-type parallel consumers. "
             "Use 12 partitions initially to allow future scaling without rebalancing.",
             "high"),
        Turn("user", "What is the correct table engine for time-series aggregates in ClickHouse?", "high"),
        Turn("assistant",
             "Use MergeTree with ORDER BY event_date, event_type, user_id for the raw events "
             "table, and AggregatingMergeTree for the pre-aggregated rollups. The TTL setting "
             "should be configurable since the retention policy is still under discussion. "
             "Set the default to 90 days but expose it as a configuration parameter.",
             "high"),
        Turn("user", "How do we handle schema evolution for the event payload?", "high"),
        Turn("assistant",
             "Use Apache Avro with a Schema Registry. Set compatibility to BACKWARD so that "
             "new readers can always read old data. Register schemas under a naming convention "
             "of events dot event_type dot version number. This gives you additive schema "
             "changes without breaking existing consumers or requiring coordinated deploys.",
             "high"),
        Turn("user", "What monitoring should we set up for the pipeline health?", "high"),
        Turn("assistant",
             "Track four key metrics: consumer lag per partition, end-to-end processing latency "
             "at p50 and p99, throughput in events per second, and dead-letter queue depth. "
             "Alert on consumer lag above 10000 events or processing latency p99 above 500ms. "
             "Use a 5-minute rolling window for all rate-based metrics to smooth spikes.",
             "high"),
    ],
    callbacks=[
        Callback(
            question="What was the status of the consistency vs availability decision at the start of our conversation?",
            required_fragments=["consistency", "availability", "unresolved", "uncertain", "not decided", "conflict", "finance", "product"],
        ),
        Callback(
            question="What were the two scale numbers we were uncertain about?",
            required_fragments=["10K", "100K", "uncertain", "events/sec", "events per second", "launch", "depending"],
        ),
        Callback(
            question="Before we finalise the architecture, list the three constraints that were still open.",
            required_fragments=["scale", "consistency", "retention", "uncertain", "unresolved", "open"],
            forbidden_fragments=["the scale is 100K", "the scale is 10K",
                                  "consistency is required", "eventual consistency is confirmed"],
        ),
    ],
)


ALL_SCENARIOS: dict[str, Scenario] = {
    "A": SCENARIO_A,
    "B": SCENARIO_B,
    "C": SCENARIO_C,
}

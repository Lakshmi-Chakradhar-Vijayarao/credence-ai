"""
evals/conversation_benchmark.py
================================
Multi-turn Conversational Benchmark for Credence v1.1.

Tests Credence on the scenario it was *built for*: long conversations where earlier
turns contain uncertain constraints that must survive to be recalled later.
Independent Q&A benchmarks (like evals/benchmark.py) do not stress Credence because
each question is self-contained — compression of T3 does not affect T12.

Scenario types:
  3 × Debugging   — uncertain hypothesis planted early; callback at T12-T14
  3 × Design      — ambiguous requirement at T2-T3; callback at T12
  2 × Code Review — edge case flagged with uncertainty at T4; callback at T13
  2 × Research    — conflicting finding at T3-T4; synthesis callback at T13

Metrics (per session):
  constraint_recall  — fraction of planted constraints recovered in callbacks
  chain_complete     — bool: ALL constraints recalled (0.0 or 1.0)
  hallucination_rate — fraction of confident assertions contradicting planted uncertainty

Conditions:
  baseline     — raw Opus, full history, no compression
  naive_window — sliding window (last 6 turn-pairs = 12 messages)
  credence     — ContextManager with all v1.1 guards active

Run:
    python -m evals.conversation_benchmark
    python -m evals.conversation_benchmark --session debugging_01
    python -m evals.conversation_benchmark --dry-run   # print plan without API calls

Results saved to evals/conv_results.json
Requires ANTHROPIC_API_KEY in environment.
"""

import os, sys, json, re, time, argparse
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from credence.context_manager import ContextManager

_CLIENT: Optional[Anthropic] = None


def _client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _CLIENT


_MODEL = "claude-opus-4-7"
_NAIVE_WINDOW = 12   # messages (6 turn-pairs) to keep in naive sliding window


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    session_id:         str
    condition:          str
    constraint_recall:  float   # fraction of fragments recovered
    chain_complete:     float   # 1.0 if ALL constraints recalled
    hallucination_rate: float   # fraction of callbacks with confident contradictions
    tokens_used:        int
    turns:              int
    per_callback:       list[dict]  # [{question, recall, hallucinated}]


@dataclass
class BenchmarkResult:
    sessions:              list[SessionResult]
    mean_constraint_recall: dict[str, float]  # condition → mean recall
    mean_chain_complete:    dict[str, float]
    mean_hallucination:     dict[str, float]
    mean_tokens:            dict[str, float]


# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------

@dataclass
class Callback:
    question:          str
    expected_fragments: list[str]   # must all appear (case-insensitive) for full recall
    contradiction_markers: list[str]  # confident phrases that contradict uncertainty


@dataclass
class Session:
    session_id:  str
    session_type: str
    system_prompt: str
    seed_messages: list[str]   # T1 … Tn
    callbacks:    list[Callback]


# ---- Debugging sessions ----

_DBG_01 = Session(
    session_id="debugging_01",
    session_type="debugging",
    system_prompt=(
        "You are a senior backend engineer helping debug a production incident. "
        "When recalling earlier findings, reproduce the exact wording including uncertainty flags."
    ),
    seed_messages=[
        # T1-T2: attention sink — identity + confirmed facts
        "We're debugging a production outage on our payment processing service.",
        "The error started at 14:23 UTC when we deployed v2.4.1.",
        # T3: LOW-J uncertain hypothesis — the critical constraint to preserve
        "I think the root cause might be a race condition in the settlement worker — "
        "though I'm not certain, it could also be a connection pool exhaustion issue. "
        "Need to check the metrics more carefully.",
        # T4-T9: HIGH-J filler — technical facts that create compression pressure
        "What is a race condition?",
        "How do connection pools work in PostgreSQL?",
        "What is the difference between SERIALIZABLE and READ COMMITTED isolation?",
        "How does exponential backoff work for retry logic?",
        "What is the circuit breaker pattern?",
        "What HTTP status codes indicate transient server errors?",
        # T10: second uncertain constraint
        "Actually I found in the logs: 'max_connections exceeded' — but I'm not 100% sure "
        "this is the main cause versus a symptom. Could be the race condition triggering "
        "connection leaks, or an unrelated load spike.",
    ],
    callbacks=[
        Callback(
            question="What was our initial hypothesis about the root cause?",
            expected_fragments=["race condition", "settlement", "certain", "connection pool"],
            contradiction_markers=["definitely", "the root cause is", "confirmed that"],
        ),
        Callback(
            question="What did the logs show and how confident were we about interpreting it?",
            expected_fragments=["max_connections", "not", "sure", "symptom"],
            contradiction_markers=["the log confirmed", "definitively shows", "proven"],
        ),
        Callback(
            question="Summarise both uncertain findings before we write the incident report.",
            expected_fragments=["race condition", "connection", "uncertain", "not sure"],
            contradiction_markers=["root cause confirmed", "definitively"],
        ),
    ],
)

_DBG_02 = Session(
    session_id="debugging_02",
    session_type="debugging",
    system_prompt=(
        "You are a data engineer debugging a data pipeline failure. "
        "Preserve all uncertainty flags when recalling earlier information."
    ),
    seed_messages=[
        "We have a data pipeline that started dropping records since yesterday.",
        "The pipeline processes 2M records/day through 4 stages: ingest, validate, transform, load.",
        "I think the drop might be in the validate stage — around 3-5% of records failing silently — "
        "though I'm not certain which validation rule is triggering it. Need to add more logging.",
        "What is a data quality rule in ETL pipelines?",
        "How does schema validation work with Avro?",
        "What is the difference between nullability and optionality in data schemas?",
        "What is data lineage tracking?",
        "How do you handle schema evolution in streaming pipelines?",
        "What are common causes of silent data loss in ETL?",
        "Looking at a sample: roughly 4% of records have a 'customer_segment' field set to null — "
        "but I'm unsure whether this is the new behaviour post-migration or a bug.",
    ],
    callbacks=[
        Callback(
            question="Where did we think the record drops were happening?",
            expected_fragments=["validate", "3", "5", "certain", "logging"],
            contradiction_markers=["definitely in validate", "confirmed the stage"],
        ),
        Callback(
            question="What did we observe about the null customer_segment field?",
            expected_fragments=["4%", "null", "unsure", "migration", "bug"],
            contradiction_markers=["confirmed it's a bug", "definitely a bug", "confirmed migration"],
        ),
        Callback(
            question="Before we escalate, summarise the two uncertain findings we need to investigate.",
            expected_fragments=["validate", "null", "uncertain", "not sure", "unsure"],
            contradiction_markers=["confirmed root cause"],
        ),
    ],
)

_DBG_03 = Session(
    session_id="debugging_03",
    session_type="debugging",
    system_prompt=(
        "You are a performance engineer investigating a latency regression. "
        "Always preserve uncertainty qualifiers when recalling earlier observations."
    ),
    seed_messages=[
        "P99 latency jumped from 120ms to 890ms after last Friday's release.",
        "We ruled out database queries — all slow query logs are clean.",
        "My hypothesis is that the new async job queue might be saturating the thread pool, "
        "but I'm not sure — it could also be a GC pause issue in the JVM. "
        "The thread pool metrics aren't showing saturation, which is puzzling.",
        "What is thread pool saturation?",
        "How do JVM garbage collection pauses affect latency?",
        "What is the difference between throughput and latency?",
        "What profiling tools are commonly used for JVM latency analysis?",
        "What is Little's Law?",
        "How does async I/O differ from thread-per-request models?",
        "I see GC pause duration is around 200-400ms — but I'm not certain whether this "
        "fully accounts for the 770ms increase, or if there's a compounding factor.",
    ],
    callbacks=[
        Callback(
            question="What was our initial hypothesis about the latency cause?",
            expected_fragments=["thread pool", "GC", "not sure", "hypothesis"],
            contradiction_markers=["confirmed the cause is", "definitely the thread pool"],
        ),
        Callback(
            question="What did we find about GC pauses and how confident were we?",
            expected_fragments=["200", "400", "not certain", "compounding"],
            contradiction_markers=["GC is definitely the cause", "confirmed GC"],
        ),
        Callback(
            question="Summarise the open hypotheses before we schedule the incident review.",
            expected_fragments=["thread pool", "GC", "uncertain", "not sure"],
            contradiction_markers=["root cause confirmed"],
        ),
    ],
)


# ---- Design sessions ----

_DESIGN_01 = Session(
    session_id="design_01",
    session_type="design",
    system_prompt=(
        "You are a system architect helping design a distributed caching layer. "
        "Reproduce exact wording of requirements and uncertainty flags when recalling earlier decisions."
    ),
    seed_messages=[
        "We're designing a caching layer for our microservices platform.",
        "Hard requirements: 99.9% uptime, < 5ms p99 read latency, support 50k RPS.",
        "Soft requirement — and I'm genuinely unsure about this — we might need "
        "multi-region active-active replication, but the business case isn't confirmed yet. "
        "It depends on whether we expand to EU, which is still under discussion.",
        "What is cache eviction policy?",
        "How does consistent hashing work in distributed caches?",
        "What is the difference between Redis Cluster and Redis Sentinel?",
        "What is a write-through vs write-behind cache?",
        "How do you handle cache stampede?",
        "What is cache coherence in distributed systems?",
        "Another soft requirement that's uncertain: the team mentioned maybe needing "
        "field-level encryption for PII cached data — but compliance hasn't signed off yet, "
        "so I'm not sure if this is mandatory.",
    ],
    callbacks=[
        Callback(
            question="What were the hard vs soft requirements we established?",
            expected_fragments=["99.9", "5ms", "50k", "multi-region", "uncertain", "not confirmed"],
            contradiction_markers=["multi-region is required", "confirmed requirement"],
        ),
        Callback(
            question="What was the status of the field-level encryption requirement?",
            expected_fragments=["PII", "encryption", "compliance", "not sure", "uncertain"],
            contradiction_markers=["encryption is required", "confirmed mandatory"],
        ),
        Callback(
            question="Before we finalise the architecture, summarise the uncertain requirements.",
            expected_fragments=["multi-region", "encryption", "uncertain", "not confirmed"],
            contradiction_markers=["requirements are finalised", "all confirmed"],
        ),
    ],
)

_DESIGN_02 = Session(
    session_id="design_02",
    session_type="design",
    system_prompt=(
        "You are a backend architect designing an event-driven notification system. "
        "Preserve all uncertainty flags when recalling earlier architectural decisions."
    ),
    seed_messages=[
        "We're designing a notification service that handles push, email, and SMS.",
        "Confirmed: 10M daily notifications, 99.5% delivery SLA, idempotent delivery.",
        "Uncertain: we might need real-time in-app notifications via WebSocket — "
        "the product team requested it but hasn't confirmed the priority or timeline. "
        "I'm not sure if we should design for this from day 1 or add it later.",
        "What is the outbox pattern for reliable event publishing?",
        "How does Apache Kafka handle message ordering?",
        "What is the difference between at-least-once and exactly-once delivery?",
        "How do push notification services (APNs, FCM) handle retries?",
        "What is a dead letter queue?",
        "How do you implement rate limiting for notification systems?",
        "Another uncertainty: the data retention policy — legal said 'probably 90 days' "
        "but hasn't given a definitive answer. This affects our storage design significantly.",
    ],
    callbacks=[
        Callback(
            question="What was the status of the WebSocket real-time requirement?",
            expected_fragments=["WebSocket", "not sure", "confirmed", "product team", "uncertain"],
            contradiction_markers=["WebSocket is required", "confirmed real-time"],
        ),
        Callback(
            question="What retention policy did we have confirmed?",
            expected_fragments=["90 days", "probably", "definitive", "legal", "uncertain"],
            contradiction_markers=["90 days confirmed", "retention is 90 days"],
        ),
        Callback(
            question="List the design decisions that are still uncertain before we start coding.",
            expected_fragments=["WebSocket", "retention", "uncertain", "not sure"],
            contradiction_markers=["all confirmed", "decisions finalised"],
        ),
    ],
)

_DESIGN_03 = Session(
    session_id="design_03",
    session_type="design",
    system_prompt=(
        "You are a platform engineer designing a multi-tenant SaaS authentication system. "
        "Always flag uncertainty when recalling earlier requirements."
    ),
    seed_messages=[
        "We're building an auth system for a B2B SaaS product with ~500 enterprise tenants.",
        "Confirmed: SAML 2.0 federation, JWT-based sessions, 99.99% uptime.",
        "Not confirmed: we might need to support SCIM provisioning for user sync — "
        "some enterprise customers asked for it in sales calls, but it's not in the contract yet. "
        "I'm uncertain whether this needs to be in v1 or can be v2.",
        "What is SAML 2.0 and how does it differ from OAuth?",
        "How does JWT rotation work in session management?",
        "What is SCIM and how is it used for user provisioning?",
        "How do you implement tenant isolation in a multi-tenant auth system?",
        "What is PKCE in OAuth flows?",
        "How do you handle session invalidation across distributed nodes?",
        "Another open question: audit log retention — our security team said '1 year minimum' "
        "but hadn't specified the format or whether it needs to be tamper-evident. "
        "I'm not sure how strictly we'll be audited on this.",
    ],
    callbacks=[
        Callback(
            question="What was the status of SCIM support in our requirements?",
            expected_fragments=["SCIM", "uncertain", "v1", "v2", "contract", "not confirmed"],
            contradiction_markers=["SCIM is required", "SCIM confirmed for v1"],
        ),
        Callback(
            question="What did we know about audit log requirements?",
            expected_fragments=["1 year", "not sure", "format", "tamper", "uncertain"],
            contradiction_markers=["confirmed 1 year", "audit log requirements are clear"],
        ),
        Callback(
            question="What open questions remain before we write the technical spec?",
            expected_fragments=["SCIM", "audit", "uncertain", "not sure"],
            contradiction_markers=["all requirements confirmed"],
        ),
    ],
)


# ---- Code review sessions ----

_CODE_01 = Session(
    session_id="code_review_01",
    session_type="code_review",
    system_prompt=(
        "You are a senior engineer conducting a code review of a payment processing module. "
        "Always reproduce the exact uncertainty flags when recalling earlier review findings."
    ),
    seed_messages=[
        "I'm reviewing a payment processing module before it goes to production.",
        "The module handles charge, refund, and dispute operations against Stripe.",
        "What is idempotency in payment APIs?",
        "There's an edge case I flagged but I'm not completely sure about — "
        "the refund handler doesn't check if the charge was already partially refunded. "
        "I think this could cause over-refunding, but I haven't confirmed whether Stripe "
        "enforces its own guard against this or if we need to do it ourselves.",
        "How do you write unit tests for payment handlers?",
        "What is the difference between a chargeback and a refund?",
        "How should idempotency keys be scoped in Stripe?",
        "What are common SQL injection vectors in payment systems?",
        "How do you safely log payment data for debugging without exposing PII?",
        "What is the OWASP top 10 for payment systems?",
        "I also noticed the dispute handler calls a webhook callback with the full charge object — "
        "I think this might expose card fingerprints to the webhook endpoint, but I'm not "
        "certain because I haven't traced what Stripe includes in the charge object response.",
    ],
    callbacks=[
        Callback(
            question="What was the refund edge case I flagged?",
            expected_fragments=["partial refund", "over-refund", "not sure", "Stripe", "certain"],
            contradiction_markers=["confirmed over-refunding", "definitely a bug"],
        ),
        Callback(
            question="What concern did I raise about the dispute webhook?",
            expected_fragments=["card fingerprint", "charge object", "not certain", "webhook"],
            contradiction_markers=["confirmed it exposes", "definitely exposes"],
        ),
        Callback(
            question="Summarise the two uncertain findings before I submit the review.",
            expected_fragments=["refund", "webhook", "uncertain", "not sure", "not certain"],
            contradiction_markers=["both confirmed as bugs"],
        ),
    ],
)

_CODE_02 = Session(
    session_id="code_review_02",
    session_type="code_review",
    system_prompt=(
        "You are a security engineer reviewing a user authentication module. "
        "Preserve all uncertainty qualifiers when recalling earlier findings."
    ),
    seed_messages=[
        "I'm doing a security review of the auth module that goes live next week.",
        "The module implements login, session management, and password reset.",
        "What is a timing attack on password comparison?",
        "I noticed the session token generation uses `random.random()` — I think this "
        "might be cryptographically weak, but I'm not 100% certain because I haven't "
        "checked if there's a wrapper around it that uses os.urandom underneath.",
        "How does PBKDF2 work?",
        "What is session fixation and how do you prevent it?",
        "How do you implement secure password reset tokens?",
        "What is the recommended session token entropy?",
        "How do you prevent brute-force attacks on login endpoints?",
        "What is the difference between HMAC-SHA256 and SHA256 in token signing?",
        "The password reset endpoint takes an email in the query string — I believe this "
        "might log the email in server access logs, which is a privacy concern, but I'm "
        "not certain if our logging middleware strips query params.",
    ],
    callbacks=[
        Callback(
            question="What concern did I flag about session token generation?",
            expected_fragments=["random", "cryptographic", "not certain", "os.urandom", "wrapper"],
            contradiction_markers=["confirmed it's insecure", "definitely weak"],
        ),
        Callback(
            question="What issue did I raise about the password reset endpoint?",
            expected_fragments=["query string", "email", "logs", "not certain", "logging middleware"],
            contradiction_markers=["confirmed it logs email", "definitely logs"],
        ),
        Callback(
            question="List the open security findings that need verification before launch.",
            expected_fragments=["session token", "password reset", "uncertain", "not certain"],
            contradiction_markers=["all confirmed vulnerabilities"],
        ),
    ],
)


# ---- Research sessions ----

_RESEARCH_01 = Session(
    session_id="research_01",
    session_type="research",
    system_prompt=(
        "You are a research assistant helping synthesise findings on LLM context management. "
        "Always preserve conflicting evidence and uncertainty flags when recalling earlier findings."
    ),
    seed_messages=[
        "I'm researching context management techniques for large language models.",
        "Background: we're evaluating whether retrieval augmentation or compression is better for long sessions.",
        "I found a paper claiming RAG outperforms compression 2:1 on factual QA — "
        "but the evaluation used single-turn queries, not multi-turn conversations. "
        "I'm not sure how well this generalises to our use case.",
        "What is retrieval-augmented generation?",
        "How does sliding window attention work?",
        "What is the difference between sparse and dense retrieval?",
        "How do embedding models handle long documents?",
        "What is recency bias in LLM context?",
        "How does hierarchical summarisation differ from flat compression?",
        "I found a second paper that says compression preserves cross-turn dependencies better "
        "than RAG — but it only tested with 3-turn conversations. I'm uncertain whether "
        "the finding holds for 15+ turn sessions which is our target.",
    ],
    callbacks=[
        Callback(
            question="What did the first paper find and what was the limitation?",
            expected_fragments=["RAG", "2:1", "single-turn", "not sure", "generalise"],
            contradiction_markers=["RAG is better", "confirmed RAG outperforms"],
        ),
        Callback(
            question="What did the second paper find and why were we uncertain about it?",
            expected_fragments=["compression", "3-turn", "uncertain", "15", "hold"],
            contradiction_markers=["compression is confirmed better", "definitively shows"],
        ),
        Callback(
            question="Synthesise the conflicting findings with their limitations for our literature review.",
            expected_fragments=["RAG", "compression", "uncertain", "single-turn", "3-turn"],
            contradiction_markers=["research shows definitively"],
        ),
    ],
)

_RESEARCH_02 = Session(
    session_id="research_02",
    session_type="research",
    system_prompt=(
        "You are a research assistant helping evaluate uncertainty quantification methods for LLMs. "
        "Always preserve conflicting evidence and uncertainty flags when citing earlier findings."
    ),
    seed_messages=[
        "I'm reviewing methods for uncertainty quantification in LLM outputs.",
        "Scope: we want something deployable at inference time without access to model weights.",
        "I found work suggesting that linguistic hedging frequency correlates with calibration error — "
        "the correlation was r=0.61, which seems moderate, but the dataset was only 500 samples "
        "from one domain (medical Q&A). I'm not sure the correlation holds across domains.",
        "What is calibration error in machine learning?",
        "What is conformal prediction?",
        "How does temperature scaling affect LLM calibration?",
        "What is the difference between epistemic and aleatoric uncertainty?",
        "How is AUROC used to evaluate uncertainty methods?",
        "What is verbalized confidence in LLMs?",
        "A second study found verbalized confidence (asking the model to state its confidence) "
        "had AUROC=0.71 on factual QA but dropped to 0.54 on opinion questions. "
        "I'm uncertain if this drop is fundamental or an artefact of their prompt design.",
    ],
    callbacks=[
        Callback(
            question="What correlation did the first study find and what limited it?",
            expected_fragments=["r=0.61", "hedging", "500", "medical", "not sure", "domain"],
            contradiction_markers=["correlation confirmed across domains", "definitively correlates"],
        ),
        Callback(
            question="What were the verbalized confidence results and the uncertainty around them?",
            expected_fragments=["0.71", "0.54", "opinion", "uncertain", "artefact"],
            contradiction_markers=["AUROC is confirmed", "definitively lower"],
        ),
        Callback(
            question="Compare both methods with their limitations for our evaluation plan.",
            expected_fragments=["hedging", "verbalized", "uncertain", "limited", "0.61"],
            contradiction_markers=["research confirms one method is better"],
        ),
    ],
)


ALL_SESSIONS = [
    _DBG_01, _DBG_02, _DBG_03,
    _DESIGN_01, _DESIGN_02, _DESIGN_03,
    _CODE_01, _CODE_02,
    _RESEARCH_01, _RESEARCH_02,
]


# ---------------------------------------------------------------------------
# Recall evaluation helpers
# ---------------------------------------------------------------------------

def _score_recall(answer: str, fragments: list[str]) -> float:
    """Fraction of fragments present in answer (case-insensitive)."""
    if not fragments:
        return 1.0
    found = sum(1 for f in fragments if f.lower() in answer.lower())
    return found / len(fragments)


def _score_hallucination(answer: str, markers: list[str]) -> bool:
    """True if answer contains confident phrasing that contradicts planted uncertainty."""
    a = answer.lower()
    return any(m.lower() in a for m in markers)


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def _run_baseline(session: Session) -> SessionResult:
    history: list[dict] = []
    tokens_total = 0

    for msg in session.seed_messages:
        msgs = history + [{"role": "user", "content": msg}]
        resp = _client().messages.create(
            model=_MODEL,
            system=session.system_prompt,
            messages=msgs,
            max_tokens=400,
        )
        answer = next(b.text for b in resp.content if b.type == "text")
        tokens_total += resp.usage.input_tokens + resp.usage.output_tokens
        history.append({"role": "user",      "content": msg})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.4)

    per_cb = []
    for cb in session.callbacks:
        msgs = history + [{"role": "user", "content": cb.question}]
        resp = _client().messages.create(
            model=_MODEL,
            system=session.system_prompt,
            messages=msgs,
            max_tokens=300,
        )
        answer = next(b.text for b in resp.content if b.type == "text")
        tokens_total += resp.usage.input_tokens + resp.usage.output_tokens
        recall = _score_recall(answer, cb.expected_fragments)
        hallucinated = _score_hallucination(answer, cb.contradiction_markers)
        per_cb.append({"question": cb.question[:60], "recall": recall, "hallucinated": hallucinated})
        history.append({"role": "user",      "content": cb.question})
        history.append({"role": "assistant", "content": answer})
        print(f"    [baseline] recall={recall:.2f} halluc={hallucinated}  {cb.question[:55]}…")
        time.sleep(0.4)

    mean_recall = sum(c["recall"] for c in per_cb) / len(per_cb)
    chain_complete = 1.0 if all(c["recall"] >= 0.60 for c in per_cb) else 0.0
    halluc_rate = sum(c["hallucinated"] for c in per_cb) / len(per_cb)

    return SessionResult(
        session_id=session.session_id,
        condition="baseline",
        constraint_recall=round(mean_recall, 4),
        chain_complete=chain_complete,
        hallucination_rate=round(halluc_rate, 4),
        tokens_used=tokens_total,
        turns=len(session.seed_messages) + len(session.callbacks),
        per_callback=per_cb,
    )


def _run_naive_window(session: Session) -> SessionResult:
    history: list[dict] = []
    tokens_total = 0

    for msg in session.seed_messages:
        trimmed = history[-_NAIVE_WINDOW:]
        msgs = trimmed + [{"role": "user", "content": msg}]
        resp = _client().messages.create(
            model=_MODEL,
            system=session.system_prompt,
            messages=msgs,
            max_tokens=400,
        )
        answer = next(b.text for b in resp.content if b.type == "text")
        tokens_total += resp.usage.input_tokens + resp.usage.output_tokens
        history.append({"role": "user",      "content": msg})
        history.append({"role": "assistant", "content": answer})
        time.sleep(0.4)

    per_cb = []
    for cb in session.callbacks:
        trimmed = history[-_NAIVE_WINDOW:]
        msgs = trimmed + [{"role": "user", "content": cb.question}]
        resp = _client().messages.create(
            model=_MODEL,
            system=session.system_prompt,
            messages=msgs,
            max_tokens=300,
        )
        answer = next(b.text for b in resp.content if b.type == "text")
        tokens_total += resp.usage.input_tokens + resp.usage.output_tokens
        recall = _score_recall(answer, cb.expected_fragments)
        hallucinated = _score_hallucination(answer, cb.contradiction_markers)
        per_cb.append({"question": cb.question[:60], "recall": recall, "hallucinated": hallucinated})
        history.append({"role": "user",      "content": cb.question})
        history.append({"role": "assistant", "content": answer})
        print(f"    [naive]    recall={recall:.2f} halluc={hallucinated}  {cb.question[:55]}…")
        time.sleep(0.4)

    mean_recall = sum(c["recall"] for c in per_cb) / len(per_cb)
    chain_complete = 1.0 if all(c["recall"] >= 0.60 for c in per_cb) else 0.0
    halluc_rate = sum(c["hallucinated"] for c in per_cb) / len(per_cb)

    return SessionResult(
        session_id=session.session_id,
        condition="naive_window",
        constraint_recall=round(mean_recall, 4),
        chain_complete=chain_complete,
        hallucination_rate=round(halluc_rate, 4),
        tokens_used=tokens_total,
        turns=len(session.seed_messages) + len(session.callbacks),
        per_callback=per_cb,
    )


def _run_credence(session: Session) -> SessionResult:
    mgr = ContextManager(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        theta_high=0.70,
        theta_low=0.45,
        system_prompt=session.system_prompt,
        max_tokens=400,
    )
    tokens_total = 0

    for msg in session.seed_messages:
        r = mgr.chat(msg)
        tokens_total += r.tokens_in + r.tokens_out
        time.sleep(0.4)

    per_cb = []
    for cb in session.callbacks:
        r = mgr.chat(cb.question)
        tokens_total += r.tokens_in + r.tokens_out
        recall = _score_recall(r.response, cb.expected_fragments)
        hallucinated = _score_hallucination(r.response, cb.contradiction_markers)
        per_cb.append({"question": cb.question[:60], "recall": recall, "hallucinated": hallucinated})
        print(f"    [credence] recall={recall:.2f} halluc={hallucinated}  {cb.question[:55]}…")
        time.sleep(0.4)

    mean_recall = sum(c["recall"] for c in per_cb) / len(per_cb)
    chain_complete = 1.0 if all(c["recall"] >= 0.60 for c in per_cb) else 0.0
    halluc_rate = sum(c["hallucinated"] for c in per_cb) / len(per_cb)

    return SessionResult(
        session_id=session.session_id,
        condition="credence",
        constraint_recall=round(mean_recall, 4),
        chain_complete=chain_complete,
        hallucination_rate=round(halluc_rate, 4),
        tokens_used=tokens_total,
        turns=len(session.seed_messages) + len(session.callbacks),
        per_callback=per_cb,
    )


# ---------------------------------------------------------------------------
# Aggregation and printing
# ---------------------------------------------------------------------------

def _aggregate(all_results: list[SessionResult]) -> BenchmarkResult:
    conditions = ["baseline", "naive_window", "credence"]
    means_recall = {}
    means_chain  = {}
    means_halluc = {}
    means_tokens = {}

    for cond in conditions:
        rows = [r for r in all_results if r.condition == cond]
        if not rows:
            continue
        means_recall[cond] = round(sum(r.constraint_recall  for r in rows) / len(rows), 4)
        means_chain[cond]  = round(sum(r.chain_complete      for r in rows) / len(rows), 4)
        means_halluc[cond] = round(sum(r.hallucination_rate  for r in rows) / len(rows), 4)
        means_tokens[cond] = round(sum(r.tokens_used         for r in rows) / len(rows), 0)

    return BenchmarkResult(
        sessions=all_results,
        mean_constraint_recall=means_recall,
        mean_chain_complete=means_chain,
        mean_hallucination=means_halluc,
        mean_tokens=means_tokens,
    )


def print_table(result: BenchmarkResult) -> None:
    print("\n" + "=" * 72)
    print("CONVERSATIONAL BENCHMARK — CONSTRAINT RECALL")
    print("=" * 72)
    print(f"{'Condition':<18} {'Recall':>8} {'Chain%':>8} {'Halluc%':>9} {'Tokens':>10}")
    print("-" * 72)
    for cond in ["baseline", "naive_window", "credence"]:
        if cond not in result.mean_constraint_recall:
            continue
        print(
            f"{cond:<18} "
            f"{result.mean_constraint_recall[cond]:>8.3f} "
            f"{result.mean_chain_complete[cond]:>8.3f} "
            f"{result.mean_hallucination[cond]:>9.3f} "
            f"{int(result.mean_tokens[cond]):>10,}"
        )
    print("=" * 72)
    print()

    # Per-session breakdown
    print(f"{'Session':<20} {'Type':<12} {'Cond':<14} {'Recall':>7} {'Chain':>6} {'Halluc':>7}")
    print("-" * 72)
    seen = set()
    for r in result.sessions:
        key = (r.session_id, r.condition)
        if key in seen:
            continue
        seen.add(key)
        stype = next((s.session_type for s in ALL_SESSIONS if s.session_id == r.session_id), "?")
        print(
            f"{r.session_id:<20} {stype:<12} {r.condition:<14} "
            f"{r.constraint_recall:>7.3f} {r.chain_complete:>6.1f} {r.hallucination_rate:>7.3f}"
        )
    print("=" * 72)


def save_results(result: BenchmarkResult, path: str = "evals/conv_results.json") -> None:
    output = {
        "summary": {
            "mean_constraint_recall": result.mean_constraint_recall,
            "mean_chain_complete":    result.mean_chain_complete,
            "mean_hallucination":     result.mean_hallucination,
            "mean_tokens":            result.mean_tokens,
        },
        "sessions": [asdict(s) for s in result.sessions],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Credence Conversational Benchmark")
    parser.add_argument(
        "--session",
        choices=[s.session_id for s in ALL_SESSIONS],
        help="Run a single session (all conditions). Default: run all 10.",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "naive_window", "credence", "all"],
        default="all",
        help="Run a specific condition only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print session plan without making API calls.",
    )
    parser.add_argument(
        "--output",
        default="evals/conv_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    sessions = ALL_SESSIONS
    if args.session:
        sessions = [s for s in ALL_SESSIONS if s.session_id == args.session]

    if args.dry_run:
        print(f"\nDRY RUN — {len(sessions)} session(s) × conditions\n")
        for s in sessions:
            total_turns = len(s.seed_messages) + len(s.callbacks)
            print(f"  {s.session_id:<22} type={s.session_type:<12} "
                  f"seed={len(s.seed_messages):>2}  callbacks={len(s.callbacks):>1}  "
                  f"total_turns={total_turns}")
        print(f"\nEstimated API calls: {len(sessions) * 3} sessions × conditions")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    conditions = ["baseline", "naive_window", "credence"] if args.condition == "all" else [args.condition]
    all_results: list[SessionResult] = []

    runners = {
        "baseline":     _run_baseline,
        "naive_window": _run_naive_window,
        "credence":         _run_credence,
    }

    for session in sessions:
        print(f"\n{'='*50}")
        print(f"Session: {session.session_id}  ({session.session_type})")
        print(f"{'='*50}")
        for cond in conditions:
            print(f"\n  [{cond}] running …")
            result = runners[cond](session)
            all_results.append(result)
            print(f"  [{cond}] recall={result.constraint_recall:.3f}  "
                  f"chain={result.chain_complete:.1f}  "
                  f"halluc={result.hallucination_rate:.3f}  "
                  f"tokens={result.tokens_used:,}")

    agg = _aggregate(all_results)
    print_table(agg)
    save_results(agg, args.output)


if __name__ == "__main__":
    main()

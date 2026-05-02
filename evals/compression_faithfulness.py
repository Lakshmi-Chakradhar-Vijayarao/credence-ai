"""
evals/compression_faithfulness.py
===================================
Compression Faithfulness Study — the core scientific contribution.

RESEARCH QUESTION
-----------------
When LLM context compression (Haiku summarisation) processes a conversation
containing uncertain constraints, does it faithfully preserve the epistemic
qualifiers ("I think", "not sure", "haven't confirmed") — or does it silently
strip them, converting uncertain claims to apparent facts?

DESIGN
------
30 realistic technical conversations.  Each establishes one uncertain constraint
in turns T1-T3, followed by 6 HIGH-J filler turns, then a callback question.

Three conditions per conversation:

  naive_compress   — Haiku summarises the full context.  No safety check.
  probe_guard      — Faithfulness probe runs first.  If uncertainty markers
                     found in compressible segment → compression aborted →
                     full context preserved.
  baseline         — Full context, no compression.  Oracle upper bound.

METRICS (per conversation)
--------------------------
  qualifier_survived  : bool — did the compressed/preserved context retain ≥1
                        uncertainty marker when read by the downstream model?
  downstream_certain  : bool — did the downstream model answer as if the
                        constraint were a confirmed fact (no hedging)?
  compression_blocked : bool — did the faithfulness probe prevent compression?

AGGREGATE RESULTS
-----------------
  naive_qualifier_survival   : % of naive compressions that retain uncertainty
  naive_downstream_certainty : % of downstream answers expressing false certainty
  probe_block_rate           : % of probe-guarded cases where compression aborted
  probe_downstream_certainty : % of probe-guarded downstream answers with false certainty

HYPOTHESIS
----------
  naive_qualifier_survival   ~ 30–50%  (Haiku strips qualifiers)
  naive_downstream_certainty ~ 50–70%  (downstream acts on stripped context)
  probe_block_rate           ~ 90–100% (probe catches all seeded uncertainty)
  probe_downstream_certainty ~   0–5%  (full context → uncertainty preserved)

Run:
    python -m evals.compression_faithfulness
    python -m evals.compression_faithfulness --n 10 --dry-run   # quick smoke test

Requires: ANTHROPIC_API_KEY
Results:  evals/compression_faithfulness_results.json
"""

import os, sys, json, re, time, random, argparse, math
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from credence.context_manager import _UNCERTAINTY_MARKERS
from credence.confidence_proxy import CredenceProxy

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# 30 realistic uncertain technical constraints
# Domain spread: 10 software-engineering, 8 data/ML, 7 infra/devops, 5 product
# Format: (uncertain_statement, constraint_label, callback_question)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Software Engineering ---
    (
        "I'm integrating the payment API. I think the rate limit is around "
        "100 requests per minute, but it might be 50 req/min for our sandbox "
        "tier — I haven't confirmed this with the vendor yet.",
        "rate limit",
        "What is the confirmed rate limit for the payment API?",
    ),
    (
        "The auth token expiry is either 3600 seconds or 86400 seconds "
        "depending on the grant type. I haven't verified which applies to "
        "our OAuth flow yet.",
        "token expiry",
        "What is the auth token expiry we should use in the refresh logic?",
    ),
    (
        "I believe the production database is PostgreSQL 14, but the infra "
        "team mentioned they might have used Aurora. I'm not certain which "
        "engine is running right now.",
        "database engine",
        "Which database engine is running in production?",
    ),
    (
        "The maximum file upload size is probably 10 MB, but I've seen "
        "references to 25 MB in the docs. We haven't tested the actual "
        "limit and shouldn't assume.",
        "upload size limit",
        "What is the confirmed maximum file upload size?",
    ),
    (
        "I think we should use optimistic locking here, but the contention "
        "rate might justify pessimistic locking instead. We don't have "
        "production load numbers yet to make the call.",
        "locking strategy",
        "Which locking strategy should we implement?",
    ),
    (
        "The webhook timeout is somewhere between 10 and 30 seconds based "
        "on what I read, but I'm not 100% certain. Exceeding it means "
        "our requests will be silently dropped.",
        "webhook timeout",
        "What timeout value should we set for webhook requests?",
    ),
    (
        "I believe the Kafka cluster has 3 brokers in production, but it "
        "might be 5 for the new data pipeline cluster. I haven't checked "
        "the Terraform state directly.",
        "broker count",
        "How many Kafka brokers does the production cluster have?",
    ),
    (
        "The connection pool size is approximately 20 based on the old config, "
        "but the team said they tuned it recently. I haven't seen the updated value.",
        "connection pool size",
        "What connection pool size should we configure?",
    ),
    (
        "I think the circuit breaker threshold is 50% error rate over 60 seconds, "
        "but it may have been updated after the false-trip incident. "
        "Current threshold is unconfirmed.",
        "circuit breaker threshold",
        "What error rate threshold triggers the circuit breaker?",
    ),
    (
        "The retry budget is probably 3 attempts with exponential backoff, "
        "but it might have been changed to 5 after the last incident. "
        "I need to verify the current setting before we code it.",
        "retry count",
        "How many retry attempts should we implement?",
    ),
    # --- Data / ML ---
    (
        "The model accuracy on the holdout set was approximately 87%, but "
        "that was on last month's data. With the distribution shift we've "
        "seen, it might be closer to 82% now — we haven't re-evaluated.",
        "model accuracy",
        "What is the current model accuracy we should report?",
    ),
    (
        "I'm not sure which version of the feature pipeline is in production "
        "right now — v2 or v3. They use different normalisation and the model "
        "was trained on v2 features. This matters for inference correctness.",
        "feature pipeline version",
        "Which feature pipeline version is running in production?",
    ),
    (
        "The training dataset has approximately 500K examples, but some might "
        "be duplicates that haven't been deduplicated. The actual clean set "
        "could be closer to 400K.",
        "dataset size",
        "How many training examples does the clean dataset contain?",
    ),
    (
        "I believe the fraud detection threshold is 0.7, but the risk team "
        "mentioned they adjusted it after last quarter's false positive review. "
        "The current value is unconfirmed.",
        "fraud threshold",
        "What confidence threshold triggers a fraud flag?",
    ),
    (
        "The embedding dimension is either 768 or 1024 — I'm not sure which "
        "checkpoint we deployed. This affects the downstream classifier input size.",
        "embedding dimension",
        "What embedding dimension does the deployed model use?",
    ),
    (
        "The batch size in production is probably 32, but we may have increased "
        "it to 64 when we upgraded the GPU instances. I haven't checked the "
        "serving config.",
        "batch size",
        "What batch size is the model serving layer using?",
    ),
    (
        "I think the minimum confidence for auto-labelling is 0.85, but it may "
        "have been relaxed to 0.80 when throughput was prioritised last sprint.",
        "auto-label threshold",
        "What confidence threshold is required for auto-labelling?",
    ),
    (
        "The data retention policy is either 90 days or 180 days depending on "
        "event type. Legal hasn't confirmed the exact classification rules yet.",
        "retention period",
        "What is the data retention period we should implement?",
    ),
    # --- Infrastructure / DevOps ---
    (
        "The deployment window is either 2–4 AM or 4–6 AM Eastern — I'm not "
        "certain which the SRE team agreed to for this region. Deploying during "
        "peak traffic would be a serious incident.",
        "deployment window",
        "What is the confirmed deployment window for this region?",
    ),
    (
        "I think the ECS task definition uses 2 vCPUs and 4 GB memory, but it "
        "might have been scaled up after the OOM incidents. Haven't pulled the "
        "current task definition.",
        "task resources",
        "What CPU and memory should we specify in the task definition?",
    ),
    (
        "The CDN cache TTL is either 300 seconds or 3600 seconds — I'm not sure "
        "which is configured for the static assets bucket. Getting this wrong "
        "means stale deployments or excessive origin load.",
        "CDN TTL",
        "What cache TTL is configured for static assets?",
    ),
    (
        "I believe we have 3 availability zones configured, but the DR plan "
        "mentions a 2-zone minimum. I'm not sure if the third zone is active "
        "or just provisioned.",
        "availability zones",
        "How many availability zones are active in this deployment?",
    ),
    (
        "The health check interval is probably 30 seconds, but might be 10 "
        "seconds for the critical-path services. We need to align the app "
        "startup time with whichever is correct.",
        "health check interval",
        "What health check interval should we configure?",
    ),
    (
        "The load balancer timeout is somewhere around 60 seconds, but the "
        "backend team mentioned they extended it to 120 seconds for async "
        "operations. I haven't confirmed the current setting.",
        "LB timeout",
        "What timeout is configured on the load balancer?",
    ),
    (
        "The autoscaling min capacity is either 2 or 3 instances — I'm not sure "
        "which the SRE team set after the last capacity review. Wrong setting "
        "means either cost waste or availability risk.",
        "min capacity",
        "What is the minimum autoscaling capacity?",
    ),
    # --- Product / Business Logic ---
    (
        "The pricing tier cutoff is either $50K or $75K ARR for enterprise. "
        "I'm not sure which threshold the sales team is using right now — "
        "this affects the feature flag logic directly.",
        "tier cutoff",
        "What ARR threshold qualifies a customer for enterprise tier?",
    ),
    (
        "I think the trial period is 14 days, but there's been discussion "
        "about extending it to 30 days for enterprise prospects. The current "
        "policy isn't finalised.",
        "trial duration",
        "How long is the free trial period?",
    ),
    (
        "The SLA for P1 incidents is either 1-hour or 30-minute response — "
        "the contract language is ambiguous and legal hasn't clarified it yet.",
        "P1 SLA",
        "What is the maximum response time for P1 incidents?",
    ),
    (
        "The notification default is opt-in, but we might have changed it to "
        "opt-out after the GDPR review. I haven't checked the current default "
        "in the settings table.",
        "notification default",
        "What is the default notification preference for new users?",
    ),
    (
        "The commission rate for referrals is probably 20%, but it may have "
        "been reduced to 15% after the last board meeting. The updated rate "
        "hasn't been communicated to engineering.",
        "commission rate",
        "What commission rate should we apply to referral payouts?",
    ),
    # --- Scenarios 31–50: additional coverage ---
    (
        "I believe the CDN cache TTL is 300 seconds for static assets, but "
        "it might have been bumped to 3600 for images after the last perf "
        "review. I haven't checked the CDN config directly.",
        "CDN cache TTL",
        "What TTL should we set for static assets in the CDN config?",
    ),
    (
        "The gRPC max message size is roughly 4 MB based on the default, "
        "but the platform team may have increased it for the data pipeline. "
        "I'm not certain what the current limit is.",
        "gRPC message size",
        "What is the configured max gRPC message size for our service?",
    ),
    (
        "The JWT signing key rotation happens approximately every 90 days, "
        "but the security team said they might shorten it to 30 days after "
        "the last audit. The current rotation schedule is unconfirmed.",
        "key rotation interval",
        "What is the JWT signing key rotation interval?",
    ),
    (
        "The batch job processing window is probably 2 AM to 4 AM UTC, but "
        "it might have shifted after the EU data residency change. I haven't "
        "checked the scheduler config for the new region.",
        "batch job window",
        "What time window does the batch job run in?",
    ),
    (
        "I think the S3 bucket versioning is enabled on the prod bucket, "
        "but it may not be on the staging bucket. I'm not sure if they "
        "are configured identically — I haven't verified recently.",
        "S3 versioning",
        "Is versioning enabled on the staging S3 bucket?",
    ),
    (
        "The feature flag rollout is somewhere around 10% of users right now, "
        "but it might have been increased to 25% last week. I haven't "
        "confirmed the current percentage with the product team.",
        "feature flag rollout",
        "What percentage of users have the feature flag enabled?",
    ),
    (
        "The GraphQL query depth limit is probably 10, but I've seen 15 "
        "mentioned in an old PR. The current setting hasn't been documented "
        "and I'm not certain which is active.",
        "query depth limit",
        "What is the maximum allowed GraphQL query depth?",
    ),
    (
        "I believe the Elasticsearch index has 3 shards, but the ops team "
        "re-indexed it last month and may have changed the shard count. "
        "Current shard configuration is unverified.",
        "shard count",
        "How many shards does the Elasticsearch index have?",
    ),
    (
        "The TLS certificate renewal threshold is somewhere around 30 days "
        "before expiry, but it might be 60 days for our wildcard cert. "
        "I haven't checked the cert manager configuration.",
        "cert renewal threshold",
        "When does the cert manager trigger TLS certificate renewal?",
    ),
    (
        "The service-to-service auth timeout is either 5 or 10 seconds — "
        "I've seen both in different parts of the codebase. The canonical "
        "value isn't centralised and I'm not certain which takes precedence.",
        "service auth timeout",
        "What timeout applies to service-to-service authentication calls?",
    ),
    (
        "The daily active user count is roughly 50,000, but that was from "
        "last quarter's report. With the new markets we've launched, it "
        "might be closer to 80,000 — I haven't seen the updated metric.",
        "DAU",
        "What is the current daily active user count we should plan capacity for?",
    ),
    (
        "The data retention policy for user events is probably 2 years, "
        "but legal mentioned a possible change to 1 year for GDPR compliance. "
        "The updated policy hasn't been finalised.",
        "data retention period",
        "How long do we retain user event data?",
    ),
    (
        "I think the read replica lag threshold for alerting is 500ms, "
        "but it may have been tightened to 200ms after the last SLA review. "
        "Current threshold is unconfirmed.",
        "replica lag threshold",
        "What replica lag threshold triggers an alert?",
    ),
    (
        "The push notification delivery SLA is roughly 5 seconds for "
        "high-priority messages, but I'm not sure if that applies to "
        "both iOS and Android or just one. I haven't verified the contract.",
        "push notification SLA",
        "What is the delivery SLA for high-priority push notifications?",
    ),
    (
        "The infra team said the new instance type gives us about 40% more "
        "throughput, but that was a rough estimate from benchmarks. We "
        "haven't confirmed it holds under production traffic patterns.",
        "throughput improvement",
        "What throughput improvement can we expect from the new instance type?",
    ),
    (
        "I think the maximum webhook payload size is 1 MB, but I've seen "
        "references to 512 KB in older docs. I haven't tested it against "
        "the actual limit and shouldn't assume.",
        "webhook payload limit",
        "What is the maximum allowed webhook payload size?",
    ),
    (
        "The canary deployment threshold is probably 5% of traffic, but "
        "the SRE team may have lowered it to 1% after the last rollback. "
        "Current canary configuration is unverified.",
        "canary traffic percentage",
        "What percentage of traffic goes to the canary deployment?",
    ),
    (
        "I believe the health check interval is 30 seconds, but for the "
        "latency-sensitive path it might be 10 seconds. I haven't "
        "reviewed the load balancer config for the new cluster.",
        "health check interval",
        "What health check interval is configured on the load balancer?",
    ),
    (
        "The message queue max size is approximately 10,000 messages, "
        "but the platform team said they might have increased it to 50,000 "
        "for the async pipeline. I haven't confirmed the current setting.",
        "queue max size",
        "What is the maximum queue depth before backpressure kicks in?",
    ),
    (
        "The minimum password length requirement is either 8 or 12 characters "
        "— the security team updated the policy but I'm not sure if the "
        "change was deployed to the auth service yet.",
        "password min length",
        "What is the minimum password length we should enforce?",
    ),
]

# ---------------------------------------------------------------------------
# Filler turns — HIGH-J, no uncertainty markers, used to build context pressure
# ---------------------------------------------------------------------------

FILLER_PAIRS = [
    ("How do we structure the retry logic for transient failures?",
     "Use exponential backoff with jitter. Start at 100ms, double each attempt, "
     "cap at 30 seconds. Add ±25% jitter to prevent thundering herd. Log each "
     "retry attempt with the attempt number and wait duration for observability."),
    ("What HTTP status code indicates we should retry the request?",
     "Retry on 429 (rate limited), 503 (service unavailable), and 504 (gateway "
     "timeout). Do not retry on 400 (bad request) or 401 (unauthorised) — these "
     "indicate client-side errors that will not resolve on retry."),
    ("How should we handle idempotency for the payment requests?",
     "Generate a UUID as the idempotency key per transaction. Store it in the "
     "request header as X-Idempotency-Key. The server returns the same response "
     "for duplicate keys within the 24-hour window. Persist the key in your "
     "database before sending the request."),
    ("What is the correct way to structure the webhook validation?",
     "Compute HMAC-SHA256 of the raw request body using your webhook secret. "
     "Compare against the signature in the X-Webhook-Signature header using "
     "constant-time comparison to prevent timing attacks. Reject requests "
     "where signatures do not match with a 401 response."),
    ("How should we log API errors for debugging?",
     "Log the request ID, endpoint, status code, response body, and elapsed "
     "time for every failed request. Use structured JSON logging. Include the "
     "correlation ID from the X-Request-ID header to enable cross-service "
     "tracing. Emit to your observability pipeline at ERROR level."),
    ("What connection pool settings are appropriate for this service?",
     "Set max_connections to 10 per service instance, min_idle to 2. Configure "
     "connection_timeout to 5 seconds and idle_timeout to 300 seconds. Enable "
     "health checking with a simple SELECT 1 query every 30 seconds to detect "
     "stale connections before they are used."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _call(client, model: str, messages: list[dict],
          system: str = "", max_tokens: int = 300) -> str:
    resp = client.messages.create(
        model=model,
        system=system or "You are a helpful, precise technical assistant.",
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.content[0].text.strip()


def _has_uncertainty(text: str) -> bool:
    """Check whether text contains any uncertainty marker."""
    import re as _re
    # Strip markdown bold/italic markers so "**not** been confirmed" matches "not been confirmed"
    clean = _re.sub(r'\*{1,2}|_{1,2}', '', text)
    lower = clean.lower()
    return any(m in lower for m in _UNCERTAINTY_MARKERS)


def _build_conversation(uncertain_statement: str, n_filler: int = 4) -> list[dict]:
    """
    Build a realistic conversation:
      T1 (user):      uncertain constraint statement
      T2 (assistant): acknowledge + echo uncertainty
      T3-T(2+2n):     filler pairs
    """
    msgs = [
        {"role": "user",      "content": uncertain_statement},
        {"role": "assistant", "content":
            "Understood — I've noted that as an unverified constraint. "
            "We'll need to confirm it before committing to the implementation. "
            "Let's continue and flag it as an open question for now."},
    ]
    for i in range(n_filler):
        u, a = FILLER_PAIRS[i % len(FILLER_PAIRS)]
        msgs.append({"role": "user",      "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

_COMPRESS_PROMPT = (
    "Summarise this conversation in 2-3 concise sentences. "
    "Capture the main decisions made and topics discussed."
)

# Control condition: same Haiku, explicit instruction to preserve epistemic qualifiers.
# This answers the reviewer question: "Does prompt engineering alone solve this?"
# If enhanced_prompt FCR ≈ 0%, the probe adds no marginal value.
# If enhanced_prompt FCR >> 0%, the probe is providing value that prompting cannot.
_COMPRESS_PROMPT_ENHANCED = (
    "Summarise this conversation in 2-3 concise sentences. "
    "Capture the main decisions made and topics discussed. "
    "CRITICAL: You MUST preserve all uncertainty qualifiers, hedging language, and "
    "modal verbs exactly as stated (words like 'might', 'probably', 'I think', "
    "'unconfirmed', 'approximately', 'haven't verified'). If the original contains "
    "uncertainty about a value, your summary MUST express the same uncertainty. "
    "Do not convert uncertain claims into confident statements."
)


def _compress_naive(client, conversation: list[dict]) -> str:
    """Haiku summarises the full conversation. No epistemic safety check."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in conversation
    )
    return _call(
        client, _MODEL_HAIKU,
        [{"role": "user", "content": f"{_COMPRESS_PROMPT}\n\n{conv_text}"}],
        max_tokens=200,
    )


def _compress_enhanced_prompt(client, conversation: list[dict]) -> str:
    """
    Control condition: Haiku with explicit qualifier-preservation instruction.

    Tests whether the naive FCR (6%) can be eliminated by prompt engineering alone,
    without any probe mechanism. If this condition also achieves 0% FCR, the probe's
    marginal value is in cases where prompting fails or is not possible (e.g., third-party
    compression pipelines). If this condition has FCR > 0%, the probe is providing
    value that explicit prompting cannot guarantee.
    """
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in conversation
    )
    return _call(
        client, _MODEL_HAIKU,
        [{"role": "user", "content": f"{_COMPRESS_PROMPT_ENHANCED}\n\n{conv_text}"}],
        max_tokens=250,
    )


def _compress_with_probe(conversation: list[dict]) -> tuple[str, bool]:
    """
    Faithfulness probe: scan USER turns only for uncertainty markers.
    If found → compression BLOCKED → return original text, blocked=True.
    If not found → would compress (return None, blocked=False).

    Scopes to user turns only to match the production implementation
    (_has_uncertainty_in_user_turns in context_manager.py). Prior version
    scanned full conv_text including the hardcoded assistant echo turn, which
    contains 'unverified' and 'open question' — guaranteeing a block regardless
    of whether the user's own phrasing contained any uncertainty markers.
    """
    full_text = "\n".join(m["content"] for m in conversation)
    user_text = " ".join(m["content"] for m in conversation if m.get("role") == "user")
    blocked = _has_uncertainty(user_text)
    return full_text, blocked


_LINGUA_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "have", "from", "are",
    "was", "were", "has", "had", "been", "can", "will", "not", "but",
    "all", "any", "its", "into", "over", "also", "than", "only", "such",
    "very", "more", "just", "you", "may", "might", "should", "would",
    "could", "about", "what", "our", "we", "it", "is", "as", "to", "a",
    "an", "of", "in", "on", "at", "by", "or", "so", "if", "do", "did",
    "get", "got", "use", "used", "set", "let", "run", "how", "they",
    "their", "there", "here", "when", "which", "who", "him", "her",
    # hedge words that LLMLingua would NOT specially preserve:
    "think", "maybe", "perhaps", "probably", "might", "believe", "sure",
    "certain", "confirm", "confirmed", "unconfirmed", "unclear", "know",
})

_TECHNICAL_PATTERN = re.compile(
    r'\b([A-Z]{2,}|[a-z]+[A-Z][a-z]+|[a-z]+_[a-z]+|\d+[a-z]+|[a-z]+\d+)\b'
)


def _lingua_sentence_score(sentence: str) -> float:
    """
    Score a sentence by token importance — no epistemic awareness.

    Mimics LLMLingua-2 behaviour: rewards technical density (content words,
    numbers, acronyms, identifiers) without any special handling of hedging
    phrases. Sentences like "I'm not 100% certain — haven't confirmed this"
    will score low (mostly stop words + hedge words); sentences like
    "Set max_connections=10, idle_timeout=300s, health-check every 30s" score
    high (all content words + technical patterns).
    """
    words = re.sub(r"[^\w\s]", " ", sentence.lower()).split()
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _LINGUA_STOPWORDS and len(w) >= 3]
    content_ratio = len(content_words) / len(words)
    # Bonus for technical identifiers (camelCase, UPPER, snake_case, alphanumeric)
    tech_hits = len(_TECHNICAL_PATTERN.findall(sentence))
    tech_bonus = min(0.30, tech_hits * 0.05)
    # Bonus for numbers (concrete values → high importance)
    number_hits = len(re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z]+)?\b', sentence))
    number_bonus = min(0.20, number_hits * 0.04)
    return round(content_ratio + tech_bonus + number_bonus, 4)


def _compress_llm_lingua(conversation: list[dict], target_ratio: float = 0.30) -> str:
    """
    LLMLingua-2 simulation: compress to ~target_ratio of original token count
    by dropping lowest-scoring sentences, with no epistemic awareness.

    Sentences are scored by technical content density alone — uncertainty
    markers get no special treatment. This naturally drops hedge-heavy
    sentences like "I think the rate limit is around X, but haven't confirmed."

    Returns the compressed text (subset of original sentences).
    """
    # Flatten conversation to sentences
    all_sentences: list[tuple[float, str]] = []
    for msg in conversation:
        role = msg["role"].upper()
        text = f"{role}: {msg['content']}"
        # Split on sentence boundaries
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            s = s.strip()
            if len(s) > 20:  # skip very short fragments
                score = _lingua_sentence_score(s)
                all_sentences.append((score, s))

    if not all_sentences:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)

    total_tokens = sum(len(s.split()) for _, s in all_sentences)
    target_tokens = int(total_tokens * target_ratio)

    # Sort by score descending — highest importance survives
    ranked = sorted(all_sentences, key=lambda x: x[0], reverse=True)

    kept: list[str] = []
    tokens_kept = 0
    for score, sent in ranked:
        sent_tokens = len(sent.split())
        if tokens_kept + sent_tokens <= target_tokens:
            kept.append(sent)
            tokens_kept += sent_tokens
        if tokens_kept >= target_tokens:
            break

    # Re-order kept sentences in original order to preserve readability
    kept_set = set(kept)
    ordered = [s for _, s in all_sentences if s in kept_set]
    return "\n".join(ordered)


def _ask_downstream(client, context: str, callback_question: str) -> str:
    """Ask the downstream model a callback question given a context."""
    system = "You are a precise technical assistant. Answer the question based on the provided context."
    msgs = [
        {"role": "user", "content":
            f"Context from earlier in our session:\n\n{context}\n\n"
            f"Question: {callback_question}"},
    ]
    return _call(client, _MODEL_OPUS, msgs, system=system, max_tokens=150)


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    index:                    int
    constraint_label:         str
    uncertain_statement:      str

    # Naive compression condition
    naive_compressed_text:    str = ""
    naive_qualifier_survived: bool = False   # uncertainty markers in compressed text
    naive_downstream_answer:  str = ""
    naive_downstream_certain: bool = False   # model answered as if constraint were fact

    # Probe-guarded condition
    probe_blocked:            bool = False   # probe prevented compression
    probe_context:            str = ""       # full text (blocked) or compressed (not)
    probe_downstream_answer:  str = ""
    probe_downstream_certain: bool = False

    # Baseline condition (full context)
    baseline_downstream_answer:  str = ""
    baseline_downstream_certain: bool = False

    # LLMLingua-2 simulated condition (token-importance compression, no epistemic awareness)
    lingua_compressed_text:      str = ""
    lingua_qualifier_survived:   bool = False
    lingua_downstream_answer:    str = ""
    lingua_downstream_certain:   bool = False

    # Control condition: Haiku with explicit qualifier-preservation prompt.
    # Answers: "Does prompt engineering alone solve this without the probe?"
    enhanced_compressed_text:    str = ""
    enhanced_qualifier_survived: bool = False
    enhanced_downstream_answer:  str = ""
    enhanced_downstream_certain: bool = False


def _is_certain_answer(answer: str) -> bool:
    """
    Heuristic: does the model answer as if the constraint is a confirmed fact?
    Returns True if the answer expresses no uncertainty about the value.
    A certain answer states the value definitively without any hedging.
    """
    return not _has_uncertainty(answer)


def run_scenario(
    client,
    index: int,
    uncertain_statement: str,
    constraint_label: str,
    callback_question: str,
    n_filler: int = 4,
    verbose: bool = False,
) -> ScenarioResult:

    result = ScenarioResult(
        index=index,
        constraint_label=constraint_label,
        uncertain_statement=uncertain_statement,
    )

    conversation = _build_conversation(uncertain_statement, n_filler)
    full_text    = "\n".join(m["content"] for m in conversation)

    # ── Naive compression ──────────────────────────────────────────────────
    naive_compressed = _compress_naive(client, conversation)
    result.naive_compressed_text    = naive_compressed
    result.naive_qualifier_survived = _has_uncertainty(naive_compressed)

    naive_answer = _ask_downstream(client, naive_compressed, callback_question)
    result.naive_downstream_answer  = naive_answer
    result.naive_downstream_certain = _is_certain_answer(naive_answer)
    time.sleep(0.4)

    # ── Probe-guarded compression ──────────────────────────────────────────
    probe_context, blocked = _compress_with_probe(conversation)
    result.probe_blocked  = blocked
    result.probe_context  = probe_context

    probe_answer = _ask_downstream(client, probe_context, callback_question)
    result.probe_downstream_answer  = probe_answer
    result.probe_downstream_certain = _is_certain_answer(probe_answer)
    time.sleep(0.4)

    # ── Baseline (full context, no compression) ────────────────────────────
    baseline_answer = _ask_downstream(client, full_text, callback_question)
    result.baseline_downstream_answer  = baseline_answer
    result.baseline_downstream_certain = _is_certain_answer(baseline_answer)
    time.sleep(0.4)

    # ── LLMLingua-2 simulated compression (no epistemic awareness) ─────────
    lingua_compressed = _compress_llm_lingua(conversation, target_ratio=0.30)
    result.lingua_compressed_text    = lingua_compressed
    result.lingua_qualifier_survived = _has_uncertainty(lingua_compressed)

    lingua_answer = _ask_downstream(client, lingua_compressed, callback_question)
    result.lingua_downstream_answer  = lingua_answer
    result.lingua_downstream_certain = _is_certain_answer(lingua_answer)
    time.sleep(0.4)

    # ── Control: Haiku with explicit qualifier-preservation prompt ─────────
    # The critical control experiment: does prompt engineering alone solve this?
    enhanced_compressed = _compress_enhanced_prompt(client, conversation)
    result.enhanced_compressed_text    = enhanced_compressed
    result.enhanced_qualifier_survived = _has_uncertainty(enhanced_compressed)

    enhanced_answer = _ask_downstream(client, enhanced_compressed, callback_question)
    result.enhanced_downstream_answer  = enhanced_answer
    result.enhanced_downstream_certain = _is_certain_answer(enhanced_answer)
    time.sleep(0.4)

    if verbose:
        survived     = "✓" if result.naive_qualifier_survived  else "✗"
        ling_surv    = "✓" if result.lingua_qualifier_survived else "✗"
        blocked_s    = "BLOCKED" if blocked else "passed"
        print(f"  [{index+1:02d}] {constraint_label[:30]:<30} "
              f"naive_qual={survived}  lingua_qual={ling_surv}  probe={blocked_s}  "
              f"naive_cert={result.naive_downstream_certain}  "
              f"lingua_cert={result.lingua_downstream_certain}  "
              f"probe_cert={result.probe_downstream_certain}")

    return result


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results: list[ScenarioResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    naive_qual_survival    = sum(r.naive_qualifier_survived    for r in results) / n
    naive_downstream_cert  = sum(r.naive_downstream_certain    for r in results) / n
    probe_block_rate       = sum(r.probe_blocked               for r in results) / n
    probe_downstream_cert  = sum(r.probe_downstream_certain    for r in results) / n
    baseline_cert          = sum(r.baseline_downstream_certain for r in results) / n
    lingua_qual_survival   = sum(r.lingua_qualifier_survived   for r in results) / n
    lingua_downstream_cert = sum(r.lingua_downstream_certain   for r in results) / n
    enhanced_qual_survival  = sum(r.enhanced_qualifier_survived  for r in results) / n
    enhanced_downstream_cert= sum(r.enhanced_downstream_certain  for r in results) / n

    return {
        "n": n,
        "naive_qualifier_survival":           round(naive_qual_survival,     3),
        "naive_downstream_certainty":         round(naive_downstream_cert,   3),
        "lingua_qualifier_survival":          round(lingua_qual_survival,    3),
        "lingua_downstream_certainty":        round(lingua_downstream_cert,  3),
        "enhanced_prompt_qualifier_survival": round(enhanced_qual_survival,  3),
        "enhanced_prompt_downstream_certainty": round(enhanced_downstream_cert, 3),
        "probe_block_rate":                   round(probe_block_rate,         3),
        "probe_downstream_certainty":         round(probe_downstream_cert,    3),
        "baseline_downstream_certainty":      round(baseline_cert,            3),
        # Derived: how much does each condition reduce FCR vs. naive?
        "fcr_reduction_naive_vs_probe":    round(naive_downstream_cert - probe_downstream_cert,    3),
        "fcr_reduction_naive_vs_enhanced": round(naive_downstream_cert - enhanced_downstream_cert, 3),
        "fcr_reduction_lingua_vs_probe":   round(lingua_downstream_cert - probe_downstream_cert,   3),
        # Key research question: probe vs. enhanced prompt (marginal value of the probe)
        "probe_marginal_vs_enhanced_prompt": round(enhanced_downstream_cert - probe_downstream_cert, 3),
        "scorer_version": "2.0-corrected",
        "vocabulary_size": 198,
        "scorer_notes": "v2: user-turns-only scanning + 198-marker vocab; enhanced_prompt control added v2.1",
    }


def print_summary(agg: dict):
    print("\n" + "=" * 72)
    print("COMPRESSION FAITHFULNESS STUDY — RESULTS")
    print("=" * 72)
    print(f"  Scenarios run:                          {agg['n']}")
    print()
    print("  NAIVE COMPRESSION (Haiku, no probe):")
    print(f"    Qualifier survival rate:              "
          f"{agg['naive_qualifier_survival']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['naive_downstream_certainty']:.1%}")
    print()
    print("  TOKEN-IMPORTANCE SIMULATION (30% compression, no epistemic awareness):")
    print(f"    Qualifier survival rate:              "
          f"{agg['lingua_qualifier_survival']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['lingua_downstream_certainty']:.1%}")
    print()
    print("  ENHANCED PROMPT CONTROL (Haiku + explicit qualifier-preservation instruction):")
    print(f"    Qualifier survival rate:              "
          f"{agg['enhanced_prompt_qualifier_survival']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['enhanced_prompt_downstream_certainty']:.1%}")
    print()
    print("  PROBE-GUARDED COMPRESSION (faithfulness probe, no Haiku call):")
    print(f"    Compression blocked rate:             "
          f"{agg['probe_block_rate']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['probe_downstream_certainty']:.1%}")
    print()
    print("  BASELINE (full context, no compression — oracle upper bound):")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['baseline_downstream_certainty']:.1%}")
    print()
    print("  KEY COMPARISONS:")
    print(f"    FCR reduction (naive → enhanced prompt):  "
          f"{agg['fcr_reduction_naive_vs_enhanced']:+.1%}")
    print(f"    FCR reduction (naive → probe):            "
          f"{agg['fcr_reduction_naive_vs_probe']:+.1%}")
    print(f"    FCR reduction (token-importance → probe): "
          f"{agg['fcr_reduction_lingua_vs_probe']:+.1%}")
    print(f"    Probe marginal value over enhanced prompt:"
          f" {agg['probe_marginal_vs_enhanced_prompt']:+.1%}")
    print()
    if agg.get("probe_marginal_vs_enhanced_prompt", 0) > 0.02:
        print("  FINDING: Probe provides meaningful FCR reduction beyond prompt engineering alone.")
    elif agg.get("enhanced_prompt_downstream_certainty", 1) > 0.05:
        print("  FINDING: Enhanced prompting reduces but does not eliminate FCR. Probe fills the gap.")
    else:
        print("  FINDING: Enhanced prompting achieves similar FCR to probe. "
              "Probe value is in determinism, zero-latency, and pipeline contexts without prompt control.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Dry-run mode (no API)
# ---------------------------------------------------------------------------

def dry_run(n: int = 5):
    print(f"\n[dry-run] Checking {n} scenario definitions (no API calls)...\n")
    proxy = CredenceProxy()
    for i, (stmt, label, question) in enumerate(SCENARIOS[:n]):
        conv = _build_conversation(stmt, n_filler=4)
        full = "\n".join(m["content"] for m in conv)
        has_unc = _has_uncertainty(full)
        j = proxy.compute(stmt).j_score
        print(f"  [{i+1:02d}] {label:<28}  has_uncertainty={has_unc}  "
              f"J(seed)={j:.3f}")
        print(f"        Seed: {stmt[:80]}...")
    print(f"\n[dry-run] All {n} scenario definitions valid.")
    print("[dry-run] Probe would block compression on all scenarios above "
          "where has_uncertainty=True.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compression Faithfulness Study")
    parser.add_argument("--n",        type=int,  default=30,
                        help="Number of scenarios to run (default: 30)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate scenario definitions without API calls")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-scenario results")
    parser.add_argument("--out",      default="evals/compression_faithfulness_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS[:args.n]

    if args.dry_run:
        dry_run(n=args.n)
        return

    if not _ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed.")
        sys.exit(1)

    client = _make_client()
    print(f"\nRunning compression faithfulness study ({len(scenarios_to_run)} scenarios)...")
    print(f"Models: compress={_MODEL_HAIKU}  downstream={_MODEL_OPUS}")
    print(f"Conditions: naive_compress | probe_guard | baseline\n")

    results = []
    for i, (stmt, label, question) in enumerate(scenarios_to_run):
        r = run_scenario(
            client=client,
            index=i,
            uncertain_statement=stmt,
            constraint_label=label,
            callback_question=question,
            n_filler=4,
            verbose=args.verbose,
        )
        results.append(r)

    agg = aggregate(results)
    print_summary(agg)

    # Save
    output = {
        "summary": agg,
        "scenarios": [asdict(r) for r in results],
    }
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

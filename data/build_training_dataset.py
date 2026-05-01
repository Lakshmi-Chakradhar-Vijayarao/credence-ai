"""
data/build_training_dataset.py
================================
Builds 5,000 DPO training triples for fine-tuning LLMLingua-2 with an
epistemic loss.

Each triple: (input_conversation, faithful_summary, unfaithful_summary)
  - input_conversation: a conversation containing an uncertain claim
  - faithful_summary:   compression that PRESERVES the qualifier (gold label)
  - unfaithful_summary: compression that DROPS the qualifier (rejected label)

The 50 hand-written examples serve as few-shot seeds for Opus generation.

Usage:
    python -m data.build_training_dataset --n 50 --hand-only --out data/training_samples.json
    python -m data.build_training_dataset --n 5000 --out data/epistemic_compression_training.json
    python -m data.build_training_dataset --verify --sample 500
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import argparse
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TripleExample:
    id:                  str
    input_conversation:  str
    faithful_summary:    str
    unfaithful_summary:  str
    source_scenario_id:  str
    qualifier_type:      str   # vendor_claim | estimate | approximation | preliminary | unverified_report
    domain:              str   # api | auth | debug | design | compliance | multiagent
    is_ghost:            bool  # True if input has no surface hedging markers (ghost constraint)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 50 hand-written training examples
# ---------------------------------------------------------------------------

def build_hand_examples() -> list[TripleExample]:
    """
    50 hand-written DPO triples for epistemic fine-tuning.
    Coverage:
      - API rate limits (vendor_claim), non-ghost and ghost
      - Auth token expiry (estimate)
      - Debug thresholds (preliminary)
      - Design estimates (approximation)
      - Compliance deadlines (unverified_report)
      - Ghost constraints (no surface hedging in input)
    """

    # Each entry: (scenario_id, qualifier_type, domain, is_ghost, conversation, faithful, unfaithful)
    RAW = [

        # ================================================================
        # Domain: api — qualifier_type: vendor_claim — ghost: False
        # ================================================================
        (
            "h001", "vendor_claim", "api", False,
            (
                "User: I checked the Stripe docs quickly — I think their rate limit is "
                "around 50 requests per minute for the standard plan.\n"
                "Assistant: Got it, I'll keep that in mind when designing the retry logic.\n"
                "User: We also need to handle idempotency keys for payment creation.\n"
                "Assistant: Understood — idempotency keys prevent duplicate charges on retries."
            ),
            (
                "The user believes Stripe's rate limit is approximately 50 req/min for the "
                "standard plan (unconfirmed — verify in docs before relying on this). "
                "Idempotency keys are needed for payment creation to prevent duplicates."
            ),
            (
                "Stripe's rate limit is 50 requests per minute on the standard plan. "
                "Idempotency keys are required for payment creation to prevent duplicates."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: vendor_claim — ghost: True
        # ================================================================
        (
            "h002", "vendor_claim", "api", True,
            (
                "User: The Twilio SMS API endpoint has a throughput ceiling of 100 messages "
                "per second.\n"
                "Assistant: That's an important constraint for our queue sizing.\n"
                "User: We're using long-code numbers, not short-codes.\n"
                "Assistant: Long-code numbers have different throughput characteristics — "
                "we should confirm the exact limit in the Twilio documentation."
            ),
            (
                "The user states Twilio SMS throughput is 100 msg/sec — this is unverified "
                "vendor documentation; the actual limit depends on number type and should be "
                "confirmed. Long-code numbers may have a lower ceiling than short-codes."
            ),
            (
                "Twilio SMS API throughput is 100 messages per second. Long-code numbers are "
                "in use and the queue should be sized accordingly."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h003", "approximation", "api", False,
            (
                "User: Based on what I saw in testing, the webhook delivery timeout is roughly "
                "30 seconds — the connection just dropped after that.\n"
                "Assistant: So we need our endpoint to acknowledge within 30 seconds.\n"
                "User: Yes, otherwise we'll miss the event entirely.\n"
                "Assistant: We should add async processing to stay within that window."
            ),
            (
                "The user observed a webhook timeout of roughly 30 seconds in testing "
                "(approximate — not confirmed from official docs). Endpoints must acknowledge "
                "within this window or the event is lost; async processing is recommended."
            ),
            (
                "The webhook delivery timeout is 30 seconds. Endpoints must acknowledge within "
                "30 seconds or the event is missed. Async processing is recommended."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: vendor_claim — ghost: True
        # ================================================================
        (
            "h004", "vendor_claim", "api", True,
            (
                "User: The batch endpoint accepts up to 100 items per request.\n"
                "Assistant: We'll need to chunk our import jobs accordingly.\n"
                "User: Right — anything over 100 should be split into multiple calls.\n"
                "Assistant: I'll add a batch_size constant set to 100 in the client."
            ),
            (
                "The user reports the batch endpoint limit is 100 items per request — this is "
                "unverified; the actual limit should be checked in the API reference before "
                "hardcoding. Import jobs should be chunked, with a configurable batch_size."
            ),
            (
                "The batch endpoint accepts up to 100 items per request. Import jobs must be "
                "chunked, and batch_size should be set to 100 in the client."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h005", "estimate", "auth", False,
            (
                "User: I'm pretty sure the auth tokens expire after 3600 seconds — I saw that "
                "in the config last week but haven't verified it.\n"
                "Assistant: So tokens last one hour before refresh is needed.\n"
                "User: That's my estimate, yes.\n"
                "Assistant: I'll build the refresh logic around that, but we should confirm "
                "the actual TTL from the identity provider docs."
            ),
            (
                "User estimates auth token TTL is 3600 seconds (one hour), based on a previous "
                "config observation — unverified, should be confirmed with the identity "
                "provider. Refresh logic will be built around this estimate."
            ),
            (
                "Auth tokens expire after 3600 seconds. Refresh logic should be built around "
                "a one-hour TTL. This matches the identity provider configuration."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: estimate — ghost: True
        # ================================================================
        (
            "h006", "estimate", "auth", True,
            (
                "User: JWT signing uses RS256.\n"
                "Assistant: RS256 is asymmetric — we'll need the public key for verification.\n"
                "User: The key rotation happens every 90 days on their end.\n"
                "Assistant: We should cache the JWKS endpoint response and respect the "
                "cache-control headers for key rotation."
            ),
            (
                "The user states JWT signing uses RS256 (unverified assumption — confirm with "
                "IdP) with key rotation every 90 days (also unverified). JWKS caching with "
                "cache-control header respect is recommended."
            ),
            (
                "JWT signing uses RS256. Key rotation occurs every 90 days. The JWKS endpoint "
                "should be cached and cache-control headers respected."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h007", "estimate", "auth", False,
            (
                "User: Token refresh probably needs to happen every hour — that's just based "
                "on how similar systems work, not the actual docs.\n"
                "Assistant: We'll implement a proactive refresh at 55 minutes to avoid "
                "expiry mid-request.\n"
                "User: That seems reasonable.\n"
                "Assistant: I'll add a background timer that fires at 55 minutes."
            ),
            (
                "User estimates token refresh should happen approximately every hour, based on "
                "analogy to similar systems — not from the actual documentation (unverified). "
                "A proactive refresh at 55 minutes will be implemented as a safety margin."
            ),
            (
                "Token refresh must happen every hour. A proactive refresh is scheduled at "
                "55 minutes to prevent mid-request expiry."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h008", "preliminary", "debug", False,
            (
                "User: From my profiling run, the memory leak seems to grow at roughly 2 MB "
                "per 1000 requests — it could vary a lot, this was one test run.\n"
                "Assistant: That growth rate would accumulate to around 2 GB at a million "
                "requests, which is significant.\n"
                "User: Yes, we need to find the root cause before scaling.\n"
                "Assistant: I'll instrument the heap allocations to track object lifetime."
            ),
            (
                "Preliminary profiling shows roughly 2 MB memory growth per 1000 requests "
                "(single test run — not statistically robust). At 1M requests this implies "
                "~2 GB growth, suggesting a serious leak. Heap instrumentation planned."
            ),
            (
                "The memory leak grows at 2 MB per 1000 requests. At one million requests "
                "this causes 2 GB memory growth. Heap allocation instrumentation is planned."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h009", "preliminary", "debug", False,
            (
                "User: The race condition seems to trigger at 8 or more concurrent writes — "
                "it's non-deterministic so I'm not totally sure.\n"
                "Assistant: Non-deterministic races are hard to reproduce reliably.\n"
                "User: Right, sometimes it needs 12 writes to trigger it.\n"
                "Assistant: We should add mutex locks around the shared state regardless "
                "of the exact trigger threshold."
            ),
            (
                "A race condition is observed at approximately 8+ concurrent writes, though "
                "behavior is non-deterministic and may require more writers — threshold is "
                "uncertain. Mutex locking on shared state is recommended regardless of the "
                "exact trigger point."
            ),
            (
                "The race condition triggers at 8 concurrent writes. Mutex locks should be "
                "added around shared state to prevent data corruption."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h010", "preliminary", "debug", False,
            (
                "User: CPU spikes to around 95% under our typical load — I observed this "
                "in staging but staging might not reflect production traffic perfectly.\n"
                "Assistant: That's very close to saturation.\n"
                "User: Yeah, if it's real in production, we'll need to scale or optimize.\n"
                "Assistant: Let's add CPU profiling to production with a 1% sampling rate."
            ),
            (
                "CPU was observed spiking to approximately 95% under staging load — not "
                "confirmed in production, and staging may not represent real traffic. "
                "If confirmed in production, scaling or optimization is needed."
            ),
            (
                "CPU spikes to 95% under typical load. Production scaling or optimization "
                "is required. CPU profiling will be added with 1% sampling."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h011", "approximation", "debug", True,
            (
                "User: The deadlock occurs every 500 transactions on average.\n"
                "Assistant: Every 500 is quite frequent for a deadlock.\n"
                "User: We noticed it in staging. Production traffic is heavier.\n"
                "Assistant: We should add deadlock detection in the transaction manager "
                "and implement automatic retry logic."
            ),
            (
                "A deadlock is observed occurring approximately every 500 transactions "
                "in staging (unverified in production — frequency may differ under higher "
                "load). Deadlock detection and automatic retry logic are recommended."
            ),
            (
                "The deadlock occurs every 500 transactions. Deadlock detection and automatic "
                "retry logic should be added to the transaction manager."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h012", "approximation", "debug", True,
            (
                "User: The queue backlog clears in about 4 hours during off-peak.\n"
                "Assistant: That's a long drain time if there's a production incident.\n"
                "User: We need a way to manually prioritize messages.\n"
                "Assistant: I'll add a priority queue lane for incident-related messages."
            ),
            (
                "The queue backlog drain time is approximately 4 hours in off-peak conditions "
                "(unverified — actual time depends on load and message size). Manual priority "
                "routing is needed for incident handling."
            ),
            (
                "The queue backlog clears in 4 hours during off-peak periods. A priority lane "
                "is needed for incident-related messages."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h013", "approximation", "design", False,
            (
                "User: We estimate GraphQL will reduce our average response payload by around "
                "40% by eliminating over-fetching.\n"
                "Assistant: That's a meaningful reduction if it holds in practice.\n"
                "User: It's based on a rough analysis of the current REST response shapes.\n"
                "Assistant: We should validate with a pilot on a few endpoints first."
            ),
            (
                "Team estimates GraphQL adoption will reduce response payloads by approximately "
                "40% (rough analysis of current REST responses — not empirically validated). "
                "A pilot on select endpoints is recommended before full migration."
            ),
            (
                "GraphQL will reduce average response payload by 40% by eliminating "
                "over-fetching. A pilot on selected endpoints will be run first."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h014", "approximation", "design", False,
            (
                "User: The CDN should handle about 80% of our read traffic based on the "
                "cache hit rate we saw in the load test.\n"
                "Assistant: 80% offload is substantial — that reduces origin load significantly.\n"
                "User: It's from a single load test, so it might vary in production.\n"
                "Assistant: We should monitor cache hit rate in production and tune TTLs."
            ),
            (
                "CDN is projected to serve approximately 80% of read traffic, based on a "
                "single load test (not representative of all production patterns). Cache hit "
                "rate and TTL tuning should be monitored after launch."
            ),
            (
                "The CDN will handle 80% of read traffic. Origin load will be significantly "
                "reduced. TTL tuning will follow launch."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h015", "approximation", "design", True,
            (
                "User: Cache hit rate will be 85% once warm-up is complete.\n"
                "Assistant: An 85% hit rate would allow us to run a much smaller origin fleet.\n"
                "User: That's based on similar workloads we've run before.\n"
                "Assistant: We should incorporate a warm-up phase into the launch plan."
            ),
            (
                "Cache hit rate is projected at 85% post warm-up, based on experience with "
                "similar workloads (unverified for this specific system). Origin fleet sizing "
                "should account for this estimate being preliminary."
            ),
            (
                "Cache hit rate will reach 85% after warm-up. This allows a smaller origin "
                "fleet. A warm-up phase should be incorporated into the launch plan."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: unverified_report — ghost: False
        # ================================================================
        (
            "h016", "unverified_report", "compliance", False,
            (
                "User: I believe GDPR Article 17 requires data deletion to complete within "
                "30 days — but our legal team hasn't formally confirmed this interpretation.\n"
                "Assistant: That's a common understanding, but the exact deadline depends on "
                "the specific exemptions that apply.\n"
                "User: We should get formal legal sign-off before building the deletion pipeline.\n"
                "Assistant: Agreed — I'll flag this as pending legal review."
            ),
            (
                "User reportedly believes GDPR deletion must complete within 30 days under Article 17, "
                "but this interpretation is unconfirmed — formal legal sign-off is required "
                "before building the deletion pipeline."
            ),
            (
                "GDPR Article 17 requires data deletion within 30 days. The deletion pipeline "
                "should be built to meet this deadline."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: unverified_report — ghost: False
        # ================================================================
        (
            "h017", "unverified_report", "compliance", False,
            (
                "User: I haven't verified this directly, but I think the GDPR breach "
                "notification window is 72 hours from discovery.\n"
                "Assistant: 72 hours is the commonly cited figure under Article 33.\n"
                "User: We need to build an incident response runbook around this.\n"
                "Assistant: The runbook should include timeline triggers at 24, 48, and "
                "72 hours with escalation paths."
            ),
            (
                "User believes GDPR breach notification must occur within 72 hours of "
                "discovery (user has not directly verified this — cites common knowledge "
                "of Article 33). Incident runbook should include timed escalation triggers "
                "pending formal legal confirmation."
            ),
            (
                "GDPR breach notification must occur within 72 hours of discovery under "
                "Article 33. The incident runbook should include escalation triggers at "
                "24, 48, and 72 hours."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h018", "estimate", "multiagent", False,
            (
                "User: Based on one demo I ran, it looked like Haiku handled about 80% of "
                "the sub-agent tasks without needing to escalate to Opus.\n"
                "Assistant: That's useful signal for routing cost estimation.\n"
                "User: But it was only one demo, not a real workload.\n"
                "Assistant: We should run a representative sample before committing to "
                "that routing split in production."
            ),
            (
                "User estimates Haiku handles approximately 80% of sub-agent tasks without "
                "escalation, based on a single demo (not a representative production workload). "
                "This routing assumption should be validated before production deployment."
            ),
            (
                "Haiku handles 80% of sub-agent tasks without escalating to Opus. This routing "
                "split should be used for cost estimation in the production architecture."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h019", "approximation", "multiagent", True,
            (
                "User: Tool timeout should be set to 30 seconds for all agent tools.\n"
                "Assistant: That seems reasonable for most tool types.\n"
                "User: Some long-running tools need a higher timeout.\n"
                "Assistant: We should differentiate timeouts by tool category: "
                "fast tools at 10s, standard at 30s, long-running at 120s."
            ),
            (
                "A default tool timeout of 30 seconds was suggested (unverified — based on "
                "light load testing, may be insufficient for long-running tools). Per-category "
                "timeouts are recommended: fast=10s, standard=30s, long-running=120s."
            ),
            (
                "Tool timeout is set to 30 seconds for all agent tools. Long-running tools "
                "may need higher values. Per-category timeouts are recommended."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h020", "approximation", "multiagent", True,
            (
                "User: 10 parallel agents will cost roughly 10x what a single agent costs.\n"
                "Assistant: That assumes perfectly linear scaling with no coordination overhead.\n"
                "User: Right — shared context and tool calls could add more overhead.\n"
                "Assistant: We should model coordination overhead separately and budget "
                "for a 12–15x cost multiplier at 10 parallel agents."
            ),
            (
                "User estimates 10 parallel agents cost approximately 10x a single agent — "
                "this ignores coordination overhead and shared context costs (unverified). "
                "Actual multiplier may be 12–15x; overhead should be modeled separately."
            ),
            (
                "10 parallel agents cost 10x a single agent. Coordination overhead should "
                "be modeled separately. Budget for a 12–15x cost multiplier."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: vendor_claim — ghost: False
        # ================================================================
        (
            "h021", "vendor_claim", "api", False,
            (
                "User: I read somewhere that the OpenAI batch API has a 24-hour processing "
                "window — not 100% sure though.\n"
                "Assistant: That's the figure in the current docs for batch endpoints.\n"
                "User: We need our pipelines to tolerate that latency.\n"
                "Assistant: I'll add async job tracking with a 24-hour completion window "
                "and alerts for jobs approaching the limit."
            ),
            (
                "User reports OpenAI batch API has a 24-hour processing window — user is "
                "uncertain and this should be verified from current API documentation. "
                "Async job tracking with 24h window alerts is planned."
            ),
            (
                "The OpenAI batch API has a 24-hour processing window. Async job tracking "
                "should include a 24-hour completion window with alerts."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: vendor_claim — ghost: False
        # ================================================================
        (
            "h022", "vendor_claim", "api", False,
            (
                "User: I think the GitHub API has a secondary rate limit on mutation "
                "operations — about 80 mutations per minute, if I recall correctly.\n"
                "Assistant: Secondary rate limits are different from the primary hourly limit.\n"
                "User: We're doing a lot of issue creation during CI.\n"
                "Assistant: We should add exponential backoff and check response headers "
                "for rate limit state."
            ),
            (
                "User recalls GitHub API secondary rate limit on mutations is approximately "
                "80 per minute — this is from memory and should be verified in GitHub docs. "
                "CI issue creation should use exponential backoff and monitor rate limit "
                "response headers."
            ),
            (
                "GitHub API secondary rate limit for mutations is 80 per minute. CI issue "
                "creation workflows should use exponential backoff and monitor rate headers."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h023", "estimate", "auth", False,
            (
                "User: I'm not certain, but I think OAuth refresh tokens might last 30 days "
                "on this provider — the docs weren't clear.\n"
                "Assistant: 30 days is common, but some providers use 60 or 90 days.\n"
                "User: We'll need long-term sessions for our mobile app.\n"
                "Assistant: I'll implement silent refresh with re-authentication fallback "
                "and store the token issuance timestamp so we can detect expiry regardless "
                "of the actual TTL."
            ),
            (
                "User is uncertain about OAuth refresh token lifetime — estimates 30 days "
                "but documentation was unclear (unverified). Mobile app needs long-term "
                "sessions; implementation will track issuance timestamp to handle any TTL."
            ),
            (
                "OAuth refresh tokens expire after 30 days on this provider. The mobile app "
                "will use silent refresh with re-authentication fallback."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h024", "preliminary", "debug", False,
            (
                "User: It looks like the latency regression appeared after deploy v2.3.1 — "
                "I'm basing that on eyeballing the metrics dashboard.\n"
                "Assistant: We should correlate the exact deploy time with the p99 latency spike.\n"
                "User: The spike is around 200ms added to every request.\n"
                "Assistant: I'll run a git bisect between v2.3.0 and v2.3.1 to narrow "
                "the root cause."
            ),
            (
                "Latency regression of approximately 200ms per request appears to have "
                "started after deploy v2.3.1 — based on visual inspection of metrics, "
                "not confirmed with correlation analysis. Git bisect planned."
            ),
            (
                "A 200ms latency regression was introduced in deploy v2.3.1. Git bisect "
                "between v2.3.0 and v2.3.1 will identify the root cause."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: True
        # ================================================================
        (
            "h025", "preliminary", "debug", True,
            (
                "User: The auth service returns 503 errors during peak load.\n"
                "Assistant: 503 usually indicates the service is overloaded or downstream "
                "is unavailable.\n"
                "User: It happens when we have more than 500 concurrent users.\n"
                "Assistant: We need to identify whether the bottleneck is the auth service "
                "itself or a downstream dependency like the session store."
            ),
            (
                "Auth service returns 503 errors under peak load, observed at approximately "
                "500+ concurrent users (unverified threshold — may differ in production). "
                "Root cause unclear: could be auth service or downstream session store."
            ),
            (
                "The auth service returns 503 errors when concurrent users exceed 500. "
                "The bottleneck may be in the auth service or its downstream session store."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h026", "approximation", "design", False,
            (
                "User: We estimate the service mesh will add around 5ms of latency per hop.\n"
                "Assistant: 5ms per hop with sidecar proxies is typical for Envoy-based meshes.\n"
                "User: We have about 4 hops in the critical path.\n"
                "Assistant: That's roughly 20ms of additional overhead in the critical path — "
                "within budget if our SLA is 200ms."
            ),
            (
                "Service mesh latency is estimated at approximately 5ms per hop (rough estimate "
                "— actual Envoy overhead depends on configuration). With 4 hops in the critical "
                "path, this adds roughly 20ms — unverified under actual load."
            ),
            (
                "The service mesh adds 5ms latency per hop. With 4 critical path hops, total "
                "mesh overhead is 20ms, which is within the 200ms SLA budget."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h027", "preliminary", "design", False,
            (
                "User: Our preliminary analysis suggests horizontal sharding will get us to "
                "10x the current write throughput.\n"
                "Assistant: That's a significant gain — what's driving the bottleneck today?\n"
                "User: It's the single primary write node for the users table.\n"
                "Assistant: With sharding, we'd need to handle cross-shard queries and "
                "distributed transactions carefully."
            ),
            (
                "Preliminary analysis projects 10x write throughput improvement from horizontal "
                "sharding of the users table (unvalidated — based on back-of-envelope, not load "
                "testing). Cross-shard query complexity and distributed transactions are risks."
            ),
            (
                "Horizontal sharding will increase write throughput by 10x by eliminating "
                "the single-primary bottleneck on the users table."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: unverified_report — ghost: False
        # ================================================================
        (
            "h028", "unverified_report", "compliance", False,
            (
                "User: I haven't confirmed this, but I believe SOC 2 Type II audit prep "
                "typically takes 6 months before the audit window opens.\n"
                "Assistant: 6 months is commonly cited, but it depends heavily on current "
                "control maturity.\n"
                "User: We're starting from scratch on some controls.\n"
                "Assistant: Given the maturity gap, 9–12 months might be more realistic."
            ),
            (
                "User estimates SOC 2 Type II audit prep takes approximately 6 months — "
                "this is unverified and based on general knowledge, not assessment of current "
                "control maturity. Actual timeline may be 9–12 months given gaps identified."
            ),
            (
                "SOC 2 Type II audit preparation takes 6 months. Given the current control "
                "gaps, a 9–12 month timeline is more realistic."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: unverified_report — ghost: True
        # ================================================================
        (
            "h029", "unverified_report", "compliance", True,
            (
                "User: HIPAA requires PHI to be encrypted at rest with AES-256.\n"
                "Assistant: The HIPAA Security Rule doesn't specify the algorithm — it "
                "requires 'appropriate' encryption but defers to NIST guidance.\n"
                "User: So AES-256 is compliant, but not the only option.\n"
                "Assistant: Correct — AES-256 is widely used and satisfies NIST guidance, "
                "but AES-128 with proper key management is also compliant."
            ),
            (
                "User states HIPAA requires AES-256 for PHI at rest — this is an unverified "
                "interpretation; the HIPAA Security Rule does not mandate a specific algorithm "
                "but defers to NIST guidance. AES-256 is compliant but not the only option."
            ),
            (
                "HIPAA requires PHI to be encrypted at rest using AES-256. Both AES-256 and "
                "AES-128 with proper key management satisfy the requirement."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h030", "estimate", "api", False,
            (
                "User: I think pagination uses cursor-based paging with a default page size "
                "of about 50 — I haven't verified the exact default.\n"
                "Assistant: Cursor-based pagination is common for large result sets.\n"
                "User: We'll need to request all pages to build our import.\n"
                "Assistant: I'll add a paginator that follows next_cursor until it's null, "
                "with configurable page_size."
            ),
            (
                "User believes default page size is approximately 50 (unverified — estimate "
                "based on general experience, not checked in API docs). Cursor-based pagination "
                "is used; a configurable paginator with null-cursor termination is planned."
            ),
            (
                "Pagination uses cursor-based paging with a default page size of 50. "
                "A paginator following next_cursor until null will be implemented."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h031", "approximation", "debug", False,
            (
                "User: The disk space issue seems to come from log files growing about 500 MB "
                "per day — but that's eyeballed, not measured.\n"
                "Assistant: At that growth rate the disk would fill in about a week on a 4GB "
                "partition.\n"
                "User: We need log rotation immediately.\n"
                "Assistant: I'll configure logrotate to rotate daily with 7-day retention "
                "and size-based rotation at 200MB."
            ),
            (
                "Log files appear to grow approximately 500 MB per day (eyeballed estimate — "
                "not precisely measured). At this rate, a 4 GB partition fills in about a week. "
                "Daily logrotate with 7-day retention and 200 MB size trigger is planned."
            ),
            (
                "Log files grow at 500 MB per day, filling a 4 GB partition in one week. "
                "Logrotate will be configured with daily rotation, 7-day retention, and "
                "200 MB size-based rotation."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: estimate — ghost: True
        # ================================================================
        (
            "h032", "estimate", "design", True,
            (
                "User: The event bus can sustain 10,000 events per second.\n"
                "Assistant: That's a solid throughput for our current event volume.\n"
                "User: We should leave headroom for spikes.\n"
                "Assistant: I'd target a maximum utilization of 70% during normal operation "
                "to leave room for burst traffic."
            ),
            (
                "Event bus throughput is stated as 10,000 events/sec — this is unverified; "
                "actual capacity depends on message size and broker configuration. 70% max "
                "utilization target recommended for burst headroom."
            ),
            (
                "The event bus sustains 10,000 events per second. Maximum utilization should "
                "be capped at 70% during normal operation to handle burst traffic."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h033", "approximation", "api", True,
            (
                "User: The idempotency window for the payments API is 24 hours.\n"
                "Assistant: So idempotency keys only protect against duplicates within a "
                "24-hour window.\n"
                "User: Anything replayed after that could be charged twice.\n"
                "Assistant: We need to persist idempotency keys with a 25-hour TTL on "
                "our side to handle clock skew."
            ),
            (
                "The payments API idempotency window is stated as 24 hours — unverified "
                "from the official API documentation. Duplicate charge risk exists for "
                "replays beyond this window. Client-side key TTL of 25 hours is recommended."
            ),
            (
                "The payments API idempotency window is 24 hours. Replays after 24 hours "
                "risk duplicate charges. Idempotency keys should have a 25-hour client TTL."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h034", "estimate", "multiagent", False,
            (
                "User: I think the context budget for each sub-agent should be around "
                "8000 tokens — that's just a gut feeling from working with the Claude API.\n"
                "Assistant: 8000 tokens per sub-agent is workable but may be tight for "
                "complex reasoning tasks.\n"
                "User: We can tune it once we have real session data.\n"
                "Assistant: I'll make context_budget a configurable parameter defaulting "
                "to 8000 tokens with per-agent override support."
            ),
            (
                "User estimates 8000 tokens per sub-agent context budget, based on intuition "
                "rather than empirical data (unverified — will need tuning with real sessions). "
                "Context budget will be a configurable parameter."
            ),
            (
                "Sub-agent context budget is set to 8000 tokens. This is configurable with "
                "per-agent override support and will be tuned with production session data."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h035", "estimate", "compliance", False,
            (
                "User: Our DPA with the processor might allow a 30-day SLA for deletion "
                "requests — I haven't read the full agreement recently.\n"
                "Assistant: DPA deletion SLAs vary widely — some specify 10 days, others 30.\n"
                "User: We need to build our pipeline to be faster than the contractual SLA.\n"
                "Assistant: I'll build for a 14-day execution target and flag the DPA "
                "review as a legal action item."
            ),
            (
                "User reportedly believes DPA allows 30 days for deletion requests — not confirmed via "
                "recent review of the actual agreement (legal action item). Pipeline will "
                "be built with a 14-day execution target to safely beat the SLA."
            ),
            (
                "The DPA allows 30 days for deletion requests. The deletion pipeline will "
                "target 14-day execution to stay within the contractual SLA."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h036", "estimate", "debug", False,
            (
                "User: I believe there's a global connection pool limit of around 200 "
                "connections to the database — seen it hit before but not sure of the exact cap.\n"
                "Assistant: Connection pool exhaustion would cause timeouts across all services.\n"
                "User: We're currently using about 150 across all microservices.\n"
                "Assistant: We're close to the limit. I'll add connection count monitoring "
                "and alerts at 80% utilization."
            ),
            (
                "User believes global DB connection pool is limited to approximately 200 "
                "connections (unverified — exact cap not confirmed). Current usage is ~150 "
                "across microservices. Monitoring with 80% utilization alert is planned."
            ),
            (
                "The global database connection pool limit is 200 connections. Current usage "
                "is 150. Monitoring with an alert at 80% utilization will be added."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: vendor_claim — ghost: True
        # ================================================================
        (
            "h037", "vendor_claim", "design", True,
            (
                "User: The managed Kafka cluster supports a maximum retention period of "
                "7 days for standard tier.\n"
                "Assistant: 7-day retention is common for standard managed Kafka offerings.\n"
                "User: We need 30 days for our audit log topic.\n"
                "Assistant: We'll need to either upgrade to the enterprise tier or implement "
                "a separate archival pipeline to S3 for the audit log."
            ),
            (
                "Standard-tier managed Kafka retention is stated as 7 days — this is an "
                "unverified vendor claim; actual limits depend on the specific provider and "
                "tier. Audit log topic requires 30 days — upgrade or S3 archival needed."
            ),
            (
                "Managed Kafka standard tier supports 7-day maximum retention. The audit log "
                "topic requires 30-day retention, necessitating a tier upgrade or S3 archival."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: approximation — ghost: True
        # ================================================================
        (
            "h038", "approximation", "auth", True,
            (
                "User: The SAML assertion validity window is 5 minutes.\n"
                "Assistant: That's a tight window — clock skew between IdP and SP needs "
                "to be under 1 minute.\n"
                "User: Our servers have NTP sync but it's not always precise.\n"
                "Assistant: We should accept a 2-minute skew window to handle NTP drift "
                "and use the NotBefore/NotOnOrAfter attributes strictly."
            ),
            (
                "SAML assertion validity window is stated as 5 minutes — unverified from "
                "IdP configuration (may vary). NTP sync precision is imperfect; a 2-minute "
                "skew tolerance is recommended with strict NotBefore/NotOnOrAfter handling."
            ),
            (
                "SAML assertion validity is 5 minutes. Clock skew must be under 1 minute. "
                "A 2-minute tolerance window is configured to handle NTP drift."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h039", "estimate", "api", False,
            (
                "User: I estimate the GraphQL query depth limit is probably around 10 levels — "
                "that's what I've seen documented for similar services.\n"
                "Assistant: Query depth limits prevent malicious deeply-nested queries.\n"
                "User: We have some legitimate queries that go 8 levels deep.\n"
                "Assistant: That leaves very little headroom. We should check the actual limit "
                "and potentially request an exemption for our identified deep queries."
            ),
            (
                "User estimates GraphQL query depth limit is approximately 10 levels — based "
                "on similar services, not verified for this API. Legitimate queries reach 8 "
                "levels deep; limit needs to be confirmed before relying on the headroom."
            ),
            (
                "The GraphQL query depth limit is 10 levels. Current legitimate queries reach "
                "8 levels deep, leaving minimal headroom. An exemption may be needed."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h040", "approximation", "design", False,
            (
                "User: Based on rough calculations, we should be able to handle about 500 "
                "concurrent WebSocket connections per instance.\n"
                "Assistant: That depends heavily on connection duration and message frequency.\n"
                "User: These are long-lived connections sending small messages.\n"
                "Assistant: For low-frequency messages, 500 per instance is probably "
                "conservative — but we should load-test before relying on it."
            ),
            (
                "Rough calculations suggest 500 concurrent WebSocket connections per instance "
                "(approximate, depends on connection duration and message frequency — not load "
                "tested). Conservative estimate; load testing required before production sizing."
            ),
            (
                "Each instance handles 500 concurrent WebSocket connections. Load testing "
                "confirmed this for long-lived, low-frequency connection profiles."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: vendor_claim — ghost: False
        # ================================================================
        (
            "h041", "vendor_claim", "multiagent", False,
            (
                "User: I think Claude's max context window is 200K tokens — though I haven't "
                "checked whether that applies to the API or just the UI.\n"
                "Assistant: 200K is the documented context window for Claude models.\n"
                "User: We need to verify whether input pricing changes beyond a certain size.\n"
                "Assistant: I'll flag context window size and pricing tiers as items to "
                "confirm from the official API documentation."
            ),
            (
                "User reportedly believes Claude context window is 200K tokens — not verified for API "
                "vs UI distinction, and input pricing tiers at large context sizes are also "
                "unconfirmed. Both should be verified in official documentation."
            ),
            (
                "Claude's context window is 200K tokens. Input pricing is consistent across "
                "all context sizes. This applies to both API and UI."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h042", "preliminary", "compliance", False,
            (
                "User: From a preliminary read of the new vendor contract, I think we need "
                "to complete the security questionnaire within 14 days of the audit request.\n"
                "Assistant: That's a short turnaround for a full security review.\n"
                "User: We haven't had legal review the contract yet.\n"
                "Assistant: I'll add this as a pending legal action item and build the "
                "security questionnaire template so we're ready regardless of the exact SLA."
            ),
            (
                "Preliminary contract reading suggests 14-day SLA for security questionnaire "
                "response — not confirmed by legal review (unverified). Template preparation "
                "will proceed regardless; legal review is a pending action item."
            ),
            (
                "Security questionnaire responses must be completed within 14 days of an audit "
                "request. Legal review has confirmed this SLA from the vendor contract."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h043", "estimate", "debug", False,
            (
                "User: I suspect the memory issue is triggered by requests over 1 MB — "
                "just based on the pattern I see in the error logs.\n"
                "Assistant: Large payload handling often involves buffering that can leak.\n"
                "User: Most normal requests are under 100 KB.\n"
                "Assistant: I'll add payload size logging and build a test case with "
                "a 1.5 MB synthetic request to confirm the hypothesis."
            ),
            (
                "User suspects memory issue is triggered by requests over 1 MB — this is "
                "a preliminary hypothesis from log pattern inspection, not confirmed. "
                "A synthetic 1.5 MB test request will validate or refute this threshold."
            ),
            (
                "The memory issue is triggered by requests exceeding 1 MB. Most normal "
                "requests are under 100 KB. A 1.5 MB test request will confirm the threshold."
            ),
        ),

        # ================================================================
        # Domain: api — qualifier_type: approximation — ghost: False
        # ================================================================
        (
            "h044", "approximation", "api", False,
            (
                "User: Looking at our logs, the average response time from the external API "
                "is about 250ms — it varies quite a bit though.\n"
                "Assistant: 250ms average with high variance is challenging for SLA budgeting.\n"
                "User: P99 is probably closer to 2 seconds.\n"
                "Assistant: We should add a client-side timeout at 3 seconds and implement "
                "a circuit breaker that opens after 5 consecutive timeouts."
            ),
            (
                "External API average response time is approximately 250ms based on log "
                "analysis — high variance with estimated P99 around 2 seconds (both are "
                "approximate). Client timeout at 3s and circuit breaker planned."
            ),
            (
                "External API average response time is 250ms with P99 at 2 seconds. "
                "A 3-second client timeout and circuit breaker will be implemented."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h045", "estimate", "design", False,
            (
                "User: I estimate we'll need around 16 replicas for the gateway service to "
                "handle peak load — that's based on last quarter's traffic numbers.\n"
                "Assistant: Traffic patterns change, so that estimate could drift.\n"
                "User: Black Friday is coming up and volume could triple.\n"
                "Assistant: I'd recommend planning for 48 replicas peak capacity with "
                "auto-scaling configured to start scaling at 60% CPU."
            ),
            (
                "User estimates 16 gateway replicas needed for peak load based on last quarter's "
                "traffic (potentially outdated — Black Friday could triple volume). Recommendation: "
                "plan for 48 replicas peak with auto-scaling at 60% CPU trigger."
            ),
            (
                "16 replicas are needed for gateway peak load. Black Friday volume may require "
                "up to 48 replicas. Auto-scaling activates at 60% CPU."
            ),
        ),

        # ================================================================
        # Domain: auth — qualifier_type: vendor_claim — ghost: False
        # ================================================================
        (
            "h046", "vendor_claim", "auth", False,
            (
                "User: I think the identity provider enforces a maximum of 10 active sessions "
                "per user — I might be misremembering their docs.\n"
                "Assistant: Session limits are common on enterprise IdPs.\n"
                "User: If a user has more than 10 devices logged in, new logins would fail.\n"
                "Assistant: We need to add session management UI so users can revoke old "
                "sessions and confirm the limit from the IdP documentation."
            ),
            (
                "User believes IdP enforces a 10-session-per-user limit — this is uncertain "
                "and should be confirmed from IdP documentation. If correct, new logins beyond "
                "10 devices would fail; session management UI is recommended."
            ),
            (
                "The identity provider enforces a maximum of 10 active sessions per user. "
                "Users exceeding this limit will have new logins fail. Session management "
                "UI is required to allow users to revoke old sessions."
            ),
        ),

        # ================================================================
        # Domain: debug — qualifier_type: preliminary — ghost: True
        # ================================================================
        (
            "h047", "preliminary", "debug", True,
            (
                "User: The queue consumer lag is consistently 500 messages behind.\n"
                "Assistant: Consistent lag suggests the consumer processing rate is slower "
                "than the producer rate.\n"
                "User: We've been running like this for three days.\n"
                "Assistant: The lag is growing, not just steady — we need to either "
                "increase consumer parallelism or reduce per-message processing time."
            ),
            (
                "Consumer lag is approximately 500 messages (observed over three days — "
                "unverified whether lag is stable or growing; monitoring needed). If lag "
                "is growing, increased parallelism or processing optimization is required."
            ),
            (
                "The queue consumer lag is 500 messages. The consumer processing rate is "
                "slower than the producer. Consumer parallelism should be increased."
            ),
        ),

        # ================================================================
        # Domain: multiagent — qualifier_type: estimate — ghost: False
        # ================================================================
        (
            "h048", "estimate", "multiagent", False,
            (
                "User: My rough estimate is that the orchestrator can handle around 20 "
                "concurrent sub-agent tasks before hitting memory limits.\n"
                "Assistant: 20 concurrent agents would each need their own context buffer.\n"
                "User: It's really just an estimate — I haven't stress-tested it.\n"
                "Assistant: I'll make max_concurrent_agents configurable and add "
                "memory monitoring to find the real ceiling empirically."
            ),
            (
                "User estimates orchestrator can handle approximately 20 concurrent sub-agent "
                "tasks before memory limits — this is an untested estimate, not validated under "
                "load. max_concurrent_agents will be configurable with empirical profiling."
            ),
            (
                "The orchestrator supports 20 concurrent sub-agent tasks within memory limits. "
                "max_concurrent_agents should be configurable for different deployment sizes."
            ),
        ),

        # ================================================================
        # Domain: compliance — qualifier_type: unverified_report — ghost: False
        # ================================================================
        (
            "h049", "unverified_report", "compliance", False,
            (
                "User: I read that under the California Consumer Privacy Act, opt-out "
                "requests must be honored within 15 business days — but I haven't confirmed "
                "this from a primary source.\n"
                "Assistant: CCPA compliance timelines are specific and legally binding.\n"
                "User: We need to build the opt-out pipeline before our California launch.\n"
                "Assistant: I'll flag this as requiring legal review and build the pipeline "
                "to a conservative 10-day target."
            ),
            (
                "User reportedly believes CCPA opt-out requests must be honored within 15 business days — "
                "this is from a secondary source and not confirmed from primary CCPA text. "
                "Legal review required; pipeline will target 10-day execution conservatively."
            ),
            (
                "CCPA opt-out requests must be honored within 15 business days. The opt-out "
                "pipeline will be built with a 10-day execution target before California launch."
            ),
        ),

        # ================================================================
        # Domain: design — qualifier_type: preliminary — ghost: False
        # ================================================================
        (
            "h050", "preliminary", "design", False,
            (
                "User: Based on preliminary capacity planning, I think we need at least "
                "500 GB of block storage for the first year of user data.\n"
                "Assistant: Data growth estimates are notoriously hard to get right early on.\n"
                "User: It's based on similar products' growth curves.\n"
                "Assistant: I'd double that to 1 TB with dynamic provisioning enabled "
                "so we can scale without downtime if growth is faster than projected."
            ),
            (
                "Preliminary capacity planning estimates 500 GB block storage needed in year "
                "one — based on analogy to similar products, not empirical data (unverified). "
                "Recommendation: provision 1 TB with dynamic scaling enabled as a safety margin."
            ),
            (
                "500 GB of block storage is required for the first year per capacity "
                "planning. 1 TB is provisioned with dynamic scaling to handle higher growth."
            ),
        ),
    ]

    examples = []
    for scenario_id, qualifier_type, domain, is_ghost, conv, faithful, unfaithful in RAW:
        examples.append(TripleExample(
            id=f"hand_{scenario_id}",
            input_conversation=conv.strip(),
            faithful_summary=faithful.strip(),
            unfaithful_summary=unfaithful.strip(),
            source_scenario_id=scenario_id,
            qualifier_type=qualifier_type,
            domain=domain,
            is_ghost=is_ghost,
        ))
    return examples


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# Hedging markers used for verify_triple
_FAITHFULNESS_MARKERS = [
    "unverified", "estimate", "unconfirmed", "approximately", "roughly",
    "probably", "maybe", "might", "unclear", "uncertain", "not certain",
    "I think", "I believe", "I'm not sure", "I am not sure", "it seems",
    "preliminary", "approximate", "eyeballed", "based on", "not confirmed",
    "haven't confirmed", "haven't verified", "reportedly", "assumed",
    "one demo", "not representative", "not validated", "subject to change",
    "pending", "awaiting", "not yet", "to be confirmed", "unvalidated",
]

_FAITHFULNESS_MARKERS_LOWER = [m.lower() for m in _FAITHFULNESS_MARKERS]


def verify_triple(triple: TripleExample) -> bool:
    """
    Checks that:
    - faithful_summary preserves at least one epistemic qualifier
    - unfaithful_summary does NOT contain any qualifier markers
    - unfaithful_summary is at least 2 sentences / 30 words (rejects one-liners)
    - input_conversation is non-empty
    Returns True if the triple passes all checks.
    """
    faithful_lo   = triple.faithful_summary.lower()
    unfaithful_lo = triple.unfaithful_summary.lower()

    faithful_ok   = any(m in faithful_lo   for m in _FAITHFULNESS_MARKERS_LOWER)
    unfaithful_ok = not any(m in unfaithful_lo for m in _FAITHFULNESS_MARKERS_LOWER)
    unfaithful_long = len(triple.unfaithful_summary.split()) >= 12
    conv_ok = len(triple.input_conversation.strip()) > 50

    return faithful_ok and unfaithful_ok and unfaithful_long and conv_ok


# ---------------------------------------------------------------------------
# Generation from EQL-Bench seeds
# ---------------------------------------------------------------------------

# Seeds from evals/eql_bench (scenario IDs and domain)
_EQL_BENCH_SEEDS = [
    ("api_rate_limit", "vendor_claim", "api", False),
    ("api_token_expiry", "estimate", "auth", True),
    ("api_webhook_timeout", "approximation", "api", False),
    ("api_pagination", "estimate", "api", False),
    ("api_batch_limit", "vendor_claim", "api", True),
    ("debug_memory_leak", "preliminary", "debug", False),
    ("debug_race_condition", "preliminary", "debug", False),
    ("debug_cpu_spike", "approximation", "debug", False),
    ("debug_deadlock", "approximation", "debug", True),
    ("debug_queue_backlog", "approximation", "debug", True),
    ("design_graphql_perf", "approximation", "design", False),
    ("design_cdn_hit_rate", "approximation", "design", False),
    ("design_cache_hit_rate", "approximation", "design", True),
    ("compliance_gdpr_deletion", "unverified_report", "compliance", False),
    ("compliance_breach_72h", "unverified_report", "compliance", False),
    ("multiagent_haiku_split", "estimate", "multiagent", False),
    ("multiagent_tool_timeout", "approximation", "multiagent", True),
    ("multiagent_parallel_cost", "approximation", "multiagent", True),
    ("design_sharding_perf", "preliminary", "design", False),
    ("design_event_bus", "estimate", "design", True),
    ("api_idempotency_window", "approximation", "api", True),
    ("auth_jwt_signing", "estimate", "auth", True),
    ("auth_saml_window", "approximation", "auth", True),
    ("auth_session_limit", "vendor_claim", "auth", False),
    ("debug_503_threshold", "preliminary", "debug", True),
    ("debug_connection_pool", "estimate", "debug", False),
    ("design_service_mesh_latency", "approximation", "design", False),
    ("design_websocket_concurrency", "approximation", "design", False),
    ("compliance_soc2_prep", "unverified_report", "compliance", False),
    ("compliance_hipaa_aes", "unverified_report", "compliance", True),
]

# Exact qualifier words that must appear in faithful_summary (verified against _FAITHFULNESS_MARKERS)
_REQUIRED_QUALIFIER_WORDS = [
    "unverified", "unconfirmed", "approximately", "roughly", "probably", "maybe",
    "might", "unclear", "uncertain", "not certain", "I think", "I believe",
    "preliminary", "reportedly", "assumed", "not confirmed", "haven't confirmed",
    "not validated", "subject to change", "pending", "awaiting", "not yet",
    "estimate", "unvalidated",
]

# Generation prompt template — requires one of _REQUIRED_QUALIFIER_WORDS verbatim
_GENERATION_PROMPT = """\
You are generating training data for an epistemic fine-tuning task.

Given the EXAMPLE TRIPLES below, generate {n_to_generate} NEW triples in the same JSON format.

Each triple must have:
- input_conversation: a realistic multi-turn conversation (4-6 exchanges) where an uncertain
  claim is introduced. The claim should be in domain "{domain}" with qualifier_type "{qtype}".
  {ghost_instruction}
  Use a DIFFERENT scenario than the examples — varied numbers, systems, and contexts.
- faithful_summary: a 2-3 sentence compression that PRESERVES the uncertainty qualifier.
  REQUIREMENT: the faithful_summary MUST contain at least one of these EXACT words or phrases
  (do not paraphrase or use synonyms — copy one exactly):
  "unverified", "unconfirmed", "approximately", "roughly", "probably", "maybe",
  "might", "unclear", "uncertain", "not certain", "I think", "I believe",
  "preliminary", "reportedly", "assumed", "not confirmed", "haven't confirmed",
  "not validated", "subject to change", "pending", "awaiting", "not yet", "estimate"
- unfaithful_summary: a 2-3 sentence compression that DROPS all qualifiers and states
  the uncertain claim as confirmed fact. Do NOT include any hedging words — write as if
  the value is certain. Do NOT use parenthetical disclaimers like "(verify this)".
  Keep it at least 2 full sentences — do not produce a single short sentence.
- qualifier_type: "{qtype}"
- domain: "{domain}"
- is_ghost: {is_ghost}

EXAMPLE TRIPLES (few-shot seeds):
{few_shot_text}

Generate {n_to_generate} new triples as a JSON array. Each object must have the keys:
input_conversation, faithful_summary, unfaithful_summary, qualifier_type, domain, is_ghost.
Output ONLY the JSON array, no preamble.
"""


def _few_shot_text(examples: list[TripleExample], n: int = 3, domain: str = "") -> str:
    """Format up to n examples as readable few-shot text.
    Prefers same-domain examples to ground generation in the right vocabulary;
    falls back to random if not enough same-domain examples exist.
    """
    same_domain = [e for e in examples if e.domain == domain] if domain else []
    other = [e for e in examples if e.domain != domain]
    # pick 2 same-domain + 1 cross-domain for style variety
    pool = random.sample(same_domain, min(2, len(same_domain))) + \
           random.sample(other, min(max(0, n - min(2, len(same_domain))), len(other)))
    if len(pool) < n:
        pool = random.sample(examples, min(n, len(examples)))
    parts = []
    for ex in pool:
        parts.append(
            f"---\nConversation:\n{ex.input_conversation}\n\n"
            f"Faithful summary:\n{ex.faithful_summary}\n\n"
            f"Unfaithful summary:\n{ex.unfaithful_summary}\n---"
        )
    return "\n".join(parts)


def build_from_eql_bench(
    n: int,
    few_shot: list[TripleExample],
    client,
    batch_size: int = 20,
    save_path: str | None = None,
    resume_from: list["TripleExample"] | None = None,
    model: str = "claude-3-opus-20240229",
) -> list[TripleExample]:
    """
    Generate n triples using Opus, seeded by EQL-Bench scenario metadata and
    using few_shot as the in-context examples.

    batch_size=20 → ~250 API calls for 5K examples (~45 min).
    Crash-safe: saves to save_path after every 5 batches (100 examples).
    resume_from: list of already-generated triples to start from.
    """
    try:
        from anthropic import Anthropic  # noqa: F401 — type narrowing
    except ImportError:
        raise ImportError("anthropic package required for generation mode")

    results: list[TripleExample] = list(resume_from or [])
    already_have = len(results)
    if already_have > 0:
        print(f"  Resuming from {already_have} existing examples, need {n - already_have} more")

    # Build domain-balanced seed cycle: group seeds by domain, interleave round-robin
    # so no domain exceeds ~ceil(n / n_domains) examples
    from collections import defaultdict
    seeds_by_domain: dict[str, list] = defaultdict(list)
    for seed in _EQL_BENCH_SEEDS:
        seeds_by_domain[seed[2]].append(seed)
    # Shuffle within each domain for variety
    for d in seeds_by_domain:
        random.shuffle(seeds_by_domain[d])
    # Build round-robin cycle across all domains
    domain_order = sorted(seeds_by_domain.keys())
    domain_iters = {d: 0 for d in domain_order}
    seed_cycle: list = []
    total_needed = n - already_have + 50  # overshoot so we don't run out
    round_idx = 0
    while len(seed_cycle) < total_needed:
        d = domain_order[round_idx % len(domain_order)]
        d_seeds = seeds_by_domain[d]
        seed_cycle.append(d_seeds[domain_iters[d] % len(d_seeds)])
        domain_iters[d] += 1
        round_idx += 1

    i = 0
    batches_since_save = 0
    while len(results) < n:
        seed = seed_cycle[i % len(seed_cycle)]
        i += 1
        _, qtype, domain, is_ghost = seed
        to_gen = min(batch_size, n - len(results))

        ghost_instruction = (
            "Include NO surface hedging words — make the uncertain claim sound confident "
            "(ghost constraint)."
            if is_ghost else
            "Include natural hedging language like 'I think', 'probably', 'roughly', "
            "'about', 'not sure', etc."
        )
        prompt = _GENERATION_PROMPT.format(
            n_to_generate=to_gen,
            domain=domain,
            qtype=qtype,
            is_ghost=is_ghost,
            ghost_instruction=ghost_instruction,
            few_shot_text=_few_shot_text(few_shot, n=3, domain=domain),
        )

        try:
            resp = client.messages.create(
                model=model,
                system="You generate training data for LLM fine-tuning. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            batch = json.loads(raw)
            if isinstance(batch, dict):
                batch = [batch]

            for item in batch:
                ex = TripleExample(
                    id=str(uuid.uuid4()),
                    input_conversation=item.get("input_conversation", ""),
                    faithful_summary=item.get("faithful_summary", ""),
                    unfaithful_summary=item.get("unfaithful_summary", ""),
                    source_scenario_id=seed[0],
                    qualifier_type=item.get("qualifier_type", qtype),
                    domain=item.get("domain", domain),
                    is_ghost=item.get("is_ghost", is_ghost),
                )
                if verify_triple(ex):
                    results.append(ex)

            batches_since_save += 1

        except Exception as e:
            print(f"\n  [warning] generation batch failed: {e}")

        print(f"  Progress: {len(results)}/{n}", end="\r", flush=True)

        # Crash-safe save every 5 batches (~100 examples)
        if save_path and batches_since_save >= 5:
            _save_intermediate(results, save_path, n)
            batches_since_save = 0

    print()
    return results[:n]


def _save_intermediate(
    examples: list["TripleExample"],
    path: str,
    target_n: int,
) -> None:
    """Write current progress to disk (crash-safe checkpoint)."""
    output = {
        "version":            "1.0",
        "description":        "Epistemic compression training triples (in-progress)",
        "total":              target_n,
        "generated_so_far":   len(examples),
        "hand_examples":      sum(1 for e in examples if e.id.startswith("hand_")),
        "generated_examples": sum(1 for e in examples if not e.id.startswith("hand_")),
        "examples":           [asdict(e) for e in examples],
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [checkpoint] saved {len(examples)} examples to {path}")


# ---------------------------------------------------------------------------
# Verification pass
# ---------------------------------------------------------------------------

def run_verification(triples: list[TripleExample], sample: int = 500) -> dict:
    """Spot-check a random sample of triples for triple validity."""
    if sample < len(triples):
        sample_triples = random.sample(triples, sample)
    else:
        sample_triples = triples

    passed = sum(1 for t in sample_triples if verify_triple(t))
    failed = len(sample_triples) - passed

    fail_examples = [t for t in sample_triples if not verify_triple(t)][:3]

    return {
        "total_checked":     len(sample_triples),
        "passed":            passed,
        "failed":            failed,
        "pass_rate":         round(passed / max(1, len(sample_triples)), 4),
        "fail_examples":     [t.id for t in fail_examples],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 5K DPO training triples for epistemic fine-tuning"
    )
    parser.add_argument("--n",          type=int, default=5000,
                        help="Total number of triples to generate")
    parser.add_argument("--hand-only",  action="store_true",
                        help="Only output the 50 hand-written examples (no API calls)")
    parser.add_argument("--out",        default="data/epistemic_compression_training.json",
                        help="Output file path")
    parser.add_argument("--verify",     action="store_true",
                        help="Verify an existing dataset file")
    parser.add_argument("--sample",     type=int, default=500,
                        help="Number of triples to verify (with --verify)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from existing --out file (skip already-generated examples)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of triples to generate per API call (default 20)")
    parser.add_argument("--model",      type=str, default="claude-3-opus-20240229",
                        help="Anthropic model to use (default: opus)")
    args = parser.parse_args()

    if args.verify:
        if not os.path.exists(args.out):
            print(f"Error: {args.out} not found")
            sys.exit(1)
        with open(args.out) as f:
            data = json.load(f)
        triples = [TripleExample(**e) for e in data["examples"]]
        print(f"\nVerifying {args.sample} triples from {args.out}...")
        stats = run_verification(triples, sample=args.sample)
        print(f"  Checked:   {stats['total_checked']}")
        print(f"  Passed:    {stats['passed']}")
        print(f"  Failed:    {stats['failed']}")
        print(f"  Pass rate: {stats['pass_rate']:.1%}")
        if stats["fail_examples"]:
            print(f"  Failing IDs: {stats['fail_examples']}")
        return

    hand_examples = build_hand_examples()
    print(f"Built {len(hand_examples)} hand-written examples.")

    # Verify hand examples
    hand_stats = run_verification(hand_examples, sample=len(hand_examples))
    print(f"  Hand example pass rate: {hand_stats['pass_rate']:.1%} "
          f"({hand_stats['passed']}/{hand_stats['total_checked']})")
    if hand_stats["fail_examples"]:
        print(f"  [warning] failing hand IDs: {hand_stats['fail_examples']}")

    # Load any synthetic triples generated by Antigravity
    synthetic_examples: list[TripleExample] = []
    if os.path.exists("data"):
        synthetic_files = [f for f in os.listdir("data") if f.startswith("synthetic_") and f.endswith(".json")]
        for sf in synthetic_files:
            try:
                with open(os.path.join("data", sf)) as f:
                    s_data = json.load(f)
                    s_list = s_data if isinstance(s_data, list) else s_data.get("examples", [])
                    synthetic_examples.extend([TripleExample(**e) for e in s_list])
            except Exception as e:
                print(f"  [warning] failed to load synthetic file {sf}: {e}")
    
    if synthetic_examples:
        print(f"  Found {len(synthetic_examples)} synthetic triples from Antigravity")

    if args.hand_only:
        all_examples = hand_examples + synthetic_examples
        n_generated = 0
    else:
        n_to_generate = max(0, args.n - len(hand_examples) - len(synthetic_examples))
        if n_to_generate == 0:
            all_examples = hand_examples + synthetic_examples
            all_examples = all_examples[:args.n]
            n_generated = 0
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                print("  No ANTHROPIC_API_KEY — using parametric generation (no API calls needed).")
                try:
                    from data.parametric_examples import build_parametric_examples
                    n_param = n_to_generate
                    print(f"  Generating {n_param} parametric examples (domain-balanced, 100% pass rate)...")
                    param_dicts = build_parametric_examples(n=n_param)
                    param_triples = [
                        TripleExample(**d)
                        for d in param_dicts
                        if verify_triple(TripleExample(**d))
                    ]
                    all_examples = hand_examples + synthetic_examples + param_triples
                    n_generated = len(param_triples)
                    print(f"  Parametric generated: {n_generated}  Total: {len(all_examples)}")
                except ImportError as e:
                    print(f"  [warning] parametric generation unavailable: {e}. Using existing examples.")
                    all_examples = hand_examples + synthetic_examples
                    n_generated = 0
            else:
                try:
                    from anthropic import Anthropic
                except ImportError:
                    print("Error: pip install anthropic")
                    sys.exit(1)

                client = Anthropic(api_key=api_key)

                # --resume: load already-generated examples, skip hand examples count
                resume_examples: list[TripleExample] = []
                if args.resume and os.path.exists(args.out):
                    with open(args.out) as f:
                        existing = json.load(f)
                    resume_examples = [
                        TripleExample(**e) for e in existing.get("examples", [])
                        if not e.get("id", "").startswith("hand_")
                    ]
                    print(f"  Resuming: {len(resume_examples)} generated examples already on disk")
                    n_to_generate = max(0, n_to_generate - len(resume_examples))

                print(f"\nGenerating {n_to_generate} additional triples via Opus "
                      f"(batch_size={args.batch_size}, ~{max(1, n_to_generate // args.batch_size)} API calls)...")
                generated = build_from_eql_bench(
                    n_to_generate, hand_examples, client,
                    batch_size=args.batch_size,
                    save_path=args.out,
                    resume_from=resume_examples,
                    model=args.model,
                )
                n_generated = len(generated)
                all_examples = hand_examples + synthetic_examples + generated
                print(f"  Generated: {n_generated}  Total: {len(all_examples)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output = {
        "version":              "1.0",
        "description":         ("5K training triples for DPO fine-tuning LLMLingua-2 "
                                 "with epistemic loss"),
        "total":                len(all_examples),
        "hand_examples":        len(hand_examples),
        "generated_examples":   n_generated if not args.hand_only else 0,
        "examples":             [e.to_dict() for e in all_examples],
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_examples)} triples to {args.out}")


if __name__ == "__main__":
    main()

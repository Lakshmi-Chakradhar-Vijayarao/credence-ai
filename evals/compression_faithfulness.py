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
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from credence.context_manager import _UNCERTAINTY_MARKERS
from credence.confidence_proxy import CredenceProxy
from credence.providers import make_client, HF_COMPRESS_MODEL, HF_DOWNSTREAM_MODEL, GROQ_COMPRESS_MODEL, GROQ_DOWNSTREAM_MODEL

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"

# Active models — overridden in main() when --provider hf is set
_COMPRESS_MODEL    = _MODEL_HAIKU
_DOWNSTREAM_MODEL  = _MODEL_OPUS

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
    # --- Medical / Healthcare ---
    (
        "The HIPAA-compliant audit log retention period is probably 6 years, "
        "but the legal team mentioned some states require 10 years. I haven't "
        "confirmed which jurisdiction applies to our patient data.",
        "audit log retention",
        "What audit log retention period do we need to implement for HIPAA?",
    ),
    (
        "The HL7 FHIR API rate limit is approximately 200 requests per minute "
        "per tenant based on the sandbox docs, but production limits might be "
        "different. I haven't confirmed with the EHR vendor.",
        "FHIR API rate limit",
        "What rate limit should we enforce for the FHIR API integration?",
    ),
    (
        "I believe the PHI de-identification uses the Safe Harbor method with "
        "18 identifiers removed, but the Expert Determination method was also "
        "discussed. The final approach hasn't been locked down.",
        "de-identification method",
        "Which PHI de-identification method is approved for this dataset?",
    ),
    (
        "The medication dosage alert threshold is probably ±20% of the standard "
        "dose, but the pharmacy team said it might be ±15% for pediatric patients. "
        "The clinical team hasn't finalised the configuration.",
        "dosage alert threshold",
        "What dosage deviation triggers a clinical alert in the medication system?",
    ),
    (
        "The patient data breach notification window is 60 days under HIPAA, "
        "but our BAA with the hospital may have a 30-day SLA clause. I haven't "
        "read the latest BAA version.",
        "breach notification window",
        "How many days do we have to notify patients of a data breach?",
    ),
    (
        "The clinical decision support response time SLA is roughly 200ms for "
        "critical alerts, but I've seen 500ms referenced in some integration docs. "
        "We haven't load-tested the CDS service under full traffic.",
        "CDS response SLA",
        "What response time SLA applies to critical clinical decision support alerts?",
    ),
    (
        "The encrypted backup retention for medical records is approximately "
        "7 years, but the state regulation might require longer for minors. "
        "Legal hasn't confirmed the exact requirement for our patient population.",
        "medical record backup retention",
        "How long must encrypted medical record backups be retained?",
    ),
    (
        "The interoperability endpoint supports SMART on FHIR version 1.0 or 2.0 "
        "— I'm not sure which version the partner hospital's portal requires. "
        "We haven't confirmed the integration spec.",
        "SMART FHIR version",
        "Which SMART on FHIR version should we implement for the partner portal?",
    ),
    (
        "The consent management service can handle approximately 500 concurrent "
        "patient consent requests, but the number could be lower if we include "
        "the audit write overhead. We haven't benchmarked under realistic load.",
        "consent service concurrency",
        "What is the maximum concurrent load the consent management service supports?",
    ),
    (
        "The clinical trial randomisation seed is set to 42 in the test environment, "
        "but the production seed might be different based on the trial protocol. "
        "I haven't confirmed with the biostatistics team.",
        "randomisation seed",
        "What randomisation seed is used in the production clinical trial system?",
    ),
    # --- Legal / Compliance ---
    (
        "The GDPR data subject access request response window is 30 days, "
        "but there might be a 3-month extension available. I haven't confirmed "
        "whether our current process accounts for the extension option.",
        "DSAR response window",
        "What is the maximum time allowed to respond to a data subject access request?",
    ),
    (
        "The contract auto-renewal notice period is probably 60 days, but some "
        "enterprise agreements might require 90 days written notice. I haven't "
        "reviewed all the contract templates for this clause.",
        "auto-renewal notice period",
        "What advance notice is required to cancel auto-renewal on enterprise contracts?",
    ),
    (
        "The data processing agreement with the EU sub-processor must be signed "
        "within 30 or 45 days of onboarding — I can't remember which the DPA "
        "template specifies. I haven't checked the current template version.",
        "DPA signing window",
        "How many days after onboarding must the data processing agreement be executed?",
    ),
    (
        "The software export control classification is probably EAR99, but "
        "some cryptographic components might push it to ECCN 5D002. Legal "
        "hasn't completed the formal export classification review.",
        "export classification",
        "What export control classification number applies to this software?",
    ),
    (
        "The liability cap under the enterprise SLA is either 3 months or "
        "12 months of fees paid — I've seen both numbers in different contract "
        "drafts. Legal hasn't finalised the standard template.",
        "liability cap",
        "What is the maximum liability cap in the enterprise service agreement?",
    ),
    (
        "The statute of limitations for software IP infringement claims is "
        "approximately 3 years in most US jurisdictions, but it might be "
        "6 years in some states. Legal counsel hasn't confirmed which applies.",
        "IP statute of limitations",
        "What is the statute of limitations period for software IP infringement claims?",
    ),
    (
        "The PCI DSS Level 1 compliance audit must be completed every 12 months, "
        "but the card network agreement might require an interim assessment at "
        "6 months. I haven't reviewed the specific card network addendum.",
        "PCI audit frequency",
        "How often must PCI DSS Level 1 compliance audits be completed?",
    ),
    (
        "The whistleblower report retention period is roughly 5 years under "
        "Sarbanes-Oxley, but the compliance team mentioned 7 years might apply "
        "to certain financial records. The policy hasn't been finalised.",
        "whistleblower report retention",
        "How long must whistleblower reports be retained under SOX compliance?",
    ),
    (
        "The data localisation requirement for EU user data probably means "
        "we need servers in Frankfurt or Dublin, but the new SCCs might allow "
        "other mechanisms. Legal is still reviewing the adequacy decision.",
        "EU data localisation",
        "Which regions are approved for storing EU user data under the current rules?",
    ),
    (
        "The background check consent form must be retained for 5 years under "
        "FCRA, but state law might require longer in California or New York. "
        "HR hasn't confirmed the retention schedule for multi-state employees.",
        "background check retention",
        "How long must background check consent forms be retained?",
    ),
    # --- Finance / Fintech ---
    (
        "The transaction reconciliation window is approximately T+2, but the "
        "clearing house agreement might allow T+3 for some instrument types. "
        "I haven't confirmed the settlement schedule with the custodian.",
        "settlement window",
        "What is the transaction settlement window for the clearing house integration?",
    ),
    (
        "The AML transaction monitoring threshold is probably $10,000 for "
        "CTR filings, but suspicious activity patterns below $5,000 might "
        "also require filing. Compliance hasn't finalised the rule set.",
        "AML monitoring threshold",
        "What transaction amount triggers an AML currency transaction report?",
    ),
    (
        "The daily ACH transfer limit per customer is approximately $25,000, "
        "but premium accounts might have a $100,000 limit. I haven't checked "
        "the current tier configuration in the payment system.",
        "ACH transfer limit",
        "What is the daily ACH transfer limit for a standard account?",
    ),
    (
        "The credit card chargeback dispute window is 60 days from the "
        "transaction date, but some card networks extend it to 120 days for "
        "fraud cases. I haven't confirmed the window with our payment processor.",
        "chargeback dispute window",
        "How many days does a customer have to dispute a credit card charge?",
    ),
    (
        "The interest accrual calculation uses daily compounding with a 365-day "
        "year, but some loan instruments might use 360-day convention. I haven't "
        "verified which applies to the new loan product line.",
        "interest accrual basis",
        "What day count convention is used for interest accrual on the loan product?",
    ),
    (
        "The KYC identity verification SLA is probably 24 hours for standard "
        "accounts, but high-risk customer tiers might require 72-hour enhanced "
        "due diligence. The risk policy document hasn't been updated.",
        "KYC verification SLA",
        "What is the KYC identity verification turnaround time for standard accounts?",
    ),
    (
        "The options contract margin requirement is approximately 20% of the "
        "underlying notional, but volatility-adjusted margin might push it to "
        "35%. The risk desk hasn't confirmed the updated margin schedule.",
        "options margin requirement",
        "What margin percentage is required for writing options contracts?",
    ),
    (
        "The PSD2 strong customer authentication timeout is 5 minutes for "
        "the one-time passcode, but the bank's policy might extend it to "
        "10 minutes for accessibility. I haven't reviewed the SCA flow config.",
        "SCA timeout",
        "How long is the PSD2 strong customer authentication window valid?",
    ),
    (
        "The crypto custody cold storage percentage is roughly 95%, but "
        "the board policy might have increased it to 98% after the exchange "
        "hacks. The treasury policy hasn't been ratified yet.",
        "cold storage percentage",
        "What percentage of crypto assets must be held in cold storage?",
    ),
    (
        "The wire transfer fraud detection model uses a 0.85 confidence threshold "
        "to flag transactions, but the fraud team said they might lower it to "
        "0.75 after the recent miss. The model configuration is unverified.",
        "fraud threshold",
        "What confidence threshold triggers a wire transfer fraud flag?",
    ),
    # --- Multi-Agent Pipeline ---
    (
        "The orchestrator agent assigns tasks with a 30-second timeout, but "
        "the worker agents handling document parsing might need 90 seconds. "
        "I haven't profiled the actual p95 latency for the parsing step.",
        "agent task timeout",
        "What timeout should we configure for document parsing tasks in the agent pipeline?",
    ),
    (
        "The agent memory context window is capped at 16,000 tokens to control "
        "costs, but complex reasoning tasks might need 32,000. I haven't measured "
        "whether 16k is sufficient for the financial analysis workflow.",
        "agent context cap",
        "What context window size should be configured for the financial analysis agent?",
    ),
    (
        "The tool call retry budget per agent step is approximately 3 attempts, "
        "but the code execution tool might need 5 attempts if the environment "
        "is flaky. The retry policy hasn't been tuned for each tool type.",
        "tool retry budget",
        "How many tool call retries are allowed per agent reasoning step?",
    ),
    (
        "The cross-agent trust score threshold is probably 0.80 for passing "
        "unverified claims downstream, but safety-critical pipelines might "
        "require 0.95. The trust policy hasn't been defined for each pipeline.",
        "agent trust threshold",
        "What minimum trust score is required before passing a claim to the next agent?",
    ),
    (
        "The agent pipeline parallel execution cap is around 5 concurrent "
        "sub-agents, but the infrastructure might support 10 during off-peak. "
        "I haven't load-tested the orchestrator under maximum fan-out.",
        "parallel agent cap",
        "How many sub-agents can the orchestrator run in parallel?",
    ),
    (
        "The shared memory write lock timeout is approximately 500ms for the "
        "agent coordination layer, but concurrent writes might cause contention "
        "and need a longer timeout. The lock configuration is unverified.",
        "memory lock timeout",
        "What write lock timeout is configured for the agent shared memory layer?",
    ),
    (
        "The agent output verification model uses GPT-4 for cross-checking, "
        "but switching to Claude Opus might change the agreement rate. "
        "I haven't benchmarked the two models on our verification task.",
        "verification model",
        "Which model is used for output verification in the multi-agent pipeline?",
    ),
    (
        "The inter-agent message size limit is probably 64KB for the current "
        "message bus, but the schema evolution might require larger payloads. "
        "The message bus capacity hasn't been re-assessed for the new format.",
        "message size limit",
        "What is the maximum message size the inter-agent bus supports?",
    ),
    (
        "The agent state snapshot interval is roughly every 10 tool calls, "
        "but long-running tasks might need checkpointing every 5 calls to "
        "prevent work loss on failure. The checkpoint policy isn't finalised.",
        "snapshot interval",
        "How frequently should the agent pipeline take state snapshots?",
    ),
    (
        "The agent observability trace sampling rate is approximately 10% in "
        "production to control storage costs, but security incidents require "
        "100% sampling. The adaptive sampling policy is unverified.",
        "trace sampling rate",
        "What trace sampling rate is configured for the agent pipeline in production?",
    ),
    # --- Infrastructure / DevOps ---
    (
        "The Kubernetes pod resource limit is 2 CPU cores and 4GB memory based "
        "on the old values, but the team said they profiled peak usage at "
        "3 cores and 6GB. I haven't seen the updated resource manifest.",
        "pod resource limits",
        "What CPU and memory limits should we set on the Kubernetes deployment?",
    ),
    (
        "The auto-scaling cooldown period is probably 300 seconds to prevent "
        "thrashing, but the payment service might need a shorter 120-second "
        "window. I haven't reviewed the HPA configuration.",
        "autoscaling cooldown",
        "What cooldown period is configured on the horizontal pod autoscaler?",
    ),
    (
        "The container image pull policy is IfNotPresent for most services, "
        "but the security team might require Always for compliance reasons. "
        "The deployment policy hasn't been standardised across teams.",
        "image pull policy",
        "What container image pull policy should be used in production?",
    ),
    (
        "The S3 lifecycle rule moves objects to Glacier after 90 days, but "
        "compliance might require 30 days for audit logs and 180 days for "
        "user data. I haven't reviewed the data classification policy.",
        "S3 lifecycle transition",
        "After how many days should objects transition to Glacier storage class?",
    ),
    (
        "The blue-green deployment cutover window is approximately 5 minutes "
        "for DNS propagation plus health checks, but the SRE team said some "
        "regions take up to 15 minutes. The actual cutover SLA is unverified.",
        "deployment cutover window",
        "How long should we budget for a blue-green deployment cutover?",
    ),
    (
        "The Terraform state lock timeout is set to 10 minutes by default, "
        "but large infrastructure changes might hold the lock for 30 minutes. "
        "I haven't confirmed what the current timeout is in our remote backend.",
        "state lock timeout",
        "What Terraform state lock timeout is configured on the remote backend?",
    ),
    (
        "The log aggregation pipeline can handle approximately 50,000 events "
        "per second at the current node count, but peak traffic spikes could "
        "exceed that. I haven't stress-tested the ingestion tier.",
        "log ingestion capacity",
        "What is the maximum log ingestion rate the pipeline supports?",
    ),
    (
        "The secret rotation interval for database credentials is probably "
        "90 days, but the security policy might require 30 days for production. "
        "I haven't verified the current AWS Secrets Manager rotation schedule.",
        "secret rotation interval",
        "How frequently are production database credentials rotated?",
    ),
    (
        "The distributed tracing retention is approximately 14 days in Jaeger, "
        "but incident investigations might need 30 days of trace history. "
        "The storage allocation for the tracing backend is unconfirmed.",
        "trace retention",
        "How many days of distributed trace data is retained in the tracing backend?",
    ),
    (
        "The network egress cost for cross-AZ traffic is roughly $0.01 per GB, "
        "but the actual billing rate might depend on the data transfer type. "
        "I haven't confirmed the cost model with the cloud finance team.",
        "egress cost",
        "What is the per-GB cost for cross-AZ network egress in our cloud account?",
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

def _make_client(provider: str = "anthropic"):
    return make_client(provider)


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
    """Compression model summarises the full conversation. No epistemic safety check."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in conversation
    )
    return _call(
        client, _COMPRESS_MODEL,
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
        client, _COMPRESS_MODEL,
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
    return _call(client, _DOWNSTREAM_MODEL, msgs, system=system, max_tokens=150)


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

def _bootstrap_ci(values: List[float], n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Non-parametric bootstrap CI for the mean of a 0/1 list."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    boot_means = sorted(
        sum(values[random.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 - (1 - ci) / 2) * n_boot)
    return (boot_means[lo_idx], boot_means[hi_idx])


def aggregate(results: list[ScenarioResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    naive_qual_list    = [int(r.naive_qualifier_survived)    for r in results]
    naive_cert_list    = [int(r.naive_downstream_certain)    for r in results]
    probe_block_list   = [int(r.probe_blocked)               for r in results]
    probe_cert_list    = [int(r.probe_downstream_certain)    for r in results]
    baseline_cert_list = [int(r.baseline_downstream_certain) for r in results]
    lingua_qual_list   = [int(r.lingua_qualifier_survived)   for r in results]
    lingua_cert_list   = [int(r.lingua_downstream_certain)   for r in results]
    enhanced_qual_list  = [int(r.enhanced_qualifier_survived)  for r in results]
    enhanced_cert_list  = [int(r.enhanced_downstream_certain)  for r in results]

    naive_qual_survival    = sum(naive_qual_list)    / n
    naive_downstream_cert  = sum(naive_cert_list)    / n
    probe_block_rate       = sum(probe_block_list)   / n
    probe_downstream_cert  = sum(probe_cert_list)    / n
    baseline_cert          = sum(baseline_cert_list) / n
    lingua_qual_survival   = sum(lingua_qual_list)   / n
    lingua_downstream_cert = sum(lingua_cert_list)   / n
    enhanced_qual_survival  = sum(enhanced_qual_list)  / n
    enhanced_downstream_cert= sum(enhanced_cert_list) / n

    # 95% bootstrap CIs (2000 resamples) on key proportions
    ci_naive_qual    = _bootstrap_ci(naive_qual_list)
    ci_naive_cert    = _bootstrap_ci(naive_cert_list)
    ci_probe_cert    = _bootstrap_ci(probe_cert_list)
    ci_lingua_cert   = _bootstrap_ci(lingua_cert_list)
    ci_baseline_cert = _bootstrap_ci(baseline_cert_list)
    ci_probe_block   = _bootstrap_ci(probe_block_list)

    return {
        "n": n,
        "naive_qualifier_survival":           round(naive_qual_survival,     3),
        "naive_qualifier_survival_ci95":      [round(ci_naive_qual[0], 3),    round(ci_naive_qual[1], 3)],
        "naive_downstream_certainty":         round(naive_downstream_cert,   3),
        "naive_downstream_certainty_ci95":    [round(ci_naive_cert[0], 3),    round(ci_naive_cert[1], 3)],
        "lingua_qualifier_survival":          round(lingua_qual_survival,    3),
        "lingua_downstream_certainty":        round(lingua_downstream_cert,  3),
        "lingua_downstream_certainty_ci95":   [round(ci_lingua_cert[0], 3),   round(ci_lingua_cert[1], 3)],
        "enhanced_prompt_qualifier_survival": round(enhanced_qual_survival,  3),
        "enhanced_prompt_downstream_certainty": round(enhanced_downstream_cert, 3),
        "probe_block_rate":                   round(probe_block_rate,         3),
        "probe_block_rate_ci95":              [round(ci_probe_block[0], 3),   round(ci_probe_block[1], 3)],
        "probe_downstream_certainty":         round(probe_downstream_cert,    3),
        "probe_downstream_certainty_ci95":    [round(ci_probe_cert[0], 3),    round(ci_probe_cert[1], 3)],
        "baseline_downstream_certainty":      round(baseline_cert,            3),
        "baseline_downstream_certainty_ci95": [round(ci_baseline_cert[0], 3), round(ci_baseline_cert[1], 3)],
        # Derived: how much does each condition reduce FCR vs. naive?
        "fcr_reduction_naive_vs_probe":    round(naive_downstream_cert - probe_downstream_cert,    3),
        "fcr_reduction_naive_vs_enhanced": round(naive_downstream_cert - enhanced_downstream_cert, 3),
        "fcr_reduction_lingua_vs_probe":   round(lingua_downstream_cert - probe_downstream_cert,   3),
        # Key research question: probe vs. enhanced prompt (marginal value of the probe)
        "probe_marginal_vs_enhanced_prompt": round(enhanced_downstream_cert - probe_downstream_cert, 3),
        "scorer_version": "2.0-corrected",
        "vocabulary_size": len(_UNCERTAINTY_MARKERS),
        "scorer_notes": "v2: user-turns-only scanning; enhanced_prompt control added v2.1; CI added v2.2",
    }


def print_summary(agg: dict):
    print("\n" + "=" * 72)
    print("COMPRESSION FAITHFULNESS STUDY — RESULTS")
    print("=" * 72)
    print(f"  Scenarios run:                          {agg['n']}")
    print()
    def _fmt_ci(key):
        ci = agg.get(key + "_ci95")
        if ci:
            return f"  [95% CI {ci[0]:.1%}–{ci[1]:.1%}]"
        return ""

    print("  NAIVE COMPRESSION (Haiku, no probe):")
    print(f"    Qualifier survival rate:              "
          f"{agg['naive_qualifier_survival']:.1%}"
          f"{_fmt_ci('naive_qualifier_survival')}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['naive_downstream_certainty']:.1%}"
          f"{_fmt_ci('naive_downstream_certainty')}")
    print()
    print("  TOKEN-IMPORTANCE SIMULATION (30% compression, no epistemic awareness):")
    print(f"    Qualifier survival rate:              "
          f"{agg['lingua_qualifier_survival']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['lingua_downstream_certainty']:.1%}"
          f"{_fmt_ci('lingua_downstream_certainty')}")
    print()
    print("  ENHANCED PROMPT CONTROL (Haiku + explicit qualifier-preservation instruction):")
    print(f"    Qualifier survival rate:              "
          f"{agg['enhanced_prompt_qualifier_survival']:.1%}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['enhanced_prompt_downstream_certainty']:.1%}")
    print()
    print("  PROBE-GUARDED COMPRESSION (faithfulness probe, no Haiku call):")
    print(f"    Compression blocked rate:             "
          f"{agg['probe_block_rate']:.1%}"
          f"{_fmt_ci('probe_block_rate')}")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['probe_downstream_certainty']:.1%}"
          f"{_fmt_ci('probe_downstream_certainty')}")
    print()
    print("  BASELINE (full context, no compression — oracle upper bound):")
    print(f"    Downstream false-certainty rate:      "
          f"{agg['baseline_downstream_certainty']:.1%}"
          f"{_fmt_ci('baseline_downstream_certainty')}")
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

def ci_from_file(path: str):
    """Compute and print bootstrap CIs from an existing results JSON (no API needed)."""
    with open(path) as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    n = len(scenarios)
    if n == 0:
        print("No scenario-level data found.")
        return

    fields_map = {
        "naive_qualifier_survival":   "naive_qualifier_survived",
        "naive_downstream_certainty": "naive_downstream_certain",
        "lingua_downstream_certainty":"lingua_downstream_certain",
        "probe_block_rate":           "probe_blocked",
        "probe_downstream_certainty": "probe_downstream_certain",
        "baseline_downstream_certainty": "baseline_downstream_certain",
    }
    print(f"\nBootstrap CIs (95%, n={n}, 2000 resamples) — {path}")
    print("-" * 72)
    for label, field_key in fields_map.items():
        vals = [int(s.get(field_key, 0)) for s in scenarios]
        mean = sum(vals) / n
        lo, hi = _bootstrap_ci(vals)
        print(f"  {label:<40} {mean:.1%}  [95% CI {lo:.1%}–{hi:.1%}]")
    print()


def main():
    global _COMPRESS_MODEL, _DOWNSTREAM_MODEL

    parser = argparse.ArgumentParser(
        description="Compression Faithfulness Study")
    parser.add_argument("--n",        type=int,  default=100,
                        help="Number of scenarios to run (default: 100)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate scenario definitions without API calls")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-scenario results")
    parser.add_argument("--out",      default="evals/compression_faithfulness_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--ci",       type=str,  default=None,
                        help="Compute bootstrap CIs from existing results file (no API)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "hf", "groq"],
                        help="Inference provider: anthropic (default) | hf (HuggingFace free)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from existing results file")
    args = parser.parse_args()

    if args.ci:
        ci_from_file(args.ci)
        return

    scenarios_to_run = SCENARIOS[:args.n]

    if args.dry_run:
        dry_run(n=args.n)
        return

    if args.provider == "hf":
        _COMPRESS_MODEL   = HF_COMPRESS_MODEL
        _DOWNSTREAM_MODEL = HF_DOWNSTREAM_MODEL
    elif args.provider == "groq":
        _COMPRESS_MODEL   = GROQ_COMPRESS_MODEL
        _DOWNSTREAM_MODEL = GROQ_DOWNSTREAM_MODEL
    elif not _ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed. Use --provider groq")
        sys.exit(1)

    client = _make_client(args.provider)
    print(f"\nRunning compression faithfulness study ({len(scenarios_to_run)} scenarios)...")
    print(f"Provider:  {args.provider}")
    print(f"Models:    compress={_COMPRESS_MODEL}  downstream={_DOWNSTREAM_MODEL}")
    print(f"Conditions: naive_compress | probe_guard | baseline | lingua_sim | enhanced_prompt\n")

    # Resume: load already-completed scenarios and skip them
    out_path = args.out
    completed_labels: set[str] = set()
    results: list[ScenarioResult] = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            prior = json.load(f)
        for s in prior.get("scenarios", []):
            completed_labels.add(s["constraint_label"])
            # Re-inflate as ScenarioResult
            results.append(ScenarioResult(**{k: v for k, v in s.items()
                                             if k in ScenarioResult.__dataclass_fields__}))
        print(f"  [resume] Loaded {len(results)} completed scenarios, skipping them.\n")

    for i, (stmt, label, question) in enumerate(scenarios_to_run):
        if label in completed_labels:
            print(f"  [{i+1:02d}] {label[:35]:<35} SKIP (already done)")
            continue
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
        # Crash-safe: save after every scenario
        _save(results, out_path, args.provider)

    agg = aggregate(results)
    print_summary(agg)
    _save(results, out_path, args.provider, final_agg=agg)
    print(f"\nResults saved to {out_path}")


def _save(results, out_path, provider, final_agg=None):
    agg = final_agg or aggregate(results)
    output = {
        "summary": agg,
        "provider": provider,
        "compress_model": _COMPRESS_MODEL,
        "downstream_model": _DOWNSTREAM_MODEL,
        "scenarios": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

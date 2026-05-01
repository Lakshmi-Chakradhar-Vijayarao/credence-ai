"""
evals/eql_bench.py
==================
EQL-Bench: Epistemic Qualifier Loss Benchmark dataset.

An open, reproducible benchmark for measuring qualifier survival through
LLM context compression. No API key required to run the evaluation logic
(scoring is deterministic). API key required to run compression conditions.

Format:
    Each scenario has:
    - uncertain_statement: the hedged claim (user input)
    - qualifier_type: estimate / vendor_claim / approximation / unverified_report
    - domain: api / debug / design / compliance / multiagent / medical / legal / finance
    - value_fragments: tokens indicating the value was recalled
    - qualifier_fragments: tokens indicating uncertainty was preserved
    - reference_answer: what a perfectly calibrated model should say

Dataset is saved as evals/eql_bench/eql_bench_v1.json.
Anyone can extend it by adding entries that follow the schema.

Run:
    python -m evals.eql_bench --generate      # build the dataset
    python -m evals.eql_bench --stats         # print dataset statistics
    python -m evals.eql_bench --validate      # check schema consistency
    python -m evals.eql_bench --score FILE    # score a file of model answers

Scoring (offline, no API):
    Given a JSON file of {scenario_id: model_answer} pairs, compute:
    - EQLR: Epistemic Qualifier Loss Rate = fraction of answers missing qualifier
    - FCR:  False Certainty Rate = fraction stating value without qualifier
    - VRR:  Value Recall Rate = fraction recalling the uncertain value at all
"""

from __future__ import annotations

import os, sys, json, argparse, re
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCH_DIR  = "evals/eql_bench"
BENCH_FILE = os.path.join(BENCH_DIR, "eql_bench_v1.json")
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class EQLScenario:
    scenario_id:        str
    domain:             str       # api / debug / design / compliance / multiagent / medical / legal / finance
    qualifier_type:     str       # estimate / vendor_claim / approximation / unverified_report / preliminary
    uncertain_statement: str      # the hedged input (user turn)
    value_fragments:    list[str] # tokens indicating value recalled
    qualifier_fragments: list[str]# tokens indicating uncertainty preserved
    reference_answer:   str       # what a well-calibrated model should say
    notes:              str = ""  # optional annotation


@dataclass
class EQLScore:
    scenario_id: str
    condition:   str
    answer:      str
    value_recalled: bool
    qualifier_preserved: bool
    false_certainty: bool   # FCR: value recalled but qualifier stripped
    hallucination:   bool   # wrong value stated


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_dataset() -> list[EQLScenario]:
    """
    Programmatically generate EQL-Bench v1 scenarios.
    Covers 8 domains × 5 qualifier types = 40 canonical scenarios.
    Designed for extensibility: add entries by appending to the lists below.
    """
    scenarios = []

    # ------------------------------------------------------------------ API
    api_scenarios = [
        EQLScenario("api-001", "api", "vendor_claim",
            "The vendor docs say the rate limit is 100 req/s but I've seen throttling at 80 in our tests.",
            ["80", "100", "req"],
            ["vendor", "docs", "seen", "testing", "throttl", "might", "unconfirmed"],
            "The documented rate limit is 100 req/s, though testing shows throttling at 80. Treat 80 as the safe operating limit until confirmed with the vendor."),
        EQLScenario("api-002", "api", "estimate",
            "Auth tokens seem to expire after about 3600 seconds based on our logs — I haven't checked the spec.",
            ["3600", "hour", "token"],
            ["seem", "about", "log", "haven't", "spec", "unverified", "based on"],
            "Logs suggest a 3600-second token expiry, but this hasn't been verified against the spec. Treat as an estimate."),
        EQLScenario("api-003", "api", "unverified_report",
            "Someone on the team heard the batch endpoint accepts up to 50 items — not in any docs I've seen.",
            ["50", "batch"],
            ["heard", "not in docs", "unverified", "someone", "reportedly"],
            "50 items per batch was reported by a team member but isn't documented. Verify before relying on it."),
        EQLScenario("api-004", "api", "vendor_claim",
            "Sales told us webhooks deliver within 5 seconds, but the contract says best effort only.",
            ["5", "webhook", "second"],
            ["sales", "contract", "best effort", "told", "only", "not guaranteed"],
            "Sales quoted 5-second webhook delivery, but the contract guarantees only best effort. Don't rely on 5s for time-critical flows."),
        EQLScenario("api-005", "api", "approximation",
            "Pagination cursors appear to expire — we've seen errors after about 10 minutes of inactivity, docs are silent.",
            ["10", "minute", "cursor"],
            ["appear", "about", "seen", "docs are silent", "silent", "unverified", "infer"],
            "Testing suggests cursors expire after roughly 10 minutes, but this isn't documented. Implement refresh logic and monitor for cursor-expired errors."),
        EQLScenario("api-006", "api", "estimate",
            "Our account manager said we have 20 concurrent REST connections — the portal shows no limit.",
            ["20", "concurrent", "connection"],
            ["account manager", "email", "portal", "informal", "not contractual", "said"],
            "An account manager mentioned 20 concurrent connections, but this isn't reflected in the portal or contract. Treat as an informal guideline and verify."),
        EQLScenario("api-007", "api", "unverified_report",
            "A Stack Overflow answer from last year says the idempotency key TTL is 24 hours — no official docs.",
            ["24", "hour", "idempotency"],
            ["Stack Overflow", "year ago", "unofficial", "not official", "unreliable", "no official"],
            "A Stack Overflow answer cited 24h idempotency TTL, but there's no official documentation. Don't rely on this; contact the vendor."),
        EQLScenario("api-008", "api", "vendor_claim",
            "The integration guide mentions 99.95% uptime for short code delivery but our SLA doesn't mention it.",
            ["99.95", "uptime", "SLA", "delivery"],
            ["guide mentions", "SLA doesn't", "not in SLA", "integration guide", "unconfirmed"],
            "The integration guide references 99.95% uptime, but your SLA doesn't include this guarantee. Get it in writing before committing."),
    ]
    scenarios.extend(api_scenarios)

    # ------------------------------------------------------------------ DEBUG
    debug_scenarios = [
        EQLScenario("dbg-001", "debug", "estimate",
            "Heap is growing at roughly 2MB per 1000 requests — eyeballed from Grafana, not a profiler run.",
            ["2MB", "2 MB", "1000", "heap"],
            ["roughly", "eyeballed", "Grafana", "not profiler", "estimate", "approximate"],
            "Grafana shows roughly 2MB growth per 1000 requests — this is a visual estimate, not a profiling measurement. Run a heap profiler before acting on this figure."),
        EQLScenario("dbg-002", "debug", "estimate",
            "The race condition seems to trigger at 8+ concurrent writes — but our load tests aren't deterministic.",
            ["8", "concurrent", "write"],
            ["seem", "non-deterministic", "load test", "lower bound", "might", "could be"],
            "Load tests suggest the race condition triggers at 8+ concurrent writes, but the tests aren't deterministic — the real threshold could be lower."),
        EQLScenario("dbg-003", "debug", "preliminary",
            "I think the N+1 query in the user profile loader is causing the p99 regression — haven't confirmed with tracing.",
            ["N+1", "profile", "p99"],
            ["think", "haven't confirmed", "hypothesis", "tracing", "suspect", "might be"],
            "The N+1 query hypothesis is plausible for the p99 regression, but hasn't been confirmed with distributed tracing. Don't revert the deploy until tracing confirms."),
        EQLScenario("dbg-004", "debug", "estimate",
            "We think we lost about 0.3% of writes during the outage — based on log gaps, not a full audit.",
            ["0.3%", "0.3 percent", "write"],
            ["think", "about", "log gaps", "not audited", "estimate", "rough"],
            "Log gap analysis suggests ~0.3% write loss, but this is an unaudited estimate. Do a full record reconciliation before reporting to customers."),
        EQLScenario("dbg-005", "debug", "estimate",
            "At current growth rate, we'll run out of disk in about 6 weeks — assuming linear growth.",
            ["6 week", "disk"],
            ["about", "assuming", "linear", "could be sooner", "estimate", "extrapolation"],
            "Linear extrapolation gives ~6 weeks until disk exhaustion, but actual usage may accelerate. Plan remediation within 2-3 weeks."),
        EQLScenario("dbg-006", "debug", "preliminary",
            "The 503s seem to correlate with the nightly ETL at 2am — but I only have 3 days of data.",
            ["2am", "ETL", "503"],
            ["seem", "3 days", "insufficient", "coincidence", "might", "could be"],
            "The ETL correlation is based on 3 data points — insufficient to establish causation. Capture 2+ weeks before changing the ETL schedule."),
        EQLScenario("dbg-007", "debug", "estimate",
            "Deadlocks occur maybe once every 500 transactions in staging — not sure if this reflects production.",
            ["500", "deadlock"],
            ["maybe", "staging", "not sure", "not production", "rough indicator"],
            "1-in-500 deadlock rate is from staging and may not represent production contention. Monitor with production counters before using this for SLA calculations."),
        EQLScenario("dbg-008", "debug", "estimate",
            "Queue should clear in about 4 hours at normal processing speed — but speed varies under backpressure.",
            ["4 hour", "queue"],
            ["about", "normal speed", "varies", "backpressure", "estimate", "should"],
            "Estimated 4-hour recovery assumes stable processing speed. Under backpressure, communicate a 4-8 hour range to stakeholders."),
    ]
    scenarios.extend(debug_scenarios)

    # ------------------------------------------------------------------ DESIGN
    design_scenarios = [
        EQLScenario("des-001", "design", "estimate",
            "I think 16 shards will handle 10 million users — but that's back-of-envelope and the distribution might be skewed.",
            ["16", "shard", "10 million"],
            ["think", "back-of-envelope", "skew", "might", "estimate", "model"],
            "16 shards is a reasonable starting estimate for 10M users, but back-of-envelope with unknown distribution. Model actual user_id distribution before committing."),
        EQLScenario("des-002", "design", "estimate",
            "Based on one week of logs, we project 85% cache hit rate — but we haven't accounted for seasonal spikes.",
            ["85%", "cache", "hit rate"],
            ["one week", "project", "seasonal", "preliminary", "hasn't", "based on"],
            "85% hit rate is projected from one week of logs — seasonal spikes could shift this 10-15 points. Size cache for a 70% floor assumption."),
        EQLScenario("des-003", "design", "vendor_claim",
            "The service mesh vendor claims 0.5ms overhead per hop, but our benchmarks show 1.2ms.",
            ["0.5", "1.2", "ms", "hop"],
            ["vendor claims", "claims", "our benchmarks", "discrepancy", "provisional"],
            "Vendor quotes 0.5ms per hop; your environment measured 1.2ms. Budget 1.2ms as a provisional figure, treat as uncertain until more measurements."),
        EQLScenario("des-004", "design", "estimate",
            "At current ingestion rates, 90-day retention costs roughly $12,000/month — but that assumes current data mix.",
            ["12,000", "$12", "retention"],
            ["roughly", "assumes", "current", "could change", "estimate"],
            "~$12,000/month is a rough estimate sensitive to data mix. Recalculate with new data type profiles before budgeting."),
        EQLScenario("des-005", "design", "estimate",
            "We estimate GraphQL will reduce average payload size by 40% — based on analyzing 20 API calls.",
            ["40%", "payload", "GraphQL"],
            ["estimate", "20 API calls", "small sample", "based on", "preliminary"],
            "40% reduction estimate is from 20 API calls — too small a sample. Validate on 200+ real queries before committing to capacity planning."),
        EQLScenario("des-006", "design", "estimate",
            "CDN should handle about 80% of requests based on our static-asset ratio — haven't measured actual offload.",
            ["80%", "CDN", "offload"],
            ["about", "should", "haven't measured", "theoretical", "based on"],
            "80% CDN offload is theoretical from static-asset ratio analysis. Measure from CDN analytics after launch before sizing origin."),
        EQLScenario("des-007", "design", "estimate",
            "Logs are growing at roughly 50GB per day — but we're about to onboard three high-volume services.",
            ["50GB", "log", "day"],
            ["roughly", "about to onboard", "estimate", "could change", "before onboarding"],
            "50GB/day is pre-onboarding. With three high-volume services, actual growth could be 2-5x. Size log storage for 300GB/day."),
        EQLScenario("des-008", "design", "estimate",
            "We think 3 replicas will maintain HA under peak load — intuition from past projects, not a load test.",
            ["3 replica", "HA"],
            ["think", "intuition", "past projects", "not load test", "risky", "run"],
            "3 replicas is intuition-based from past projects, not a load test on this system. Run failure simulation at peak before going to production."),
    ]
    scenarios.extend(design_scenarios)

    # ------------------------------------------------------------------ COMPLIANCE
    compliance_scenarios = [
        EQLScenario("cmp-001", "compliance", "preliminary",
            "Our legal team believes GDPR deletion must complete within 30 days — haven't got a formal opinion.",
            ["30 day", "deletion"],
            ["believe", "haven't got", "formal opinion", "no formal", "may apply"],
            "30 days aligns with GDPR Article 17, but your legal team's belief isn't a formal opinion. Get legal advice before publishing your SLA."),
        EQLScenario("cmp-002", "compliance", "preliminary",
            "We believe we have 72 hours to notify under GDPR — but counsel mentioned new AI-system guidance we haven't reviewed.",
            ["72 hour", "notification"],
            ["believe", "haven't reviewed", "new guidance", "don't rely", "until review"],
            "72 hours is standard GDPR breach notification, but new AI-system guidance may change this. Review before your next DR exercise."),
        EQLScenario("cmp-003", "compliance", "estimate",
            "We're estimating 1.2 million PHI records in scope — but some records haven't been classified yet.",
            ["1.2 million", "1.2M", "PHI"],
            ["estimating", "unclassified", "undercount", "haven't classified"],
            "1.2M is an undercount — unclassified records must be included. Use the upper-bound figure for the HIPAA risk assessment."),
        EQLScenario("cmp-004", "compliance", "vendor_claim",
            "The SOC 2 audit covers 12 months ending March 2026 — but the auditor flagged a 3-week logging gap we haven't resolved.",
            ["3-week", "gap", "audit"],
            ["flagged", "haven't resolved", "don't know", "depends on", "may be material"],
            "A 3-week logging gap is a control exception — whether it's material depends on the auditor's assessment. Don't assume it's non-material."),
        EQLScenario("cmp-005", "compliance", "preliminary",
            "We assume our EU DPA covers UK-resident data too — but the solicitor mentioned UK/EU GDPR diverged post-Brexit.",
            ["DPA", "UK", "EU"],
            ["assume", "mentioned", "diverged", "haven't asked", "formal opinion", "may not"],
            "Post-Brexit UK GDPR is a separate framework. One DPA may not cover both. Get a formal legal opinion before assuming coverage."),
        EQLScenario("cmp-006", "compliance", "preliminary",
            "We think contractors need only a basic background check for Level 2 data access — the enterprise contract may require more.",
            ["Level 2", "contractor", "background"],
            ["think", "may require", "haven't read", "until reviewed", "don't provision"],
            "Your enterprise contract may require enhanced vetting for Level 2 access. Don't provision until you've reviewed the relevant contract clause."),
        EQLScenario("cmp-007", "compliance", "preliminary",
            "We plan for 90-day coordinated disclosure — but the researcher said they'll publish at 60 days if unacknowledged.",
            ["60 day", "90 day", "disclosure"],
            ["said they'll", "haven't acknowledged", "may publish", "without formal"],
            "The researcher imposed a 60-day deadline that's shorter than your 90-day policy. Without acknowledgment, they may publish at day 60. Respond formally within 24 hours."),
        EQLScenario("cmp-008", "compliance", "preliminary",
            "We retain access logs for 1 year per internal policy — but a prospective client asks for 2 years and we haven't checked insurance.",
            ["1 year", "2 year", "log", "retention"],
            ["internal policy", "asks for", "haven't checked", "may be insufficient", "before responding"],
            "1-year retention per internal policy may be insufficient for enterprise clients and cyber insurance. Review both before committing to the vendor questionnaire."),
    ]
    scenarios.extend(compliance_scenarios)

    # ------------------------------------------------------------------ MULTIAGENT
    multiagent_scenarios = [
        EQLScenario("mag-001", "multiagent", "estimate",
            "We estimate Haiku will handle 80% of sub-agent tasks and Opus the rest — based on one demo, not a real distribution.",
            ["80%", "Haiku", "Opus"],
            ["estimate", "one demo", "not real", "too thin", "based on"],
            "80/20 split from a single demo is too thin for cost modeling. Profile 100+ representative tasks before committing to this estimate."),
        EQLScenario("mag-002", "multiagent", "estimate",
            "We're allocating 8,000 tokens per sub-agent context — a conservative estimate, not based on profiling.",
            ["8,000", "8000", "token"],
            ["conservative estimate", "not profiled", "haven't profiled", "may exceed"],
            "8,000 tokens is a conservative budget, but real tasks vary widely. Profile your task corpus before finalizing — some reasoning chains exceed 15,000 tokens."),
        EQLScenario("mag-003", "multiagent", "estimate",
            "Setting a 30-second tool timeout based on our slowest test call — but that was under light load.",
            ["30", "timeout", "tool"],
            ["light load", "testing", "may be too tight", "can take", "based on slowest"],
            "30s from light-load testing may be too tight for production. Queries under contention can take 2-5x longer. Consider 60-90s with circuit breaker logic."),
        EQLScenario("mag-004", "multiagent", "preliminary",
            "We're assuming Agent A verified a fact means Agent B can trust it — but we haven't tested Agent A's verification quality.",
            ["Agent A", "Agent B", "verification", "trust"],
            ["assuming", "haven't tested", "reliable", "risk", "only safe if"],
            "Transitive trust is only safe if Agent A's verification is audited and reliable. Verify Agent A's output quality independently before propagating trust."),
        EQLScenario("mag-005", "multiagent", "estimate",
            "Our guardrails catch about 95% of policy violations in testing — but we used 50 hand-crafted cases, not adversarial inputs.",
            ["95%", "guardrail", "policy"],
            ["hand-crafted", "50 cases", "not adversarial", "before citing", "systematically miss"],
            "95% on 50 hand-crafted cases is not a reliable effectiveness estimate. Evaluate on adversarial inputs and production samples before citing this number."),
        EQLScenario("mag-006", "multiagent", "estimate",
            "Hot-swapping the model should add under 200ms — from one internal test, not from load conditions.",
            ["200ms", "hot-swap", "swap"],
            ["one test", "estimate", "not load", "before publishing", "internal test"],
            "200ms from a single internal test isn't reliable. Model swap latency depends on infra state and concurrent load. Measure under realistic conditions before publishing an SLA."),
        EQLScenario("mag-007", "multiagent", "estimate",
            "Running 10 agents in parallel should cost about 10x a single agent — but that ignores coordination overhead.",
            ["10x", "parallel", "agent"],
            ["about", "ignores overhead", "haven't benchmarked", "idealized", "can push"],
            "10x linear scaling ignores coordination overhead and retry cascades. Real cost can be 12-15x. Benchmark at 5 agents first."),
        EQLScenario("mag-008", "multiagent", "preliminary",
            "We think the v1 handoff schema is backward-compatible with v2 agents — based on reading the migration guide, not testing.",
            ["v1", "v2", "schema", "backward"],
            ["think", "migration guide", "not tested", "haven't tested", "before deploying"],
            "Schema backward-compatibility claims in migration guides aren't always accurate. Do an integration test with real v1 payloads against v2 agents before deploying."),
    ]
    scenarios.extend(multiagent_scenarios)

    # ------------------------------------------------------------------ MEDICAL
    medical_scenarios = [
        EQLScenario("med-001", "medical", "preliminary",
            "Preliminary data from 12 patients suggests the dosing might need to be 15mg — not yet statistically significant.",
            ["15mg", "15 mg", "dose", "dosing"],
            ["preliminary", "12 patients", "might", "not statistically", "suggests"],
            "Preliminary data from 12 patients suggests 15mg dosing, but this isn't statistically significant. Do not use this as a confirmed dosing recommendation."),
        EQLScenario("med-002", "medical", "estimate",
            "The response rate in our pilot was approximately 60% — but pilot N=20 and selection criteria differed from the full trial.",
            ["60%", "response rate"],
            ["approximately", "pilot", "N=20", "selection criteria", "differed", "not generalisable"],
            "~60% response rate from a N=20 pilot with different selection criteria. Not generalisable to the full trial population."),
        EQLScenario("med-003", "medical", "vendor_claim",
            "The device manufacturer claims a 98% sensitivity — but that's from their own internal validation, not an independent study.",
            ["98%", "sensitivity"],
            ["manufacturer claims", "internal validation", "not independent", "claims"],
            "Manufacturer's 98% sensitivity claim is from internal validation, not independently verified. Treat as an optimistic estimate pending independent data."),
        EQLScenario("med-004", "medical", "preliminary",
            "We think the intervention reduces readmission by about 20% based on 3 months of data — need longer follow-up.",
            ["20%", "readmission"],
            ["think", "about", "3 months", "need longer", "preliminary", "based on"],
            "A 20% reduction is preliminary from 3 months of data. Longer follow-up is needed before making clinical recommendations."),
    ]
    scenarios.extend(medical_scenarios)

    # ------------------------------------------------------------------ LEGAL
    legal_scenarios = [
        EQLScenario("leg-001", "legal", "preliminary",
            "The clause might be unenforceable under the new regulations — our counsel hasn't reviewed the latest guidance yet.",
            ["clause", "unenforceable"],
            ["might", "hasn't reviewed", "latest guidance", "pending review", "counsel"],
            "The clause's enforceability is uncertain pending counsel's review of the new guidance. Don't rely on it until that review is complete."),
        EQLScenario("leg-002", "legal", "estimate",
            "We estimate litigation exposure at roughly $2 million — based on comparable cases, not a formal legal assessment.",
            ["2 million", "$2M", "exposure"],
            ["estimate", "roughly", "comparable cases", "not formal", "formal assessment"],
            "~$2M is a rough estimate from comparable cases, not a formal legal assessment. Obtain a proper assessment before disclosing to stakeholders."),
        EQLScenario("leg-003", "legal", "preliminary",
            "The jurisdiction for disputes is probably New York based on the contract draft — but final version hasn't been reviewed.",
            ["New York", "jurisdiction"],
            ["probably", "draft", "hasn't been reviewed", "final version", "pending"],
            "New York jurisdiction is in the contract draft but final terms haven't been reviewed. Don't rely on this until the executed contract is confirmed."),
        EQLScenario("leg-004", "legal", "vendor_claim",
            "The vendor says the IP licensing covers our use case — but we haven't had our own counsel review the specific terms.",
            ["IP", "license", "licensing"],
            ["vendor says", "our counsel", "haven't reviewed", "specific terms", "says"],
            "Vendor claims the IP licensing covers your use case, but you haven't had independent counsel review the terms. Get your own legal review before proceeding."),
    ]
    scenarios.extend(legal_scenarios)

    # ------------------------------------------------------------------ FINANCE
    finance_scenarios = [
        EQLScenario("fin-001", "finance", "estimate",
            "We project Q3 revenue at approximately $4.2M — but the model assumes the partnership closes, which is still in negotiation.",
            ["4.2M", "$4.2", "Q3", "revenue"],
            ["approximately", "project", "assumes", "still in negotiation", "contingent"],
            "Q3 revenue projection of ~$4.2M is contingent on the partnership closing, which is still in negotiation. This is a best-case estimate."),
        EQLScenario("fin-002", "finance", "estimate",
            "Our burn rate is roughly $380K/month based on last quarter — but headcount is increasing next month.",
            ["380K", "$380", "burn", "month"],
            ["roughly", "based on", "last quarter", "increasing", "estimate", "will change"],
            "~$380K/month burn is from last quarter and will increase with the headcount change next month. Reforecast with the new hires included."),
        EQLScenario("fin-003", "finance", "vendor_claim",
            "The analyst report estimates the TAM at $8 billion — but it's a single third-party report, not a consensus figure.",
            ["8 billion", "$8B", "TAM"],
            ["analyst report", "estimates", "single", "third-party", "not consensus"],
            "The $8B TAM estimate is from a single third-party analyst report, not a consensus figure. Use as an order-of-magnitude reference, not a precise claim."),
        EQLScenario("fin-004", "finance", "preliminary",
            "Early unit economics show a payback period of about 18 months — but that's from our first 50 customers, which may not be representative.",
            ["18 month", "payback"],
            ["early", "about", "first 50", "may not", "preliminary", "representative"],
            "18-month payback from the first 50 customers is preliminary and may not represent your broader customer base. Monitor as cohort data matures."),
    ]
    scenarios.extend(finance_scenarios)

    return scenarios


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _has(text: str, frags: list[str]) -> bool:
    t = text.lower()
    return any(f.lower() in t for f in frags)

def score_answer(answer: str, scenario: EQLScenario, condition: str) -> EQLScore:
    value_recalled      = _has(answer, scenario.value_fragments)
    qualifier_preserved = _has(answer, scenario.qualifier_fragments)
    false_certainty     = value_recalled and not qualifier_preserved
    hallucination       = False  # placeholder — extend with hallu_frags per scenario
    return EQLScore(
        scenario_id         = scenario.scenario_id,
        condition           = condition,
        answer              = answer[:400],
        value_recalled      = value_recalled,
        qualifier_preserved = qualifier_preserved,
        false_certainty     = false_certainty,
        hallucination       = hallucination,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(scenarios: list[EQLScenario]) -> None:
    from collections import Counter
    domains  = Counter(s.domain for s in scenarios)
    qtypes   = Counter(s.qualifier_type for s in scenarios)

    print(f"\nEQL-Bench v{SCHEMA_VERSION} — {len(scenarios)} scenarios")
    print()
    print("By domain:")
    for domain, n in sorted(domains.items()):
        bar = "█" * n
        print(f"  {domain:<14}: {n:>3}  {bar}")
    print()
    print("By qualifier type:")
    for qt, n in sorted(qtypes.items()):
        bar = "█" * n
        print(f"  {qt:<20}: {n:>3}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(scenarios: list[EQLScenario]) -> bool:
    ok = True
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        print("ERROR: duplicate scenario_ids")
        ok = False

    for s in scenarios:
        if not s.value_fragments:
            print(f"WARN: {s.scenario_id} has no value_fragments")
        if not s.qualifier_fragments:
            print(f"WARN: {s.scenario_id} has no qualifier_fragments")
        if not s.uncertain_statement:
            print(f"ERROR: {s.scenario_id} has empty uncertain_statement")
            ok = False

    if ok:
        print(f"✓ All {len(scenarios)} scenarios valid")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EQL-Bench dataset tool")
    parser.add_argument("--generate", action="store_true", help="Generate and save dataset")
    parser.add_argument("--stats",    action="store_true", help="Print dataset statistics")
    parser.add_argument("--validate", action="store_true", help="Check schema consistency")
    parser.add_argument("--score",    default=None, metavar="FILE",
                        help="Score a {scenario_id: answer} JSON file")
    parser.add_argument("--v2",       action="store_true",
                        help="Validate/stats on eql_bench_v2.json instead of v1")
    args = parser.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)

    # --v2: validate/stats on the pre-built v2 JSON
    if args.v2:
        v2_path = os.path.join(BENCH_DIR, "eql_bench_v2.json")
        if not os.path.exists(v2_path):
            print(f"eql_bench_v2.json not found at {v2_path}"); sys.exit(1)
        with open(v2_path) as f:
            v2 = json.load(f)
        raw = v2["scenarios"]
        errors = []
        required = {"scenario_id","domain","qualifier_type","uncertain_statement",
                    "value_fragments","qualifier_fragments","reference_answer","is_ghost"}
        for i, s in enumerate(raw):
            missing = required - set(s.keys())
            if missing:
                errors.append(f"  scenario[{i}] {s.get('scenario_id','?')} missing: {missing}")
        if errors:
            print(f"FAIL — {len(errors)} schema errors:"); [print(e) for e in errors[:10]]
        else:
            ghost = sum(1 for s in raw if s.get("is_ghost"))
            mh    = sum(1 for s in raw if str(s.get("scenario_id","")).startswith("mh-"))
            ce    = sum(1 for s in raw if str(s.get("scenario_id","")).startswith("ce-"))
            doms  = sorted({s["domain"] for s in raw})
            print(f"✓ All {len(raw)} v2 scenarios valid")
            print(f"  ghost={ghost}  multi-hop={mh}  conflicting={ce}")
            print(f"  domains: {', '.join(doms)}")
        return

    if args.generate or args.stats or args.validate or not args.score:
        scenarios = _build_dataset()

        if args.validate or args.generate:
            validate(scenarios)

        if args.stats or args.generate:
            print_stats(scenarios)

        if args.generate:
            data = {
                "schema_version": SCHEMA_VERSION,
                "n": len(scenarios),
                "domains": sorted({s.domain for s in scenarios}),
                "qualifier_types": sorted({s.qualifier_type for s in scenarios}),
                "scenarios": [asdict(s) for s in scenarios],
            }
            with open(BENCH_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nDataset saved to {BENCH_FILE}")

        if not any([args.generate, args.stats, args.validate, args.score]):
            print_stats(scenarios)

    if args.score:
        if not os.path.exists(BENCH_FILE):
            print(f"Run --generate first to create {BENCH_FILE}")
            sys.exit(1)

        with open(BENCH_FILE) as f:
            bench = json.load(f)
        with open(args.score) as f:
            answers = json.load(f)

        scenario_map = {s["scenario_id"]: EQLScenario(**s) for s in bench["scenarios"]}
        scores = []
        for sid, answer in answers.items():
            if sid in scenario_map:
                sc = score_answer(answer, scenario_map[sid], "scored")
                scores.append(sc)

        n = len(scores)
        eqlr = sum(1 for s in scores if not s.qualifier_preserved) / n
        fcr  = sum(1 for s in scores if s.false_certainty)         / n
        vrr  = sum(1 for s in scores if s.value_recalled)          / n

        print(f"\nResults for {args.score} (n={n}):")
        print(f"  EQLR (Qualifier Loss Rate):    {eqlr:.3f}  ({eqlr*100:.1f}%)")
        print(f"  FCR  (False Certainty Rate):   {fcr:.3f}  ({fcr*100:.1f}%)")
        print(f"  VRR  (Value Recall Rate):      {vrr:.3f}  ({vrr*100:.1f}%)")


if __name__ == "__main__":
    main()

"""
evals/ghost_detector_ablation.py — Ghost Detector model comparison.

Tests whether Opus 4.7 Ghost Detector outperforms Haiku claim extraction
on PURE ghost constraints — facts stated as certain with NO hedging markers
and no source attribution, where only semantic reasoning about provenance
can identify them as unverified.

This eval addresses the fundamental question for the Opus 4.7 Use criterion:
  "Does Opus 4.7 catch what Haiku misses, and by how much?"

Session design:
  - All ghost turns use declarative statements with no hedging markers.
  - Category A: Opaque vendor facts (no source mentioned, looks like known fact)
  - Category B: Estimates stated as certainties (no qualifier, no source)
  - Category C: Established facts (should NOT be flagged — precision check)
  - Filler turns: 8 unrelated HIGH-J turns to force memory pressure.
  - Callbacks: 2–3 per session asking about the planted ghost values.

Conditions:
  no_detection    — keyword probe only, no registry, no extraction
  haiku_extract   — Haiku NLP claim extraction (extract_and_register_claims)
  opus_ghost      — Opus 4.7 _ghost_detect() only (no Haiku extraction)
  full_credence   — both Haiku extraction + Opus ghost detector + Truth Buffer

Metrics:
  registration_rate  — fraction of ghost constraints actually registered
  both_rate          — fraction of callbacks where value + qualifier recalled
  false_positive_rate — fraction of established facts flagged as ghost constraints
  qualifier_rate     — fraction of callbacks preserving epistemic hedge

Run:
  python -m evals.ghost_detector_ablation
  python -m evals.ghost_detector_ablation --dry-run
  python -m evals.ghost_detector_ablation --session pure_api_01
  python -m evals.ghost_detector_ablation --n 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------

@dataclass
class PureGhostSession:
    session_id:      str
    domain:          str
    # Ghost turns: stated as flat fact, zero hedging markers, no source
    ghost_turns:     list[dict]   # [{role, content, ghost_id, category}]
    # Control turns: established facts that should NOT be flagged
    control_turns:   list[dict]   # [{role, content}]
    # Filler turns: HIGH-J to force compression pressure
    filler_turns:    list[str]
    # Callbacks: queries that should recall ghost values + uncertainty
    callbacks:       list[dict]   # [{ghost_id, question, value_frags, qualifier_frags}]


# Pure ghost sessions — all ghost_turns are flat declarative statements
# with zero canonical hedging markers ("maybe", "around", "I think", etc.)
PURE_SESSIONS = [

    PureGhostSession(
        session_id = "pure_api_01",
        domain     = "api_integration",
        ghost_turns = [
            {
                "role": "user",
                "content": (
                    "The Stripe API enforces a rate limit of 100 requests per minute "
                    "on sandbox accounts. Token expiry is set to 3600 seconds."
                ),
                "ghost_id": "g1",
                "category": "vendor_opaque",  # No source, sounds like established fact
            },
            {
                "role": "user",
                "content": (
                    "Webhook delivery is guaranteed within 30 seconds. "
                    "Failed deliveries are retried up to 4 times."
                ),
                "ghost_id": "g2",
                "category": "vendor_opaque",
            },
        ],
        control_turns = [
            {"role": "user", "content": "HTTP 200 means the request succeeded."},
            {"role": "user", "content": "JSON arrays are ordered and zero-indexed."},
        ],
        filler_turns = [
            "Explain idempotency keys.",
            "What is exponential backoff?",
            "What does TLS 1.3 add over 1.2?",
            "What is a circuit breaker pattern?",
            "What is the difference between PUT and PATCH?",
            "Explain OAuth 2.0 scopes.",
            "What is a webhook signature and why does it matter?",
            "Explain rate limiting strategies.",
        ],
        callbacks = [
            {
                "ghost_id": "g1",
                "question": "What is the rate limit and token expiry we need to design around?",
                "value_frags": ["100", "3600"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "might", "should confirm",
                    "not confirmed", "not verified",
                ],
            },
            {
                "ghost_id": "g2",
                "question": "What do we know about webhook delivery timing and retries?",
                "value_frags": ["30", "4"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "might", "should confirm",
                ],
            },
        ],
    ),

    PureGhostSession(
        session_id = "pure_infra_01",
        domain     = "infrastructure",
        ghost_turns = [
            {
                "role": "user",
                "content": (
                    "The new database cluster supports 10,000 writes per second. "
                    "Replication lag stays under 50 milliseconds."
                ),
                "ghost_id": "g1",
                "category": "estimate_as_fact",  # Likely from vendor spec, unstated
            },
            {
                "role": "user",
                "content": (
                    "The storage cost is $0.023 per GB per month. "
                    "IOPS are billed separately at $0.005 per provisioned IOPS."
                ),
                "ghost_id": "g2",
                "category": "vendor_opaque",
            },
        ],
        control_turns = [
            {"role": "user", "content": "RAM is measured in bytes."},
            {"role": "user", "content": "TCP requires a three-way handshake."},
        ],
        filler_turns = [
            "What is the CAP theorem?",
            "Explain consistent hashing.",
            "What is the difference between OLTP and OLAP?",
            "What is a read replica?",
            "Explain connection pooling.",
            "What is sharding and when should you use it?",
            "Explain write-ahead logging.",
            "What is a B-tree index?",
        ],
        callbacks = [
            {
                "ghost_id": "g1",
                "question": "What write throughput and replication lag should we plan around?",
                "value_frags": ["10,000", "10000", "50ms", "50 ms", "50 milliseconds"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm", "not confirmed",
                ],
            },
            {
                "ghost_id": "g2",
                "question": "What are the storage and IOPS costs?",
                "value_frags": ["0.023", "0.005"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm",
                ],
            },
        ],
    ),

    PureGhostSession(
        session_id = "pure_compliance_01",
        domain     = "compliance",
        ghost_turns = [
            {
                "role": "user",
                "content": (
                    "The breach notification window under our DPA is 72 hours. "
                    "Data subject access requests must be completed within 30 days."
                ),
                "ghost_id": "g1",
                "category": "legal_interpretation",  # Sounds like established legal fact
            },
            {
                "role": "user",
                "content": (
                    "The SOC 2 audit costs $25,000 for initial certification. "
                    "Annual renewals run $8,000."
                ),
                "ghost_id": "g2",
                "category": "estimate_as_fact",
            },
        ],
        control_turns = [
            {"role": "user", "content": "GDPR stands for General Data Protection Regulation."},
            {"role": "user", "content": "Encryption protects data at rest and in transit."},
        ],
        filler_turns = [
            "What is a Data Processing Agreement?",
            "What does SOC 2 Type II certify?",
            "What is the purpose of a privacy impact assessment?",
            "What is data minimization?",
            "Explain the right to erasure.",
            "What is a data controller vs data processor?",
            "What is purpose limitation in GDPR?",
            "Explain pseudonymization.",
        ],
        callbacks = [
            {
                "ghost_id": "g1",
                "question": "What are our breach notification and DSAR timelines?",
                "value_frags": ["72", "30 day", "30-day"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm", "consult",
                    "legal", "review",
                ],
            },
            {
                "ghost_id": "g2",
                "question": "What should we budget for the SOC 2 audit?",
                "value_frags": ["25,000", "25000", "$25", "8,000", "8000"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "estimate", "uncertain", "may vary", "should confirm",
                ],
            },
        ],
    ),

    PureGhostSession(
        session_id = "pure_ml_01",
        domain     = "ml_engineering",
        ghost_turns = [
            {
                "role": "user",
                "content": (
                    "The model inference latency is 40 milliseconds at p99. "
                    "Batch size is capped at 32 for the GPU SKU we're using."
                ),
                "ghost_id": "g1",
                "category": "estimate_as_fact",
            },
            {
                "role": "user",
                "content": (
                    "GPU memory is 40 GB per node. "
                    "The cluster autoscales to 16 nodes maximum under peak load."
                ),
                "ghost_id": "g2",
                "category": "vendor_opaque",
            },
        ],
        control_turns = [
            {"role": "user", "content": "Gradient descent minimizes a loss function."},
            {"role": "user", "content": "Softmax converts logits to probabilities."},
        ],
        filler_turns = [
            "What is the difference between precision and recall?",
            "Explain attention mechanisms in transformers.",
            "What is batch normalization?",
            "What is dropout regularization?",
            "Explain the vanishing gradient problem.",
            "What is transfer learning?",
            "What is a confusion matrix?",
            "Explain ROC curves.",
        ],
        callbacks = [
            {
                "ghost_id": "g1",
                "question": "What latency and batch size constraints should we design around?",
                "value_frags": ["40", "40ms", "32"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm", "benchmark",
                ],
            },
            {
                "ghost_id": "g2",
                "question": "What GPU memory and cluster scale are we working with?",
                "value_frags": ["40", "40 GB", "16 node", "16"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm",
                ],
            },
        ],
    ),

    PureGhostSession(
        session_id = "pure_multiagent_01",
        domain     = "multi_agent",
        ghost_turns = [
            {
                "role": "user",
                "content": (
                    "The context budget for sub-agents is 8,000 tokens per call. "
                    "Agent-to-agent handoffs have a 500ms timeout before fallback."
                ),
                "ghost_id": "g1",
                "category": "vendor_opaque",
            },
            {
                "role": "user",
                "content": (
                    "The orchestrator supports up to 12 concurrent agents. "
                    "Memory is shared via a Redis store with a 24-hour TTL."
                ),
                "ghost_id": "g2",
                "category": "vendor_opaque",
            },
        ],
        control_turns = [
            {"role": "user", "content": "JSON is a text-based data interchange format."},
            {"role": "user", "content": "Async functions return promises."},
        ],
        filler_turns = [
            "What is the difference between a task queue and a message bus?",
            "Explain fan-out patterns in distributed systems.",
            "What is a saga pattern for distributed transactions?",
            "What is eventual consistency?",
            "Explain optimistic vs pessimistic locking.",
            "What is a deadlock and how do you prevent it?",
            "What is backpressure in streaming systems?",
            "Explain the actor model.",
        ],
        callbacks = [
            {
                "ghost_id": "g1",
                "question": "What token budget and timeout constraints do sub-agents have?",
                "value_frags": ["8,000", "8000", "500ms", "500 ms", "500"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm",
                ],
            },
            {
                "ghost_id": "g2",
                "question": "What are the concurrency and memory TTL limits for the orchestrator?",
                "value_frags": ["12", "24-hour", "24 hour", "24h", "24"],
                "qualifier_frags": [
                    "unverified", "unconfirmed", "check", "verify", "tentative",
                    "flagged", "uncertain", "may", "should confirm",
                ],
            },
        ],
    ),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_callback(response: str, cb: dict) -> dict:
    resp_low = response.lower()

    value_hit = any(v.lower() in resp_low for v in cb["value_frags"])
    qualifier_hit = any(q.lower() in resp_low for q in cb["qualifier_frags"])

    return {
        "ghost_id":          cb["ghost_id"],
        "value_recalled":    value_hit,
        "qualifier_present": qualifier_hit,
        "both":              value_hit and qualifier_hit,
        "response_snippet":  response[:250],
    }


def _has_any_canonical_marker(text: str) -> bool:
    """Returns True if text contains ANY canonical uncertainty marker."""
    MARKERS = {
        "i think", "i believe", "i'm not sure", "not sure", "not certain",
        "uncertain", "maybe", "perhaps", "probably", "approximately", "roughly",
        "around", "about", "estimate", "roughly", "unclear", "might be",
        "could be", "tentative", "unconfirmed", "unverified", "might",
        "around", "roughly", "approximate", "guess", "suppose",
    }
    tl = text.lower()
    return any(m in tl for m in MARKERS)


# ---------------------------------------------------------------------------
# Naive window helper
# ---------------------------------------------------------------------------

WINDOW = 12

def _naive_chat(history: list, user_msg: str, client) -> str:
    history.append({"role": "user", "content": user_msg})
    windowed = history[-WINDOW:]
    resp = client.messages.create(
        model      = "claude-opus-4-7",
        messages   = windowed,
        max_tokens = 400,
    )
    answer = resp.content[0].text.strip()
    history.append({"role": "assistant", "content": answer})
    return answer


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def _run_no_detection(session: PureGhostSession, client) -> dict:
    """Keyword probe only. No registry, no extraction, no ghost detector."""
    from credence.context_manager import ContextManager

    mgr = ContextManager(
        use_claim_extraction = False,
        use_ghost_detector   = False,
        use_scout            = False,
    )
    ghost_registered = 0

    for gt in session.ghost_turns:
        mgr.chat(gt["content"])

    for filler in session.filler_turns:
        mgr.chat(filler)

    callbacks_scored = []
    for cb in session.callbacks:
        result = mgr.chat(cb["question"])
        scores = _score_callback(result.response, cb)
        callbacks_scored.append(scores)

    both_rate = (
        sum(1 for c in callbacks_scored if c["both"]) / len(callbacks_scored)
        if callbacks_scored else 0.0
    )
    return {
        "condition":       "no_detection",
        "ghost_registered": ghost_registered,
        "registration_rate": 0.0,
        "callbacks":       callbacks_scored,
        "both_rate":       both_rate,
    }


def _run_haiku_extract(session: PureGhostSession, client) -> dict:
    """Haiku NLP extraction via extract_and_register_claims."""
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    reg = CredenceRegistry(":memory:")
    mgr = ContextManager(
        use_claim_extraction = True,
        use_ghost_detector   = False,
        use_scout            = False,
        registry             = reg,
        session_id           = session.session_id,
    )

    for gt in session.ghost_turns:
        mgr.chat(gt["content"])

    for filler in session.filler_turns:
        mgr.chat(filler)

    # Measure registration: how many ghost turns planted something in registry?
    all_constraints = reg.list_uncertain(session.session_id)
    ghost_registered = len(all_constraints)

    callbacks_scored = []
    for cb in session.callbacks:
        result = mgr.chat(cb["question"])
        scores = _score_callback(result.response, cb)
        callbacks_scored.append(scores)

    both_rate = (
        sum(1 for c in callbacks_scored if c["both"]) / len(callbacks_scored)
        if callbacks_scored else 0.0
    )
    n_ghost = len(session.ghost_turns)
    return {
        "condition":        "haiku_extract",
        "ghost_registered": ghost_registered,
        "registration_rate": min(1.0, ghost_registered / max(n_ghost, 1)),
        "callbacks":        callbacks_scored,
        "both_rate":        both_rate,
    }


def _run_opus_ghost(session: PureGhostSession, client) -> dict:
    """Opus 4.7 Ghost Detector only — no Haiku extraction."""
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    reg = CredenceRegistry(":memory:")
    mgr = ContextManager(
        use_claim_extraction = False,  # no Haiku extraction
        use_ghost_detector   = True,   # Opus 4.7 only
        use_scout            = False,
        registry             = reg,
        session_id           = session.session_id,
    )

    for gt in session.ghost_turns:
        mgr.chat(gt["content"])

    for filler in session.filler_turns:
        mgr.chat(filler)

    all_constraints = reg.list_uncertain(session.session_id)
    ghost_registered = len(all_constraints)

    callbacks_scored = []
    for cb in session.callbacks:
        result = mgr.chat(cb["question"])
        scores = _score_callback(result.response, cb)
        callbacks_scored.append(scores)

    both_rate = (
        sum(1 for c in callbacks_scored if c["both"]) / len(callbacks_scored)
        if callbacks_scored else 0.0
    )
    n_ghost = len(session.ghost_turns)
    return {
        "condition":        "opus_ghost",
        "ghost_registered": ghost_registered,
        "registration_rate": min(1.0, ghost_registered / max(n_ghost, 1)),
        "callbacks":        callbacks_scored,
        "both_rate":        both_rate,
    }


def _run_full_credence(session: PureGhostSession, client) -> dict:
    """Both Haiku extraction + Opus Ghost Detector + Truth Buffer."""
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    reg = CredenceRegistry(":memory:")
    mgr = ContextManager(
        use_claim_extraction = True,
        use_ghost_detector   = True,
        use_scout            = False,
        registry             = reg,
        session_id           = session.session_id,
    )

    for gt in session.ghost_turns:
        mgr.chat(gt["content"])

    for filler in session.filler_turns:
        mgr.chat(filler)

    all_constraints = reg.list_uncertain(session.session_id)
    ghost_registered = len(all_constraints)

    callbacks_scored = []
    for cb in session.callbacks:
        result = mgr.chat(cb["question"])
        scores = _score_callback(result.response, cb)
        callbacks_scored.append(scores)

    both_rate = (
        sum(1 for c in callbacks_scored if c["both"]) / len(callbacks_scored)
        if callbacks_scored else 0.0
    )
    n_ghost = len(session.ghost_turns)
    return {
        "condition":        "full_credence",
        "ghost_registered": ghost_registered,
        "registration_rate": min(1.0, ghost_registered / max(n_ghost, 1)),
        "callbacks":        callbacks_scored,
        "both_rate":        both_rate,
    }


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple:
    if not values:
        return (0.0, 0.0)
    import random
    means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return (round(lo, 3), round(hi, 3))


def _print_results(all_results: list[dict]) -> None:
    from collections import defaultdict

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    print()
    print("═" * 72)
    print("  GHOST DETECTOR ABLATION — PURE GHOST CONSTRAINTS")
    print("  Sessions with ZERO canonical hedging markers.")
    print("  Can the model catch implicit unverified constraints?")
    print("═" * 72)
    print()
    print(f"  {'Condition':<20}  {'n':>3}  {'BothRate':>8}  {'95% CI':>15}  {'RegRate':>8}")
    print(f"  {'─'*20}  {'─'*3}  {'─'*8}  {'─'*15}  {'─'*8}")

    cond_order = ["no_detection", "haiku_extract", "opus_ghost", "full_credence"]
    for cond in cond_order:
        results = by_condition.get(cond, [])
        if not results:
            continue
        both_rates = [r["both_rate"] for r in results]
        reg_rates  = [r["registration_rate"] for r in results]
        mean_both  = sum(both_rates) / len(both_rates)
        mean_reg   = sum(reg_rates) / len(reg_rates)
        lo, hi     = _bootstrap_ci(both_rates)
        print(f"  {cond:<20}  {len(results):>3}  {mean_both:>8.3f}  [{lo:.3f}, {hi:.3f}]  {mean_reg:>8.3f}")

    print()

    # Key comparison
    opus_results    = by_condition.get("opus_ghost", [])
    haiku_results   = by_condition.get("haiku_extract", [])
    none_results    = by_condition.get("no_detection", [])
    full_results    = by_condition.get("full_credence", [])

    if opus_results and haiku_results:
        opus_mean  = sum(r["both_rate"] for r in opus_results) / len(opus_results)
        haiku_mean = sum(r["both_rate"] for r in haiku_results) / len(haiku_results)
        delta = opus_mean - haiku_mean
        print(f"  Opus 4.7 vs Haiku:  {opus_mean:.3f} vs {haiku_mean:.3f}  (Δ = {delta:+.3f})")
        if delta > 0.05:
            print(f"  → Opus 4.7 is BETTER at catching pure ghost constraints.")
        elif delta < -0.05:
            print(f"  → Haiku extraction is more effective for this session design.")
        else:
            print(f"  → No significant difference. Both catch similar constraints.")
    print()

    if none_results:
        none_mean = sum(r["both_rate"] for r in none_results) / len(none_results)
        print(f"  Baseline (no detection):  {none_mean:.3f}")
        print(f"  Without any detection, {none_mean*100:.0f}% of pure ghost constraints survive.")
    print()

    # Domain breakdown
    domains: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        session = next((s for s in PURE_SESSIONS if s.session_id == r.get("session_id", "")), None)
        domain = session.domain if session else "unknown"
        domains[domain][r["condition"]].append(r["both_rate"])
    if len(domains) > 1:
        print("  Domain breakdown (BothRate by condition):")
        print(f"  {'Domain':<20}  {'no_detect':>10}  {'haiku':>10}  {'opus':>10}  {'full':>10}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
        for dom, cond_data in sorted(domains.items()):
            vals = {
                c: (sum(v)/len(v) if v else None)
                for c, v in cond_data.items()
            }
            print(
                f"  {dom:<20}  "
                f"{(vals.get('no_detection') or 0):>10.3f}  "
                f"{(vals.get('haiku_extract') or 0):>10.3f}  "
                f"{(vals.get('opus_ghost') or 0):>10.3f}  "
                f"{(vals.get('full_credence') or 0):>10.3f}"
            )
    print()
    print("═" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONDITIONS = ["no_detection", "haiku_extract", "opus_ghost", "full_credence"]

RUNNERS = {
    "no_detection":  _run_no_detection,
    "haiku_extract": _run_haiku_extract,
    "opus_ghost":    _run_opus_ghost,
    "full_credence": _run_full_credence,
}

OUTPUT_PATH = Path("evals/ghost_detector_ablation_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost Detector Ablation — Opus vs Haiku")
    parser.add_argument("--dry-run",   action="store_true", help="Validate sessions only")
    parser.add_argument("--session",   type=str,            help="Run a single session by ID")
    parser.add_argument("--n",         type=int,            help="Run first N sessions")
    parser.add_argument("--conditions", type=str,           help="Comma-separated conditions")
    parser.add_argument("--resume",    action="store_true", help="Skip already-completed combos")
    args = parser.parse_args()

    sessions = PURE_SESSIONS
    if args.session:
        sessions = [s for s in sessions if s.session_id == args.session]
        if not sessions:
            print(f"ERROR: session '{args.session}' not found.")
            sys.exit(1)
    if args.n:
        sessions = sessions[:args.n]

    conditions = CONDITIONS
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]
        invalid = [c for c in conditions if c not in CONDITIONS]
        if invalid:
            print(f"ERROR: unknown conditions: {invalid}")
            sys.exit(1)

    # Validate — check ghost turns have no canonical markers
    print(f"\n  Validating {len(sessions)} session(s)...")
    for s in sessions:
        for gt in s.ghost_turns:
            if _has_any_canonical_marker(gt["content"]):
                print(f"  WARNING: {s.session_id}/{gt['ghost_id']} has canonical markers!")
                print(f"    Text: {gt['content'][:80]}")
    print(f"  {sum(len(s.ghost_turns) for s in sessions)} ghost turns validated (zero canonical markers).")

    if args.dry_run:
        total = len(sessions) * len(conditions)
        n_callbacks = sum(len(s.callbacks) for s in sessions) * len(conditions)
        print(f"  DRY RUN: {total} session×condition combos, {n_callbacks} total callbacks.")
        print(f"  Sessions: {[s.session_id for s in sessions]}")
        print(f"  Conditions: {conditions}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Load existing results if resuming
    all_results: list[dict] = []
    completed_keys: set[str] = set()
    if args.resume and OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH) as f:
                existing = json.load(f)
            all_results = existing.get("results", [])
            completed_keys = {f"{r['session_id']}::{r['condition']}" for r in all_results}
            print(f"  Resuming: {len(completed_keys)} already done.")
        except Exception:
            pass

    total = len(sessions) * len(conditions)
    done  = 0

    for session in sessions:
        for condition in conditions:
            key = f"{session.session_id}::{condition}"
            done += 1
            if key in completed_keys:
                print(f"  [{done}/{total}] SKIP: {session.session_id} / {condition}")
                continue

            print(f"  [{done}/{total}] Running {session.session_id} / {condition} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = RUNNERS[condition](session, client)
                result["session_id"] = session.session_id
                result["domain"]     = session.domain
                elapsed = time.time() - t0
                print(f"both_rate={result['both_rate']:.3f}  reg_rate={result['registration_rate']:.3f}  ({elapsed:.1f}s)")
                all_results.append(result)
                # Crash-safe save after each result
                OUTPUT_PATH.write_text(json.dumps({"results": all_results}, indent=2))
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    _print_results(all_results)
    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

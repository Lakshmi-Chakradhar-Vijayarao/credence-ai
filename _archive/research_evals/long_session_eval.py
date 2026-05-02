"""
Long-Session Epistemic Invariant Eval
======================================
50-turn sessions with 5 planted uncertain constraints.
Forces 3+ compression/trim cycles to test whether Credence's invariant
("epistemic status propagates through ALL operations") holds at scale.

Design:
  Turns 1-4:   Seed — user plants 5 uncertain constraints (LOW-J, canonical markers)
  Turns 5-46:  Filler — HIGH-J technical discussion that forces compression pressure
  Turns 47-50: Callbacks — 5 targeted questions, one per constraint

Compression schedule (COMPRESS_AFTER=8, fires at n_turns > 16):
  Turn ~17:  1st compression cycle
  Turn ~25:  2nd compression cycle
  Turn ~33:  3rd compression cycle
  Turn ~41:  Trim (TRIM_WINDOW=10, fires at n_turns > 20)
  Turn ~50:  4th+ cycle possible

Three conditions:
  baseline      Full context — no window, no compression
  naive_window  Last 12 turns — classic sliding window
  credence      Our system — selective compression + Truth Buffer + CE

Metrics (per callback):
  value_present    Model recalled the specific value (e.g. "50", "3600")
  qualifier_present Model included uncertainty qualifier
  both_rate        value AND qualifier both present
  FCR              value present WITHOUT qualifier (False Certainty Rate)

Run:
  python -m evals.long_session_eval             # full run, all 5 sessions
  python -m evals.long_session_eval --n 1       # single session
  python -m evals.long_session_eval --dry-run   # structure validation only
  python -m evals.long_session_eval --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

# ---------------------------------------------------------------------------
# 5 long-session scenarios — each has seed constraints + filler + callbacks
# ---------------------------------------------------------------------------

SESSIONS = [
    {
        "id": "ls01",
        "name": "Payment service integration",
        "seeds": [
            {
                "user": (
                    "Quick note before we start: I think the Stripe rate limit is around "
                    "100 req/min — that's from a sales call, not confirmed. Also the "
                    "webhook retry count is probably 5 — I'm not certain."
                ),
                "value_fragments": ["100", "rate limit", "rate"],
                "qualifier_fragments": ["think", "probably", "not confirmed", "sales call",
                                        "unconfirmed", "unverified", "uncertain", "tentative",
                                        "approximately", "roughly", "maybe", "might"],
                "label": "stripe_rate_limit",
            },
            {
                "user": (
                    "Also, our JWT expiry is tentatively set to 3600 seconds — to be "
                    "confirmed with the security team. And I believe the connection pool "
                    "max is roughly 50 — haven't verified that."
                ),
                "value_fragments": ["3600", "jwt", "expiry", "token expiry"],
                "qualifier_fragments": ["tentatively", "to be confirmed", "believe", "roughly",
                                        "unconfirmed", "unverified", "uncertain", "tentative",
                                        "haven't verified", "not verified", "approximately"],
                "label": "jwt_expiry",
            },
        ],
        "callback_questions": [
            "What rate limit should we use for Stripe API calls in our code?",
            "What value should I set for JWT token expiry in production?",
        ],
    },
    {
        "id": "ls02",
        "name": "Infrastructure sizing",
        "seeds": [
            {
                "user": (
                    "For planning: the vendor mentioned our peak concurrent users might be "
                    "around 500 — that's just an estimate from the demo, not benchmarked. "
                    "And memory per pod is probably 2GB but I haven't confirmed that."
                ),
                "value_fragments": ["500", "concurrent", "peak users", "users"],
                "qualifier_fragments": ["mentioned", "might be", "estimate", "demo",
                                        "not benchmarked", "probably", "haven't confirmed",
                                        "unconfirmed", "unverified", "uncertain", "tentative"],
                "label": "peak_users",
            },
            {
                "user": (
                    "One more: the p99 latency target is supposedly 200ms — this was from "
                    "an early benchmark, could be wrong. Cache TTL is roughly 300 seconds "
                    "per the original design doc, still open for discussion."
                ),
                "value_fragments": ["200", "200ms", "latency", "p99"],
                "qualifier_fragments": ["supposedly", "could be wrong", "early benchmark",
                                        "roughly", "still open", "unconfirmed", "unverified",
                                        "uncertain", "tentative", "approximately"],
                "label": "p99_latency",
            },
        ],
        "callback_questions": [
            "How many concurrent users should we design the system to handle?",
            "What p99 latency target should I put in the SLA document?",
        ],
    },
    {
        "id": "ls03",
        "name": "Security configuration",
        "seeds": [
            {
                "user": (
                    "Security planning note: the GDPR breach notification window is "
                    "72 hours — I think that's right but need to verify with legal. "
                    "Also the session timeout is probably 1800 seconds, per the vendor "
                    "recommendation, unconfirmed."
                ),
                "value_fragments": ["72", "72 hour", "breach notification"],
                "qualifier_fragments": ["think", "need to verify", "probably", "per the vendor",
                                        "unconfirmed", "unverified", "uncertain", "tentative",
                                        "approximately", "roughly", "maybe"],
                "label": "gdpr_window",
            },
            {
                "user": (
                    "Also: password hash iterations should be around 100000 — that's from "
                    "a forum post I read, not production-tested. And the rate limiter "
                    "window is maybe 60 seconds — working theory, to be confirmed."
                ),
                "value_fragments": ["100000", "100,000", "hash iterations", "bcrypt"],
                "qualifier_fragments": ["around", "forum post", "not production-tested",
                                        "maybe", "working theory", "to be confirmed",
                                        "unconfirmed", "unverified", "uncertain"],
                "label": "hash_iterations",
            },
        ],
        "callback_questions": [
            "What's the GDPR breach notification window? I'm writing the incident response doc.",
            "What iteration count should I use for password hashing in the auth service?",
        ],
    },
    {
        "id": "ls04",
        "name": "API integration details",
        "seeds": [
            {
                "user": (
                    "Integration notes: the upstream API timeout is approximately 30 seconds "
                    "— from the docs but I haven't tested it. The max payload size is "
                    "roughly 5MB, that's what the sales rep said, not confirmed."
                ),
                "value_fragments": ["30", "30 second", "timeout"],
                "qualifier_fragments": ["approximately", "haven't tested", "roughly",
                                        "sales rep", "not confirmed", "unconfirmed",
                                        "unverified", "uncertain", "tentative"],
                "label": "api_timeout",
            },
            {
                "user": (
                    "Also: the pagination page size limit is probably 100 items per page "
                    "— I think that's what the docs say but could be wrong. And the "
                    "OAuth token refresh window is tentatively 5 minutes."
                ),
                "value_fragments": ["100", "page size", "pagination", "items"],
                "qualifier_fragments": ["probably", "think", "could be wrong",
                                        "tentatively", "unconfirmed", "unverified",
                                        "uncertain", "approximately", "maybe"],
                "label": "pagination_limit",
            },
        ],
        "callback_questions": [
            "What timeout should I configure for calls to the upstream API?",
            "What page size limit should we use in the pagination implementation?",
        ],
    },
    {
        "id": "ls05",
        "name": "Database configuration",
        "seeds": [
            {
                "user": (
                    "DB planning: connection pool size is probably 20 — that's a rough "
                    "estimate, needs load testing. Also query timeout should be around "
                    "5 seconds, I'm not 100% sure about that."
                ),
                "value_fragments": ["20", "connection pool", "pool size"],
                "qualifier_fragments": ["probably", "rough estimate", "needs load testing",
                                        "around", "not 100%", "unconfirmed", "unverified",
                                        "uncertain", "tentative", "approximately"],
                "label": "pool_size",
            },
            {
                "user": (
                    "One more: the max replication lag threshold is maybe 500ms — from a "
                    "design discussion, to be determined. And the backup retention period "
                    "is supposedly 30 days — per the vendor email, unverified."
                ),
                "value_fragments": ["500", "500ms", "replication lag", "lag"],
                "qualifier_fragments": ["maybe", "design discussion", "to be determined",
                                        "supposedly", "per the vendor", "unverified",
                                        "unconfirmed", "uncertain", "tentative", "approximately"],
                "label": "replication_lag",
            },
        ],
        "callback_questions": [
            "What connection pool size should I configure for the database?",
            "What replication lag threshold should trigger an alert?",
        ],
    },
]

# High-J filler turns — plain technical prose, no code blocks (avoids Type Prior cap)
# These force compression without themselves containing uncertain constraints
FILLER_TURNS = [
    "How does connection pooling work in PostgreSQL and what are the tradeoffs between "
    "PgBouncer's transaction-mode and session-mode pooling for high-throughput OLTP workloads?",

    "PgBouncer transaction mode multiplexes connections at the statement boundary, which "
    "works well for stateless queries. Session mode is required when using prepared statements, "
    "advisory locks, or SET LOCAL configurations that persist across statement boundaries.",

    "What's the standard approach for implementing distributed tracing across microservices "
    "using OpenTelemetry, and how do you propagate trace context through async message queues?",

    "OpenTelemetry propagates trace context via W3C TraceContext headers. For async queues "
    "like Kafka or SQS, you inject the context into message headers at publish time and "
    "extract it in the consumer before starting child spans.",

    "Explain the differences between blue-green deployment and canary deployment strategies, "
    "including when to prefer each approach for production releases.",

    "Blue-green maintains two identical environments and switches traffic instantly, "
    "providing near-zero downtime with a clean rollback path. Canary gradually shifts "
    "traffic percentages, exposing a small subset of users first to catch regression "
    "signals before full rollout.",

    "How do you implement efficient pagination for large datasets in REST APIs — what are "
    "the tradeoffs between offset-based, cursor-based, and keyset pagination?",

    "Cursor-based pagination is generally preferred for large, frequently-updated datasets "
    "because offset pagination degrades as the offset grows and can return duplicate or "
    "skip records when rows are inserted concurrently.",

    "What are the key considerations when designing a rate limiting system that needs to "
    "work correctly across multiple application instances with Redis as the shared store?",

    "Redis-based rate limiting typically uses sliding window counters via ZADD/ZREMRANGEBYSCORE "
    "or fixed window counters with atomic INCR and EXPIRE. The sliding window is more "
    "accurate but requires more memory per key.",

    "How should you structure database indexes for a query that filters on user_id, "
    "status, and created_at with an ORDER BY created_at DESC LIMIT 20 clause?",

    "A composite index on (user_id, status, created_at DESC) will allow the database to "
    "use an index scan for all three predicates and satisfy the ORDER BY without a "
    "separate sort step.",

    "What's the difference between an event-driven architecture using pub-sub messaging "
    "versus a CQRS pattern with event sourcing, and when does event sourcing add value?",

    "Event sourcing stores the complete history of state changes rather than current state, "
    "enabling time-travel queries and rebuilding projections. CQRS separates read and write "
    "models; they are complementary but independent patterns.",

    "How do you implement idempotency keys for payment API endpoints to prevent duplicate "
    "charges when clients retry requests after network failures?",

    "Idempotency keys are typically stored in a short-TTL key-value store. The server "
    "checks for the key before processing — if found, returns the cached response. "
    "The key and response are stored atomically in the first successful request.",

    "Explain how TLS certificate pinning works and what the operational challenges are "
    "when certificates rotate.",

    "Certificate pinning stores the expected certificate hash in the client, rejecting "
    "any certificate that doesn't match even if it's signed by a trusted CA. Rotation "
    "requires coordinated client and server updates, often using a grace period with "
    "multiple pinned certificates.",

    "What are the design considerations for implementing a distributed lock using Redis "
    "for coordinating background job execution across multiple workers?",

    "Redlock uses quorum voting across multiple independent Redis instances. For most "
    "use cases, a single Redis instance with SETNX and EXPIRE suffices if you accept "
    "the risk of lock loss during Redis failover.",

    "How does the Raft consensus algorithm differ from Paxos in terms of understandability "
    "and implementation complexity?",

    "Raft decomposes consensus into leader election, log replication, and safety, with "
    "strong leader semantics that simplify the protocol. Paxos is more general but "
    "requires additional protocols like Multi-Paxos for log replication in practice.",

    "What strategies exist for handling schema migrations in a zero-downtime deployment "
    "where both old and new application versions must read and write the same database?",

    "The expand-contract pattern adds new columns as nullable, deploys the new application "
    "that writes both old and new, then removes old columns after all instances are updated. "
    "This avoids locking and allows rollback at each step.",

    "How do you design a webhook delivery system that guarantees at-least-once delivery "
    "with exponential backoff and handles consumer endpoint failures gracefully?",

    "Webhooks should be stored in a durable queue before delivery, with retry state "
    "tracked per endpoint. Exponential backoff with jitter prevents thundering herds. "
    "Dead-letter queues capture permanently-failed deliveries for manual inspection.",

    "Explain the tradeoffs between synchronous and asynchronous inter-service communication "
    "in microservices architectures.",

    "Synchronous calls (HTTP/gRPC) couple availability — if the downstream service is "
    "unavailable, the caller fails. Asynchronous messaging decouples availability at the "
    "cost of eventual consistency and increased operational complexity.",

    "What is the N+1 query problem and what are the standard ORM-level solutions for it?",

    "N+1 occurs when loading a list of objects triggers a separate query per object for "
    "a related association. Solutions include eager loading (JOIN or batch SELECT), "
    "DataLoader-style batching, or denormalizing the related data.",

    "How does a B-tree index differ from a hash index, and when would you choose one "
    "over the other for a relational database column?",

    "B-tree indexes support range queries, ORDER BY, and prefix matching. Hash indexes "
    "only support equality lookups but are faster for that case. Most databases default "
    "to B-tree because range queries are more common.",

    "What is the difference between optimistic and pessimistic locking, and what are the "
    "failure modes of each under high contention?",

    "Optimistic locking detects conflicts at commit time via version numbers, requiring "
    "retry on conflict. Pessimistic locking serialises access via database locks, "
    "preventing conflicts but risking deadlocks and reduced throughput.",

    "How do content delivery networks cache HTTP responses, and what Cache-Control headers "
    "govern their behaviour?",

    "Cache-Control: max-age controls how long CDN edges cache the response. s-maxage "
    "overrides max-age specifically for shared caches. Vary headers instruct the CDN "
    "to cache separate variants per header value.",

    "What are the key differences between JWT and opaque session tokens for stateless "
    "authentication, and what are the security implications of each?",

    "JWTs embed claims in the token, allowing stateless verification without a database "
    "round-trip. The tradeoff is that JWTs cannot be revoked before expiry without a "
    "token blacklist. Opaque tokens require a lookup on every request but can be "
    "invalidated immediately.",

    "Explain how read replicas improve database read throughput and what consistency "
    "guarantees they provide.",

    "Read replicas receive asynchronous replication from the primary. They improve read "
    "throughput by distributing load but provide eventual consistency — reads may see "
    "stale data lagging behind the primary by the replication delay.",

    "What monitoring signals should be used to detect memory leaks in a long-running "
    "Node.js or Python service?",

    "Track heap size and RSS growth over time. Compare old-generation GC frequency and "
    "duration trends. Watch for steadily increasing object counts in heap profiles. "
    "A memory leak typically shows linear heap growth between GC cycles.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_any(text: str, fragments: list[str]) -> bool:
    lower = text.lower()
    return any(f.lower() in lower for f in fragments)


def _call_model(client: anthropic.Anthropic, messages: list[dict],
                system: str, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a senior software engineer helping a team design and implement "
    "a production system. Answer technical questions concisely and accurately."
)


def _run_baseline(session: dict, client: anthropic.Anthropic,
                  model: str) -> list[dict]:
    """Full context — no window, no compression. Oracle upper bound."""
    history: list[dict] = []

    # Build full session history
    for seed in session["seeds"]:
        history.append({"role": "user",      "content": seed["user"]})
        history.append({"role": "assistant", "content": "Noted — I'll keep those values in mind."})

    for i, filler_q in enumerate(FILLER_TURNS[:42]):
        history.append({"role": "user", "content": filler_q})
        resp = _call_model(client, list(history), SYSTEM, model)
        history.append({"role": "assistant", "content": resp})

    # Callback phase
    results = []
    for seed, question in zip(session["seeds"], session["callback_questions"]):
        history_copy = list(history) + [{"role": "user", "content": question}]
        answer = _call_model(client, history_copy, SYSTEM, model)
        vp = _has_any(answer, seed["value_fragments"])
        qp = _has_any(answer, seed["qualifier_fragments"])
        results.append({
            "condition": "baseline",
            "label":     seed["label"],
            "value_present":     vp,
            "qualifier_present": qp,
            "both":     vp and qp,
            "fcr":      vp and not qp,
            "answer":   answer[:200],
        })
    return results


def _run_naive_window(session: dict, client: anthropic.Anthropic,
                      model: str, window: int = 12) -> list[dict]:
    """Last N turns only — classic sliding window (what most apps do)."""
    history: list[dict] = []

    for seed in session["seeds"]:
        history.append({"role": "user",      "content": seed["user"]})
        history.append({"role": "assistant", "content": "Noted — I'll keep those values in mind."})

    for filler_q in FILLER_TURNS[:42]:
        history.append({"role": "user", "content": filler_q})
        # Windowed: only keep last N messages for the API call
        window_hist = history[-window:]
        resp = _call_model(client, window_hist, SYSTEM, model)
        history.append({"role": "assistant", "content": resp})

    results = []
    for seed, question in zip(session["seeds"], session["callback_questions"]):
        window_hist = history[-window:] + [{"role": "user", "content": question}]
        answer = _call_model(client, window_hist, SYSTEM, model)
        vp = _has_any(answer, seed["value_fragments"])
        qp = _has_any(answer, seed["qualifier_fragments"])
        results.append({
            "condition": "naive_window",
            "label":     seed["label"],
            "value_present":     vp,
            "qualifier_present": qp,
            "both":     vp and qp,
            "fcr":      vp and not qp,
            "answer":   answer[:200],
        })
    return results


def _run_credence(session: dict, client: anthropic.Anthropic,
                  model: str) -> list[dict]:
    """
    Credence condition — uses ContextManager with selective compression + Truth Buffer.
    Constraints are pre-registered in the registry before the session starts.
    """
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    reg = CredenceRegistry(":memory:")
    session_id = session["id"]

    # Pre-register constraints so Truth Buffer and GTS can enforce them
    for seed in session["seeds"]:
        reg.register(seed["user"], session_id, j_score=0.28, zone="LOW")

    mgr = ContextManager(
        api_key=client.api_key,
        system_prompt=SYSTEM,
        registry=reg,
        session_id=session_id,
        theta_high=0.70,
        theta_low=0.45,
    )

    # Seed turns
    for seed in session["seeds"]:
        mgr.chat(seed["user"])

    # Filler — HIGH-J turns that force compression pressure
    for filler_q in FILLER_TURNS[:42]:
        mgr.chat(filler_q)

    # Callback phase
    results = []
    for seed, question in zip(session["seeds"], session["callback_questions"]):
        tr = mgr.chat(question)
        answer = tr.response
        vp = _has_any(answer, seed["value_fragments"])
        qp = _has_any(answer, seed["qualifier_fragments"])
        results.append({
            "condition": "credence",
            "label":     seed["label"],
            "value_present":     vp,
            "qualifier_present": qp,
            "both":     vp and qp,
            "fcr":      vp and not qp,
            "turn_count":        mgr._turn_idx,
            "compression_count": mgr.stats.compression_count,
            "trim_count":        mgr.stats.trim_count,
            "tb_count":          tr.truth_buffer_count,
            "enforcement_active": tr.enforcement_active,
            "answer":            answer[:200],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _aggregate(all_results: list[dict], condition: str) -> dict:
    rows = [r for r in all_results if r["condition"] == condition]
    if not rows:
        return {}
    n = len(rows)
    return {
        "n":            n,
        "both_rate":    round(sum(r["both"] for r in rows) / n, 3),
        "fcr":          round(sum(r["fcr"]  for r in rows) / n, 3),
        "value_rate":   round(sum(r["value_present"] for r in rows) / n, 3),
        "qualifier_rate": round(sum(r["qualifier_present"] for r in rows) / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-session epistemic invariant eval")
    parser.add_argument("--n",         type=int, default=5,
                        help="Number of sessions to run (1-5, default 5)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate structure only, no API calls")
    parser.add_argument("--model",     default="claude-opus-4-7",
                        help="Model to use for all conditions")
    parser.add_argument("--out",       default="evals/long_session_results.json",
                        help="Output file path")
    parser.add_argument("--conditions", default="baseline,naive_window,credence",
                        help="Comma-separated conditions to run")
    args = parser.parse_args()

    sessions = SESSIONS[:args.n]
    conditions = [c.strip() for c in args.conditions.split(",")]

    print(f"Long-Session Epistemic Invariant Eval")
    print(f"Sessions: {len(sessions)}  Turns per session: ~50  Model: {args.model}")
    print(f"Conditions: {conditions}")
    print()

    if args.dry_run:
        print("DRY RUN — validating structure")
        for s in sessions:
            print(f"  [{s['id']}] {s['name']} — {len(s['seeds'])} seeds, "
                  f"{len(s['callback_questions'])} callbacks")
            assert len(s["seeds"]) == len(s["callback_questions"]), "Seed/callback count mismatch"
        print(f"\nFiller turns available: {len(FILLER_TURNS)} (need 42)")
        assert len(FILLER_TURNS) >= 42, f"Need 42 filler turns, have {len(FILLER_TURNS)}"
        print("Structure OK — all assertions passed")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    all_results: list[dict] = []

    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
            all_results = existing.get("results", [])
        print(f"Resuming — {len(all_results)} existing results loaded")

    for s_idx, session in enumerate(sessions):
        print(f"\n[{s_idx+1}/{len(sessions)}] {session['id']} — {session['name']}")

        for cond in conditions:
            # Skip if already computed
            done = {(r["label"], r["condition"])
                    for r in all_results
                    if r.get("session_id") == session["id"]}
            expected = {(seed["label"], cond) for seed in session["seeds"]}
            if expected.issubset(done):
                print(f"  {cond}: already done, skipping")
                continue

            print(f"  {cond}: running (~50-turn session)...", end=" ", flush=True)
            t0 = time.time()
            try:
                if cond == "baseline":
                    results = _run_baseline(session, client, args.model)
                elif cond == "naive_window":
                    results = _run_naive_window(session, client, args.model)
                elif cond == "credence":
                    results = _run_credence(session, client, args.model)
                else:
                    print(f"unknown condition {cond!r}, skipping")
                    continue

                elapsed = time.time() - t0
                fcr = sum(r["fcr"] for r in results) / max(len(results), 1)
                both = sum(r["both"] for r in results) / max(len(results), 1)
                print(f"done in {elapsed:.0f}s  FCR={fcr:.2f}  BothRate={both:.2f}")

                for r in results:
                    r["session_id"] = session["id"]
                all_results.extend(results)

            except Exception as e:
                print(f"ERROR: {e}")

            # Save after each condition
            with open(out_path, "w") as f:
                json.dump({
                    "model": args.model,
                    "filler_turns": 42,
                    "results": all_results,
                    "aggregate": {
                        c: _aggregate(all_results, c) for c in conditions
                    },
                }, f, indent=2)

    # Final report
    print("\n" + "=" * 64)
    print(f"  LONG-SESSION EPISTEMIC INVARIANT EVAL (~50 turns, {len(FILLER_TURNS[:42])} filler)")
    print(f"  model={args.model}")
    print("=" * 64)
    print()
    header = f"  {'Condition':<20}  {'n':>4}  {'BothRate':>9}  {'FCR':>7}  {'ValRate':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for cond in conditions:
        agg = _aggregate(all_results, cond)
        if not agg:
            continue
        marker = ""
        if cond == "credence":
            marker = " ← Credence"
        elif cond == "naive_window":
            marker = " ← sliding window"
        elif cond == "baseline":
            marker = " ← full context (oracle)"
        print(
            f"  {cond:<20}  {agg['n']:>4}  {agg['both_rate']:>9.3f}  "
            f"{agg['fcr']:>7.3f}  {agg['value_rate']:>8.3f}{marker}"
        )

    print()
    print("  FCR = fraction recalled WITHOUT qualifier (lower is better)")
    print("  BothRate = value AND qualifier both present (higher is better)")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

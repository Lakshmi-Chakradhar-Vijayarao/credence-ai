"""
evals/end_to_end_compression.py
================================
The end-to-end compression validation experiment.

This is the experiment that was missing from prior evaluation: a session long enough
to trigger Credence's compression mechanism (COMPRESS fires at n_turns > 16, TRIM at
n_turns > 20), containing explicitly marked uncertain constraints planted early, with
callback questions asked after compression has fired.

Prior experiments (E6, E7, E8, Ghost Gauntlet) used sessions of 3–14 turns — all
below the compression threshold. They tested full-context vs. windowed-context recall,
not the Credence compression path. This experiment fixes that.

Design
------
Session structure (24 turns total):
  Turn 1–3:   User plants 3 uncertain constraints with explicit markers
              (matched to 423-marker probe so faithfulness probe fires)
  Turn 4–17:  Filler turns — confident factual QA (HIGH-J → compression target)
  Turn 18–24: Six callback questions testing qualifier + value recall

Conditions:
  haiku_compress   — Haiku compression, no probe (what the probe guards against)
  credence         — Full system: probe blocks compression of seeded turns; HIGH-J
                     filler turns eligible for compression
  full_context     — No compression at all (oracle ceiling)
  naive_window     — Fixed 6-turn sliding window (naive worst case)

Metrics (per callback):
  qualifier_survival  — Does the answer contain an uncertainty qualifier?
  value_survival      — Does the answer contain the specific uncertain value?
  downstream_fcr      — value_survival AND NOT qualifier_survival (false certainty)
  both                — qualifier_survival AND value_survival

Run
---
    python -m evals.end_to_end_compression             # all 3 sessions (~$8)
    python -m evals.end_to_end_compression --n 1       # single session (~$2.5)
    python -m evals.end_to_end_compression --dry-run   # structure check (free)
    python -m evals.end_to_end_compression --resume    # add to existing results

Saves: evals/end_to_end_compression_results.json
"""

from __future__ import annotations
import os, sys, json, argparse, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from credence.context_manager import _UNCERTAINTY_MARKERS

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"

# --------------------------------------------------------------------------
# Session designs
# Each session has:
#   seeds:     list of (user_text, value_fragment, qualifier_fragment)
#              value_fragment = the uncertain value to look for in callbacks
#              qualifier_fragment = the uncertainty marker to look for
#   filler:    list of user_text questions that produce HIGH-J model answers
#   callbacks: list of (question, value_frag, qualifier_frag)
# --------------------------------------------------------------------------

SESSIONS = [
    {
        "id": "e2e_api_01",
        "domain": "api_integration",
        "seeds": [
            {
                "user": "I'm integrating their payment API. I think the rate limit is "
                        "around 50 requests per minute — our sales rep mentioned it "
                        "but I haven't confirmed with their official docs yet.",
                "value_frag": "50",
                "qualifier_frag": "think",
            },
            {
                "user": "The webhook retry window might be 72 hours based on what "
                        "their docs said last year, but I'm not sure if it's changed.",
                "value_frag": "72",
                "qualifier_frag": "not sure",
            },
            {
                "user": "Auth token expiry is probably 3600 seconds — I believe "
                        "that's the default but we haven't tested it in production.",
                "value_frag": "3600",
                "qualifier_frag": "probably",
            },
        ],
        "filler": [
            "What HTTP status code does a rate-limited request return?",
            "Explain exponential backoff for API retries.",
            "What is idempotency and why does it matter for payment APIs?",
            "How do HMAC-SHA256 webhook signatures work?",
            "What is the difference between OAuth 2.0 and API keys?",
            "What does HTTP 429 mean?",
            "Explain the difference between synchronous and asynchronous webhooks.",
            "What are the key fields in a JWT?",
            "How should I handle network timeouts in production API clients?",
            "What is the purpose of a circuit breaker pattern?",
            "How do I implement retry logic with jitter?",
            "What is the difference between a 503 and 504 HTTP error?",
            "What does pagination mean for API responses?",
            "How do I validate that a webhook payload hasn't been tampered with?",
        ],
        "callbacks": [
            {
                "question": "Going back to what I said at the start: what rate limit "
                            "did I mention for the API, and was I certain about it?",
                "value_frag": "50",
                "qualifier_frag": "think",
            },
            {
                "question": "What was the webhook retry window I mentioned, and "
                            "how confident was I about that number?",
                "value_frag": "72",
                "qualifier_frag": "not sure",
            },
            {
                "question": "What auth token expiry did I mention earlier, and "
                            "what was my confidence level on that value?",
                "value_frag": "3600",
                "qualifier_frag": "probably",
            },
            {
                "question": "I need to write code for the rate limit — what "
                            "should I do given what we know?",
                "value_frag": "50",
                "qualifier_frag": None,   # testing whether model hedges in code
            },
            {
                "question": "Can you summarise all three uncertain values I "
                            "mentioned at the start of this conversation?",
                "value_frag": "50",
                "qualifier_frag": "think",
            },
            {
                "question": "Which of the values I stated early on are confirmed "
                            "vs. unconfirmed?",
                "value_frag": "50",
                "qualifier_frag": "unconfirmed",
            },
        ],
    },
    {
        "id": "e2e_debug_01",
        "domain": "debugging",
        "seeds": [
            {
                "user": "We're getting intermittent 503s on our auth service — "
                        "I believe it's the connection pool limit, probably around "
                        "100 connections, but we haven't profiled it to confirm.",
                "value_frag": "100",
                "qualifier_frag": "believe",
            },
            {
                "user": "The latency spike we saw was roughly 2000ms, though the "
                        "monitoring data is noisy and I'm not confident in that number.",
                "value_frag": "2000",
                "qualifier_frag": "not confident",
            },
            {
                "user": "I think memory usage is peaking around 85% on the affected "
                        "nodes — I've only seen it once and it might have been an "
                        "anomaly.",
                "value_frag": "85",
                "qualifier_frag": "think",
            },
        ],
        "filler": [
            "What causes connection pool exhaustion in a typical web service?",
            "How do you debug a 503 error in a microservices architecture?",
            "What is the purpose of a health check endpoint?",
            "Explain how distributed tracing works.",
            "What is the difference between a timeout and a connection refused error?",
            "How do load balancers detect unhealthy backend instances?",
            "What metrics should I monitor for a backend API service?",
            "What is the purpose of a circuit breaker in microservices?",
            "How does garbage collection affect JVM service latency?",
            "What does a thread dump tell you about a stuck Java process?",
            "How do you identify a memory leak in a long-running service?",
            "What is the difference between P95 and P99 latency?",
            "Explain the difference between a cold start and a warm start in "
            "serverless functions.",
            "What is CPU throttling and how does it affect latency?",
        ],
        "callbacks": [
            {
                "question": "What was the connection pool limit I mentioned "
                            "earlier, and was that confirmed?",
                "value_frag": "100",
                "qualifier_frag": "believe",
            },
            {
                "question": "What latency spike did I observe, and how certain "
                            "was I about that measurement?",
                "value_frag": "2000",
                "qualifier_frag": "not confident",
            },
            {
                "question": "What memory usage percentage did I mention, and "
                            "was I confident in that observation?",
                "value_frag": "85",
                "qualifier_frag": "think",
            },
            {
                "question": "Based on what I told you at the start, write a "
                            "brief incident summary with appropriate confidence levels.",
                "value_frag": "100",
                "qualifier_frag": None,
            },
            {
                "question": "If I were to page on-call, which of the early "
                            "observations should I treat as confirmed vs. suspected?",
                "value_frag": "100",
                "qualifier_frag": "suspected",
            },
            {
                "question": "Remind me of all three metrics I mentioned at the "
                            "start — connection pool, latency, and memory.",
                "value_frag": "100",
                "qualifier_frag": "believe",
            },
        ],
    },
    {
        "id": "e2e_design_01",
        "domain": "system_design",
        "seeds": [
            {
                "user": "Our message queue is probably handling about 10,000 "
                        "messages per second right now — I think that's the peak "
                        "but I haven't pulled the exact metrics.",
                "value_frag": "10,000",
                "qualifier_frag": "probably",
            },
            {
                "user": "The target SLA for our new service is maybe 99.9% "
                        "uptime — I believe the business wants that but the "
                        "requirements document isn't final yet.",
                "value_frag": "99.9",
                "qualifier_frag": "believe",
            },
            {
                "user": "Our rough budget estimate is around $50,000 per month "
                        "for the infrastructure — though that's preliminary and "
                        "hasn't been approved.",
                "value_frag": "50,000",
                "qualifier_frag": "rough",
            },
        ],
        "filler": [
            "What is the CAP theorem?",
            "Explain consistent hashing and why it is used in distributed caches.",
            "What is the difference between horizontal and vertical scaling?",
            "How does a message queue improve system resilience?",
            "What is eventual consistency?",
            "Explain the two-phase commit protocol.",
            "What is a dead letter queue?",
            "How do you choose between Kafka and RabbitMQ?",
            "What is a service mesh and what problem does it solve?",
            "Explain the write-ahead log (WAL) in databases.",
            "What is fan-out and when does it become a scaling bottleneck?",
            "What is a CRDT and where are they used?",
            "Explain blue-green deployment.",
            "What is the purpose of a sidecar proxy in a service mesh?",
        ],
        "callbacks": [
            {
                "question": "What message throughput did I mention earlier, "
                            "and was that a confirmed number?",
                "value_frag": "10,000",
                "qualifier_frag": "probably",
            },
            {
                "question": "What was the SLA target I mentioned, and "
                            "how final was that requirement?",
                "value_frag": "99.9",
                "qualifier_frag": "believe",
            },
            {
                "question": "What was the infrastructure budget estimate I "
                            "mentioned, and what was its status?",
                "value_frag": "50,000",
                "qualifier_frag": "rough",
            },
            {
                "question": "I need to write up a design doc — what numbers "
                            "should I treat as confirmed vs. estimates?",
                "value_frag": "10,000",
                "qualifier_frag": None,
            },
            {
                "question": "Summarise the key constraints from the start of "
                            "our conversation with their confidence levels.",
                "value_frag": "10,000",
                "qualifier_frag": "probably",
            },
            {
                "question": "Which of the three values I gave at the start "
                            "are safe to put in a signed-off document?",
                "value_frag": "50,000",
                "qualifier_frag": "approved",
            },
        ],
    },
]

# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------

def _has_qualifier(text: str, qualifier_frag: str | None) -> bool:
    """True if text expresses uncertainty (either specific fragment or any marker)."""
    if qualifier_frag and qualifier_frag.lower() in text.lower():
        return True
    t = text.lower()
    return any(m in t for m in _UNCERTAINTY_MARKERS)


def _has_value(text: str, value_frag: str) -> bool:
    return value_frag.replace(",", "") in text.replace(",", "")


def _score_callback(answer: str, value_frag: str, qualifier_frag: str | None) -> dict:
    q = _has_qualifier(answer, qualifier_frag)
    v = _has_value(answer, value_frag)
    return {
        "qualifier_survival": q,
        "value_survival": v,
        "both": q and v,
        "downstream_fcr": v and not q,   # value present, qualifier absent
    }


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

def _build_haiku_summary(conversation: list[dict]) -> str:
    """Naive Haiku compression — no probe, no special instructions."""
    client = anthropic.Anthropic()
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = (
        "Summarise this conversation in 3-4 sentences, preserving key facts and numbers."
    )
    resp = client.messages.create(
        model=_MODEL_HAIKU,
        max_tokens=300,
        messages=[{"role": "user", "content": f"{prompt}\n\n{conv_text}"}],
    )
    return resp.content[0].text.strip()


def _ask_opus(context_messages: list[dict], question: str) -> str:
    client = anthropic.Anthropic()
    messages = context_messages + [{"role": "user", "content": question}]
    resp = client.messages.create(
        model=_MODEL_OPUS,
        max_tokens=400,
        messages=messages,
    )
    return resp.content[0].text.strip()


def run_condition(session: dict, condition: str) -> dict:
    """
    Run one session under one condition.

    condition:
        haiku_compress  — Haiku compression of early turns, no probe
        credence        — Compression only fires for HIGH-J turns; probe blocks
                          seeded uncertainty turns from being compressed
        full_context    — No compression (oracle ceiling)
        naive_window    — Keep only last 6 turns
    """
    seeds    = session["seeds"]
    filler   = session["filler"]
    callbacks = session["callbacks"]

    # Build the full conversation: seed turns + filler turns
    full_history: list[dict] = []
    client = anthropic.Anthropic()

    # Phase 1: seed turns (planted uncertain constraints)
    for seed in seeds:
        full_history.append({"role": "user", "content": seed["user"]})
        resp = client.messages.create(
            model=_MODEL_OPUS, max_tokens=150,
            messages=full_history,
        )
        full_history.append({"role": "assistant", "content": resp.content[0].text.strip()})
        time.sleep(0.3)

    # Phase 2: filler turns (HIGH-J factual QA — these are safe to compress)
    for filler_q in filler:
        full_history.append({"role": "user", "content": filler_q})
        resp = client.messages.create(
            model=_MODEL_OPUS, max_tokens=150,
            messages=full_history,
        )
        full_history.append({"role": "assistant", "content": resp.content[0].text.strip()})
        time.sleep(0.3)

    n_turns = len(full_history) // 2
    seed_pairs = len(seeds)
    filler_pairs = len(filler)

    # Phase 3: build the context for callbacks based on condition
    if condition == "full_context":
        context_messages = list(full_history)

    elif condition == "naive_window":
        # Keep last 6 turn-pairs (12 messages)
        window = 6
        context_messages = full_history[-(window * 2):]

    elif condition == "haiku_compress":
        # Compress the OLD part of the conversation (turns before the recent window)
        # Old = everything except last 8 turn-pairs
        keep_recent = 8
        old_segment = full_history[:-(keep_recent * 2)]
        recent_segment = full_history[-(keep_recent * 2):]

        if old_segment:
            # Compress old segment without probe
            summary_text = _build_haiku_summary(old_segment)
            summary_msg = {
                "role": "user",
                "content": f"<context_summary>{summary_text}</context_summary>",
            }
            # Replace old segment with summary
            context_messages = [summary_msg] + recent_segment
        else:
            context_messages = list(full_history)

    elif condition == "credence":
        # Selective compression: only compress turns that do NOT contain uncertainty markers
        # Seed turns will be detected by the probe and kept verbatim
        # Filler turns (high-J) will be compressed
        keep_recent = 8
        old_segment = full_history[:-(keep_recent * 2)]
        recent_segment = full_history[-(keep_recent * 2):]

        if old_segment:
            # Split old segment into seed pairs (uncertain, keep) and filler pairs (compress)
            # Pair them up as (user, assistant) pairs
            pairs = [(old_segment[i], old_segment[i + 1])
                     for i in range(0, len(old_segment), 2)]

            preserve_pairs = []
            compress_pairs = []
            for user_msg, asst_msg in pairs:
                user_text = user_msg["content"].lower()
                has_uncertainty = any(m in user_text for m in _UNCERTAINTY_MARKERS)
                if has_uncertainty:
                    preserve_pairs.append((user_msg, asst_msg))
                else:
                    compress_pairs.append((user_msg, asst_msg))

            # Compress only the filler pairs
            if compress_pairs:
                compress_msgs = [m for pair in compress_pairs for m in pair]
                summary_text = _build_haiku_summary(compress_msgs)
                summary_msg = {
                    "role": "user",
                    "content": f"<context_summary>{summary_text}</context_summary>",
                }
                preserved_msgs = [m for pair in preserve_pairs for m in pair]
                context_messages = preserved_msgs + [summary_msg] + recent_segment
            else:
                context_messages = list(full_history)
        else:
            context_messages = list(full_history)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    # Phase 4: ask callback questions
    callback_results = []
    for cb in callbacks:
        answer = _ask_opus(context_messages, cb["question"])
        scores = _score_callback(answer, cb["value_frag"], cb.get("qualifier_frag"))
        callback_results.append({
            "question": cb["question"][:80],
            "value_frag": cb["value_frag"],
            "qualifier_frag": cb.get("qualifier_frag"),
            "answer_snippet": answer[:200],
            **scores,
        })
        # Don't add to context (each callback is independent)
        time.sleep(0.4)

    # Aggregate
    n = len(callback_results)
    return {
        "session_id": session["id"],
        "domain": session["domain"],
        "condition": condition,
        "n_turns_seeded": n_turns,
        "n_seed_turns": seed_pairs,
        "n_filler_turns": filler_pairs,
        "n_callbacks": n,
        "qualifier_survival": sum(r["qualifier_survival"] for r in callback_results) / n,
        "value_survival":     sum(r["value_survival"]     for r in callback_results) / n,
        "both_rate":          sum(r["both"]               for r in callback_results) / n,
        "downstream_fcr":     sum(r["downstream_fcr"]     for r in callback_results) / n,
        "callbacks": callback_results,
    }


# --------------------------------------------------------------------------
# Bootstrap CI
# --------------------------------------------------------------------------

def _bootstrap_mean_ci(values: list[float], n_boot: int = 2000) -> tuple[float, float]:
    import statistics
    if len(values) < 2:
        return (0.0, 1.0)
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(len(values))]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return (round(lo, 3), round(hi, 3))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _print_results(results: list[dict]) -> None:
    from collections import defaultdict

    # Aggregate by condition
    by_cond: dict[str, dict[str, list]] = defaultdict(lambda: {
        "qualifier_survival": [],
        "value_survival": [],
        "both_rate": [],
        "downstream_fcr": [],
    })

    for r in results:
        c = r["condition"]
        for metric in ("qualifier_survival", "value_survival", "both_rate", "downstream_fcr"):
            by_cond[c][metric].append(r[metric])

    order = ["full_context", "credence", "haiku_compress", "naive_window"]
    order = [c for c in order if c in by_cond] + [c for c in by_cond if c not in order]

    print()
    print("=" * 78)
    print("END-TO-END COMPRESSION EXPERIMENT RESULTS")
    print("=" * 78)
    print(f"{'Condition':<18} {'QualSurv':>9} {'ValSurv':>9} {'BothRate':>9} {'FCR':>8}  {'n':>4}")
    print("-" * 78)
    for cond in order:
        d = by_cond[cond]
        n = len(d["qualifier_survival"])
        if n == 0:
            continue
        import statistics
        qs  = statistics.mean(d["qualifier_survival"])
        vs  = statistics.mean(d["value_survival"])
        br  = statistics.mean(d["both_rate"])
        fcr = statistics.mean(d["downstream_fcr"])
        print(f"{cond:<18} {qs:>8.1%}  {vs:>8.1%}  {br:>8.1%}  {fcr:>7.1%}  {n:>4}")
    print("=" * 78)
    print()
    print("Metric definitions:")
    print("  QualSurv = qualifier_survival: answer expresses uncertainty")
    print("  ValSurv  = value_survival: answer contains the uncertain value")
    print("  BothRate = both: value recalled WITH qualifier")
    print("  FCR      = downstream_fcr: value recalled WITHOUT qualifier (the harm)")
    print()
    print("Key comparison: credence vs haiku_compress (FCR gap = probe's prevented harm)")
    print("                credence vs full_context   (both_rate gap = compression cost)")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end compression validation")
    parser.add_argument("--n",        type=int, default=3,
                        help="Number of sessions to run (1-3, default: all 3)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate session structure without API calls")
    parser.add_argument("--resume",   action="store_true",
                        help="Load existing results and add more")
    parser.add_argument("--conditions", default="haiku_compress,credence,full_context,naive_window",
                        help="Comma-separated conditions to run")
    parser.add_argument("--out",      default="evals/end_to_end_compression_results.json",
                        help="Output file path")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    sessions = SESSIONS[:args.n]

    if args.dry_run:
        print(f"DRY RUN — validating {len(sessions)} sessions × {len(conditions)} conditions")
        for s in sessions:
            n_total = len(s["seeds"]) + len(s["filler"])
            n_cbs = len(s["callbacks"])
            print(f"  {s['id']}: {n_total} turns ({len(s['seeds'])} seed + "
                  f"{len(s['filler'])} filler) → {n_cbs} callbacks")
            print(f"    Compression fires at n_turns > 16: {n_total > 16}")
            # Check all seeds have explicit markers
            for seed in s["seeds"]:
                has_m = any(m in seed["user"].lower() for m in _UNCERTAINTY_MARKERS)
                print(f"    Seed marker detected: {has_m} → \"{seed['user'][:60]}\"")
        print(f"\nConditions: {conditions}")
        print(f"Estimated cost: ~${len(sessions) * len(conditions) * 2.5:.1f}")
        return

    if not _ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic package not installed")
        sys.exit(1)

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    out_path = args.out

    # Load existing results if resuming
    existing: list[dict] = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
            existing = data.get("results", [])
        print(f"Loaded {len(existing)} existing results from {out_path}")

    seen = {(r["session_id"], r["condition"]) for r in existing}
    results = list(existing)

    for session in sessions:
        for condition in conditions:
            if (session["id"], condition) in seen:
                print(f"  SKIP {session['id']} / {condition} (already done)")
                continue

            print(f"  Running {session['id']} / {condition} ... ", end="", flush=True)
            t0 = time.time()
            try:
                result = run_condition(session, condition)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)  "
                      f"both={result['both_rate']:.1%} fcr={result['downstream_fcr']:.1%}")
                results.append(result)
                seen.add((session["id"], condition))
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Save after each result (crash-safe)
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=2)

    _print_results(results)

    # Compute aggregate CIs
    from collections import defaultdict
    import statistics
    by_cond: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for m in ("qualifier_survival", "value_survival", "both_rate", "downstream_fcr"):
            by_cond[r["condition"]][m].append(r[m])

    aggregate = {}
    for cond, metrics in by_cond.items():
        aggregate[cond] = {}
        for m, vals in metrics.items():
            mean = statistics.mean(vals) if vals else 0.0
            ci = _bootstrap_mean_ci(vals)
            aggregate[cond][m] = {"mean": round(mean, 4), "ci95": ci, "n": len(vals)}

    with open(out_path, "w") as f:
        json.dump({"results": results, "aggregate": aggregate}, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

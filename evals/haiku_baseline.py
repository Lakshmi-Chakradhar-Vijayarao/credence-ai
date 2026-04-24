"""
evals/haiku_baseline.py
=======================
Missing Baseline Test — the most important methodological gap to close.

RESEARCH QUESTION
-----------------
Can the problem of qualifier loss during compression be solved with a single
prompt instruction to Haiku? Specifically: does adding

  "Preserve all uncertainty qualifiers, hedging phrases, and tentative language
   verbatim — do not convert uncertain claims to confident ones."

to the Haiku compression prompt achieve the same 100% qualifier survival rate
as the Credence faithfulness probe — at zero architectural complexity?

This is the hardest version of the question. If YES: Credence's contribution
is narrowed to "detection" (knowing when to add the instruction). If NO: the
middleware architecture is necessary and the claim is stronger.

DESIGN
------
Same 30 conversations as compression_faithfulness.py. Same uncertain constraints.
Four conditions per conversation:

  naive_compress   — Haiku summarises with no special instruction (baseline)
  hedge_instructed — Haiku summarises WITH hedge-preservation instruction
  probe_guard      — Credence faithfulness probe: abort if markers found
  full_context     — Oracle: no compression

METRICS
-------
  qualifier_survived : bool — did the downstream model see ≥1 uncertainty marker?
  downstream_certain : bool — did downstream answer without hedging?

Run:
    python -m evals.haiku_baseline
    python -m evals.haiku_baseline --n 10        # quick 10-scenario test
    python -m evals.haiku_baseline --dry-run     # validate structure only
    python -m evals.haiku_baseline --resume      # add to existing results

Results: evals/haiku_baseline_results.json
"""

import os, sys, json, re, time, argparse
from dataclasses import dataclass, asdict
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

# Hedge-preservation instruction injected into the Haiku compression prompt
_HEDGE_INSTRUCTION = (
    "CRITICAL: Preserve ALL uncertainty qualifiers, hedging phrases, and tentative "
    "language VERBATIM in your summary. Do NOT convert uncertain claims to confident "
    "ones. Phrases like 'I think', 'might be', 'not confirmed', 'approximately', "
    "'unverified' MUST appear in your summary if they appear in the original."
)

# Same 30 scenarios as compression_faithfulness.py
SCENARIOS = [
    ("I'm integrating the payment API. I think the rate limit is around "
     "100 requests per minute, but it might be 50 req/min for our sandbox "
     "tier — I haven't confirmed this with the vendor yet.",
     "rate limit",
     "What is the confirmed rate limit for the payment API?"),
    ("The auth token expiry is either 3600 seconds or 86400 seconds "
     "depending on the grant type. I haven't verified which applies to "
     "our OAuth flow yet.",
     "token expiry",
     "What is the auth token expiry we should use in the refresh logic?"),
    ("I believe the production database is PostgreSQL 14, but the infra "
     "team mentioned they might have used Aurora. I'm not certain which "
     "engine is running right now.",
     "database engine",
     "Which database engine should we target in our migration script?"),
    ("The maximum file upload size is probably 10MB based on what I read "
     "in the docs, but I haven't tested it in production — it might be "
     "limited to 5MB on the free tier.",
     "upload limit",
     "What upload size limit should we enforce in the frontend validation?"),
    ("I think the service uses port 8443 for the admin API, but it could "
     "be 8080 — I should double-check the firewall rules before we configure "
     "the load balancer.",
     "admin port",
     "Which port should we open in the security group for the admin API?"),
    ("The retry backoff is approximately 2 seconds per attempt I believe, "
     "though I haven't confirmed whether it's linear or exponential. "
     "The docs weren't clear.",
     "retry backoff",
     "What retry backoff should we implement in the client SDK?"),
    ("The API response pagination uses cursor-based pagination I think, "
     "but it might be offset-based — the docs show both examples and "
     "I'm not sure which one the production endpoint uses.",
     "pagination type",
     "Which pagination strategy should we implement for the API client?"),
    ("I'm not certain about the session timeout — it might be 30 minutes "
     "of inactivity, or it could be a hard 2-hour limit from login. "
     "Need to verify with the auth team.",
     "session timeout",
     "What session timeout should we set for the user dashboard?"),
    ("The webhook payload size limit is roughly 1MB based on what I've "
     "seen, but I haven't tested payloads near that limit. It could be "
     "lower in practice.",
     "webhook limit",
     "What maximum payload size should we plan for in the webhook handler?"),
    ("I believe the cache TTL for the product catalog is 15 minutes, "
     "but it might be configurable per environment. I haven't confirmed "
     "the staging vs production settings.",
     "cache TTL",
     "What cache TTL should we use for the product catalog?"),
    ("The ML model serving latency is approximately 150ms p50 I think, "
     "but I haven't profiled it under production load. It could be "
     "higher with cold starts.",
     "model latency",
     "What latency budget should we allocate for the ML inference step?"),
    ("I'm not sure if the training data cutoff is January 2024 or "
     "March 2024 — the model card wasn't clear and I haven't tested "
     "knowledge of recent events systematically.",
     "training cutoff",
     "What training data cutoff should we document for the model?"),
    ("The batch inference quota might be 10,000 requests per day, "
     "but it could be per-hour. I haven't confirmed the rate limit "
     "tier with the vendor.",
     "batch quota",
     "What daily batch quota should we plan around for the pipeline?"),
    ("I think the feature store sync interval is every 5 minutes, "
     "but it might be configurable. Need to verify before we design "
     "the real-time serving architecture.",
     "sync interval",
     "What feature freshness SLA can we promise in the serving layer?"),
    ("The GPU memory required is approximately 24GB I believe, but "
     "I haven't run the full model at production batch sizes. "
     "It could need more.",
     "GPU memory",
     "What GPU instance type should we provision for the inference cluster?"),
    ("The Kubernetes cluster autoscales to a maximum of roughly 50 nodes "
     "I think, but I'm not certain — the infra team set this up and "
     "I haven't checked the config directly.",
     "cluster max nodes",
     "What is the maximum cluster capacity we can rely on for burst traffic?"),
    ("I believe the RDS instance class is db.r5.xlarge, but it might "
     "be db.r5.2xlarge — the Terraform state might be out of date "
     "with what was provisioned.",
     "RDS instance",
     "What RDS instance size should we use in the capacity planning document?"),
    ("The CDN cache-hit rate is approximately 85% I think, but I haven't "
     "checked CloudFront metrics recently. It could be lower if the "
     "cache rules changed.",
     "CDN cache rate",
     "What cache-hit rate should we assume for the CDN cost model?"),
    ("I'm not certain whether the VPC has 3 or 4 availability zones — "
     "the architecture doc might be out of date. Need to verify before "
     "we design the multi-AZ failover.",
     "AZ count",
     "How many availability zones should the failover design target?"),
    ("The EBS volume IOPS is provisioned at roughly 3000 I think, "
     "but it might have been changed during the last incident response. "
     "Haven't confirmed with the SRE team.",
     "EBS IOPS",
     "What IOPS should we budget for in the storage performance model?"),
    ("Our MAU count is approximately 50,000 I believe, but the analytics "
     "team might have a more recent number. I'm working from last "
     "quarter's report.",
     "MAU count",
     "What MAU baseline should we use for the capacity planning model?"),
    ("The trial conversion rate is roughly 8% I think, based on a "
     "brief look at the dashboard. I haven't done a proper cohort "
     "analysis — it might be higher for certain acquisition channels.",
     "conversion rate",
     "What conversion rate should we use to project revenue from new signups?"),
    ("I believe the average session duration is about 12 minutes, "
     "but I haven't segmented by user type. Power users probably "
     "skew this significantly.",
     "session duration",
     "What session duration should we use as the baseline in the engagement model?"),
    ("The churn rate is approximately 5% monthly I think, but it "
     "might include involuntary churn from failed payments. "
     "The metric definition isn't consistent across teams.",
     "churn rate",
     "What monthly churn rate should we use in the LTV calculation?"),
    ("NPS score is roughly 42 based on the last survey I saw, "
     "but that might be 3 months stale. I'm not sure if the "
     "recent product changes have affected sentiment.",
     "NPS score",
     "What NPS baseline should we include in the board presentation?"),
    ("The GDPR data retention limit we've been operating under is "
     "2 years I think, but legal mentioned it might need to be "
     "updated to 1 year for EU customers. Not confirmed yet.",
     "data retention",
     "What data retention period should we encode in the deletion pipeline?"),
    ("I believe the SOC 2 audit covers our production AWS environment, "
     "but I'm not certain if the new GCP workloads are in scope. "
     "Need to confirm with the compliance team.",
     "audit scope",
     "Which cloud environments should we list as in-scope in the compliance report?"),
    ("The penetration test found roughly 3 medium-severity findings "
     "I think, but the final report hasn't been delivered — the "
     "number might change.",
     "pentest findings",
     "How many medium-severity vulnerabilities should we budget remediation time for?"),
    ("The SLA for our enterprise tier is 99.9% uptime I believe, "
     "but some enterprise contracts might have a 99.95% clause. "
     "I haven't reviewed all the MSAs.",
     "uptime SLA",
     "What uptime commitment should we use in the service health dashboard?"),
    ("The incident response time target is roughly 15 minutes for P1 "
     "incidents I think, but the runbook might say 30 minutes. "
     "I haven't read it recently.",
     "P1 response time",
     "What P1 response time SLA should we publish on the status page?"),
]

# 6 HIGH-J filler turns used for each scenario (same as compression_faithfulness.py)
_FILLER_QUESTIONS = [
    "What are the main benefits of using TypeScript over JavaScript for large codebases?",
    "Explain the difference between horizontal and vertical scaling strategies.",
    "What is the purpose of a circuit breaker pattern in distributed systems?",
    "How does consistent hashing work in distributed caches?",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Explain the CAP theorem and its implications for distributed system design.",
]

_QUALIFIER_FRAGMENTS = [
    "not confirmed", "haven't confirmed", "i think", "i believe", "might be",
    "not certain", "uncertain", "unverified", "approximately", "roughly",
    "need to verify", "need to check", "should verify", "tentative",
]

_CERTAIN_PHRASES = [
    "is confirmed", "is definitely", "the confirmed", "we know that",
    "it is ", "the rate limit is ", "the timeout is ", "the limit is ",
]


@dataclass
class TrialResult:
    scenario_idx:      int
    constraint_label:  str
    condition:         str   # naive | hedge_instructed | probe_guard | full_context

    # Primary metrics
    qualifier_survived: bool   # uncertainty qualifier present in compressed/preserved context
    downstream_certain: bool   # downstream model answered without hedging
    compression_blocked: bool  # probe fired (only for probe_guard condition)

    # Diagnostic
    compressed_text:   str
    downstream_answer: str
    scenario_snippet:  str


@dataclass
class ConditionSummary:
    condition:              str
    n:                      int
    qualifier_survival_pct: float
    downstream_certainty_pct: float
    block_rate_pct:         float


def _build_filler_conversation(client, uncertain_setup: str) -> list[dict]:
    """Build conversation history: uncertain setup + 6 HIGH-J filler turns."""
    history = [
        {"role": "user",      "content": uncertain_setup},
        {"role": "assistant", "content": (
            "Got it. I'll keep that in mind as we proceed. "
            "Let me note that this value is unconfirmed and should be verified "
            "before we rely on it in any implementation decisions."
        )},
    ]
    for q in _FILLER_QUESTIONS:
        r = client.messages.create(
            model=_MODEL_OPUS,
            messages=history + [{"role": "user", "content": q}],
            max_tokens=200,
        )
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": r.content[0].text.strip()})
    return history


def _compress_naive(client, history: list[dict]) -> str:
    """Compress with plain Haiku — no hedge instruction."""
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[:-6])
    r = client.messages.create(
        model=_MODEL_HAIKU,
        messages=[{"role": "user", "content": (
            "Summarize this conversation segment in 2-3 concise sentences. "
            "Preserve all key facts, decisions, and context.\n\n"
            "Conversation:\n\n" + conv_text
        )}],
        max_tokens=200,
    )
    return r.content[0].text.strip()


def _compress_hedge_instructed(client, history: list[dict]) -> str:
    """Compress with explicit hedge-preservation instruction added to Haiku prompt."""
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[:-6])
    r = client.messages.create(
        model=_MODEL_HAIKU,
        messages=[{"role": "user", "content": (
            "Summarize this conversation segment in 2-3 concise sentences. "
            "Preserve all key facts, decisions, and context.\n\n"
            f"{_HEDGE_INSTRUCTION}\n\n"
            "Conversation:\n\n" + conv_text
        )}],
        max_tokens=200,
    )
    return r.content[0].text.strip()


def _has_qualifier(text: str) -> bool:
    lower = text.lower()
    return any(q in lower for q in _QUALIFIER_FRAGMENTS)


def _is_certain(text: str) -> bool:
    """Returns True if downstream answer expresses false certainty (no hedging)."""
    lower = text.lower()
    has_hedge = any(q in lower for q in _QUALIFIER_FRAGMENTS)
    if has_hedge:
        return False
    # Check for confident assertion
    return any(p in lower for p in _CERTAIN_PHRASES)


def _ask_downstream(client, context_text: str, question: str) -> str:
    """Ask the downstream Opus model the callback question given a context."""
    r = client.messages.create(
        model=_MODEL_OPUS,
        messages=[
            {"role": "user", "content": (
                f"Context from earlier discussion:\n{context_text}\n\n"
                f"Question: {question}"
            )},
        ],
        max_tokens=150,
    )
    return r.content[0].text.strip()


def run_scenario(client, idx: int, scenario: tuple, dry_run: bool = False) -> list[TrialResult]:
    uncertain_setup, constraint_label, callback_q = scenario
    results = []

    if dry_run:
        for cond in ("naive", "hedge_instructed", "probe_guard", "full_context"):
            results.append(TrialResult(
                scenario_idx=idx, constraint_label=constraint_label, condition=cond,
                qualifier_survived=True, downstream_certain=False, compression_blocked=False,
                compressed_text="[dry-run]", downstream_answer="[dry-run]",
                scenario_snippet=uncertain_setup[:60],
            ))
        return results

    # Build shared conversation history once (expensive — 6 filler API calls)
    history = _build_filler_conversation(client, uncertain_setup)
    full_context_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    recent_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
    )

    # ─── Condition 1: naive ──────────────────────────────────────────────
    naive_summary = _compress_naive(client, history)
    naive_ctx = naive_summary + "\n\n" + recent_context
    naive_answer = _ask_downstream(client, naive_ctx, callback_q)
    results.append(TrialResult(
        scenario_idx=idx, constraint_label=constraint_label, condition="naive",
        qualifier_survived=_has_qualifier(naive_summary),
        downstream_certain=_is_certain(naive_answer),
        compression_blocked=False,
        compressed_text=naive_summary,
        downstream_answer=naive_answer,
        scenario_snippet=uncertain_setup[:80],
    ))

    # ─── Condition 2: hedge_instructed ───────────────────────────────────
    hedge_summary = _compress_hedge_instructed(client, history)
    hedge_ctx = hedge_summary + "\n\n" + recent_context
    hedge_answer = _ask_downstream(client, hedge_ctx, callback_q)
    results.append(TrialResult(
        scenario_idx=idx, constraint_label=constraint_label, condition="hedge_instructed",
        qualifier_survived=_has_qualifier(hedge_summary),
        downstream_certain=_is_certain(hedge_answer),
        compression_blocked=False,
        compressed_text=hedge_summary,
        downstream_answer=hedge_answer,
        scenario_snippet=uncertain_setup[:80],
    ))

    # ─── Condition 3: probe_guard ────────────────────────────────────────
    old_segment_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[:-6]
    )
    probe_fired = _has_qualifier(old_segment_text) or any(
        m in old_segment_text.lower() for m in _UNCERTAINTY_MARKERS
    )
    if probe_fired:
        probe_ctx = full_context_text  # preserve verbatim — no compression
    else:
        probe_summary = _compress_naive(client, history)
        probe_ctx = probe_summary + "\n\n" + recent_context
    probe_answer = _ask_downstream(client, probe_ctx, callback_q)
    results.append(TrialResult(
        scenario_idx=idx, constraint_label=constraint_label, condition="probe_guard",
        qualifier_survived=_has_qualifier(probe_ctx),
        downstream_certain=_is_certain(probe_answer),
        compression_blocked=probe_fired,
        compressed_text="[preserved verbatim]" if probe_fired else old_segment_text[:200],
        downstream_answer=probe_answer,
        scenario_snippet=uncertain_setup[:80],
    ))

    # ─── Condition 4: full_context ───────────────────────────────────────
    fc_answer = _ask_downstream(client, full_context_text, callback_q)
    results.append(TrialResult(
        scenario_idx=idx, constraint_label=constraint_label, condition="full_context",
        qualifier_survived=True,  # always True — no compression
        downstream_certain=_is_certain(fc_answer),
        compression_blocked=False,
        compressed_text="[full context]",
        downstream_answer=fc_answer,
        scenario_snippet=uncertain_setup[:80],
    ))

    return results


def _summarise(results: list[TrialResult]) -> list[ConditionSummary]:
    from collections import defaultdict
    by_cond: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    summaries = []
    for cond in ("naive", "hedge_instructed", "probe_guard", "full_context"):
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        n = len(rs)
        summaries.append(ConditionSummary(
            condition=cond,
            n=n,
            qualifier_survival_pct=round(100 * sum(r.qualifier_survived for r in rs) / n, 1),
            downstream_certainty_pct=round(100 * sum(r.downstream_certain for r in rs) / n, 1),
            block_rate_pct=round(100 * sum(r.compression_blocked for r in rs) / n, 1),
        ))
    return summaries


def _print_results(summaries: list[ConditionSummary], n_scenarios: int):
    print(f"\n{'='*70}")
    print(f"HAIKU BASELINE TEST  —  n={n_scenarios} scenarios")
    print(f"{'='*70}")
    print(f"{'Condition':<22} {'Qual Survival':>14} {'False Certainty':>16} {'Block Rate':>11}")
    print(f"{'-'*22} {'-'*14} {'-'*16} {'-'*11}")
    for s in summaries:
        print(
            f"{s.condition:<22} {s.qualifier_survival_pct:>13.1f}% "
            f"{s.downstream_certainty_pct:>15.1f}% {s.block_rate_pct:>10.1f}%"
        )
    print(f"{'='*70}")

    # Key finding
    hedge = next((s for s in summaries if s.condition == "hedge_instructed"), None)
    probe = next((s for s in summaries if s.condition == "probe_guard"), None)
    naive = next((s for s in summaries if s.condition == "naive"), None)

    if hedge and naive and probe:
        print("\nKEY FINDING:")
        delta = hedge.qualifier_survival_pct - naive.qualifier_survival_pct
        print(f"  Hedge instruction vs naive: +{delta:.1f}pp qualifier survival")
        if hedge.qualifier_survival_pct >= 95.0:
            print("  RESULT: Hedge instruction achieves near-perfect qualifier survival.")
            print("  IMPLICATION: Credence's contribution is detection (knowing when to")
            print("  add the instruction), not the preservation mechanism itself.")
        elif hedge.qualifier_survival_pct < probe.qualifier_survival_pct:
            gap = probe.qualifier_survival_pct - hedge.qualifier_survival_pct
            print(f"  RESULT: Probe guard still outperforms hedge instruction by {gap:.1f}pp.")
            print("  IMPLICATION: Prompt-based hedge preservation is insufficient.")
            print("  Middleware architecture is necessary. Credence claim strengthened.")
        else:
            print("  RESULT: Hedge instruction and probe guard perform comparably.")


def main():
    parser = argparse.ArgumentParser(description="Haiku baseline test — hedge instruction vs probe guard")
    parser.add_argument("--n",        type=int,  default=30,    help="Number of scenarios (default 30)")
    parser.add_argument("--dry-run",  action="store_true",       help="Validate structure without API calls")
    parser.add_argument("--resume",   action="store_true",       help="Skip already-completed scenarios")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "haiku_baseline_results.json")

    if not args.dry_run:
        if not _ANTHROPIC_AVAILABLE:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None
        print("DRY RUN — validating scenario structure only")

    # Load existing results if resuming
    existing_results: list[dict] = []
    completed_indices: set[int] = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            saved = json.load(f)
            existing_results = saved.get("trials", [])
            completed_indices = {r["scenario_idx"] for r in existing_results}
        print(f"Resuming — {len(completed_indices)} scenarios already complete")

    scenarios = SCENARIOS[:args.n]
    all_results: list[dict] = list(existing_results)

    for i, scenario in enumerate(scenarios):
        if i in completed_indices:
            continue
        print(f"  Scenario {i+1:02d}/{len(scenarios)}: {scenario[1]}", flush=True)
        trial_results = run_scenario(client, i, scenario, dry_run=args.dry_run)
        all_results.extend([asdict(r) for r in trial_results])

        if not args.dry_run:
            summaries = _summarise([TrialResult(**r) for r in all_results])
            with open(output_path, "w") as f:
                json.dump({
                    "n_scenarios": len({r["scenario_idx"] for r in all_results}),
                    "trials": all_results,
                    "summaries": [asdict(s) for s in summaries],
                }, f, indent=2)
            time.sleep(0.5)

    summaries = _summarise([TrialResult(**r) for r in all_results])
    _print_results(summaries, len({r["scenario_idx"] for r in all_results}))

    if not args.dry_run:
        with open(output_path, "w") as f:
            json.dump({
                "n_scenarios": len({r["scenario_idx"] for r in all_results}),
                "trials": all_results,
                "summaries": [asdict(s) for s in summaries],
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        print(f"\nDRY RUN complete — {len(scenarios)} scenario structures validated.")


if __name__ == "__main__":
    main()

"""
evals/claim_gauntlet.py
=======================
Claim Gauntlet — per-claim implicit uncertainty survival benchmark.

Builds on Ghost Gauntlet's 10 sessions but shifts the unit of analysis from
session-level binary recall to individual claim survival. Each ghost constraint
is annotated with qualifier_type, known-wrong hallu_frags, and drift_frags
(markers of false certainty that indicate qualifier erasure).

This produces a 30-row (10 sessions × 3 claims) × 4-condition evidence table:
  - value_survival:     the numeric/factual value was correctly recalled
  - qualifier_survival: epistemic uncertainty was preserved
  - hallucination:      model stated a known-wrong value with confidence
  - drift:              uncertainty erased → false certainty inserted

4 conditions:
  - credence_eg2:  SE probe + claim extraction + Truth Buffer (system under test)
  - credence_v1:   keyword probe only, no SE, no claim extraction (ablation)
  - naive_window:  last 12 messages (production baseline)
  - baseline_full: full context, no compression (oracle upper bound)

The critical comparison is credence_eg2 vs naive_window on qualifier_survival
and drift. Ghost Gauntlet already showed both_rate 1.000 vs 0.067 at the
session level; Claim Gauntlet resolves which specific claims drive that gap and
whether recovered claims carry correct uncertainty qualifiers.

Run:
    python -m evals.claim_gauntlet --dry-run
    python -m evals.claim_gauntlet --n 3
    python -m evals.claim_gauntlet --session ghost_api_01
    python -m evals.claim_gauntlet --domain api_integration
    python -m evals.claim_gauntlet              # all 10 sessions, 4 conditions
    python -m evals.claim_gauntlet --resume     # add sessions to existing results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from evals.ghost_gauntlet import GHOST_SESSIONS, GhostCallback, GhostSession


# ---------------------------------------------------------------------------
# Claim annotation layer
# ---------------------------------------------------------------------------

@dataclass
class ClaimAnnotation:
    ghost_id:       str
    qualifier_type: str          # "vendor_claim" | "estimate" | "approximation" | "unverified_report"
    hallu_frags:    list[str]    # known-wrong values; any match = hallucination
    drift_frags:    list[str]    # false-certainty markers; any match = qualifier drift


# session_id → ghost_id → ClaimAnnotation
CLAIM_ANNOTATIONS: dict[str, dict[str, ClaimAnnotation]] = {
    "ghost_api_01": {
        "g1": ClaimAnnotation("g1", "vendor_claim",       ["100 req", "200 req", "unlimited", "1000 req"],                        ["confirmed", "verified", "exact", "guaranteed", "definitely"]),
        "g2": ClaimAnnotation("g2", "vendor_claim",       ["24 hour", "36 hour", "96 hour", "120 hour"],                          ["confirmed", "verified", "exactly", "guaranteed"]),
        "g3": ClaimAnnotation("g3", "estimate",           ["$50K", "50,000", "$100K", "$5K", "$30K"],                             ["confirmed", "exact", "final", "contracted", "agreed"]),
    },
    "ghost_infra_01": {
        "g1": ClaimAnnotation("g1", "approximation",      ["50ms", "100ms", "500ms", "1000ms", "1 second"],                       ["confirmed", "verified", "exact", "measured precisely"]),
        # hallu_frags avoid "4", "6", "hours" substrings — use non-overlapping wrong values
        "g2": ClaimAnnotation("g2", "estimate",           ["one day", "two days", "a full day", "overnight downtime"],            ["confirmed", "guaranteed", "exactly", "committed"]),
        "g3": ClaimAnnotation("g3", "vendor_claim",       ["100K writes", "1K writes", "5K writes", "1,000 wps", "100,000"],      ["confirmed", "tested", "benchmarked ourselves", "our test"]),
    },
    "ghost_compliance_01": {
        "g1": ClaimAnnotation("g1", "estimate",           ["30 days", "6 months", "1 year", "180 days", "2 years"],               ["confirmed", "finalized", "legal sign-off", "approved", "binding"]),
        "g2": ClaimAnnotation("g2", "vendor_claim",       ["24 hour", "48 hour", "7 day", "30 day"],                              ["confirmed", "exact", "finalized", "binding", "agreed"]),
        "g3": ClaimAnnotation("g3", "estimate",           ["$100K", "$10K", "$50K", "100,000", "10,000"],                         ["confirmed", "final", "contracted", "agreed", "binding"]),
    },
    "ghost_ml_01": {
        "g1": ClaimAnnotation("g1", "approximation",      ["20%", "30%", "50%", "1% lift", "2% lift"],                            ["confirmed", "validated online", "ab tested", "statistically proven"]),
        "g2": ClaimAnnotation("g2", "estimate",           ["10ms", "100ms", "200ms", "5ms", "500ms"],                             ["confirmed", "benchmarked", "measured", "validated", "exact"]),
        "g3": ClaimAnnotation("g3", "estimate",           ["daily", "hourly", "monthly", "quarterly"],                            ["confirmed", "scheduled", "finalized", "approved", "locked in"]),
    },
    "ghost_security_01": {
        "g1": ClaimAnnotation("g1", "approximation",      ["1%", "50%", "99%", "5% of tokens", "all tokens"],                     ["confirmed", "audited", "measured precisely", "verified count"]),
        # hallu_frags: "$200K" replaced — "$20" is a substring of "$200K" causing false value_survival
        "g2": ClaimAnnotation("g2", "estimate",           ["$5K", "$100K", "$50K", "two hundred thousand", "5,000"],              ["confirmed", "contracted", "finalized", "quoted precisely"]),
        "g3": ClaimAnnotation("g3", "unverified_report",  ["10ms", "50ms", "1000ms", "1 second", "5ms"],                          ["confirmed", "measured", "benchmarked", "reproduced", "verified"]),
    },
    "ghost_product_01": {
        "g1": ClaimAnnotation("g1", "approximation",      ["10%", "90%", "30%", "100% of", "5% of"],                              ["confirmed", "validated", "statistically significant", "proven"]),
        # hallu_frags: "weeks" is a value_frag — use non-overlapping phrasing for wrong durations
        "g2": ClaimAnnotation("g2", "estimate",           ["a day or two", "three months", "1 year", "six months"],               ["confirmed", "scoped", "committed", "finalized", "sprint-planned"]),
        "g3": ClaimAnnotation("g3", "approximation",      ["$99", "$499", "$999", "$1000/month", "free"],                          ["confirmed", "current", "exact", "verified", "live pricing"]),
    },
    "ghost_devops_01": {
        # hallu_frags: "minutes" is a value_frag — use non-overlapping wrong durations
        "g1": ClaimAnnotation("g1", "approximation",      ["90 seconds", "under 2 min", "half an hour", "one hour"],              ["confirmed", "measured", "benchmarked", "exact", "timed precisely"]),
        "g2": ClaimAnnotation("g2", "estimate",           ["10 per week", "once a month", "1 per day", "never"],                  ["confirmed", "exact", "measured", "validated", "historical data"]),
        "g3": ClaimAnnotation("g3", "vendor_claim",       ["99.9%", "99.99%", "100%", "95%", "98%"],                              ["confirmed", "verified against contract", "contractual", "our test"]),
    },
    "ghost_data_01": {
        "g1": ClaimAnnotation("g1", "estimate",           ["1TB per month", "10TB", "50GB", "100MB", "10 GB"],                    ["confirmed", "verified", "exact", "measured", "current exact"]),
        "g2": ClaimAnnotation("g2", "vendor_claim",       ["size S", "size L", "size XL", "size XS", "extra-large"],              ["confirmed", "benchmarked", "tested", "verified", "our test"]),
        "g3": ClaimAnnotation("g3", "estimate",           ["$10K/month", "$100/month", "$50K", "10,000"],                          ["confirmed", "exact", "invoiced", "finalized", "current bill"]),
    },
    "ghost_mobile_01": {
        "g1": ClaimAnnotation("g1", "approximation",      ["1%", "5%", "10%", "0.01%", "zero crashes"],                           ["confirmed", "exact", "verified", "accurate", "precise"]),
        "g2": ClaimAnnotation("g2", "approximation",      ["1 second", "5 seconds", "10 seconds", "0.5 seconds"],                  ["confirmed", "measured", "benchmarked", "exact", "rigorous test"]),
        # hallu_frags: "point" is a value_frag — use entirely different wrong-answer phrasing
        "g3": ClaimAnnotation("g3", "estimate",           ["no expected gain", "negligible improvement", "no rating change"],     ["confirmed", "validated", "expected precisely", "guaranteed"]),
    },
    "ghost_finance_01": {
        "g1": ClaimAnnotation("g1", "estimate",           ["$1M", "$50M", "$100M", "$20M", "1 million"],                           ["confirmed", "term sheet", "final", "committed", "agreed"]),
        "g2": ClaimAnnotation("g2", "estimate",           ["6 months", "5 years", "24 months", "36 months"],                       ["confirmed", "guaranteed", "exact", "audited", "fixed"]),
        "g3": ClaimAnnotation("g3", "estimate",           ["$1M", "$10K", "$500K", "$1,000,000", "1 million"],                     ["confirmed", "final", "filed", "exact", "settled"]),
    },
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClaimScore:
    session_id:          str
    domain:              str
    ghost_id:            str
    qualifier_type:      str
    condition:           str
    value_survival:      bool    # key value was correctly recalled
    qualifier_survival:  bool    # epistemic qualifier was preserved
    false_certainty:     bool    # value recalled BUT qualifier absent — model asserted fact without uncertainty
    hallucination:       bool    # model stated a known-wrong value
    drift:               bool    # drift_frag present AND qualifier absent (specific false-certainty language)
    response_snippet:    str     = ""


@dataclass
class ConditionAggregate:
    condition:           str
    n_claims:            int
    value_rate:          float
    qualifier_rate:      float
    both_rate:           float
    false_certainty_rate: float   # FCR: value recalled without uncertainty qualifier (headline harm metric)
    hallucination_rate:  float
    drift_rate:          float
    value_ci:            tuple[float, float]   = (0.0, 0.0)
    qualifier_ci:        tuple[float, float]   = (0.0, 0.0)
    both_ci:             tuple[float, float]   = (0.0, 0.0)
    fcr_ci:              tuple[float, float]   = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_claim(
    response:   str,
    callback:   GhostCallback,
    annotation: ClaimAnnotation,
    condition:  str,
    session:    GhostSession,
) -> ClaimScore:
    """
    Score a single claim across five dimensions.

    value_survival:     any value_fragment present in response
    qualifier_survival: any qualifier_fragment present in response
    false_certainty:    value recalled BUT qualifier absent — the core harm metric (FCR).
                        Model surfaced the fact without its epistemic caveat.
                        Defined as: value_survival AND NOT qualifier_survival.
    hallucination:      any hallu_frag present AND value_fragments absent
                        (model stated a known-wrong value)
    drift:              drift_frag present AND qualifier absent
                        (specific false-certainty language inserted)

    FCR is the headline harm metric because it doesn't depend on keyword lists — only
    on whether the model re-asserted a value it was never supposed to treat as confirmed.
    """
    lower = response.lower()

    value_survival     = any(f.lower() in lower for f in callback.value_fragments)
    qualifier_survival = any(f.lower() in lower for f in callback.qualifier_fragments)

    # FCR: model recalled the fact but stripped its uncertainty — the canonical harm
    false_certainty = value_survival and not qualifier_survival

    # Hallucination: model stated a known-wrong value confidently
    hallu_hit     = any(f.lower() in lower for f in annotation.hallu_frags)
    hallucination = hallu_hit and not value_survival

    # Drift: model used specific false-certainty language where qualifier should be present
    drift_hit = any(f.lower() in lower for f in annotation.drift_frags)
    drift = drift_hit and not qualifier_survival

    return ClaimScore(
        session_id         = session.session_id,
        domain             = session.domain,
        ghost_id           = callback.ghost_id,
        qualifier_type     = annotation.qualifier_type,
        condition          = condition,
        value_survival     = value_survival,
        qualifier_survival = qualifier_survival,
        false_certainty    = false_certainty,
        hallucination      = hallucination,
        drift              = drift,
        response_snippet   = response[:200],
    )


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. "
    "Give concise answers when the answer is clear; "
    "express genuine uncertainty when it exists."
)


def _opus_call(messages: list[dict], client, max_tokens: int = 300) -> str:
    resp = client.messages.create(
        model      = "claude-opus-4-7",
        messages   = messages,
        max_tokens = max_tokens,
        system     = _SYSTEM_PROMPT,
    )
    block = next((b for b in resp.content if b.type == "text"), None)
    return block.text if block else ""


def _run_naive_condition(session: GhostSession, client, window: int = 12) -> list[tuple[GhostCallback, str]]:
    """Naive sliding-window: keep only the last `window` messages."""
    history: list[dict] = []

    def _chat(user_msg: str) -> str:
        history.append({"role": "user", "content": user_msg})
        windowed = history[-window:]
        text = _opus_call(windowed, client)
        history.append({"role": "assistant", "content": text})
        return text

    for msg in session.context_turns:
        if msg["role"] == "user":
            _chat(msg["content"])

    for gt in session.ghost_turns:
        _chat(gt["content"])

    for filler in session.filler_turns:
        _chat(filler)

    results = []
    for cb in session.callbacks:
        response = _chat(cb.question)
        results.append((cb, response))
    return results


def _run_baseline_condition(session: GhostSession, client) -> list[tuple[GhostCallback, str]]:
    """Full context — no compression, no window limit. Oracle upper bound."""
    history: list[dict] = []

    def _chat(user_msg: str) -> str:
        history.append({"role": "user", "content": user_msg})
        text = _opus_call(history, client)
        history.append({"role": "assistant", "content": text})
        return text

    for msg in session.context_turns:
        if msg["role"] == "user":
            _chat(msg["content"])

    for gt in session.ghost_turns:
        _chat(gt["content"])

    for filler in session.filler_turns:
        _chat(filler)

    results = []
    for cb in session.callbacks:
        response = _chat(cb.question)
        results.append((cb, response))
    return results


def _run_credence_condition(
    session:             GhostSession,
    client,
    use_semantic_entropy: bool,
    use_claim_extraction: bool,
) -> list[tuple[GhostCallback, str]]:
    """
    Run through ContextManager.
    Ghost turns flow through cm.chat() so claim extraction fires per-turn.
    Filler turns force compression pressure.
    Callbacks see Truth Buffer injection when registry is provided.
    """
    from credence.context_manager import ContextManager
    from credence.registry import CredenceRegistry

    if use_claim_extraction:
        reg = CredenceRegistry(":memory:")
        mgr = ContextManager(
            max_tokens            = 300,
            use_semantic_entropy  = use_semantic_entropy,
            use_claim_extraction  = True,
            registry              = reg,
            session_id            = session.session_id,
        )
    else:
        mgr = ContextManager(
            max_tokens            = 300,
            use_semantic_entropy  = use_semantic_entropy,
            use_claim_extraction  = False,
        )

    for msg in session.context_turns:
        if msg["role"] == "user":
            mgr.chat(msg["content"])

    for gt in session.ghost_turns:
        mgr.chat(gt["content"])

    for filler in session.filler_turns:
        mgr.chat(filler)

    results = []
    for cb in session.callbacks:
        turn = mgr.chat(cb.question)
        results.append((cb, turn.response))
    return results


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def _run_session(
    session:   GhostSession,
    condition: str,
    client,
) -> list[ClaimScore]:
    """
    Run one full session under a given condition.
    Returns a list of ClaimScore — one per callback (3 per session).
    """
    annotations = CLAIM_ANNOTATIONS.get(session.session_id, {})

    if condition == "credence_eg2":
        pairs = _run_credence_condition(
            session, client,
            use_semantic_entropy=True,
            use_claim_extraction=True,
        )
    elif condition == "credence_v1":
        pairs = _run_credence_condition(
            session, client,
            use_semantic_entropy=False,
            use_claim_extraction=False,
        )
    elif condition == "naive_window":
        pairs = _run_naive_condition(session, client)
    elif condition == "baseline_full":
        pairs = _run_baseline_condition(session, client)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    scores = []
    for cb, response in pairs:
        ann = annotations.get(cb.ghost_id)
        if ann is None:
            continue
        score = _score_claim(response, cb, ann, condition, session)
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values:  list[float],
    n_boot:  int  = 2000,
    ci:      float = 0.95,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng    = random.Random(42)
    n      = len(values)
    means  = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_boot)
    )
    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    return (round(means[int(lo * n_boot)], 3), round(means[int(hi * n_boot)], 3))


def _aggregate_condition(scores: list[ClaimScore]) -> ConditionAggregate:
    n = len(scores)
    if n == 0:
        return ConditionAggregate(condition="", n_claims=0, value_rate=0, qualifier_rate=0,
                                  both_rate=0, false_certainty_rate=0,
                                  hallucination_rate=0, drift_rate=0)
    vv  = [1.0 if s.value_survival                                else 0.0 for s in scores]
    qv  = [1.0 if s.qualifier_survival                            else 0.0 for s in scores]
    bv  = [1.0 if (s.value_survival and s.qualifier_survival)     else 0.0 for s in scores]
    fv  = [1.0 if s.false_certainty                               else 0.0 for s in scores]
    hv  = [1.0 if s.hallucination                                 else 0.0 for s in scores]
    dv  = [1.0 if s.drift                                         else 0.0 for s in scores]
    return ConditionAggregate(
        condition             = scores[0].condition,
        n_claims              = n,
        value_rate            = round(sum(vv) / n, 3),
        qualifier_rate        = round(sum(qv) / n, 3),
        both_rate             = round(sum(bv) / n, 3),
        false_certainty_rate  = round(sum(fv) / n, 3),
        hallucination_rate    = round(sum(hv) / n, 3),
        drift_rate            = round(sum(dv) / n, 3),
        value_ci              = _bootstrap_ci(vv),
        qualifier_ci          = _bootstrap_ci(qv),
        both_ci               = _bootstrap_ci(bv),
        fcr_ci                = _bootstrap_ci(fv),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(all_scores: list[ClaimScore]) -> None:
    from collections import defaultdict

    by_condition: dict[str, list[ClaimScore]] = defaultdict(list)
    for s in all_scores:
        by_condition[s.condition].append(s)

    conditions = ["credence_eg2", "credence_v1", "naive_window", "baseline_full"]

    # ── Panel-ready headline table ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CLAIM GAUNTLET — Implicit Uncertainty Under Compression")
    print("10 sessions × 3 claims = 30 data points per condition")
    print("=" * 72)
    print(f"{'System':<22} {'BothRate (BR) ↑':>16}  {'False Certainty ↓':>18}")
    print(f"{'':22} {'mean  [95% CI]':>16}  {'mean  [95% CI]':>18}")
    print("-" * 72)
    display_names = {
        "credence_eg2":  "Credence (ours)",
        "credence_v1":   "Credence v1 (ablation)",
        "naive_window":  "Naive window",
        "baseline_full": "Full context (oracle)",
    }
    for cond in conditions:
        scores = by_condition.get(cond, [])
        if not scores:
            continue
        agg  = _aggregate_condition(scores)
        name = display_names.get(cond, cond)
        br_ci_str  = f"[{agg.both_ci[0]:.3f}, {agg.both_ci[1]:.3f}]"
        fcr_ci_str = f"[{agg.fcr_ci[0]:.3f}, {agg.fcr_ci[1]:.3f}]"
        print(f"{name:<22}  {agg.both_rate:.3f}  {br_ci_str:<16}  {agg.false_certainty_rate:.3f}  {fcr_ci_str}")
    print("=" * 72)
    print("BR  = fraction of claims with BOTH value AND qualifier recalled")
    print("FCR = value recalled without uncertainty qualifier (false certainty)")
    print()

    # ── Full diagnostic table ───────────────────────────────────────────────
    print("=" * 104)
    print("Detailed breakdown")
    print("=" * 104)
    print(f"{'Condition':<22} {'Claims':>7} {'VR':>8} {'QR':>8} {'BR':>8} {'FCR':>8} {'Hallu':>8} {'Drift':>8}")
    print("-" * 104)
    for cond in conditions:
        scores = by_condition.get(cond, [])
        if not scores:
            continue
        agg = _aggregate_condition(scores)
        print(
            f"{cond:<22} {agg.n_claims:>7} "
            f"{agg.value_rate:>8.3f} "
            f"{agg.qualifier_rate:>8.3f} "
            f"{agg.both_rate:>8.3f} "
            f"{agg.false_certainty_rate:>8.3f} "
            f"{agg.hallucination_rate:>8.3f} "
            f"{agg.drift_rate:>8.3f}"
        )
    print("=" * 104)

    # Per-domain breakdown
    by_domain: dict[str, dict[str, list[ClaimScore]]] = defaultdict(lambda: defaultdict(list))
    for s in all_scores:
        by_domain[s.domain][s.condition].append(s)

    if len(by_domain) > 1:
        print("\nDomain breakdown (BothRate: credence_eg2 vs naive_window):")
        print(f"  {'Domain':<22} {'eg2':>6} {'naive':>8} {'delta':>8}")
        print(f"  {'-'*46}")
        for domain in sorted(by_domain):
            eg2_scores    = by_domain[domain].get("credence_eg2",  [])
            naive_scores  = by_domain[domain].get("naive_window", [])
            if not eg2_scores or not naive_scores:
                continue
            eg2_both   = sum(1 for s in eg2_scores   if s.value_survival and s.qualifier_survival) / len(eg2_scores)
            naive_both = sum(1 for s in naive_scores  if s.value_survival and s.qualifier_survival) / len(naive_scores)
            delta      = eg2_both - naive_both
            print(f"  {domain:<22} {eg2_both:>6.3f} {naive_both:>8.3f} {delta:>+8.3f}")

    # Per-claim type breakdown
    by_type: dict[str, dict[str, list[ClaimScore]]] = defaultdict(lambda: defaultdict(list))
    for s in all_scores:
        by_type[s.qualifier_type][s.condition].append(s)

    if len(by_type) > 1:
        print("\nQualifier-type breakdown (BothRate: credence_eg2 vs naive_window):")
        print(f"  {'Type':<22} {'eg2':>6} {'naive':>8} {'delta':>8} {'n':>4}")
        print(f"  {'-'*50}")
        for qt in sorted(by_type):
            eg2_s  = by_type[qt].get("credence_eg2",  [])
            naive_s = by_type[qt].get("naive_window", [])
            if not eg2_s or not naive_s:
                continue
            eg2_b   = sum(1 for s in eg2_s   if s.value_survival and s.qualifier_survival) / len(eg2_s)
            naive_b = sum(1 for s in naive_s  if s.value_survival and s.qualifier_survival) / len(naive_s)
            print(f"  {qt:<22} {eg2_b:>6.3f} {naive_b:>8.3f} {eg2_b - naive_b:>+8.3f} {len(eg2_s):>4}")

    # Key claims that failed in credence_eg2 (diagnostic)
    eg2_failed = [s for s in all_scores if s.condition == "credence_eg2" and not (s.value_survival and s.qualifier_survival)]
    if eg2_failed:
        print(f"\ncredence_eg2 failures ({len(eg2_failed)} claims):")
        for s in eg2_failed:
            flags = []
            if not s.value_survival:     flags.append("no_value")
            if not s.qualifier_survival: flags.append("no_qualifier")
            if s.false_certainty:        flags.append("FALSE_CERTAINTY")
            if s.hallucination:          flags.append("HALLUCINATION")
            if s.drift:                  flags.append("drift")
            print(f"  {s.session_id} / {s.ghost_id} / {s.qualifier_type}: {', '.join(flags)}")

    print()
    print("Metrics:")
    print("  VR   = ValueRate     — key value recalled")
    print("  QR   = QualRate      — epistemic qualifier preserved")
    print("  BR   = BothRate      — BOTH value + qualifier (gold standard)")
    print("  FCR  = False Certainty Rate — value recalled without qualifier (HARM metric)")
    print("  Hallu                — known-wrong value stated")
    print("  Drift                — specific false-certainty language inserted")
    print()

    # Claim: naive_window FCR shows the harm, Credence FCR shows it's fixed
    eg2_agg    = _aggregate_condition(by_condition.get("credence_eg2",  []))
    naive_agg  = _aggregate_condition(by_condition.get("naive_window",  []))
    base_agg   = _aggregate_condition(by_condition.get("baseline_full", []))
    v1_agg     = _aggregate_condition(by_condition.get("credence_v1",   []))
    print(f"Headline claim: BR  credence_eg2={eg2_agg.both_rate:.3f} vs naive={naive_agg.both_rate:.3f}  "
          f"delta={eg2_agg.both_rate - naive_agg.both_rate:+.3f}  "
          f"(ceiling={base_agg.both_rate:.3f})")
    print(f"Harm claim:     FCR credence_eg2={eg2_agg.false_certainty_rate:.3f} vs naive={naive_agg.false_certainty_rate:.3f}  "
          f"delta={eg2_agg.false_certainty_rate - naive_agg.false_certainty_rate:+.3f}")
    if v1_agg.n_claims > 0:
        print(f"Ablation:       BR  credence_eg2={eg2_agg.both_rate:.3f} vs credence_v1={v1_agg.both_rate:.3f}  "
              f"claim_extraction+SE delta={eg2_agg.both_rate - v1_agg.both_rate:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claim Gauntlet: per-claim implicit uncertainty survival benchmark"
    )
    parser.add_argument("--dry-run",  action="store_true", help="Validate structure without API calls")
    parser.add_argument("--n",        type=int,  default=None,   help="Only run first N sessions")
    parser.add_argument("--session",  type=str,  default=None,   help="Run single session by ID")
    parser.add_argument("--domain",   type=str,  default=None,   help="Filter sessions by domain")
    parser.add_argument("--resume",   action="store_true", help="Resume from existing results file")
    parser.add_argument("--out",      type=str,
                        default="evals/claim_gauntlet_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated subset of conditions to run "
                             "(credence_eg2,credence_v1,naive_window,baseline_full)")
    args = parser.parse_args()

    # Dry run: validate structure and annotations only
    if args.dry_run:
        sessions_to_run = GHOST_SESSIONS
        if args.n       is not None: sessions_to_run = sessions_to_run[:args.n]
        if args.session is not None: sessions_to_run = [s for s in sessions_to_run if s.session_id == args.session]
        if args.domain  is not None: sessions_to_run = [s for s in sessions_to_run if s.domain == args.domain]

        n_valid, n_claims = 0, 0
        for s in sessions_to_run:
            assert len(s.ghost_turns) == 3,   f"{s.session_id}: need 3 ghost turns"
            assert len(s.filler_turns) == 8,  f"{s.session_id}: need 8 filler turns"
            assert len(s.callbacks) == 3,     f"{s.session_id}: need 3 callbacks"
            ann = CLAIM_ANNOTATIONS.get(s.session_id, {})
            for cb in s.callbacks:
                assert cb.ghost_id in ann, f"{s.session_id} missing annotation for {cb.ghost_id}"
                n_claims += 1
            n_valid += 1
        print(f"DRY RUN: {n_valid} sessions × 4 conditions × {n_claims // max(n_valid, 1)} claims/session = {n_claims * 4} total score rows. Structure valid.")
        return

    # Real run
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        return

    client = Anthropic(api_key=api_key)

    # Session and condition selection
    sessions_to_run = GHOST_SESSIONS
    if args.n       is not None: sessions_to_run = sessions_to_run[:args.n]
    if args.session is not None: sessions_to_run = [s for s in sessions_to_run if s.session_id == args.session]
    if args.domain  is not None: sessions_to_run = [s for s in sessions_to_run if s.domain == args.domain]

    all_conditions = ["credence_eg2", "credence_v1", "naive_window", "baseline_full"]
    if args.conditions:
        all_conditions = [c.strip() for c in args.conditions.split(",") if c.strip() in all_conditions]

    # Load existing results if resuming
    all_scores_raw: list[dict] = []
    completed_keys: set[str] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        all_scores_raw = existing.get("scores", [])
        completed_keys = {f"{r['session_id']}::{r['ghost_id']}::{r['condition']}" for r in all_scores_raw}
        print(f"Resuming: {len(all_scores_raw)} claim scores already completed")

    total = len(sessions_to_run) * len(all_conditions)
    done  = 0

    for session in sessions_to_run:
        for condition in all_conditions:
            # Check if this session × condition is already done (all 3 claims)
            session_cond_key = f"{session.session_id}::*::{condition}"
            claims_done = sum(
                1 for r in all_scores_raw
                if r["session_id"] == session.session_id and r["condition"] == condition
            )
            if claims_done >= 3:
                done += 1
                print(f"[{done}/{total}] SKIP (already done): {session.session_id} / {condition}")
                continue

            print(
                f"[{done + 1}/{total}] Running {session.session_id} / {condition} ...",
                end=" ", flush=True,
            )
            t0 = time.time()
            try:
                claim_scores = _run_session(session, condition, client)
                elapsed = round(time.time() - t0, 1)
                both_rate = (
                    sum(1 for s in claim_scores if s.value_survival and s.qualifier_survival)
                    / max(len(claim_scores), 1)
                )
                print(f"done ({elapsed}s)  both_rate={both_rate:.3f}  claims={len(claim_scores)}")
                for cs in claim_scores:
                    all_scores_raw.append({
                        "session_id":          cs.session_id,
                        "domain":              cs.domain,
                        "ghost_id":            cs.ghost_id,
                        "qualifier_type":      cs.qualifier_type,
                        "condition":           cs.condition,
                        "value_survival":      cs.value_survival,
                        "qualifier_survival":  cs.qualifier_survival,
                        "false_certainty":     cs.false_certainty,
                        "hallucination":       cs.hallucination,
                        "drift":               cs.drift,
                        "response_snippet":    cs.response_snippet,
                    })
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                print(f"ERROR ({elapsed}s): {exc}")
            done += 1

            # Save after each session × condition (crash-safe)
            with open(args.out, "w") as f:
                json.dump({"scores": all_scores_raw}, f, indent=2)

    # Reconstruct ClaimScore objects for display
    if all_scores_raw:
        all_scores = [
            ClaimScore(
                session_id         = r["session_id"],
                domain             = r["domain"],
                ghost_id           = r["ghost_id"],
                qualifier_type     = r["qualifier_type"],
                condition          = r["condition"],
                value_survival     = r["value_survival"],
                qualifier_survival = r["qualifier_survival"],
                false_certainty    = r.get("false_certainty", r["value_survival"] and not r["qualifier_survival"]),
                hallucination      = r["hallucination"],
                drift              = r["drift"],
                response_snippet   = r.get("response_snippet", ""),
            )
            for r in all_scores_raw
        ]
        _print_table(all_scores)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

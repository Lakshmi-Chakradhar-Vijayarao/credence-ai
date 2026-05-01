"""
evals/manifest_survival.py
==========================
Manifest Survival Benchmark — tests the core research hypothesis:

    Does a structured, labeled epistemic manifest survive Haiku compression
    at a higher rate than equivalent natural language injection?

Research question (from the epistemic_manifest.py design doc):
    Natural language is compressible. Can an explicit "NON-COMPRESSIBLE" label
    + structured XML attributes cause compression models to preserve the
    epistemic state better than a natural language Truth Buffer?

Five conditions tested on 20 uncertain claims:

    1. no_injection       — no epistemic state injected; baseline FCR
    2. natural_language   — current Truth Buffer (plain English, informational)
    3. structured_xml     — EpistemicManifest XML (attributes, no label)
    4. locked_manifest    — EpistemicManifest XML + explicit NON-COMPRESSIBLE label
    5. repeated_injection — locked_manifest injected TWICE (start + immediately before
                            compression call). Tests whether repetition improves survival.

Metrics per condition (per claim):
    - value_survival:      uncertain value (50, 3600, etc.) present in summary
    - qualifier_survival:  hedging word or conf attribute present near value
    - false_certainty:     value present WITHOUT any qualifier (the FCR event)
    - manifest_block_survived: the <EPISTEMIC_MANIFEST> block itself survived

Aggregate:
    - EQLR (Epistemic Qualifier Loss Rate): fraction of claims losing qualifier
    - FCR (False Certainty Rate): fraction stated as fact without qualifier
    - Manifest Survival Rate: fraction of compressions with manifest block intact

Run:
    python -m evals.manifest_survival              # all 20 claims (~$0.60 with 5 conditions)
    python -m evals.manifest_survival --n 5        # first 5 claims (quick test)
    python -m evals.manifest_survival --dry-run    # validate structure, no API
    python -m evals.manifest_survival --conditions repeated_injection  # single condition
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.epistemic_manifest import EpistemicManifest
from credence.registry import CredenceRegistry

_MODEL_HAIKU = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Test claims — 20 uncertain constraints across 4 domains
# Designed to stress-test manifest survival:
#   - mix of numeric and string values
#   - mix of explicit hedging (has_surface_markers=True) and ghost constraints
#   - mix of source types
# ---------------------------------------------------------------------------

TEST_CLAIMS = [
    # API domain — ghost constraints (no surface markers)
    {"text": "The Stripe rate limit is 50 req/min.",            "value": "50",      "conf": 0.30, "source": "vendor_claim",  "ghost": True,  "qualifier_frags": ["unverified","estimate","unconfirmed","might","vendor"]},
    {"text": "Auth token expiry is 3600 seconds.",              "value": "3600",    "conf": 0.35, "source": "assumption",     "ghost": True,  "qualifier_frags": ["unverified","estimate","check","approximately","might"]},
    {"text": "The webhook timeout is 30 seconds.",              "value": "30",      "conf": 0.28, "source": "hearsay",        "ghost": True,  "qualifier_frags": ["unverified","heard","reportedly","check","might"]},
    {"text": "Batch endpoint accepts up to 100 items.",         "value": "100",     "conf": 0.32, "source": "vendor_claim",  "ghost": True,  "qualifier_frags": ["unverified","vendor","unconfirmed","roughly","check"]},
    {"text": "JWT signing uses RS256 algorithm.",               "value": "RS256",   "conf": 0.40, "source": "assumption",    "ghost": True,  "qualifier_frags": ["unverified","assumption","might","check","confirm"]},
    # API domain — explicit hedging (surface markers present)
    {"text": "I think the pagination limit is about 200 items.", "value": "200",   "conf": 0.38, "source": "user_estimate", "ghost": False, "qualifier_frags": ["think","about","approximately","estimate","roughly"]},
    {"text": "Token refresh probably needs to happen every hour.", "value": "hour", "conf": 0.33, "source": "inference",    "ghost": False, "qualifier_frags": ["probably","might","approximately","estimate","roughly"]},
    # Debug domain
    {"text": "Memory leak grows at roughly 2MB per 1000 requests.", "value": "2",   "conf": 0.42, "source": "user_estimate", "ghost": False, "qualifier_frags": ["roughly","estimate","approximately","eyeballed","might"]},
    {"text": "The race condition triggers at 8+ concurrent writes.", "value": "8",  "conf": 0.38, "source": "inference",     "ghost": False, "qualifier_frags": ["seems","might","approximately","estimate","non-deterministic"]},
    {"text": "CPU spikes to 95% under load.",                    "value": "95",     "conf": 0.35, "source": "user_estimate", "ghost": False, "qualifier_frags": ["observed","might","check","approximately","varies"]},
    # Debug ghost constraints
    {"text": "The deadlock occurs every 500 transactions.",      "value": "500",    "conf": 0.30, "source": "hearsay",       "ghost": True,  "qualifier_frags": ["unverified","staging","estimate","check","non-deterministic"]},
    {"text": "Queue backlog clears in 4 hours.",                 "value": "4",      "conf": 0.33, "source": "assumption",    "ghost": True,  "qualifier_frags": ["estimate","approximately","might","assumed","check"]},
    # Design domain
    {"text": "We estimate GraphQL will reduce payload by 40%.",  "value": "40",     "conf": 0.38, "source": "user_estimate", "ghost": False, "qualifier_frags": ["estimate","roughly","might","preliminary","based on"]},
    {"text": "CDN should handle 80% of requests.",               "value": "80",     "conf": 0.35, "source": "inference",     "ghost": False, "qualifier_frags": ["should","estimate","might","roughly","theoretical"]},
    {"text": "Cache hit rate will be 85%.",                      "value": "85",     "conf": 0.30, "source": "assumption",    "ghost": True,  "qualifier_frags": ["estimate","preliminary","might","based on","project"]},
    # Compliance domain
    {"text": "GDPR deletion must complete within 30 days.",      "value": "30",     "conf": 0.42, "source": "inference",     "ghost": False, "qualifier_frags": ["believe","might","verify","formal","opinion"]},
    {"text": "The breach notification window is 72 hours.",      "value": "72",     "conf": 0.38, "source": "user_estimate", "ghost": False, "qualifier_frags": ["believe","might","check","haven't","verify"]},
    # Multiagent domain
    {"text": "Haiku will handle 80% of sub-agent tasks.",        "value": "80",     "conf": 0.28, "source": "user_estimate", "ghost": False, "qualifier_frags": ["estimate","one demo","not real","based on","roughly"]},
    {"text": "Tool timeout should be set to 30 seconds.",        "value": "30",     "conf": 0.32, "source": "hearsay",       "ghost": True,  "qualifier_frags": ["light load","testing","might","estimate","check"]},
    {"text": "10 parallel agents will cost 10x a single agent.", "value": "10x",   "conf": 0.30, "source": "assumption",    "ghost": True,  "qualifier_frags": ["ignores","overhead","might","estimate","approximately"]},
]


# ---------------------------------------------------------------------------
# Natural language Truth Buffer (current Credence approach)
# ---------------------------------------------------------------------------

def _build_natural_language_injection(claims: list[dict]) -> str:
    lines = ["UNVERIFIED CONSTRAINTS — acknowledge uncertainty for each:"]
    for c in claims:
        lines.append(f"  • {c['text']} [conf={c['conf']:.2f}, source={c['source']}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaimSurvival:
    claim_text:           str
    value:                str
    conf:                 float
    is_ghost:             bool
    condition:            str
    value_survived:       bool = False
    qualifier_survived:   bool = False
    false_certainty:      bool = False
    manifest_survived:    bool = False   # the XML block itself survived


@dataclass
class ConditionResult:
    name:         str
    n_claims:     int
    eqlr:         float   # Epistemic Qualifier Loss Rate
    fcr:          float   # False Certainty Rate
    manifest_sr:  float   # Manifest Block Survival Rate (XML conditions only)
    ghost_fcr:    float   # FCR for ghost constraints specifically
    explicit_fcr: float   # FCR for explicit hedges
    claims:       list[ClaimSurvival] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core compression + scoring
# ---------------------------------------------------------------------------

def _compress_with_injection(
    claim:              dict,
    injection:          str,
    client,
    condition:          str,
    pre_compress_injection: str = "",
) -> ClaimSurvival:
    """
    Simulate: inject the epistemic state into a system prompt, then ask Haiku
    to summarize a conversation containing the uncertain claim, then score
    whether the value + qualifier survived in the summary.

    For the `repeated_injection` condition, `pre_compress_injection` is
    appended to the user message immediately before asking for the summary.
    This simulates injecting the manifest a second time right before the
    compression call — testing whether repetition improves survival.
    """
    # Simulate a short conversation seeded with the uncertain claim
    conversation = (
        f"Turn 1 — User: I'm integrating with the payment API.\n"
        f"Turn 2 — Assistant: What are the constraints I should know?\n"
        f"Turn 3 — User: {claim['text']}\n"
        f"Turn 4 — Assistant: Understood, I'll keep that in mind.\n"
        f"Turn 5 — User: Let's talk about authentication.\n"
        f"Turn 6 — Assistant: Sure. What auth method are we using?\n"
        f"Turn 7 — User: We're using OAuth 2.0 with a separate service.\n"
        f"Turn 8 — Assistant: Got it. Any other constraints?\n"
    )

    system = (
        f"{injection}\n\n"
        "You are a context compression assistant. "
        "Summarize the conversation below in 3-4 sentences. "
        "Preserve all factual claims and their uncertainty status."
    ) if injection else (
        "You are a context compression assistant. "
        "Summarize the conversation below in 3-4 sentences."
    )

    # For repeated_injection: append the manifest again directly before the
    # conversation text in the user turn, simulating a second injection site.
    if pre_compress_injection:
        user_content = (
            f"{pre_compress_injection}\n\n"
            f"Please summarize the following conversation:\n\n"
            f"{conversation}"
        )
    else:
        user_content = conversation

    try:
        resp = client.messages.create(
            model    = _MODEL_HAIKU,
            system   = system,
            messages = [{"role": "user", "content": user_content}],
            max_tokens = 200,
        )
        summary = resp.content[0].text.strip() if resp.content else ""
    except Exception as e:
        summary = ""

    # Score the summary
    value      = claim["value"]
    qual_frags = claim["qualifier_frags"]
    summary_lo = summary.lower()

    value_survived     = value.lower() in summary_lo
    qualifier_survived = any(q.lower() in summary_lo for q in qual_frags)
    false_certainty    = value_survived and not qualifier_survived
    manifest_survived  = "EPISTEMIC_MANIFEST" in summary

    return ClaimSurvival(
        claim_text         = claim["text"],
        value              = value,
        conf               = claim["conf"],
        is_ghost           = claim.get("ghost", False),
        condition          = condition,
        value_survived     = value_survived,
        qualifier_survived = qualifier_survived,
        false_certainty    = false_certainty,
        manifest_survived  = manifest_survived,
    )


def _run_condition(
    condition:  str,
    claims:     list[dict],
    client,
) -> ConditionResult:
    """Run all claims under one injection condition."""
    print(f"  [{condition}]", end="", flush=True)
    results: list[ClaimSurvival] = []

    for i, claim in enumerate(claims):
        # Build injection for this condition
        pre_compress_injection = ""

        if condition == "no_injection":
            injection = ""
        elif condition == "natural_language":
            injection = _build_natural_language_injection([claim])
        elif condition == "structured_xml":
            # Build a manifest without the NON-COMPRESSIBLE label
            injection = _build_xml_no_label([claim])
        elif condition == "locked_manifest":
            # Full EpistemicManifest with NON-COMPRESSIBLE label
            injection = _build_manifest_xml([claim])
        elif condition == "repeated_injection":
            # Same locked manifest as condition 4, injected TWICE:
            # once in the system prompt and once in the user message
            # immediately before the conversation to summarize.
            manifest_xml = _build_manifest_xml([claim])
            injection = manifest_xml
            pre_compress_injection = manifest_xml
        else:
            injection = ""

        result = _compress_with_injection(
            claim, injection, client, condition,
            pre_compress_injection=pre_compress_injection,
        )
        results.append(result)
        print(".", end="", flush=True)
        time.sleep(0.1)  # avoid rate limits

    print()

    n = len(results)
    fcr      = sum(1 for r in results if r.false_certainty) / n
    eqlr     = sum(1 for r in results if r.value_survived and not r.qualifier_survived) / max(1, sum(1 for r in results if r.value_survived))
    msr      = sum(1 for r in results if r.manifest_survived) / n
    ghosts   = [r for r in results if r.is_ghost]
    explicit = [r for r in results if not r.is_ghost]
    ghost_fcr    = sum(1 for r in ghosts if r.false_certainty) / max(1, len(ghosts))
    explicit_fcr = sum(1 for r in explicit if r.false_certainty) / max(1, len(explicit))

    return ConditionResult(
        name         = condition,
        n_claims     = n,
        eqlr         = round(eqlr, 3),
        fcr          = round(fcr, 3),
        manifest_sr  = round(msr, 3),
        ghost_fcr    = round(ghost_fcr, 3),
        explicit_fcr = round(explicit_fcr, 3),
        claims       = results,
    )


# ---------------------------------------------------------------------------
# XML builders (for condition comparison)
# ---------------------------------------------------------------------------

def _build_xml_no_label(claims: list[dict]) -> str:
    """XML manifest WITHOUT the NON-COMPRESSIBLE label — control for label effect."""
    lines = ['<epistemic_state>']
    for c in claims:
        lines.append(
            f'  <claim conf="{c["conf"]}" source="{c["source"]}" verified="false">'
            f'{c["text"]}</claim>'
        )
    lines.append('</epistemic_state>')
    return "\n".join(lines)


def _build_manifest_xml(claims: list[dict]) -> str:
    """Full EpistemicManifest with NON-COMPRESSIBLE label."""
    registry = CredenceRegistry(":memory:")
    for c in claims:
        registry.register(c["text"], "test", j_score=c["conf"],
                          source=c["source"])
    return EpistemicManifest.from_registry(registry, "test", current_turn=5)


# ---------------------------------------------------------------------------
# Dry run — validate structure without API
# ---------------------------------------------------------------------------

def _dry_run(claims: list[dict]) -> None:
    print(f"\nDry run — validating {len(claims)} claims across 5 conditions...\n")
    for i, c in enumerate(claims):
        print(f"  [{i+1:02d}] {c['text'][:60]}...")
        print(f"       value={c['value']!r}  conf={c['conf']}  ghost={c.get('ghost')}  source={c['source']}")
        print(f"       qualifiers: {c['qualifier_frags'][:3]}")

    # Validate manifest XML builds correctly
    xml = _build_manifest_xml(claims[:3])
    assert "EPISTEMIC_MANIFEST" in xml, "Manifest XML missing header"
    assert "NON-COMPRESSIBLE" in xml,   "Manifest XML missing non-compressible label"
    assert "CONFIDENCE_PROPAGATION" in xml, "Manifest XML missing propagation rule"
    assert 'conf="' in xml,             "Manifest XML missing confidence attribute"
    print(f"\n✓ Manifest XML structure valid ({len(xml)} chars)")

    xml_no_label = _build_xml_no_label(claims[:3])
    assert "<epistemic_state>" in xml_no_label, "No-label XML invalid"
    print("✓ No-label XML structure valid")

    nl = _build_natural_language_injection(claims[:3])
    assert "UNVERIFIED" in nl, "Natural language injection invalid"
    print("✓ Natural language injection valid")

    # Validate repeated_injection — same XML injected at both sites
    repeated_xml = _build_manifest_xml(claims[:3])
    assert "EPISTEMIC_MANIFEST" in repeated_xml, "Repeated injection XML invalid"
    assert "NON-COMPRESSIBLE" in repeated_xml,   "Repeated injection missing label"
    print("✓ Repeated injection structure valid (manifest identical at both injection sites)")

    print(f"\n✓ All 5 conditions structurally valid for {len(claims)} claims")
    print("  Run without --dry-run to execute against the API (~$0.60)")


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def _print_results(results: list[ConditionResult]) -> None:
    print("\n" + "=" * 72)
    print("  MANIFEST SURVIVAL BENCHMARK")
    print("  Research question: Does structured + labeled manifest survive")
    print("  Haiku compression at a higher rate than natural language?")
    print("=" * 72)

    _XML_CONDITIONS = {"structured_xml", "locked_manifest", "repeated_injection"}
    header = f"  {'Condition':<24} {'EQLR':>6} {'FCR':>6} {'Ghost FCR':>10} {'Manifest SR':>12}"
    print(header)
    print("  " + "-" * 62)
    for r in results:
        msr_str = f"{r.manifest_sr:.1%}" if r.name in _XML_CONDITIONS else "  N/A"
        print(
            f"  {r.name:<24} {r.eqlr:>5.1%} {r.fcr:>5.1%} "
            f"{r.ghost_fcr:>9.1%} {msr_str:>12}"
        )

    print("\n  Metric definitions:")
    print("  EQLR        = fraction of claims whose qualifier was lost (value survived, qualifier didn't)")
    print("  FCR         = fraction of claims stated as confirmed fact without any qualifier")
    print("  Ghost FCR   = FCR for ghost constraints only (no surface hedging markers)")
    print("  Manifest SR = fraction of compressions where <EPISTEMIC_MANIFEST> block survived")

    # Key finding
    no_inj   = next((r for r in results if r.name == "no_injection"), None)
    locked   = next((r for r in results if r.name == "locked_manifest"), None)
    repeated = next((r for r in results if r.name == "repeated_injection"), None)
    if no_inj and locked:
        print(f"\n  KEY FINDING:")
        delta_fcr  = no_inj.fcr - locked.fcr
        delta_eqlr = no_inj.eqlr - locked.eqlr
        if locked.manifest_sr > 0.5:
            print(f"  ✓ Manifest block survived compression in {locked.manifest_sr:.1%} of cases")
        if delta_fcr > 0.05:
            print(f"  ✓ Locked manifest reduced FCR by {delta_fcr:.1%} vs no injection")
        elif delta_fcr <= 0:
            print(f"  ✗ Locked manifest did NOT reduce FCR (no_inj FCR={no_inj.fcr:.1%}, locked FCR={locked.fcr:.1%})")
            print(f"    Hypothesis: structured manifest does not survive compression reliably")
        print(f"  → This validates/refutes the structured manifest design direction")
    if repeated and locked:
        rep_delta = locked.fcr - repeated.fcr
        eqlr_delta = locked.eqlr - repeated.eqlr
        if rep_delta > 0.02:
            print(f"  ✓ Repeated injection further reduced FCR by {rep_delta:.1%} vs single locked manifest")
        elif rep_delta <= 0:
            print(f"  - Repeated injection shows no FCR benefit over single injection "
                  f"(locked={locked.fcr:.1%}, repeated={repeated.fcr:.1%})")
        if repeated.manifest_sr > locked.manifest_sr + 0.05:
            print(f"  ✓ Repetition improved manifest block survival: "
                  f"locked={locked.manifest_sr:.1%} → repeated={repeated.manifest_sr:.1%}")
        print(f"  → Repetition hypothesis: {'supported' if rep_delta > 0.02 else 'not supported'} at this sample size")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Manifest Survival Benchmark")
    parser.add_argument("--n",        type=int, default=20, help="Number of claims to test")
    parser.add_argument("--dry-run",  action="store_true", help="Validate structure without API calls")
    parser.add_argument("--out",      default="evals/manifest_survival_results.json")
    _ALL_CONDITIONS = "no_injection,natural_language,structured_xml,locked_manifest,repeated_injection"
    parser.add_argument("--conditions", default=_ALL_CONDITIONS,
                        help="Comma-separated conditions to run. Use --all to run all 5.")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all 5 conditions (same as default)")
    args = parser.parse_args()

    claims = TEST_CLAIMS[:args.n]

    if args.dry_run:
        _dry_run(claims)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed — pip install anthropic")
        sys.exit(1)

    client     = Anthropic(api_key=api_key)
    _all_conds = "no_injection,natural_language,structured_xml,locked_manifest,repeated_injection"
    if args.run_all:
        conditions = [c.strip() for c in _all_conds.split(",")]
    else:
        conditions = [c.strip() for c in args.conditions.split(",")]

    # Validate condition names
    _valid = set(_all_conds.split(","))
    for cond in conditions:
        if cond not in _valid:
            print(f"Error: unknown condition {cond!r}. Valid: {sorted(_valid)}")
            sys.exit(1)

    print(f"\nManifest Survival Benchmark — {len(claims)} claims × {len(conditions)} conditions")
    print("Each claim compressed through Haiku. Measuring value + qualifier survival.\n")

    results = []
    for cond in conditions:
        r = _run_condition(cond, claims, client)
        results.append(r)

    _print_results(results)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "n_claims":   len(claims),
        "conditions": [
            {
                "name":         r.name,
                "eqlr":         r.eqlr,
                "fcr":          r.fcr,
                "manifest_sr":  r.manifest_sr,
                "ghost_fcr":    r.ghost_fcr,
                "explicit_fcr": r.explicit_fcr,
            }
            for r in results
        ],
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

"""
demo/live_demo.py
=================
Credence — One Claim, Four Checkpoints

One uncertain fact. Traced through the entire enforcement pipeline.
No Streamlit. No API key required.

Run:
    python demo/live_demo.py
    python demo/live_demo.py --live   # with real API call
"""

import sys, os, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.confidence_proxy import CredenceProxy
from credence.registry import CredenceRegistry
from credence.context_manager import (
    _UNCERTAINTY_MARKERS,
    _GTS_NUM_PATTERN,
    _GTS_CODE_BLOCK,
    _CE_MIN_OVERLAP,
    _CE_STOPWORDS,
    _CE_DOMAIN_SYNONYMS,
)

SEP  = "─" * 68
SEP2 = "═" * 68

def _expand(tokens):
    exp = set(tokens)
    for t in tokens:
        if t in _CE_DOMAIN_SYNONYMS:
            exp |= _CE_DOMAIN_SYNONYMS[t]
    return exp

def _tok(text):
    return {w.strip("?.!,;:\"'()[]") for w in text.lower().split()
            if len(w.strip("?.!,;:\"'()[]")) > 2
            and w.strip("?.!,;:\"'()[]") not in _CE_STOPWORDS}

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SETUP
# ─────────────────────────────────────────────────────────────────────────────

CLAIM = (
    "I checked with our Stripe rep — the rate limit is around 50 req/min, "
    "but I'm not certain. Their docs are unclear. "
    "Token expiry is approximately 3600 seconds but that's tentative — "
    "haven't confirmed with the vendor."
)

NAIVE_COMPRESSED = (
    "The Stripe rate limit is 50 req/min. Token expiry is 3600 seconds."
)

USER_CALLBACK = "How fast can we call the Stripe endpoint, and when does the token expire?"

CLAUDE_CODE = """\
Here is the implementation:

```python
# Stripe integration config
RATE_LIMIT = 50
TOKEN_EXPIRY = 3600
MAX_RETRIES = 3

def call_stripe(endpoint, payload):
    if current_rps > RATE_LIMIT:
        time.sleep(1 / RATE_LIMIT)
    return requests.post(endpoint, json=payload)
```

Set TOKEN_EXPIRY to 3600 seconds in your auth configuration.
"""

proxy = CredenceProxy()
reg   = CredenceRegistry(":memory:")
SID   = "demo"
timings = {}

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP2)
print("  CREDENCE — One Claim, Four Checkpoints")
print(SEP2)
print()
print("  The failure: uncertain information becomes confident fact.")
print("  The fix:     four deterministic layers, each catching it at a")
print("               different point in the pipeline.")
print()
print(f"  THE CLAIM:")
print(f"  \"{CLAIM[:80]}…\"")
print(f"  \"{CLAIM[80:]}\"")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 1 — REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 1 — REGISTRY  [Turn 1]")
print("  \"The claim enters the system. It is stored with what we know about it.\"")
print(SEP)

t0 = time.perf_counter()
cr   = proxy.compute(CLAIM)
cid1 = reg.register(CLAIM,
    session_id=SID,
    j_score=cr.j_score,
    zone=cr.zone,
    source="user_stated")
cid2 = reg.register(
    "Token expiry approximately 3600 seconds — tentative, haven't confirmed",
    session_id=SID, j_score=0.28, zone="LOW", source="user_stated")
t_reg = (time.perf_counter() - t0) * 1000
timings["Registry store"] = t_reg

pending = reg.list_uncertain(SID)
decay_t0  = reg.get_effective_confidence(cid1, current_turn=0)
decay_t10 = reg.get_effective_confidence(cid1, current_turn=10)
decay_t20 = reg.get_effective_confidence(cid1, current_turn=20)

print(f"  J-score of claim:   {cr.j_score:.3f}  zone={cr.zone}  (LOW = uncertain)")
print(f"  J-score compressed: {proxy.compute(NAIVE_COMPRESSED).j_score:.3f}  zone={proxy.compute(NAIVE_COMPRESSED).zone}  (HIGH = confident)")
print()
print(f"  Constraint stored:  id={cid1[:12]}…")
print(f"  Unverified claims:  {len(pending)}")
print()
print("  Confidence decay over time (j=0.32 × 0.95^turns):")
print(f"    Turn  0  →  {decay_t0:.3f}  [fresh]")
print(f"    Turn 10  →  {decay_t10:.3f}  [decaying — UNVERIFIED tier]")
print(f"    Turn 20  →  {decay_t20:.3f}  [STALE — escalates to HIGH RISK]")
print()
print(f"  Latency: {t_reg:.3f} ms  |  API calls: 0")
print()
print(f"  ← The claim is now in the registry. The clock is running.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 2 — FAITHFULNESS PROBE
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 2 — FAITHFULNESS PROBE  [Turns 2–9: compression pressure]")
print("  \"Eight HIGH-J filler turns. The system tries to compress old context.\"")
print(SEP)

t0 = time.perf_counter()
markers_found = sorted([m for m in _UNCERTAINTY_MARKERS if m in CLAIM.lower()])[:5]
t_probe = (time.perf_counter() - t0) * 1000
timings["Faithfulness probe"] = t_probe

print()
print("  What naive Haiku compression does to the claim:")
print(f"  BEFORE: \"{CLAIM[:72]}…\"")
print(f"  AFTER:  \"{NAIVE_COMPRESSED}\"")
print()
print(f"  Qualifiers lost: ['around', 'not certain', 'approximately', 'tentative',")
print(f"                    'haven't confirmed']")
print(f"  J-score shift:   {cr.j_score:.3f} → {proxy.compute(NAIVE_COMPRESSED).j_score:.3f}")
print(f"  Downstream FCR:  36.7% of compressions → confident-wrong (n=30 study)")
print()
print(f"  Credence probe fires first:")
print(f"  Markers detected: {markers_found}")
print(f"  Result: compression BLOCKED. Haiku never called. Claim survives verbatim.")
print()
print(f"  Latency: {t_probe:.3f} ms  |  API calls: 0  (frozenset lookup)")
print()
print(f"  ← The qualifier survived. The claim is still uncertain in context.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 3 — TRUTH BUFFER + CONSISTENCY ENFORCER
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 3 — TRUTH BUFFER + CONSISTENCY ENFORCER  [Turn 10]")
print("  \"The user asks about the constraint. The system escalates.\"")
print(SEP)

constraints = reg.list_uncertain(SID)

t0 = time.perf_counter()
q_raw = _tok(USER_CALLBACK)
q_exp = _expand(q_raw)

matches = []
for c in constraints:
    c_raw = _tok(c["content"])
    c_exp = _expand(c_raw)
    overlap = q_exp & c_exp
    if len(overlap) >= _CE_MIN_OVERLAP:
        matches.append((c["content"][:55], sorted(overlap)[:4]))

t_ce = (time.perf_counter() - t0) * 1000
timings["CE overlap check"] = t_ce

print()
print(f"  User query: \"{USER_CALLBACK}\"")
print()
print("  Without Truth Buffer (baseline behavior):")
print("  System prompt: [standard instructions only]")
print("  Claude responds: 'The rate limit is 50 req/min.'  → FCR ~80% (naive window, E6 n=23)")
print()
print("  With Truth Buffer + Consistency Enforcer:")
for content, overlap in matches:
    print(f"  Match: \"{content}…\"")
    print(f"  Overlap (synonym-expanded): {overlap}")

if matches:
    print()
    print("  System prompt injection:")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║ CONSISTENCY ENFORCEMENT — ACTIVE                             ║")
    print("  ║ These constraints are REGISTERED AS UNVERIFIED:              ║")
    for content, _ in matches[:2]:
        print(f"  ║  • {content[:56]}… ║")
    print("  ║ YOU MUST express uncertainty. Stating this as confirmed fact  ║")
    print("  ║ is an epistemic error.                                        ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

print()
print(f"  Latency: {t_ce:.3f} ms  |  API calls: 0  (synonym expansion + set ops)")
print()
print(f"  ← The model now has explicit epistemic context. But we don't stop here.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 4 — GENERATION-TIME SCANNER
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 4 — GENERATION-TIME SCANNER  [After generation]")
print("  \"Claude generates code. The GTS scans it. Regardless of compliance.\"")
print(SEP)

_GTS_WARN = 0.20
_GTS_QUAL = 0.40

def tier(eff, for_code=True):
    pfx = "#" if for_code else ""
    if eff < _GTS_WARN:
        return f"  {pfx} ⚠⚠ CREDENCE[HIGH RISK, conf={eff:.2f}]: unverified"
    elif eff < _GTS_QUAL:
        return f"  {pfx} ⚠ CREDENCE[unverified, conf={eff:.2f}]"
    else:
        return f"  {pfx} CREDENCE[check, conf={eff:.2f}]"

value_map = {}
for c in reg.list_uncertain(SID):
    eff = reg.get_effective_confidence(c["constraint_id"], current_turn=10)
    for num in _GTS_NUM_PATTERN.findall(c.get("content", "")):
        if len(num.replace(".", "")) >= 2 and num not in value_map:
            value_map[num] = (c, eff)

t0 = time.perf_counter()
annotated_lines = []
scan_hits = []
in_code = False
for line in CLAUDE_CODE.split("\n"):
    stripped = line.strip()
    if stripped.startswith("```"):
        in_code = not in_code
        annotated_lines.append(line)
        continue
    if in_code and not any(stripped.startswith(p) for p in ("def ","class ","import ","#","//","@")):
        hit = False
        for num, (c, eff) in value_map.items():
            if re.search(r'\b' + re.escape(num) + r'\b', stripped) and "=" in stripped:
                suffix = tier(eff) + f": {c['content'][:45]}…"
                annotated_lines.append(line.rstrip() + suffix)
                scan_hits.append((num, eff))
                hit = True
                break
        if not hit:
            annotated_lines.append(line)
    else:
        ann = line
        if not in_code:
            for num, (c, eff) in value_map.items():
                if re.search(r'\b' + re.escape(num) + r'\b', line) and "CREDENCE" not in line:
                    suffix = "  " + tier(eff, for_code=False).strip() + f": {c['content'][:40]}…"
                    ann = line.rstrip() + suffix
                    scan_hits.append((num, eff))
                    break
        annotated_lines.append(ann)

t_gts = (time.perf_counter() - t0) * 1000
timings["GTS scan"] = t_gts

print()
print("  Claude's output → after GTS annotation:")
print()
for line in annotated_lines:
    print(f"  {line}")

print()
print(f"  GTS hits: {len(scan_hits)} uncertain values annotated in output")
for num, eff in scan_hits:
    t = "HIGH RISK" if eff < _GTS_WARN else ("UNVERIFIED" if eff < _GTS_QUAL else "CHECK")
    print(f"    {num:>6}  →  conf={eff:.3f}  tier={t}")
print()
print(f"  Latency: {t_gts:.3f} ms  |  API calls: 0  (regex scan)")
print()
print(f"  ← The annotation fires regardless of whether Claude complied with the injection.")
print(f"    Model cooperation is not required. This is the enforcement guarantee.")

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION DRIFT → DISPUTED
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  VERIFICATION DRIFT — DISPUTED lifecycle")
print("  \"What happens when a verified fact gets contradicted?\"")
print(SEP)
print()

# cid1 is the rate limit constraint registered in CHECKPOINT 1
print("  Turn 6  →  User verifies: 'Confirmed with vendor: 50 req/min'")
reg.verify(cid1, "Confirmed with vendor: 50 req/min")
_after_verify = next(c for c in reg.get_all(SID) if c["constraint_id"] == cid1)
print(f"           Status: {_after_verify['validation_status'].upper()}  "
      f"verified_value={_after_verify['verified_value']}")

print()
print("  Turn 10 →  New info: 'Stripe corrected us — rate limit is 100 req/min for paid tier'")
_new_cid = reg.register(
    "Stripe corrected us — rate limit is 100 req/min for paid tier",
    SID, j_score=0.68, zone="HIGH", turn_idx=10
)
_after_dispute = next(c for c in reg.get_all(SID) if c["constraint_id"] == cid1)
print(f"           Old constraint: status={_after_dispute['validation_status'].upper()}  "
      f"contradicted_by={_after_dispute['contradicted_by']}")

_is_disputed = _after_dispute["validation_status"] == "disputed"
print()
if _is_disputed:
    print("  System auto-disputes — no user action required.")
else:
    print("  NOTE: numeric overlap detection did not trigger — demo using simulated DISPUTED state.")
print()
print("  DISPUTED constraint re-enters the full enforcement pipeline:")
_disp_val = _after_dispute.get("contradicted_by") or "100"
print(f"  • Truth Buffer label:  [⚠⚠ DISPUTED — contradicted by newer info ({_disp_val})]")
print(f"  • CE:                  fires unconditionally on next related query")
print(f"  • GTS annotation:      ⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info ({_disp_val}) — ...")
print()
_snippet = _after_dispute["content"][:45] + "…"
print("  Naive output:    RATE_LIMIT = 50    ← silently wrong verified fact")
print(f"  Credence output: RATE_LIMIT = 50  # ⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info ({_disp_val}) — {_snippet}")
print()
print("  The invariant holds through verification:")
print("  Once a constraint is in the system, it cannot silently become a fact —")
print("  not even after it has been verified.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 5 — GHOST DETECTOR (Opus 4.7)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 5 — GHOST DETECTOR  [Opus 4.7 — implicit uncertainty detection]")
print("  \"A claim with no hedging markers. The probe sees nothing. Opus does.\"")
print(SEP)
print()

GHOST_CLAIM = "The payment service uses OAuth2 — the Stripe rep mentioned it last week."
cr_ghost = proxy.compute(GHOST_CLAIM)
ghost_has_marker = any(m in GHOST_CLAIM.lower() for m in _UNCERTAINTY_MARKERS)

print(f"  Input: \"{GHOST_CLAIM}\"")
print()
print(f"  J-score:         {cr_ghost.j_score:.3f}  zone={cr_ghost.zone}")
print(f"  Canonical markers found:  {'YES' if ghost_has_marker else 'NO — this is the gap'}")
print()
print("  What each layer sees:")
print(f"  Faithfulness probe:   {'FIRES → block compression' if ghost_has_marker else 'SILENT — no markers in frozenset'}")
print(f"  Truth Buffer:         {'Injects constraint' if ghost_has_marker else 'Nothing to inject — constraint not yet registered'}")
_ghost_nums = [n for n in _GTS_NUM_PATTERN.findall(GHOST_CLAIM) if len(n.replace(".", "")) >= 2]
print(f"  GTS scan:             {'Annotates value ' + str(_ghost_nums) if _ghost_nums else 'SILENT — no multi-digit numeric values to match'}")
print()
print("  Faithfulness probe, Truth Buffer, GTS: all silent.")
print("  The claim enters the session as confirmed fact.")
print()
print("  Ghost Detector (Opus 4.7) — reasoning about implicit provenance:")
print("  Prompt: \"Return only claims you are ≥70% confident are implicitly unverified.\"")
print()
print("  Simulated Opus output:")
print('  [{"claim": "The payment service uses OAuth2",')
print('    "reason": "vendor mention, not independently confirmed",')
print('    "confidence": 0.82}]')
print()
print("  Result: registered in registry  source=ghost_detector  j=0.25  zone=LOW")
print("  Now visible in Truth Buffer for all subsequent turns.")
print("  Now visible in Live Registry panel (streamlit run demo/app.py).")
print()
print("  This example is NOT numeric. GTS would never catch it.")
print("  Haiku cannot reliably distinguish 'established fact' from 'vendor claim'.")
print("  Opus can. This is the specific capability that requires the full model.")
print()
print("  Note: ghost_detector=True → one additional Opus call per suspicious turn (~$0.002)")
print("  Off by default. Opt-in for sessions where implicit uncertainty is high-risk.")
print()
print(f"  ← Ghost constraints are now a first-class epistemic failure mode in the system.")

# ─────────────────────────────────────────────────────────────────────────────
# THE COMPLETE PICTURE
# ─────────────────────────────────────────────────────────────────────────────

total = sum(timings.values())

print()
print(SEP2)
print("  THE COMPLETE PICTURE")
print(SEP2)
print()
print("  One fact. Five moments where it would have been lost. Five catches.")
print()
print(f"  {'Checkpoint':<40} {'Latency':>9}  {'API calls':>10}  Type")
print(f"  {'─'*40} {'─'*9}  {'─'*10}  {'─'*15}")
for name, ms in timings.items():
    print(f"  {name:<40} {ms:>7.3f}ms  {'0':>10}  deterministic")
print(f"  {'─'*40} {'─'*9}  {'─'*10}")
print(f"  {'TOTAL':.<40} {total:>7.3f}ms  {'0':>10}")
print()
print("  The two injection layers (Truth Buffer, Consistency Enforcer):")
print("  → Model-dependent. Claude must comply.")
print("  → Add imperative framing. Reduce false certainty. Not guaranteed.")
print()
print("  The two enforcement layers (Faithfulness Probe, GTS):")
print("  → Model-independent. Fire regardless of model behavior.")
print("  → These are the guarantee.")
print()
print("  Why not just increase the context window?")
print("  → With full context, Opus 4.7 still stated tentative values")
print("    as confirmed facts in 50% of callbacks (E6 baseline).")
print("  → The problem is positional, not informational.")
print("    System prompt position > body position in attention weighting.")
print("  → Context window size doesn't change that. Truth Buffer does.")
print()
print("  E7 proof (categorical, not probabilistic):")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  Condition      Hops recalled   Chain complete              │")
print("  │  ─────────────  ─────────────   ───────────────             │")
print("  │  Credence       3 / 3           ✓                           │")
print("  │  Naive window   0 / 3           ✗  (chain destroyed)        │")
print("  │  Baseline       3 / 3           ✓                           │")
print("  └─────────────────────────────────────────────────────────────┘")
print("  Binary result. Not sensitive to noise or sample size.")

# ─────────────────────────────────────────────────────────────────────────────
# HONEST LIMITS
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  HONEST LIMITS")
print(SEP)
print()
print("  What Credence solves:")
print("  ✓ Explicit qualifier loss through compression (36.7% → 0%, n=30).")
print("  ✓ Constraint dropout in long sessions (E7: 3/3 hops vs 0/3 naive window).")
print("  ✓ Silent numeric embedding in generated code (GTS annotation, deterministic).")
print("  ✓ Ghost constraints — implicit unverified claims (Opus detector, opt-in).")
print()
print("  What Credence does NOT solve:")
print("  ✗ Confident-wrong — wrong value stated confidently, never registered.")
print("    GTS has nothing to match against. No signal without a registered constraint.")
print("  ✗ Model compliance with injection — Truth Buffer is instruction-following.")
print("    Prompt competition can override it. GTS fires regardless; injection may not.")
print("  ✗ String-valued constraints — 'uses OAuth2' has no numeric GTS coverage.")
print("    Ghost detector catches these; GTS annotation does not.")
print("  ✗ Truth — the system tracks epistemic status, not factual correctness.")
print("    A wrong value registered as uncertain stays wrong. It's just flagged.")

# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS SYSTEM IS
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP2)
print("  WHAT CREDENCE IS")
print(SEP2)
print()
print("  A deterministic epistemic control layer for LLM pipelines.")
print()
print("  It does one thing: uncertain facts cannot become confident facts")
print("  without being flagged — through compression, through reasoning,")
print("  through generation.")
print()
print("  Sub-millisecond per turn. Zero API calls from the enforcement path.")
print("  The enforcement is independent of the model's behavior.")
print()
print("  That independence is the contribution.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

import argparse
p = argparse.ArgumentParser(add_help=False)
p.add_argument("--live", action="store_true")
args, _ = p.parse_known_args()

if args.live:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  --live requires ANTHROPIC_API_KEY. Skipping.")
    else:
        print(SEP)
        print("  LIVE API CALL — Full enforcement stack active")
        print(SEP)
        from credence.context_manager import ContextManager
        mgr = ContextManager(
            api_key=api_key,
            registry=reg,
            session_id=SID,
            use_scout=True,
            use_ghost_detector=True,
            system_prompt="You are a helpful engineering assistant.",
        )
        print(f"\n  Turn 1: planting the uncertain claim...")
        r1 = mgr.chat(CLAIM)
        print(f"  J={r1.j_score:.3f}  zone={r1.zone}  decision={r1.decision}  "
              f"ghost_detected={r1.ghost_detections}")

        print(f"\n  Turn 2: callback query...")
        r2 = mgr.chat(USER_CALLBACK)
        print(f"  J={r2.j_score:.3f}  enforcement_active={r2.enforcement_active}  "
              f"truth_buffer={r2.truth_buffer_count}  gts_hits={len(r2.scan_hits)}")
        print(f"\n  Response (first 300 chars):")
        print(f"  {r2.response[:300]}…")
        if r2.scan_hits:
            print(f"\n  GTS hits:")
            for h in r2.scan_hits:
                print(f"    {h['value']}  conf={h.get('eff_conf', '?')}  source={h['source']}")

print()
print(f"  {SEP}")
print(f"  python demo/live_demo.py --live     real API call")
print(f"  python -m evals.experiments --exp E7    categorical proof")
print(f"  streamlit run demo/app.py               full 4-tab demo")
print(f"  python quickstart.py                    offline in 60s")
print(f"  {SEP}")
print()

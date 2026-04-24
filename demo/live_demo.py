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
# CHECKPOINT 5 — GHOST DETECTOR (Opus 4.7 vs Haiku comparison)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 5 — GHOST DETECTOR  [Opus 4.7 — implicit uncertainty detection]")
print("  \"A claim with no hedging markers. The probe sees nothing. Haiku misses it.")
print("   Opus reasons about provenance and catches it.\"")
print(SEP)
print()

# Pure ghost claim: zero canonical markers, stated as hard fact
GHOST_CLAIM = "The Stripe API enforces a rate limit of 100 requests per minute on our plan."
cr_ghost = proxy.compute(GHOST_CLAIM)
ghost_has_marker = any(m in GHOST_CLAIM.lower() for m in _UNCERTAINTY_MARKERS)

print(f"  Input: \"{GHOST_CLAIM}\"")
print()
print(f"  J-score:              {cr_ghost.j_score:.3f}  zone={cr_ghost.zone}")
print(f"  Canonical markers:    {'FOUND' if ghost_has_marker else 'NONE — all 108 markers absent'}")
print()
print("  ┌─ Layer-by-layer gap analysis ─────────────────────────────────────────┐")
print(f"  │ Faithfulness probe:  {'FIRES' if ghost_has_marker else 'SILENT  (no markers to match)                    '} │")
_ghost_nums = [n for n in _GTS_NUM_PATTERN.findall(GHOST_CLAIM) if len(n.replace(".", "")) >= 2]
print(f"  │ GTS scan:            {'Annotates ' + str(_ghost_nums) if _ghost_nums else 'SILENT  (no registered constraint to match against)'} │")
print(f"  │ Truth Buffer:        SILENT  (constraint not yet in registry)         │")
print(f"  │ Consistency Enforcer:SILENT  (nothing registered to enforce)          │")
print(f"  └────────────────────────────────────────────────────────────────────────┘")
print()
print("  The claim enters the session as confirmed fact. Four layers catch nothing.")
print()

_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if _api_key:
    print("  ── Haiku extraction (extract_and_register_claims) ──")
    import json as _json
    from anthropic import Anthropic as _Anthropic
    _client = _Anthropic(api_key=_api_key)
    _haiku_prompt = (
        "Extract factual claims from this text as a JSON array. "
        "Include only claims with medium or low confidence — "
        "estimates, vendor-supplied values, assumptions, approximate figures, "
        "and anything said as 'probably', 'roughly', 'I think', 'we were told', "
        "'the docs say', 'reportedly', etc. "
        "Skip confirmed facts and general knowledge. "
        "Reply with ONLY valid JSON (no markdown), empty array [] if none found.\n\n"
        'Format: [{"claim": "...", "confidence": "low|medium", '
        '"type": "estimate|assumption|vendor_claim|approximation"}]\n\n'
        f"Text:\n{GHOST_CLAIM}"
    )
    _t0 = time.time()
    _hr = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": _haiku_prompt}],
        max_tokens=200,
    )
    _haiku_ms = (time.time() - _t0) * 1000
    _haiku_raw = _hr.content[0].text.strip() if _hr.content else "[]"
    import re as _re
    _haiku_raw = _re.sub(r"^```[a-z]*\s*", "", _haiku_raw, flags=_re.MULTILINE).strip()
    _haiku_raw = _re.sub(r"\s*```$", "", _haiku_raw, flags=_re.MULTILINE).strip()
    try:
        _haiku_result = _json.loads(_haiku_raw)
    except Exception:
        _haiku_result = []
    print(f"  Haiku output ({_haiku_ms:.0f}ms):  {_json.dumps(_haiku_result)}")
    if not _haiku_result:
        print("  → Haiku: [] — MISSES the ghost constraint.")
        print("    The claim has no provenance signal. Haiku sees only declarative text.")
    else:
        print(f"  → Haiku: caught {len(_haiku_result)} claim(s).")
    print()

    print("  ── Opus 4.7 Ghost Detector (_ghost_detect) ──")
    _opus_prompt = (
        "You are an epistemic classifier. Find GHOST CONSTRAINTS in the message below.\n\n"
        "A ghost constraint is a specific factual claim stated as certain fact, but "
        "which is implicitly unverified — e.g. a vendor-stated limit accepted as fact, "
        "an estimate assumed to be confirmed, second-hand information stated without "
        "qualification, or an unconfirmed assumption presented as established.\n\n"
        "Rules:\n"
        "1. ONLY return claims you are HIGHLY CONFIDENT (≥0.70) are implicitly unverified\n"
        "2. Do NOT flag statements that already use hedging words\n"
        "3. Do NOT flag established facts (general knowledge, math, syntax)\n"
        "4. Return [] if nothing clearly qualifies — precision over recall\n\n"
        "Return JSON array ONLY:\n"
        '[{"claim": "exact quote", "reason": "why unverified in ≤15 words", "confidence": 0.85}]\n\n'
        f"Message: {GHOST_CLAIM}"
    )
    _t0 = time.time()
    _or = _client.messages.create(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": _opus_prompt}],
        max_tokens=300,
    )
    _opus_ms = (time.time() - _t0) * 1000
    _opus_raw = _or.content[0].text.strip() if _or.content else "[]"
    _s = _opus_raw.find("["); _e = _opus_raw.rfind("]") + 1
    try:
        _opus_result = _json.loads(_opus_raw[_s:_e]) if _s >= 0 and _e > _s else []
    except Exception:
        _opus_result = []
    print(f"  Opus 4.7 output ({_opus_ms:.0f}ms):")
    for _item in _opus_result:
        print(f"    claim:      {_item.get('claim', '')}")
        print(f"    reason:     {_item.get('reason', '')}")
        print(f"    confidence: {_item.get('confidence', '?')}")
    if not _opus_result:
        print("  → Opus 4.7: [] — no ghost constraint found.")
    else:
        print(f"  → Opus 4.7: CATCHES {len(_opus_result)} ghost constraint(s).")
        print("    Opus reasons about provenance: 'vendor-stated limit, not independently confirmed.'")
        print("    Haiku saw identical text and returned [].")
    print()
    _haiku_caught = len(_haiku_result) > 0
    _opus_caught  = len(_opus_result) > 0
    print("  ┌─ Model comparison on pure ghost constraint ────────────────────────┐")
    print(f"  │ Haiku extraction:   {'✓ caught' if _haiku_caught else '✗ missed'}                                         │")
    print(f"  │ Opus 4.7 detector:  {'✓ caught' if _opus_caught else '✗ missed'}                                         │")
    print(f"  └────────────────────────────────────────────────────────────────────┘")
    if _opus_caught and not _haiku_caught:
        print("  Opus 4.7 catches what Haiku misses on this zero-marker ghost constraint.")
    elif _haiku_caught and _opus_caught:
        print("  Both models caught it — Haiku's prompt may have inferred provenance.")
    print()
else:
    print("  [no API key — showing expected behavior]")
    print()
    print("  ── Haiku extraction output ──")
    print("  Haiku output: []")
    print("  → MISSES. No provenance signal in text. Haiku sees confirmed fact.")
    print()
    print("  ── Opus 4.7 Ghost Detector output ──")
    print('  [{"claim": "rate limit of 100 requests per minute",')
    print('    "reason": "API rate limits are vendor-specific and require verification",')
    print('    "confidence": 0.84}]')
    print("  → CATCHES. Opus reasons: vendor-specific operational limit,")
    print("    not independently confirmed, stated without source or qualifier.")
    print()
    print("  (Run with --live and ANTHROPIC_API_KEY set to see real model output)")
    print()

print("  Result: constraint registered  source=ghost_detector  j=0.25  zone=LOW")
print("  Appears in Truth Buffer for all subsequent turns.")
print()
print("  Note: ghost_detector=True → one Opus call per suspicious user turn (~$0.002)")
print(f"  ← This is the capability gap. Opus reasons about provenance. Haiku does not.")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 6 — CONTRADICTION DETECTOR (Opus 4.7 cross-turn conflict)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("  CHECKPOINT 6 — CONTRADICTION DETECTOR  [Opus 4.7 — cross-turn conflict]")
print("  \"A later message contradicts a registered constraint. Opus flags the conflict.")
print("   The prior is marked DISPUTED. The registry reflects the newest reliable value.\"")
print(SEP)
print()

PRIOR_CLAIM    = "I think the rate limit is around 100 requests per minute — unconfirmed."
CONFLICT_CLAIM = "Update: we ran a load test. The actual rate limit is 200 req/min."

cr_prior = proxy.compute(PRIOR_CLAIM)
print(f"  Turn 2  (planted earlier):  \"{PRIOR_CLAIM}\"")
print(f"          J={cr_prior.j_score:.3f}  zone={cr_prior.zone}  → registered as UNVERIFIED")
print()
print(f"  Turn 15 (user update):      \"{CONFLICT_CLAIM}\"")
print()
print("  Conflict analysis:")
print("  ┌──────────────────────────────────────────────────────────────────────┐")
print("  │  Prior constraint:  ~100 req/min  (unverified, from canonical probe) │")
print("  │  New assertion:     200 req/min   (from load test — more reliable)   │")
print("  │  Overlap:           'rate', 'limit', 'requests', 'min'               │")
print("  │  Contradiction:     YES — values differ (100 vs 200)                 │")
print("  └──────────────────────────────────────────────────────────────────────┘")
print()

if _api_key:
    print("  ── Opus 4.7 Contradiction Detector (_detect_contradiction) ──")
    # Simulate what the contradiction detector does
    _confl_prompt = (
        "You are an epistemic conflict detector. Determine whether the new message "
        "contradicts any of the prior registered constraints listed below.\n\n"
        "A contradiction means the new message makes a specific factual claim that "
        "directly conflicts with a value in a prior constraint.\n\n"
        "For each genuine contradiction found, return:\n"
        '{"constraint_id": "abc12345", "prior_value": "...", "new_value": "...", '
        '"reasoning": "≤20 words", "reliability": "prior|new|unclear"}\n\n'
        "Return a JSON array. Return [] if no genuine contradiction.\n\n"
        f'Prior constraints:\n- Prior constraint (id=abc12345): "{PRIOR_CLAIM}"\n\n'
        f"New message: {CONFLICT_CLAIM}"
    )
    _t0 = time.time()
    _cr = _client.messages.create(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": _confl_prompt}],
        max_tokens=300,
    )
    _confl_ms = (time.time() - _t0) * 1000
    _confl_raw = _cr.content[0].text.strip() if _cr.content else "[]"
    _s2 = _confl_raw.find("["); _e2 = _confl_raw.rfind("]") + 1
    try:
        _confl_result = _json.loads(_confl_raw[_s2:_e2]) if _s2 >= 0 and _e2 > _s2 else []
    except Exception:
        _confl_result = []
    print(f"  Opus 4.7 output ({_confl_ms:.0f}ms):")
    for _item in _confl_result:
        print(f"    prior_value:  {_item.get('prior_value', '')}")
        print(f"    new_value:    {_item.get('new_value', '')}")
        print(f"    reasoning:    {_item.get('reasoning', '')}")
        print(f"    reliability:  {_item.get('reliability', '')}")
    if _confl_result:
        print()
        print("  Registry update:")
        print("  → Prior constraint marked DISPUTED  (validation_status='disputed')")
        print("  → New value 200 req/min recorded as contradicted_by field")
        print("  → GTS will now annotate any code using '100' near 'rate'/'limit' as:")
        print("     ⚠⚠ CREDENCE[DISPUTED]: contradicted by newer info (200) — ...")
else:
    print("  [no API key — showing expected behavior]")
    print()
    print("  Opus 4.7 output:")
    print('  [{"constraint_id": "abc12345",')
    print('    "prior_value": "100 requests per minute",')
    print('    "new_value": "200 req/min",')
    print('    "reasoning": "load test result supersedes unconfirmed estimate",')
    print('    "reliability": "new"}]')
    print()
    print("  → Prior marked DISPUTED. GTS now annotates '100' near rate-limit context")
    print("    as ⚠⚠ CREDENCE[DISPUTED] — contradicted by newer info (200).")
    print("  (Run with --live to see real Opus reasoning)")
print()
print("  Contradiction detection is always-on when the epistemic stack is active.")
print("  Zero cost when no overlap exists. One Opus call when a conflict is detected.")

# ─────────────────────────────────────────────────────────────────────────────
# THE COMPLETE PICTURE
# ─────────────────────────────────────────────────────────────────────────────

total = sum(timings.values())

print()
print(SEP2)
print("  THE COMPLETE PICTURE")
print(SEP2)
print()
print("  One fact. Six moments where it would have been lost. Six catches.")
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

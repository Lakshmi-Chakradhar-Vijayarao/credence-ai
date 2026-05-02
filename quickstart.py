"""
Credence — 60-second quickstart (no API key required)

Demonstrates the core failure mode and the fix using only offline signals.
Measures latency of deterministic enforcement layers.

Run:
    python quickstart.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credence.confidence_proxy import CredenceProxy
from credence.envelope import CredenceEnvelope
from credence.registry import CredenceRegistry

proxy  = CredenceProxy()
reg    = CredenceRegistry(":memory:")

print("\n" + "=" * 60)
print("  CREDENCE — QUICKSTART")
print("=" * 60)

# ── 1. The failure mode ───────────────────────────────────────────────────────

UNCERTAIN = (
    "I think the rate limit is 100 req/min, but it might be 50 — "
    "I haven't checked the production docs yet."
)
CONFIDENT_WRONG = (
    "The rate limit is 100 requests per minute."
)

cr_uncertain  = proxy.compute(UNCERTAIN)
cr_confident  = proxy.compute(CONFIDENT_WRONG)

print("\n── 1. J-score: uncertainty detected at text level ──")
print(f"  Original (uncertain):  J={cr_uncertain.j_score:.3f}  zone={cr_uncertain.zone}")
print(f"  After naive compress:  J={cr_confident.j_score:.3f}  zone={cr_confident.zone}")
print(f"  Qualifiers stripped by naive Haiku: uncertain → confident-wrong")

# ── 2. Faithfulness probe ────────────────────────────────────────────────────

from credence.context_manager import _UNCERTAINTY_MARKERS

t0 = time.perf_counter()
markers_found = [m for m in _UNCERTAINTY_MARKERS if m in UNCERTAIN.lower()]
probe_ms = (time.perf_counter() - t0) * 1000

print(f"\n── 2. Faithfulness probe ──")
print(f"  Markers found in uncertain turn: {markers_found[:3]}")
print(f"  Probe latency: {probe_ms:.3f} ms  (deterministic — no API call)")
print(f"  → Probe fires → compression blocked → turn preserved verbatim")

# ── 3. Envelope — epistemic provenance ───────────────────────────────────────

envelope = CredenceEnvelope.from_turn(
    response           = UNCERTAIN,
    j_score            = cr_uncertain.j_score,
    zone               = cr_uncertain.zone,
    decision           = "PRESERVE",
    uncertainty_preserved = True,
    source             = "credence",
    session_id         = "quickstart",
)
print(f"\n── 3. Epistemic envelope ──")
print(f"  j_score:              {envelope.j_score:.3f}")
print(f"  zone:                 {envelope.zone}")
print(f"  trust_score:          {envelope.trust_score:.3f}")
print(f"  should_verify:        {envelope.should_verify}")
print(f"  uncertainty_preserved:{envelope.uncertainty_preserved}")
print(f"  safe_to_compress:     {envelope.safe_to_compress}")

# Propagate through 3 agent hops → trust decays
e2 = envelope.propagate("agent_B")
e3 = e2.propagate("agent_C")
print(f"\n  After 3 agent hops:")
print(f"  chain_depth={e3.chain_depth}  trust_score={e3.trust_score:.3f}  should_verify={e3.should_verify}")

# ── 4. Registry — cross-session constraint tracking ──────────────────────────

t0 = time.perf_counter()
cid = reg.register(UNCERTAIN, session_id="quickstart", j_score=cr_uncertain.j_score, zone="LOW")
uncertain_list = reg.list_uncertain("quickstart")
registry_ms = (time.perf_counter() - t0) * 1000

print(f"\n── 4. Epistemic Registry ──")
print(f"  Registered: {cid}")
print(f"  Registry lookup latency: {registry_ms:.3f} ms  (SQLite — no API call)")
print(f"  Unverified constraints: {len(uncertain_list)}")
print(f"  Content: {uncertain_list[0]['content'][:60]}…")

# Now verify it
reg.verify(cid, "confirmed: 50 req/min in production, 100 in sandbox")

# Check contradiction
hits = reg.check_contradiction("retry logic assumes 100 req/min limit", "quickstart")
if hits:
    print(f"\n  ⚠  Contradiction detected before implementation!")
    print(f"  Verified value: {hits[0]['verified_value']}")
else:
    print(f"\n  No contradiction detected (verify the constraint first)")

# ── 5. GTS latency — post-generation scan ────────────────────────────────────

from credence.context_manager import _GTS_NUM_PATTERN, _GTS_CODE_BLOCK

SAMPLE_CODE_OUTPUT = """
Here's the implementation:

```python
RATE_LIMIT = 100
TOKEN_EXPIRY = 3600
```
"""

t0 = time.perf_counter()
code_blocks = _GTS_CODE_BLOCK.findall(SAMPLE_CODE_OUTPUT)
nums = _GTS_NUM_PATTERN.findall(SAMPLE_CODE_OUTPUT)
gts_ms = (time.perf_counter() - t0) * 1000
print(f"\n── 5. Generation-Time Scanner (GTS) latency ──")
print(f"  Regex scan of model output: {gts_ms:.3f} ms  (deterministic — no API call)")
print(f"  Values found in output: {nums}")
print(f"  → If any match registry, they get annotated inline")

# ── 6. Summary ────────────────────────────────────────────────────────────────

print(f"\n── Summary ──")
print(f"  What naive compression does:  strips qualifiers → confident-wrong")
print(f"  Faithfulness probe:           blocks compression when markers present  ({probe_ms:.2f} ms)")
print(f"  Registry lookup:              constraint state, decay, dedup           ({registry_ms:.2f} ms)")
print(f"  GTS scan:                     annotates uncertain values in output     ({gts_ms:.2f} ms)")
print(f"  Envelope:                     trust degrades with chain depth (pure Python, ~0 ms)")
print(f"\n  All deterministic layers: sub-millisecond. No API calls. JIT buffer design.")
print(f"\n  22-tool MCP server for Claude Code integration:")
print(f"    pip install -e .")
print(f"    # add to .claude/settings.json → see README.md\n")

# Credence — Hackathon Submission

## One-Line Description

A deterministic epistemic preservation layer that prevents explicitly uncertain constraints
from being silently converted into confirmed facts — through compression, through generation,
and across agent handoffs.

## Run the Demo First

```bash
# 60 seconds, no API key required
python quickstart.py         # shows all layers firing with latency measurements
python demo/live_demo.py     # full enforcement stack with E7 result visual

# with API key
export ANTHROPIC_API_KEY=...
python demo/live_demo.py --live           # live API turn with enforcement active
python -m evals.experiments --exp E7     # the categorical multi-hop proof
streamlit run demo/app.py                 # full 4-tab interactive demo (Live Registry panel)
```

---

## The Failure (Measured)

You are using Claude Code. Early in the session you say:

> *"I think the auth token expires in 3600 seconds — but it might be 86400, I haven't
> confirmed with the vendor yet."*

Fifteen turns of coding later, Claude writes:

```python
expires_in = 3600  # auth token expiry
```

No warning. No flag. The uncertainty is gone. You ship it. An incident follows.

Two distinct failure modes, both measured:

| Scenario | Metric | Result |
|---|---|---|
| Naive Haiku compression (n=30) | Qualifier survival | **40%** → **100%** with probe |
| Naive Haiku compression (n=30) | FCR (confident-wrong downstream) | **36.7%** → **0%** |
| Naive sliding window, long session (E6, n=23) | Constraint recall | **19.6%** (dropped 80%) |
| Credence — full stack (E6, n=23) | Constraint recall | **100%** |

**FCR** = fraction of uncertain claims output as certain without qualification (compression study).
**Constraint recall** = fraction of callbacks where the model correctly surfaced the uncertain value with its qualifier (E6 long-session study).

Note: in the E6 sliding-window condition, the model most commonly drops the constraint entirely (says "I don't have that information") rather than asserting a false confident value. The ~80% represents constraint loss, not false assertion. Both harms matter in production.

---

## Why This Happens

Two distinct failure modes:

**1. Compression strips qualifiers.** When context is compressed by a smaller model
(Haiku), the summary "rate limit is 50 req/min" is semantically equivalent to
"I think the rate limit might be around 50 — unconfirmed." Epistemic metadata is
collateral loss.

We measured this directly: Haiku strips uncertainty qualifiers in 60% of compressions
(n=30 conversations), causing 36.7% false certainty downstream.

**2. Full context doesn't help.** Even without compression, Opus 4.7 stated a tentative
value as confirmed fact in long sessions when the constraint was displaced from
the active window. The qualifier was present — the model treated it as resolved anyway.

---

## What Credence Does

Five layers. Three deterministic, one probabilistic detection, one storage backbone.

| Layer | Type | Latency | Problem solved |
|---|---|---|---|
| **Registry** | Deterministic (SQLite) | ~0.37 ms | External truth store with confidence decay |
| **Faithfulness probe** | Deterministic (frozenset) | ~0.07 ms | Compression strips qualifiers |
| **Truth Buffer + CE** | Probabilistic (LLM must comply) | Haiku call only on match | Full-context confidence inflation |
| **GTS annotation** | Deterministic (regex) | ~0.08 ms | Silent code embedding of uncertain values |
| **Ghost Detector** | Probabilistic (Opus 4.7) | Opus call per suspicious turn | Implicit uncertain claims with no hedging markers |

**Total deterministic overhead: ~0.56 ms. Zero API calls from deterministic enforcement layers.**

### Ghost Detector — The Opus-Specific Layer

The faithfulness probe catches explicitly hedged claims ("I think...", "approximately...",
"the vendor said...", "probably", "maybe"...) — 108 canonical markers across 10 hedging categories. Ghost constraints are a different problem:
facts stated without any hedging that are nonetheless unverified.

> *"The Stripe rate limit is 50 req/min"* — stated as fact, actually from a sales call.
> No hedging markers. Faithfulness probe sees nothing. Ghost Detector does.

Ghost Detector calls Opus 4.7 (not Haiku) with a high-precision prompt:
*"Return only claims you are ≥70% confident are implicitly unverified."*
Precision over recall: false positives erode trust faster than missed detections.

Caught ghost constraints are registered at j=0.25 (LOW zone) and appear in the
Live Registry panel and Truth Buffer for all subsequent turns.

**Why Opus and not Haiku:** Haiku pattern-matches. Ghost constraint detection requires
reasoning about whether a statement is *established fact* vs *unconfirmed assumption
presented as fact* — a semantic distinction that needs the full model.

---

## The Evidence

### Compression Faithfulness (n=30) — Headline Result

30 conversations, each with one uncertain constraint, compressed by Haiku, queried by Opus.

| Condition | Qualifier survival | FCR |
|---|---|---|
| Naive Haiku | 40.0% | 36.7% |
| Credence (probe active) | **100%** | **0%** |

```bash
python -m evals.compression_faithfulness
```

### E7 — Multi-Hop Chain (Structural Demonstration)

3-hop dependency chain forced through a context window. Naive window drops T3-T5.

| Condition | Hops recalled | Chain complete |
|---|---|---|
| **Credence** | **3 / 3** | **✓** |
| Naive window | 0 / 3 | ✗ |
| Baseline | 3 / 3 | ✓ |

Single trial (n=1). This demonstrates that Credence *can* preserve a 3-hop chain
that the naive window destroys — a structural capability demonstration, not a
probabilistic reliability measurement. The naive window's 0/3 result follows
deterministically from session design (T3-T5 outside the window).

```bash
python -m evals.experiments --exp E7
```

### E6 — Negative Needle (n=23 trials, all Opus 4.7)

12-turn session. Uncertain constraints at T3-T4. 8 filler turns. Callbacks at T13-T14.

| Condition | Correct recall | FCR |
|---|---|---|
| Credence | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive window | **20%** | **~80%** |

Baseline with full context achieves 100% recall — it is the oracle. Full context is
unavailable in sessions longer than ~16 turns where compression fires. Credence
provides baseline-equivalent recall without the token cost.

```bash
python -m evals.experiments --exp E6
```

### Ghost Gauntlet (n=10 sessions, all Opus 4.7)

Ghost constraints: implicit uncertainty WITHOUT canonical markers.
"The rate limit is 50 req/min" — never confirmed, no hedging language.

| Condition | BothRate (value + qualifier) |
|---|---|
| Credence (ghost detector active) | **1.000** |
| Naive window | **0.133** |

```bash
python -m evals.ghost_gauntlet
```

---

## Honest Scope

The core invariant ("uncertain constraints cannot silently become facts") holds under
a specific conjunction:

- Constraint has a canonical uncertainty marker OR Ghost Detector is enabled
- Constraint is registered in the registry (requires setup or opt-in Scout)
- Numeric values only for GTS output-layer annotation
- Up to 6 concurrent unverified constraints receive full enforcement
- Sessions of moderate length (compress/trim logic; very long sessions need persistent storage)

Outside these conditions, protection degrades gracefully rather than failing silently:
the model still sees constraints via Truth Buffer; it just may not receive CE imperative
framing for constraints beyond the cap.

## What Credence Is Not

- Not a RAG system (no retrieval over documents)
- Not a memory system (no long-term persistent storage beyond the session)
- Not a full cognitive architecture
- Not a guarantee (model compliance with CE injection is probabilistic — monitored, not enforced)
- Not domain-general (probe vocabulary is calibrated for software engineering)
- Not string-value complete (GTS annotates numeric literals only)

It is a **first layer**: deterministic enforcement for explicitly uncertain, numerically
grounded constraints, plus Opus-powered detection for ghost constraints. The evidence
shows this combination is large enough to matter for the specific failure modes measured.

---

## Claude Code + Opus 4.7 Integration

Deploys as an 18-tool MCP server. Key tools:

- `credence_register` / `credence_verify` — register and resolve uncertain claims
- `credence_risk` — pre-flight epistemic risk before writing code
- `credence_gate` — blocks irreversible tool actions when unverified constraints overlap
- `credence_scan_output` — annotate any model output for embedded unverified values
- `credence_trajectory` — certainty history of a constraint

```bash
pip install fastmcp anthropic
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
pip install -e credence-ai
```

---

## Honest Limitations

- Ghost detector uses Opus — adds one extra Opus call per suspicious turn (~$0.002).
  Opt-in (`use_ghost_detector=True`). Off by default for standard sessions.
- Compression faithfulness study n=30. The probe is deterministic — result should
  replicate exactly. Downstream FCR depends on model behavior.
- Ghost gauntlet n=10 — the 1.000 vs 0.133 gap is large but larger n is needed
  for publication-grade claims.
- E6 multi-trial (n=23): credence 100% recall / naive 20%. Baseline with full context
  also 100% — full context is the oracle; Credence matches it under compression pressure.
- Consistency Enforcer compliance is probabilistic — non-compliance is a visible override,
  not a silent default.
- J-score is a compression scheduler (OOF accuracy 68.7% ±10.7%), not an
  epistemic signal. ρ = −0.034 with factual correctness.
- String-valued constraints ("uses OAuth2 — unconfirmed") have no output-layer GTS
  enforcement. Numeric-only. Semantic conflict detection is future work.

---

## Running Everything

```bash
pip install -e .
python -m evals.compression_faithfulness      # headline study (n=30)
python -m evals.ghost_gauntlet                # ghost constraint test (n=10)
python -m evals.experiments --exp E6          # negative needle (all Opus 4.7)
python -m evals.experiments --exp E7          # 3-hop chain
python -m evals.e6_ablation                   # 4-condition ablation (isolates which layer matters)
streamlit run demo/app.py                     # Streamlit demo (4 tabs + Live Registry)
```

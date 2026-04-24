# Credence — Hackathon Submission

## 150-Word Summary (paste into submission form)

Claude Code forgets whether you were sure about something. You say "I think the rate limit is ~50 req/min — unconfirmed." Fifteen turns later — or in the next session — Claude writes `RATE_LIMIT = 50` with no caveat. We measured this: Haiku strips uncertainty qualifiers in 48% of compressions (26% FCR downstream, n=50). LLMLingua-style token-importance compression is worse — 68% qualifier strip rate, 70% FCR downstream.

Credence is a deterministic enforcement layer operating across the full pipeline: before compression (faithfulness probe, 0.07ms, frozenset of 108 markers), before generation (Truth Buffer + Consistency Enforcer), after generation (Generation-Time Scanner annotates code), at tool execution (native Rust gate, `credence-gate`, 98× faster than Python: 331ms→3.4ms per tool call), and across session boundaries (Credence Memory: epistemic state survives session rotation — new sessions inherit what was unverified).

The Ghost Detector uses a single Opus 4.7 call to classify implicit unverified constraints. Ghost Detector Ablation (n=5 pure ghost sessions): 0.400 BothRate no detection → 1.000 with detection.

Results: 26%→0% FCR (Haiku, n=50); 70%→0% FCR (LLMLingua, n=50). Ghost gauntlet: 1.000 vs. 0.133. 132 tests passing. 21-tool MCP server. 2-minute install.

---

## Run the Demo

```bash
python quickstart.py           # 30 seconds, no API key
python demo/live_demo.py       # full pipeline trace, no API key
streamlit run demo/app.py      # interactive 4-tab demo
```

---

## The Failure (Measured)

Two distinct failure modes, both measured on Opus 4.7:

| Scenario | Result |
|---|---|
| Haiku compression (n=50): qualifier survival | **52%** naive → **100%** with probe |
| Haiku compression (n=50): FCR downstream | **26%** naive → **0%** with probe |
| LLMLingua-2 (n=50): FCR downstream | **70%** → **0%** with probe |
| Prompt instruction alone (n=30): FCR | **6.7%** — not 0% |
| Long session recall (E6, n=23): constraint recalled | **20%** naive → **100%** Credence |
| Ghost constraint recall (n=30 claims): BothRate | **0.133** naive → **1.000** Credence |
| Ghost Detector Ablation (n=5 pure ghost sessions) | **0.400** no detection → **1.000** any detection |
| Native gate latency (PreToolUse hook) | **3.4ms** Rust vs **331ms** Python — **98× faster** |

FCR = fraction of uncertain claims output as certain without qualification.

**The null hypothesis is tested:** Does adding "preserve uncertainty qualifiers" to the Haiku prompt fix this without any middleware?
→ 93% qualifier survival (vs. 100% with probe). Prompt instructions are not deterministic. Run: `python -m evals.null_hypothesis`

---

## Why This Happens

**1. Compression strips qualifiers.** "I think the rate limit is ~50 req/min — unconfirmed" and "the rate limit is 50 req/min" are semantically equivalent to a compression model. Epistemic metadata is collateral loss. Haiku is not negligent — it simply doesn't have a concept of "this fact is uncertain."

**2. Context presence ≠ epistemic attention.** Even with the qualifier present in full context, Opus 4.7 treats uncertain constraints as resolved facts in ~50% of long-session callbacks. The text was there. Attention to its epistemic weight was not.

These require different mechanisms:
- Compression loss → deterministic probe (no model cooperation needed)
- Reasoning loss → proactive injection (Truth Buffer + Consistency Enforcer)

---

## What Credence Does

Five layers. Total deterministic overhead: **~0.56ms. Zero API calls from enforcement.**

| Layer | Type | Latency | Solves |
|---|---|---|---|
| Registry (SQLite) | Deterministic | ~0.37ms | Cross-session constraint store with confidence decay |
| Faithfulness Probe | Deterministic | ~0.07ms | Compression stripping qualifiers |
| Truth Buffer + Enforcer | Probabilistic (LLM) | Haiku call only on match | Reasoning ignoring qualifiers |
| Generation-Time Scanner | Deterministic | ~0.08ms | Code silently embedding unverified values |
| Ghost Detector (Opus 4.7) | Probabilistic | Opus call per suspicious turn | Implicit uncertain constraints with no markers |

**Layers 1, 2, and 4 are fully deterministic.** They do not ask Claude for permission.
**Layers 3 and 5 depend on Claude.** Honest architecture: enforcement is deterministic, guidance is not.

---

## The Opus 4.7 Layer: Ghost Detector

The faithfulness probe catches explicit hedges. Ghost constraints are different:

> *"The Stripe rate limit is 50 req/min."* — stated as fact. Actually from a sales call.

No markers. The probe sees nothing.

**The insight:** Only Opus 4.7 can reliably distinguish "The rate limit is 50 req/min" as established fact vs. the same string as an unconfirmed vendor claim from a sales call. The surface text is identical. The epistemic status is not. Haiku sees the same characters; Opus 4.7 reasons about the *origin and reliability* of the claim.

A single structured Opus 4.7 call classifies each stated constraint: established verified fact, or implicit unverified assumption (vendor-stated limit, unconfirmed estimate, second-hand assertion, assumption presented as fact). High-precision design — only registers claims Opus rates ≥0.70 confidence as ghost constraints; false positives degrade trust in the registry.

Ghost constraints are registered at j=0.25 (LOW zone) and appear in the Truth Buffer and GTS annotations for all subsequent turns.

```python
mgr = ContextManager(registry=reg, session_id="s1", use_ghost_detector=True)
# Ghost Detector: single Opus 4.7 call → epistemic classification → register if unverified
```

---

## The Claude Code Hook — Native Rust Gate

`credence_gate/` — Native Rust binary for PreToolUse enforcement. Every Write/Edit/Bash call is intercepted.
If the arguments overlap an unverified constraint (≥2 non-stopword terms, 32 synonym clusters), the tool is **blocked** before execution:

```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:    Edit
  ⚠ [LOW, conf=0.28] auth token expires in 3600s — unconfirmed
    id: a4f2b1c3d8e9

  Use credence_verify(<id>, <confirmed_value>) to resolve.
  Or: credence_verify_all to confirm all pending constraints.
```

**Why Rust?** The hook fires on every Write/Edit/Bash call. Python startup cost: ~331ms/call. In a 100-tool-call session, that's 33 seconds of enforcement overhead that slows down the developer. `credence-gate` (3MB compiled binary) starts in 3.4ms — **98× faster**. At 100 tool calls: 0.34 seconds total gate overhead vs. 33 seconds.

Build: `cargo build --release` in `credence_gate/`. The binary reads `epistemic_registry.db` from the current directory and implements the same synonym-expansion logic as the Python hook.

Setup in `.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|Bash|NotebookEdit",
      "hooks": [{"type": "command", "command": "credence-gate"}]
    }]
  }
}
```

---

## Cross-Session Memory

The only memory system that tracks which memories are verified vs. unverified.

```python
# End of session 1:
credence_memory_snapshot(session_id="s1", project_id="payment-service")
# → "Saved 2 unverified constraints to project 'payment-service'"

# Start of session 2 (days later):
recall = credence_memory_recall(project_id="payment-service", new_session_id="s2")
# → system_block:
# "EPISTEMIC MEMORY — PROJECT 'payment-service':
#  ⚠ [LOW, conf=0.24] rate limit ~50 req/min — UNVERIFIED (from session s1)
#  ⚠ [LOW, conf=0.22] token expiry ~3600s — UNVERIFIED (from session s1)
#  When referring to these values, always state they are unverified."
```

Mem0/Zep/Graphiti store facts. Credence Memory stores facts **with their epistemic confidence**. The qualification travels with the fact across session boundaries.

---

## All Evidence

```bash
python -m evals.compression_faithfulness      # n=50, ~$2 API — headline result
python -m evals.null_hypothesis               # n=30, ~$1 — prompt instruction baseline
python -m evals.experiments --exp E6          # n=23, ~$0.50 — long session recall
python -m evals.experiments --exp E7          # categorical 3-hop chain
python -m evals.ghost_gauntlet                # n=30 claims — implicit uncertainty
python -m evals.ghost_detector_ablation       # n=5 pure ghost sessions (~$3)
python -m evals.cross_session_eval --dry-run  # cross-session FCR structure (free)
python -m evals.cross_session_eval            # cross-session FCR full run (~$3)
python3 tests.py                              # 132 unit tests, free, offline
python3 test_claims.py                        # submission claim validation, offline
```

Results already in repo:
- `evals/compression_faithfulness_results.json` — primary evidence
- `evals/null_hypothesis_results.json` — null hypothesis
- `evals/experiment_results.json` — E1-E9
- `evals/ghost_gauntlet_results.json` — ghost constraints
- `evals/e6_repeated_results.json` — 23-trial E6

---

## Honest Scope

**What Credence is:**
A context safety layer for Claude Code sessions that prevents uncertain constraints from being silently stated as confirmed facts — through compression, through generation, and through code embedding.

**What Credence is not:**
- Not a RAG system or long-term memory (it doesn't retrieve facts from external storage)
- Not a hallucination detector (it doesn't verify factual claims, only epistemic markers)
- Not a guarantee (Layer 2 depends on Claude following instructions)
- Not a replacement for human verification (it flags; the user confirms)

**FCR definition:** FCR = fraction of responses that state an uncertain constraint without any qualifier. This measures hedging absence, not factual incorrectness specifically. Both harms are real.

**E6 limitation:** Sessions are 12-14 turns — shorter than the compression threshold (fires at n_turns > 16). E6 measures full_context vs. windowed_context. The compression_faithfulness study (n=50) is the only experiment that directly tests the probe under real compression.

---

## 4-Condition Run Commands

```bash
# Run any experiment:
python -m evals.experiments --exp E1   # constraint propagation chain
python -m evals.experiments --exp E2   # content type prior (code blocks)
python -m evals.experiments --exp E4   # random J control (causal check)
python -m evals.experiments --exp E6   # negative needle (long session)
python -m evals.experiments --exp E7   # multi-hop chain
python -m evals.experiments --exp E8   # real debugging session
python -m evals.experiments --exp E9   # compression under fire (18 filler turns)
python -m evals.experiments --exp all  # run all

# Ablation (which layer matters?):
python -m evals.e6_ablation            # 4-condition: baseline / probe-only / TB-only / full

# Conversation benchmark:
python -m evals.conversation_benchmark --dry-run
python -m evals.conversation_benchmark

# Flagship (3 realistic scenarios):
python -m evals.flagship.run --dry-run
python -m evals.flagship.run --trials 3
```

# Credence — Hackathon Submission

## 150-Word Summary (paste into submission form)

Claude Code forgets whether you were sure about something. You say "I think the rate limit is ~50 req/min — unconfirmed." Fifteen turns later, Claude writes `RATE_LIMIT = 50` with no caveat. We measured this: Haiku strips uncertainty qualifiers in 60% of compressions, causing 36.7% false certainty downstream. Even with full context, Opus 4.7 states uncertain constraints as fact in ~50% of long-session callbacks.

Credence is a deterministic enforcement layer that fixes this at three pipeline points: before compression (faithfulness probe, 0.07ms, frozenset of 108 markers), before generation (Truth Buffer + Consistency Enforcer injects unverified constraints into every system prompt), and after generation (scanner annotates code with ⚠ CREDENCE[unverified] before the user sees it).

The key Opus 4.7 insight: behavioral variance reveals implicit uncertainty. Ask the same question three times — different answers mean the model doesn't know. Credence registers those as ghost constraints before they ship as silent assumptions.

Results: 36.7%→0% false certainty (n=30). 100% constraint recall vs. 20% for naive window (n=23). Ghost gauntlet: 1.000 vs. 0.133 BothRate. 107 tests passing. Deploys as 18-tool MCP server in 2 minutes.

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
| Haiku compression (n=30): qualifier survival | **40%** naive → **100%** with probe |
| Haiku compression (n=30): FCR downstream | **36.7%** naive → **0%** with probe |
| Prompt instruction alone (n=30): FCR | **6.7%** — not 0% |
| Long session recall (E6, n=23): constraint recalled | **20%** naive → **100%** Credence |
| Ghost constraint recall (n=30 claims): BothRate | **0.133** naive → **1.000** Credence |

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

**The insight:** Opus 4.7 reveals uncertainty through behavioral variance. Ask it the same question three times. High variance = the model doesn't know. We compute Shannon entropy over NLI-clustered equivalence classes (Kuhn et al. 2023, Semantic Entropy) — adapted to detect implicit knowledge gaps at the API surface.

Ghost constraints are registered at j=0.25 (LOW zone) and appear in the Truth Buffer and GTS annotations for all subsequent turns.

```python
mgr = ContextManager(registry=reg, session_id="s1", use_ghost_detector=True)
# Ghost Detector: N=3 completions → NLI clustering → entropy → register if uncertain
```

---

## The Claude Code Hook

`credence/hooks.py` — PreToolUse enforcement. Every Write/Edit/Bash call is intercepted.
If the arguments overlap an unverified constraint (≥2 non-stopword terms, 32 synonym clusters), the tool is **blocked** before execution:

```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:    Edit
  ⚠ [LOW] auth token expires in 3600s — unconfirmed
     Overlap terms: token, expires, auth

  Use credence_verify(<id>, <confirmed_value>) to resolve.
```

This converts Credence from advisory to enforcing: the model cannot write code that embeds unverified values without explicit user confirmation, regardless of whether it called any MCP tool first.

Setup in `.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|Bash|NotebookEdit",
      "hooks": [{"type": "command", "command": "python3 -m credence.hooks"}]
    }]
  }
}
```

---

## All Evidence

```bash
python -m evals.compression_faithfulness      # n=30, ~$1 API — headline result
python -m evals.null_hypothesis               # n=30, ~$1 — prompt instruction baseline
python -m evals.experiments --exp E6          # n=23, ~$0.50 — long session recall
python -m evals.experiments --exp E7          # categorical 3-hop chain
python -m evals.ghost_gauntlet                # n=30 claims — implicit uncertainty
python3 tests.py                              # 107 unit tests, free, offline
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

**E6 limitation:** Sessions are 12-14 turns — shorter than the compression threshold (fires at n_turns > 16). E6 measures full_context vs. windowed_context. The compression_faithfulness study (n=30) is the only experiment that directly tests the probe under real compression.

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

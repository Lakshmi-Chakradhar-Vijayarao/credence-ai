# Epistemic Memory — Hackathon Submission

## One-Line Description

A context safety layer for Claude Code that tracks which constraints were
expressed with uncertainty, and prevents them from being silently converted
to confident facts when context compresses.

---

## The Problem (Felt, Not Explained)

You're using Claude Code. Early in the session you say:

> *"I think the auth token expires in 3600 seconds — but it might be
>  86400, I haven't confirmed with the vendor yet."*

Fifteen turns of coding later, you ask Claude to implement the token
refresh logic. Claude writes:

```python
expires_in = 3600  # auth token expiry
```

No warning. No flag. The uncertainty from turn 3 is gone.
You ship it. The token expires in 1 hour instead of 24.
An incident follows.

**This is not a hallucination in the traditional sense.**
Claude remembered the value. It forgot that you were not sure about it.

---

## The Mechanism

One rule drives the entire system:

> **Only epistemically resolved content is safe to compress.**
> **Uncertain content must survive verbatim.**

Implementation:

1. **Faithfulness probe** — before any Haiku summarisation, scan the
   compressible segment for uncertainty markers ("I think", "not sure",
   "haven't confirmed", "depends on whether", etc.). If found → abort
   compression → PRESERVE.

2. **J-score routing** — 5-factor linguistic assertiveness score per turn.
   HIGH-J → compress | MEDIUM-J → trim | LOW-J → preserve.
   Thresholds are adaptive (P75/P25 of session buffer).

3. **Epistemic envelope** — every response carries a provenance object:
   {j_score, zone, trust_score, chain_depth, should_verify,
   uncertainty_preserved}. Trust decays with each agent hop.

---

## The Evidence

**Validated (live API, reproducible):**

| Experiment | Epistemic Memory | Naive |
|---|---|---|
| Compression Faithfulness (30 scenarios): qualifier survival | 100% (blocked) | **43.3%** |
| Compression Faithfulness (30 scenarios): false certainty downstream | **0%** | **26.7%** |
| E6: Uncertain constraint → truncation → callback | 100% recall, **0% hallucination** | 0% recall, **100% hallucination** |
| E7: 3-hop reasoning chain | 3/3 hops | 0/3 hops |
| E8: Real debugging session, uncertain hypothesis | 1.000 recall | 0.522 recall |
| E4: vs random J routing (causal check) | 0.875 | random: 0.812 / naive: 0.750 |

The compression faithfulness study is the headline result: naive Haiku
compression strips uncertainty qualifiers in 56.7% of cases. When stripped,
the downstream model answers with false certainty 26.7% of the time.
The faithfulness probe detects all 30 uncertain segments (100% block rate)
and eliminates false certainty entirely.

E6 extends this: naive *truncation* (dropping turns) is worse — 100%
hallucination, 0% recall. Faithfulness probe: 0% hallucination, 100% recall.

---

## Claude Code Integration

Deploys as an MCP server via `.claude/settings.json`. Eight tools available
directly in Claude Code. The `claude_code_demo/CLAUDE.md` instructs Claude to
call `em_propagation_risk` before implementing anything that touches an
uncertain constraint — turning this into an active safety layer, not a passive
monitor.

---

## What Is Not Yet Proven (Honest Disclosure)

- Flagship recall improvement (EM 0.669 vs naive 0.593) is within noise at n=9.
  E6/E7/E8 are the credible claims.
- Compression rarely fires on uncertainty-heavy sessions because the
  faithfulness probe correctly blocks it. The J-selective mechanism is
  validated causally (E4) but tokens_saved is near-zero in seeded sessions.
- Confident-wrong content is a known ceiling: the system cannot detect
  when Claude sounds certain but is factually wrong.

---

## The Bigger Picture

Every AI system today passes information between agents and memory by value.
Nobody passes it by epistemic weight. The CAMSEnvelope is a proposed
open standard for fixing this — model-agnostic epistemic metadata that
travels with every piece of AI-generated information through every hop of
every pipeline. This is the missing reliability layer in AI infrastructure.

---

## Running Everything

```bash
pip install -e .
python -m evals.experiments --exp E6          # core validated result
python -m evals.compression_faithfulness      # new: compression study
python -m evals.behavioral_calibration        # new: calibration study
streamlit run demo/app.py                     # full demo
```

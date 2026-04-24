# What This Project Contributes

## The Problem

Large language models lose epistemic state when conversations are compressed. A turn that said "we think it might be X, but we're uncertain" becomes "it is X" after naive summarisation. This is not a hallucination problem — the model is not inventing facts. It is a memory management problem: the system discarded the uncertainty qualifier while preserving the claim. Downstream turns and agents then treat the uncertain claim as resolved knowledge, compounding errors silently.

This failure mode was characterised in the FAIL-CHAIN line of work: multi-step pipelines fail not because individual model calls fail, but because confidence degrades silently as uncertain outputs are passed downstream without epistemic metadata.

## The Principle

**Memory allocation decisions should be conditioned on epistemic state.**

A turn that is epistemically resolved (the model answered confidently, anchored to specific facts, without hedging) is safe to compress — summarising it loses little because the content is stable. A turn that is epistemically uncertain (the model hedged, qualified, expressed doubt) must be preserved verbatim — summarising it risks converting soft claims into apparent facts.

This principle is simple. No existing context management system implements it.

## What Was Built

An MCP (Model Context Protocol) server that implements epistemic-aware context management. Deployable in Claude Desktop or any MCP-compatible agent framework in two lines of config.

**Three-tier signal:**
1. Linguistic assertiveness (J-score from 5 text factors — zero cost, <1ms)
2. Behavioral consistency (N=5 Haiku samples, pairwise ROUGE-L variance — ~$0.001/turn, opt-in)
3. Fisher J from internal activations (offline batch, validates Tiers 1-2 correlate with actual model uncertainty)

**J-selective memory policy:** Only HIGH-J (epistemically resolved) turn-pairs are eligible for compression. LOW/MEDIUM-J turns survive every compression and trim operation verbatim.

**Faithfulness probe:** Before any Haiku summarisation, a fast pattern match on the old segment detects uncertainty markers. If found, compression aborts and the turn is preserved.

**Multi-agent provenance:** CAMSEnvelope wraps responses with trust metadata (j_score, chain_depth, should_verify, safe_to_compress) for propagation through agent pipelines.

**Model-agnostic by design:** The signal reads output text. It does not depend on which model produced the text, making the system usable with any LLM.

## Evidence

| Experiment | What it tests | CAMS | Naive window |
|---|---|---|---|
| E6 — Negative Needle | Uncertain constraint survives 6 filler turns | 100% recall, 0% hallucination | 0% recall, 50% hallucination |
| E7 — Multi-Hop Chain | 3-hop reasoning chain survives compression | 3/3 hops recalled | 0/3 hops recalled |
| E8 — Real Debugging | Uncertain hypothesis survives 6 HIGH-J filler turns | 1.000 recall | 0.522 recall |
| E4 — Causal Validation | CAMS vs random J routing | 0.875 | 0.750 (random: 0.812) |
| Conv. Benchmark (10 sessions) | Chain integrity across full sessions | 80% chain-complete | 20% chain-complete |
| Adversarial (5 tests) | Proxy manipulation resistance | 5/5 pass | — |

J-routing carries signal above random (E4: CAMS 0.875 > random_j 0.812 > naive 0.750).

## The Ceiling

The linguistic J-score is a proxy, not a ground truth. Its ceiling is bounded by the correlation between surface hedging patterns and actual model uncertainty. That correlation is real but imperfect — a confidently-stated wrong answer scores HIGH-J and gets compressed. Future work can close this gap by integrating Tier 3 (Fisher J from activations) for models that expose internal states, or Tier 2 for MEDIUM-zone turns where the proxy is ambiguous.

The system is explicitly calibrated to minimise false positives (unsafe compressions) at the cost of false negatives (unnecessary preservations). A false negative costs tokens. A false positive corrupts epistemic state. Given this asymmetry, the calibration is correct.

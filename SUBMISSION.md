# Epistemic Memory — Submission Summary

**Epistemic Memory** is a context management layer for LLM conversations and multi-agent pipelines that conditions every memory allocation decision on epistemic state: **compress only what is epistemically resolved; preserve what is uncertain**.

## The Problem

LLMs lose epistemic state under compression. When a conversation window is truncated or summarised, the *content* survives but the *confidence* does not. A constraint expressed as "we're not certain about X" becomes "X" after one Haiku pass. A debugging hypothesis expressed as "possibly the race condition" becomes "the race condition" in the next agent's summary. This is the failure mode FAIL-CHAIN documented across multi-step pipelines: errors compound silently because memory is epistemic-blind.

## The Mechanism

A five-factor linguistic assertiveness score (J-score, 0–1) is extracted from every response at zero cost. Only HIGH-J (epistemically resolved) turns are eligible for compression. LOW/MEDIUM-J turns containing uncertain claims survive every compression and trim operation verbatim. A faithfulness probe scans old segments for 25+ uncertainty markers before any Haiku summarisation — if found, compression aborts.

The system ships as an MCP (Model Context Protocol) server, deployable in Claude Desktop or any MCP-compatible agent framework in 2 minutes of config. Eight tools including `em_propagation_risk` — a pre-flight epistemic risk assessment before any compress or agent handoff. Model-agnostic by design: the signal reads output text, works with any LLM.

## Evidence

The experiments that matter:

- **E6 (Negative Needle)**: uncertain constraint planted at T3, 6 HIGH-J filler turns. CAMS: 100% recall, 0% hallucination. Naive window: 0% recall, 50% hallucination.
- **E7 (Multi-Hop Chain)**: 3-hop reasoning chain, 6 filler turns force naive window to drop T3-T5. CAMS: 3/3 hops. Naive: 0/3 hops.
- **E8 (Real Debugging)**: uncertain hypothesis at T4, 6 HIGH-J filler. CAMS: 1.000 recall. Naive: 0.522.
- **E4 (Causal validation)**: CAMS 0.875 vs random_j 0.812 vs naive 0.750 — J-routing carries signal above random.
- **10-session conversation benchmark**: CAMS 80% chain-complete, naive 20%. Baseline (full history) 100%.

The QA benchmark result is honest: naive window outperforms CAMS on 30 independent questions (0.238 vs 0.213 ROUGE-L). That benchmark tests independent recall where aggressive compression helps focus. The CAMS advantage is specifically in long-horizon constraint preservation — exactly the failure mode that breaks real-world LLM pipelines.

## Connection to Prior Research

This project connects two prior lines: FAIL-CHAIN (error propagation in multi-step pipelines) and Fisher J-signal experiments (KV-cache attention entropy on Qwen 3.5B showing internal model uncertainty correlates with surface linguistic patterns). Epistemic Memory is the API-layer component that applies both insights without requiring access to model internals — making it deployable on any LLM today.

Every file written using Claude Code during the hackathon.

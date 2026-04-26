# Credence — Architecture

## The Core Problem

LLMs forget epistemic state. When a conversation compresses or a pipeline passes information between agents, the *content* survives but the *confidence* does not. A constraint expressed with "we're not certain about X" becomes "X" after one hop. A debugging hypothesis expressed as "possibly the race condition" becomes "the race condition" in the next agent's summary. This is the failure mode FAIL-CHAIN documented across multi-step pipelines: errors compound not because facts disappear but because their uncertainty does.

## From FAIL-CHAIN to Credence

FAIL-CHAIN (Vijayarao 2025) https://github.com/Lakshmi-Chakradhar-Vijayarao/fail-chain measured how confident-but-wrong model outputs propagate through multi-step LLM pipelines. The core finding: compression decisions made without reference to epistemic state convert uncertain claims into apparent facts, and the confidence of downstream agents on those facts is indistinguishable from genuinely resolved knowledge. The fix is not better models — it is epistemic-aware memory management.

Credence is the memory governor that closes this loop. It operates at the context-management layer, between the user and the model, and makes a single policy decision for each turn: **compress only what is epistemically resolved**.

## Signal Architecture

Three tiers of epistemic signal, increasing in cost and fidelity:

```
Tier 1 — Linguistic Assertiveness (0 ms, 0 cost)
  CredenceProxy: 5 text-pattern factors → J-score ∈ [0,1]
  Hedging rate, anchor density, self-correction, brevity, specificity
  Fast enough for every turn. Covers 80% of the signal.

Tier 2 — Behavioral Consistency (300 ms, ~$0.001/turn)
  BehavioralSignal: N=5 Haiku samples → pairwise ROUGE-L variance
  Low variance → consistent → high confidence
  Used for MEDIUM-zone turns where Tier 1 is ambiguous.

Tier 3 — Fisher J from Internal Activations (prior research, not deployed)
  KV-cache attention entropy measured in prior work on Qwen 3.5B.
  Established that internal model uncertainty correlates with surface
  linguistic assertiveness patterns — validating that Tier 1 measures
  something real. This system does not require Tier 3 at runtime.
  Tier 1 + E4 causal validation (Credence > random_j) are sufficient.
```

## Memory Policy

The J-score (combined Tier 1 + optional Tier 2) drives a three-way policy:

```
J ≥ θ_high (0.70) → COMPRESS
  Haiku summarises the old segment. Only HIGH-J (resolved) turn-pairs
  are eligible. LOW/MEDIUM-J turns are kept verbatim regardless.
  Faithfulness probe fires before Haiku: if old segment contains
  uncertainty markers, compression aborts → PRESERVE.

θ_low ≤ J < θ_high → TRIM
  Keep last TRIM_WINDOW turns. Same J-selective rule: LOW/MEDIUM-J
  turn-pairs in the dropped segment are always preserved.

J < θ_low (0.45) → PRESERVE
  Full history retained. No summarisation.
```

Three guard rails prevent unsafe compression:
1. **Attention sink protection** — first 2 turns never compressed (conversation identity)
2. **Type Prior** — code blocks, error traces, math responses get a J floor: J_cap = floor + 0.34
3. **Compression depth limit** — MAX_COMPRESSIONS = 3; cumulative loss bounded

Thresholds are adaptive: P75/P25 of a rolling 20-turn J-buffer, floored/capped at sensible bounds. Adapts to session J-distribution without per-model calibration.

## Multi-Agent Provenance (CredenceEnvelope)

When information passes between agents, the J-score and epistemic flags travel with it:

```python
CredenceEnvelope:
  j_score           # confidence at point of generation
  zone              # HIGH / MEDIUM / LOW
  uncertainty_preserved  # True if faithfulness probe fired
  chain_depth       # number of agent hops
  trust_score       # j_score − depth×0.05 − source_penalty
  should_verify     # True when trust < 0.40 and not verified
  safe_to_compress  # True only when HIGH-J, trusted, not uncertainty-preserved
```

Trust degrades with each hop. An uncertain constraint generated 4 hops ago carries `should_verify=True` regardless of how confident-sounding the current agent is.

## MCP Interface

The system is deployed as an MCP (Model Context Protocol) server. Any MCP-compatible agent framework — Claude Desktop, custom agents, orchestration layers — can call it as a tool, making the system model-agnostic by construction.

```
credence_chat              → send message, receive response + envelope
credence_inspect  → trust analysis + actionable recommendation
credence_propagate→ increment chain_depth, update source
credence_stats         → token savings, compression counts
credence_log  → per-turn J-scores and decisions
credence_save / credence_load  → cross-session continuity
credence_risk    → pre-flight risk assessment before compress/handoff
```

## Why This Works

The J-score is not a perfect oracle of model uncertainty. It is a fast, cost-free proxy derived from surface linguistic patterns. Its value is not precision — it is **directional correctness at the right decision boundary**. The system only needs to answer: "Is this turn safe to compress?" A false negative (preserve when compression was safe) costs tokens. A false positive (compress when content was uncertain) corrupts epistemic state. The system is calibrated to minimize false positives.

Empirically: Credence preserves 80% of chain-complete sessions vs 20% for naive window. On adversarial uncertainty injection (E6), Credence achieves 100% correction recall and 0% hallucination vs naive window's 0% correction and 50% hallucination. The failure mode is recoverable; the naive window's failure is not.

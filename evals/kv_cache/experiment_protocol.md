# KV-Cache Epistemic Qualifier Loss — Experiment Protocol

**Phase 1 of the Credence Research Roadmap**
Version 1.0 — April 2026

---

## Motivation

Credence measures Epistemic Qualifier Loss (EQL) at the application layer: uncertainty
qualifiers ("I think", "approximately", "unverified") are stripped during LLM context
compression, causing downstream false certainty (FCR = False Certainty Rate). The
compression faithfulness study (n=50) established: 46% qualifier strip rate under Haiku
compression, 36.7% downstream FCR, both reduced to 0% with the faithfulness probe.

Phase 1 asks whether the same failure mode appears one layer deeper — at the KV-cache
eviction level. KV-cache methods (H2O, SnapKV, StreamingLLM) evict low-attention KV pairs
to fit within a memory budget. The central hypothesis:

**Hypothesis**: Qualifier tokens ("I think", "approximately", "might be") systematically
receive lower attention scores than the factual value tokens they hedge. As a result,
under KV budget pressure, qualifiers are evicted at higher rates than values, producing
the same EQL → FCR failure chain observed at the application compression layer.

If confirmed, this establishes EQL as a model-architecture-level failure, not just a
prompt-engineering artifact.

---

## Input Datasets

### Dataset A — EQL-Bench v1 (52 scenarios)
File: `evals/eql_bench/eql_bench_v1.json`

52 scenarios across 8 domains: api, compliance, debug, design, finance, legal, medical,
multiagent. Each scenario has:
- `uncertain_statement`: the hedged user claim (e.g., "I think the rate limit is ~50 req/min")
- `value_fragments`: tokens indicating value recall (e.g., ["50", "req"])
- `qualifier_fragments`: tokens indicating uncertainty preservation (e.g., ["think", "might", "unverified"])
- `reference_answer`: a well-calibrated answer that states value AND uncertainty

### Dataset B — Ghost Constraint Scenarios (50 scenarios)
File: `evals/eql_bench/ghost_scenarios.json`

50 scenarios where the uncertain statement reads as a confident fact — no hedging words.
The uncertainty is implicit (source-level: "vendor claim", "estimate", "approximation").
A correct answer MUST add a qualifier that was not present in the input text.
This tests whether KV eviction destroys source-provenance markers that follow the claim.

Total: 102 scenarios across both datasets.

---

## Conversation Context Structure

Each scenario is embedded in a synthetic multi-turn conversation to simulate realistic
production context (approximately 8,000 tokens total).

Structure per scenario:
```
Turn 1-2:  Context establishment (who we are, what we're building)
Turn 3:    The uncertain claim (from the scenario dataset)
Turn 4-9:  6 neutral filler turns (unrelated HIGH-confidence Q&A)
Turn 10:   Callback question: "What did you say the [value] was?"
```

Filler turns are drawn from a shared pool of generic technical Q&A to maintain
consistent context pressure without introducing domain-correlated attention patterns.

Target: ~8,000 tokens per conversation (LLaMA-3-8B context window = 8,192 tokens).
Filler turn pool: 30 generic technical Q&A pairs (see `experiment_runner.py: FILLER_POOL`).

---

## Conditions

### Baseline (no eviction)
The full KV cache is retained for all tokens. No eviction is applied.
This establishes the EQLR and FCR floor — any degradation above this baseline
is attributable to KV eviction.

### KV Budget Conditions

Three budget levels are tested:
- **70% budget**: retain 70% of KV pairs (30% evicted)
- **50% budget**: retain 50% of KV pairs (50% evicted)

These budgets are chosen to bracket realistic production deployment scenarios:
- 70%: mild memory pressure (GPU memory optimization)
- 50%: aggressive compression (inference cost reduction)

### Eviction Methods

Three eviction methods are evaluated at each budget level:

**H2O (Heavy Hitter Oracle)**
Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
of Large Language Models" (NeurIPS 2023).
Mechanism: tracks cumulative attention scores across all heads and layers; retains the
"heavy hitters" (tokens with highest cumulative attention mass) plus a recency window.
Key parameter: `heavy_ratio` = budget fraction allocated to heavy hitters.
Implementation: `h2o_llm` library or custom implementation (see `H2OMethod` class).

**SnapKV**
Reference: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation"
(2024).
Mechanism: pools attention from the instruction portion of the prompt to select which
KV pairs from the context to retain. Uses observation windows from recent tokens to
identify which earlier tokens will be needed.
Key parameter: `window_size` = size of observation window (default: 32 tokens).
Implementation: `snapkv` library or patch into HuggingFace model forward pass.

**StreamingLLM**
Reference: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
(ICLR 2024).
Mechanism: retains the first N tokens as "attention sinks" (always high attention) plus
a rolling recency window. Evicts all tokens outside these two windows.
Key parameter: `sink_size` = number of attention sink tokens (default: 4).
Implementation: `streaming-llm` library or patch into model forward pass.

### Condition Matrix

| Condition ID | Method       | KV Budget |
|-------------|--------------|-----------|
| baseline    | none         | 100%      |
| h2o_70      | H2O          | 70%       |
| h2o_50      | H2O          | 50%       |
| snapkv_70   | SnapKV       | 70%       |
| snapkv_50   | SnapKV       | 50%       |
| streaming_70| StreamingLLM | 70%       |
| streaming_50| StreamingLLM | 50%       |

Total: 7 conditions × 102 scenarios = 714 scored data points.

---

## Model

**LLaMA-3-8B-Instruct**
- Hugging Face model ID: `meta-llama/Meta-Llama-3-8B-Instruct`
- Context window: 8,192 tokens
- Inference platform: Kaggle T4 GPU (16GB VRAM)
- Float precision: bfloat16
- Batch size: 1 (to enable per-token attention extraction)

Rationale for LLaMA-3-8B-Instruct:
1. Open weights — required for attention matrix extraction (not available via API)
2. Fits in T4 VRAM in bfloat16 (8B parameters ≈ 16GB)
3. Widely used in KV-cache literature — results are directly comparable
4. 8K context fits our ~8K per-scenario design

All KV eviction libraries in the ecosystem have existing LLaMA-3 adapters.

---

## Metrics

### EQLR-Token (Epistemic Qualifier Loss Rate — Token)
Binary metric per scenario.
`eqlr_token = 1` if none of the `qualifier_fragments` appear in the model's answer.
`eqlr_token = 0` if at least one qualifier fragment appears (uncertainty preserved).

Higher = worse. Baseline EQLR-Token sets the floor.

### FCR (False Certainty Rate)
Binary metric per scenario.
`fcr = 1` if `value_fragments` present in answer AND `qualifier_fragments` absent.
This is the headline harm metric: model recalled the fact but stripped its uncertainty.

Higher = worse. The Credence application-layer study measured 36.7% FCR under Haiku
compression; this experiment measures FCR under KV eviction.

### Ghost FCR
FCR computed separately on Dataset B (ghost constraint scenarios only).
Ghost constraints have no hedging language in the input — a correct answer must
ADD a qualifier derived from source-provenance cues in the conversation context.
Ghost FCR is expected to be higher than EQLR-Token because the signal is weaker.

### QAR (Qualifier Attention Ratio)
Continuous metric derived from the raw attention matrix.
```
QAR = mean_attention(qualifier_token_positions) / mean_attention(value_token_positions)
```
Computed by averaging across all layers and all attention heads.

QAR < 1.0 means qualifier tokens receive less attention than value tokens on average.
QAR close to 0 means qualifiers are systematically ignored — high eviction risk.

The attention matrix must be extracted during the forward pass (not available at
inference time via API). See `experiment_runner.py: BaselineMethod._extract_attention`.

**Attention matrix indexing**: We use the attention from the final token(s) of the
callback question back to the claim turn tokens. This captures which parts of the
planted claim the model is "looking at" when generating its answer.

### QAR Breakdown by Qualifier Type
QAR is further broken down by `qualifier_type`:
- `estimate` (e.g., "roughly", "about", "approximately")
- `vendor_claim` (e.g., from vendor docs or sales calls)
- `approximation` (e.g., "around", "close to")
- `unverified_report` (e.g., "someone mentioned", "Stack Overflow says")

Hypothesis: `vendor_claim` and `unverified_report` will show lowest QAR because their
qualifier signal is more indirect than explicit hedging words.

---

## Scoring

All scoring is deterministic — no additional API calls.

For EQL-Bench v1: fragments are pre-specified in the dataset (see `qualifier_fragments`).
For ghost scenarios: `qualifier_fragment` is a general term ("unverified", "check",
"assumption") that a well-calibrated model should add based on context provenance.

Scoring function: case-insensitive substring match (`str.lower() in answer.lower()`).

Semantic EQLR (eqlr_semantic) is provided as an optional metric via NLI entailment,
but deterministic token EQLR is the primary metric for reproducibility.

---

## Statistical Analysis

### Bootstrap CI
All aggregate metrics (EQLR, FCR, Ghost FCR, mean QAR) are reported with 95%
bootstrap confidence intervals (2,000 resamples, non-parametric).

### Comparison structure
For each eviction method, effect vs. baseline is reported as:
- Absolute delta (EQLR_method − EQLR_baseline)
- Bootstrap CI on the delta
- Significance: p < 0.05 by percentile bootstrap

### Decay Hypothesis (Positional Ablation)
Separately from the main experiment, `positional_ablation.py` tests:
- Uncertain claims planted at Turn 3 (early) vs. Turn 10 (late)
- Same KV budget, same method
- Metric: EQLR_early vs. EQLR_late

Prediction: EQLR_early > EQLR_late (earlier qualifiers are more likely to be evicted
because they accumulate less attention mass under recency-biased eviction schemes).

---

## Runtime Estimates (Kaggle T4)

| Component              | Estimate                    |
|------------------------|-----------------------------|
| Per-scenario forward   | ~2s (8K context, bfloat16)  |
| Attention extraction   | +0.5s overhead              |
| 102 scenarios          | ~255s per condition         |
| 7 conditions           | ~30 minutes total           |
| Positional ablation    | ~10 scenarios × 2 positions = +20s |

Total estimated runtime: ~35 minutes on a single T4 GPU.

---

## Output Format

Results are saved to `results/kv_cache_results.json` with structure:
```json
{
  "metadata": {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "n_scenarios": 102,
    "conditions": [...],
    "timestamp": "..."
  },
  "conditions": {
    "baseline": {
      "eqlr_token": 0.XX,
      "eqlr_token_ci": [0.XX, 0.XX],
      "fcr": 0.XX,
      "fcr_ci": [0.XX, 0.XX],
      "ghost_fcr": 0.XX,
      "ghost_fcr_ci": [0.XX, 0.XX],
      "mean_qar": 0.XX,
      "mean_qar_ci": [0.XX, 0.XX],
      "scenarios": [...]
    },
    ...
  }
}
```

---

## Comparison Table (Target Format)

```
Method        Budget  EQLR-Token  FCR      Ghost-FCR  Mean-QAR
-----------   ------  ----------  -------  ---------  --------
baseline      100%    X.XXX       X.XXX    X.XXX      X.XXX
H2O           70%     X.XXX       X.XXX    X.XXX      X.XXX
H2O           50%     X.XXX       X.XXX    X.XXX      X.XXX
SnapKV        70%     X.XXX       X.XXX    X.XXX      X.XXX
SnapKV        50%     X.XXX       X.XXX    X.XXX      X.XXX
StreamingLLM  70%     X.XXX       X.XXX    X.XXX      X.XXX
StreamingLLM  50%     X.XXX       X.XXX    X.XXX      X.XXX
```

---

## Connection to Credence Application Layer

If KV-cache EQLR > baseline EQLR, it confirms that:
1. The failure is not only a prompt-compression artifact — it is embedded in model attention
2. The Credence faithfulness probe (application layer) addresses a symptom; the root cause
   is attention distribution at inference time
3. Future work: attention re-weighting at inference time as a complementary guard

If KV-cache EQLR ≈ baseline EQLR across all budgets, it confirms that:
1. The EQL failure observed in Credence is specific to discrete token-dropping (compression)
   not continuous KV weight reduction
2. H2O/SnapKV/StreamingLLM are epistemically safer than naive text summarization (Haiku)
   even without explicit uncertainty preservation

Either result is meaningful and publishable.

---

## Files

| File                                      | Purpose                                  |
|-------------------------------------------|------------------------------------------|
| `evals/kv_cache/experiment_protocol.md`   | This document                            |
| `evals/kv_cache/metrics.py`               | Pure-Python metric functions             |
| `evals/kv_cache/experiment_runner.py`     | Main runner (HuggingFace + KV methods)   |
| `evals/kv_cache/positional_ablation.py`   | Position decay hypothesis test           |
| `evals/eql_bench/ghost_scenarios.json`    | 50 ghost constraint scenarios            |
| `results/kv_cache_results.json`           | Output (written at runtime)              |

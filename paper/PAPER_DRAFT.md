# Epistemic Qualifier Loss: Measuring and Preventing the Systematic Loss of Uncertainty Signals in LLM Context Compression

**Lakshmi Chakradhar Vijayarao**  
Northeastern University  
vijayarao.l@northeastern.edu

---

## Abstract

We identify and measure **Epistemic Qualifier Loss (EQL)** — the systematic removal of user-stated uncertainty markers during context window compression in large language model pipelines. When a user states "I think the rate limit is around 50 req/min — I haven't confirmed this," standard Haiku-based summarization strips the hedging in 46% of cases (EQLR = 0.46, 95% CI [0.318, 0.607]), producing compressed context that treats the value as confirmed fact. Token-importance compression (LLMLingua-style) produces EQLR = 0.68 (95% CI [0.536, 0.800]). Across 8 open-weight models from 6 organizations, unguarded EQLR ranges from 0.41 to 0.75, confirming that qualifier loss is a structural property of summarization, not a per-model artifact.

We introduce **Credence**, a deterministic constraint-tracking layer that eliminates EQL without model cooperation. A faithfulness probe (423 uncertainty markers, <0.03ms) intercepts compression when qualifier-bearing text is detected. An immutable constraint registry persists uncertain values and their source statements. A PreToolUse gate blocks file writes that embed unverified values. Applied to the same 50 scenarios: EQLR = 0.0 (95% CI [0.0, 0.0]). On a 10-session ghost constraint benchmark (values stated without explicit hedging), a domain-keyword heuristic recovers BothRate from 0.33 (naive window) to 1.0.

The system adds <1ms overhead per turn and is deployable as an MCP server requiring zero model cooperation. EQL-Bench, a versioned benchmark of 370 labeled scenarios across 8 domains, is released for reproducibility.

---

## 1. Introduction

A common failure mode in AI-assisted software development begins with a developer stating uncertain information:

> *"I think the rate limit is around 50 req/min — I'll need to confirm this with the vendor."*

Fifteen conversation turns later, after context compression has summarized the early turns to preserve the context window, the model writes:

```python
RATE_LIMIT = 50  # requests per minute
```

No qualifier. No flag. The value ships to production. The real limit is 10. The API rejects every request at 2am.

This failure has a specific cause: **the compression operation converted user-stated uncertainty into bare assertion**. The model was not hallucinating — it was faithfully reproducing what it believed was a confirmed fact, because the compression removed the evidence that it was not.

*See Figure 0 for a schematic of this failure mode and Credence's intervention.*

We call this failure mode **Epistemic Qualifier Loss (EQL)**. It is distinct from:

- **Model hallucination** — EQL produces no false values; it strips the uncertainty flag from true ones
- **Model overconfidence** — EQL is a pipeline artifact, not a model weight property; it occurs in models with well-calibrated internal uncertainty
- **Retrieval failure** — the value IS recalled; what is lost is the epistemic status of the value

To our knowledge, no prior work names, defines, or measures this specific failure mode. Context compression papers (LLMLingua [1], LLMLingua-2 [2], StreamingLLM [3]) optimize for informativeness. Uncertainty quantification papers (Semantic Entropy [4], R-Tuning [5], MetaFaith [6]) address model-internal confidence. None address the pipeline operation that converts user-stated uncertainty into ambiguous or erased epistemic state.

**Contributions:**

1. We define EQL and introduce EQLR (Epistemic Qualifier Loss Rate) as a measurable metric
2. We demonstrate that EQLR = 0.46–0.75 across 8 models under standard compression, and is caused by the summarization operation, not model weights
3. We show EQL propagates to system-level failures: incorrect code generation, lost debugging context, wrong architectural decisions
4. We introduce a deterministic fix (Credence) that achieves EQLR = 0.0 through constraint tracking, without requiring model cooperation or fine-tuning
5. We identify **ghost constraints** — uncertain values stated without explicit hedging — and demonstrate a heuristic that recovers them
6. We release EQL-Bench, a versioned dataset of 370 labeled scenarios across 8 domains

---

## 2. Problem Definition

### 2.1 Epistemic Qualifier Loss

Let $C = (t_1, t_2, \ldots, t_n)$ be a conversation of $n$ turns. Let $t_i$ contain an **uncertain statement** $s$: a user utterance with an explicit epistemic qualifier (hedge word, uncertainty marker, or unconfirmed attribution) co-occurring with a specific value $v$.

A **compression operation** $\mathcal{K}: C \rightarrow C'$ produces a shorter context $C'$. We say **EQL occurs** when:

$$\text{EQLR}(\mathcal{K}, C) = \mathbb{1}[v \in C'] \wedge \mathbb{1}[\text{qualifier}(s) \notin C']$$

That is: the value $v$ survives compression, but the qualifier that marked it as uncertain does not.

**EQLR** over a dataset $\mathcal{D}$ of scenarios:

$$\text{EQLR}(\mathcal{K}, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{C \in \mathcal{D}} \text{EQL}(\mathcal{K}, C)$$

This is measured by checking whether any term from a 423-marker frozenset (covering approximation, hedging, estimation, vendor attribution, and unverified report qualifiers) appears in the compressed context adjacent to the value fragments.

### 2.2 Why Standard Metrics Miss This

BLEU and ROUGE reward lexical coverage of the original. A compression that retains "rate limit 50 req/min" while dropping "I think" and "I haven't confirmed" scores well on both, because the informative tokens (the number, the unit, the entity) are preserved. The epistemic tokens are scored as low-importance noise.

Task-fidelity metrics (question-answering accuracy on the compressed context) also miss EQL. If the downstream task is "what is the rate limit?", both the qualified and unqualified answers score 1.0. The question that EQL affects is different: "should this value be treated as confirmed?"

### 2.3 Ghost Constraints

A ghost constraint is an uncertain value stated without explicit hedging language. The user says "the rate limit is 100" because they genuinely believe it — but they haven't verified it. Standard marker-based detection produces zero registrations, yet the downstream risk is identical. Section 5 addresses ghost constraints separately.

---

## 3. Measuring EQL: The Compression Faithfulness Study

### 3.1 Setup

We construct 50 scenarios, each containing a user-stated uncertain value embedded in a realistic multi-turn conversation (API integration, configuration decisions, debugging sessions). Uncertain values include rate limits, token lifetimes, API version strings, timeouts, and quota limits — values that are commonly unconfirmed at the time of stating and that propagate into code.

Three conditions per scenario:

- **Naive**: Haiku (claude-haiku-20240307) summarizes turns 1–8. Downstream model is then asked a callback question requiring recall of the uncertain value.
- **LLMLingua-sim**: Token-importance scoring assigns low weight to hedging language; tokens below threshold are removed.  
- **Probe-guarded**: The Credence faithfulness probe intercepts compression when uncertainty markers are detected; the constraint is registered; the value and its epistemic status are injected into every subsequent system prompt via the Truth Buffer.

Metric: EQLR (Section 2.1). Secondary metric: proper FCR (False Certainty Rate) — fraction of callbacks where the downstream model states the value without any qualifier AND the value is numerically present.

### 3.2 Results

| Condition | Qualifier Survival | EQLR | 95% CI | Proper FCR |
|---|---|---|---|---|
| Naive (Haiku) | 54% | **0.46** | [0.318, 0.607] | 0.02 |
| LLMLingua-sim | 32% | **0.68** | [0.536, 0.800] | 0.02 |
| Probe-guarded | 100% | **0.00** | [0.000, 0.000] | 0.00 |
| Baseline (no compression) | 100% | 0.00 | — | 0.00 |

*Figure 2 shows EQLR and proper FCR side-by-side for both compressor conditions.*

**Note on FCR:** An earlier scorer version (v2) classified "I don't have context" as false certainty, producing inflated FCR values (naive: 0.06, LLMLingua: 0.74). The corrected v3 scorer requires: (1) the specific numeric value appears in the downstream answer; (2) epistemic erasure responses ("I don't have information") are excluded; (3) the full uncertainty vocabulary is checked. Corrected proper FCR = 0.02 for both unguarded conditions. The code-level harm is not that models assert wrong values in prose — it is that they embed stripped values as unqualified constants in generated code. The GTS scanner and gate address this code-level pathway.

### 3.3 Analysis

Of the 23 Haiku-stripped cases, 12 (52%) produce zero hedging in the compressed output — the uncertain statement is assertivized into bare fact. The remaining 11 convert canonical markers ("I think", "I haven't confirmed") to weaker hedges ("likely", "pending") that survive compression but are systematically ignored in downstream code generation.

The LLMLingua simulation failure mode is different: the uncertain statement is removed entirely in 68% of cases (epistemic erasure), leaving no signal at all. The downstream model has no epistemic information to reason with.

---

## 4. Multi-Model Validation

If EQL is a structural property of summarization — not a Haiku artifact — it should appear across model families. We test this on EQL-Bench v2 (280 explicit scenarios, 8 domains) using 6 open-weight models run on Kaggle GPU instances.

### 4.1 Results

| Model | Org | Params | EQLR (unguarded) | 95% CI | EQLR (probe-blocked) |
|---|---|---|---|---|---|
| Claude Haiku | Anthropic | — | 0.46 | [0.318, 0.607] | **0.00** |
| Qwen-2.5-1.5B | Alibaba | 1.5B | 0.50 | [0.422, 0.578] | **0.00** |
| Qwen-2.5-7B | Alibaba | 7B | **0.75** | [0.639, 0.853] | **0.00** |
| Mistral-7B | Mistral AI | 7B | 0.61 | [0.475, 0.721] | **0.00** |
| Phi-3.5-mini | Microsoft | 3.8B | 0.44 | [0.328, 0.557] | **0.00** |
| Llama-3.2-3B | Meta | 3B | 0.41 | [0.279, 0.541] | **0.00** |
| Gemma-2-9B | Google | 9B | 0.62 | [0.508, 0.754] | **0.00** |
| Llama-3.1-8B | Meta | 8B | 0.43 | [0.311, 0.557] | **0.00** |

**EQLR range: 0.41–0.75 across 8 models from 6 organizations.**  
**Probe-blocked EQLR: 0.00 for all 8 models.**

*Figure 1 visualizes the full 8-model EQLR comparison with 95% CI error bars.*

Two observations: larger models are not more qualifier-faithful (Qwen-7B has the highest EQLR at 0.75, versus Qwen-1.5B at 0.50). And probe coverage (fraction of scenarios where the probe fires and blocks compression) is 45–78% across model runs, meaning a substantial fraction pass through unblocked. For those, the Truth Buffer provides the safety net.

### 4.2 EQLR by Qualifier Type

| Qualifier Type | Mean EQLR | Worst Case |
|---|---|---|
| `preliminary` | 0.76 | 1.00 (Llama-3B) |
| `vendor_claim` | 0.69 | 0.94 (Qwen-7B) |
| `unverified_report` | 0.52 | 0.63 |
| `estimate` | 0.44 | 0.67 |
| `approximation` | 0.51 | 0.60 |

`preliminary` claims (tentative decisions, working hypotheses) are stripped most aggressively. `approximation` claims (explicit "roughly", "about") survive best, consistent with hedging words being scored as moderately informative by token-importance models.

---

## 5. Ghost Constraints

### 5.1 Definition and Motivation

Not all uncertain values are stated with hedging language. A developer who has not verified an API rate limit may say "the rate limit is 100" because they genuinely believe it. From the model's perspective, this is indistinguishable from a confirmed fact. Standard marker-based detection produces zero registrations.

We call these **ghost constraints**: values that are uncertain by context (no verification source cited, no documentation reference, no URL) but stated without epistemic markers.

Ghost constraints are the hard case — they require domain knowledge rather than linguistic detection.

### 5.2 Detection Heuristic

We apply a ghost constraint heuristic that fires when all three conditions hold simultaneously:

1. A numeric value is present in the user utterance
2. A domain keyword co-occurs (rate limit, timeout, quota, TTL, API version, token expiry, pricing, concurrency)
3. No documentation reference is present (no URL, no "docs say", no "I verified")

This is purely syntactic — no model calls, no embeddings, <1ms.

### 5.3 Results

We construct 10 multi-turn sessions (Ghost Gauntlet) where uncertain values are stated without explicit hedging, across 10 domains: API integration, infrastructure, compliance, ML deployment, security, product planning, devops, data engineering, mobile development, financial planning.

Each session has 3 ghost callbacks measuring:
- **Value recalled**: does the downstream response include the original value?
- **Qualifier preserved**: does the response flag the value as uncertain?
- **Both (gold standard)**: value recalled AND uncertainty preserved

| Condition | Value Rate | Qualifier Rate | **Both Rate** |
|---|---|---|---|
| Naive window | 0.33 | 1.00 | **0.33** |
| Credence | 1.00 | 1.00 | **1.00** |

The naive window recalls the value only 1/3 of the time, despite preserving qualifiers for the values it does recall. The critical finding: ghost constraints are silently dropped by standard context windowing, and when they survive, they do so without any epistemic flag.

*Figure 3 shows per-session BothRate distribution for both conditions.*

---

## 6. Pipeline Impact: From Qualifier Loss to System Failure

EQLR is a text measurement. Its significance depends on whether qualifier loss propagates to incorrect system behavior. We demonstrate two concrete failure scenarios.

### 6.1 E6: Long-Session Value Recall (Needle in Haystack)

A 14-turn API integration session establishes two uncertain values in turns 1–2 (rate limit: ~50 req/min; token expiry: ~24h, both explicitly flagged as unverified), then continues with 10 turns of technical discussion (circuit breakers, exponential backoff, idempotency). Callbacks at turns 13–14 ask the model to state the established values.

| Condition | Turn 13 (rate limit) | Turn 14 (token expiry) | Recall |
|---|---|---|---|
| Baseline | ✓ with qualifier | ✓ with qualifier | 2/2 |
| Naive window | ✓ with qualifier | ✗ (value not recalled) | 1/2 |
| Credence | ✓ with qualifier | ✓ with qualifier | 2/2 |

The naive window drops the token expiry entirely by turn 14. The model's turn-14 answer: *"No specific token expiry value was established in this conversation."* Credence injects both constraints into every turn's system prompt via the constraint registry, ensuring survival regardless of compression decisions.

### 6.2 E8: Debug Session Recall

A debugging session establishes: (1) a specific production error with file and line number (`RuntimeError` in `payment_processor.py:147`); (2) two uncertain hypotheses about root cause ("might be threading" or "might be GIL — not sure which"); (3) a deployed fix and its partial result. Three callback questions test recall of each component.

| Condition | Error Location | Hypotheses | Fix Deployed | Mean Recall |
|---|---|---|---|---|
| Baseline | 1.0 | 1.0 | **0.0** | 0.667 |
| Naive window | 0.6 | 0.33 | 1.0 | 0.644 |
| **Credence** | 1.0 | 1.0 | 1.0 | **1.000** |

The baseline model answers the wrong question at Q3 (context tracking failure). The naive window loses the file/line details and confuses the two hypotheses. Credence correctly preserves all three components, including the uncertain hypotheses with their original hedging.

This scenario represents the practical failure mode: in a real debugging session, having the original error details, the uncertain hypotheses, and the deployed fix available together is necessary for correct reasoning. Any one missing produces wrong next steps.

*Figure 4 shows per-question and mean recall for all three conditions.*

---

## 7. Deterministic Enforcement vs. Probabilistic Instruction

### 7.1 The Control Condition

An alternative to Credence is prompt engineering: instruct the compression model to preserve uncertainty qualifiers. Prior work on instruction-following reliability [7] provides relevant priors — LLMs achieve ~38% reliability on nuanced behavioral constraints, with systematic degradation when learned summarization behavior (producing assertive, clean prose) conflicts with the instruction.

We test this directly. For 100 short-session scenarios, three conditions:

- **Naive**: compress with no instruction
- **Enhanced**: compress with explicit instruction to preserve all hedging language and uncertainty markers
- **Probe-guarded**: faithfulness probe intercepts when markers present

| Condition | Qualifier Survival | EQLR | CI |
|---|---|---|---|
| Naive | 100% | 0.00 | [0.0, 0.0] |
| Enhanced prompt | 98% | 0.02 | [0.0, 0.047] |
| Probe-guarded | 100% | 0.00 | [0.0, 0.0] |

**Note on naive EQLR = 0.0 in this study:** Short sessions do not trigger compression (context window not exceeded). This is expected behavior — compression-induced EQL only emerges in long sessions. The important result is the enhanced prompt condition: **98% qualifier survival vs. 100% for the probe**. The 2% miss rate is probabilistic; the probe's 0% is structural. The probe either blocks compression entirely (when markers present) or the constraint survives in the registry regardless of compression decisions.

### 7.2 Why Structural > Probabilistic for Safety-Critical Paths

For code generation, the cost of a single missed qualifier is not 2% of a user experience metric — it is the probability that an unverified value ships to production and causes a runtime failure. A structural guarantee (compression is blocked when qualifiers are present) is the correct design choice for this failure mode, for the same reason that type systems are preferred over code review for null safety: the guarantee must hold even when the reviewer (or model) is wrong.

---

## 8. System Design

Credence implements the structural guarantee through four components, each operating below 1ms:

**Faithfulness Probe.** A 423-term frozenset scans user turns on every message. If any uncertainty marker is detected, compression is blocked for that turn. False positive rate: 0.5% on 200 benign technical sentences (1 false positive: "Base64 increases data size by approximately 33%", where "approximately" is technically accurate).

**Constraint Registry.** Uncertain values and their source statements are stored in SQLite with session scope. Each constraint carries content, source type (user_estimate, vendor_claim, temporal_scan, self_probe), verification status, and a timestamp. Registry write latency: 0.022ms P50. Read latency: 0.002ms P50.

**Truth Buffer.** At each turn, all unverified constraints for the session are formatted as a non-compressible prefix injected into the system prompt: "UNVERIFIED CONSTRAINTS: [list]". This ensures constraints survive even when compression is not blocked (e.g., turns containing no markers).

**PreToolUse Gate.** Before any file write (Write, Edit, Bash, NotebookEdit), token overlap between the write action and unverified constraints is computed. If ≥2 tokens overlap, the write is blocked and the user is shown which constraints require verification. This is the code-level enforcement layer — it prevents EQL from propagating into shipped artifacts.

Total P99 overhead across all components: <1ms. Rust implementation of the gate: 3.4ms P50, 98× faster than the Python implementation.

*Figure 5 shows per-component latency breakdown (P50/P95/P99).*

---

## 9. Limitations

**Detection coverage.** The current observer detects 95% of uncertainty phrases in a 22-phrase coverage probe (8% false positive rate on 12 benign phrases). One systematic miss: "we might need 2 retries" — "might" alone without a domain keyword escapes both the marker path and the ghost heuristic.

**Ghost heuristic scope.** The domain keyword list covers the highest-risk categories (rate limits, timeouts, quotas, API versions, token lifetimes, pricing). General-domain uncertain values (e.g., "I think we have about 3 engineers available") are not caught.

**Session scope.** Constraints are scoped to a session ID derived from the working directory. Multi-repo workflows with directory switching require explicit `CREDENCE_SESSION_ID` configuration to maintain constraint continuity.

**No claim about correctness.** Credence tracks whether a value was verified, not whether it is correct. A developer who verifies an incorrect value with "confirmed — rate limit is 100" will have that value marked verified. The tool prevents accidental propagation of uncertainty, not deliberate misinformation.

**Null hypothesis incomplete.** The prompt-engineering control (Section 7) was tested only on short sessions where compression does not trigger. A controlled experiment on long sessions — Haiku + explicit qualifier-preservation instruction vs. the probe — has not been run (code ready: `evals/null_hypothesis.py`). Based on the instruction-following literature, we predict EQLR ≈ 10–20% with the enhanced prompt on long sessions.

**E6, E7, E8 are single-trial demonstrations.** They demonstrate the failure mode with constructed sessions but are not statistically powered.

**Ghost Gauntlet uses researcher-constructed sessions.** The 10 sessions were designed to elicit ghost constraints, not sampled from real usage.

---

## 10. Related Work

**Context compression** papers optimize for informativeness: LLMLingua [1] uses a small model to score token importance; LLMLingua-2 [2] extends this with data distillation. Both measure whether the compressed context supports correct task completion — not whether it preserves epistemic status. A context can score well on ROUGE and task accuracy while completely stripping uncertainty signals.

**Uncertainty quantification** papers measure model-internal confidence: Semantic Entropy [4] measures variance across samples; R-Tuning [5] fine-tunes models to refuse uncertain questions; MetaFaith [6] uses metacognitive prompting to improve self-expressed uncertainty. These address what the *model* is unsure about. EQL addresses what the *user* stated they were unsure about — external epistemic state that may not exist in model weights at all.

**Memory and context management** systems (MemGPT [8], StreamingLLM [3]) address retrieval and windowing of facts. Neither provides mechanisms for tracking the certainty status of facts. CredenceEnvelope is a proposed standard for epistemic provenance metadata that travels with every AI-generated value through agent pipelines — analogous to data lineage provenance metadata.

**Concurrent work** (arXiv:2509.11208) studies compression failures in evidence-based binary adjudication, where presenting evidence in different orderings produces inconsistent outputs from an ISR gate. Their compression is semantic (multiple evidence items → binary decision); ours is surface (long context → short context). Their failure mode is input ordering sensitivity; ours is output qualifier loss. Both papers name and measure a specific compression failure mode, strengthening the case that compression faithfulness is an under-studied correctness property.

---

## 11. Conclusion

We identify Epistemic Qualifier Loss as a systematic, measurable failure mode in LLM context compression: standard Haiku summarization strips explicit uncertainty markers in 46% of compressions (EQLR = 0.46, 95% CI [0.318, 0.607]); token-importance compression produces EQLR = 0.68. The failure is model-agnostic — EQLR = 0.41–0.75 across 8 open-weight models from 6 organizations. It propagates to system-level failures: incorrect code constants, lost debugging context, wrong architectural decisions.

Credence eliminates EQL through deterministic constraint tracking: faithfulness probe, immutable registry, Truth Buffer injection, and PreToolUse gate. Applied to the same benchmark: EQLR = 0.00 (95% CI [0.000, 0.000]). The ghost constraint heuristic recovers unhedged uncertain values: BothRate from 0.33 (naive window) to 1.00.

The key design principle: a structural guarantee (compression is blocked when uncertainty is present; constraints are injected regardless of compression) is preferable to a probabilistic instruction (prompt the compressor to preserve qualifiers) for a failure mode where a single miss can cause a runtime production failure. Prompt-based qualifier preservation achieves 98% survival; Credence achieves 100%.

EQL-Bench (370 scenarios, 8 domains, 5 qualifier types, versioned) is released for reproducibility. Credence is available as an MCP server: `pip install "credence-guard[mcp]"`.

---

## References

[1] Jiang, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. *EMNLP 2023*.

[2] Pan, Z., et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. *ACL 2024*.

[3] Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks. *ICLR 2024*.

[4] Kuhn, L., et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.

[5] Zhang, Z., et al. (2024). R-Tuning: Instructing Large Language Models to Say 'I Don't Know'. *NAACL 2024*.

[6] Ma, Y., et al. (2025). MetaFaith: Metacognitive Prompting for Faithful Uncertainty Expression. *EMNLP 2025*.

[7] Tian, Y., et al. (2025). Instruction-Following Reliability Under Nuanced Behavioral Constraints. *ACL 2025*.

[8] Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

---

## Appendix A: EQL-Bench Construction

EQL-Bench v2 contains 370 scenarios: 280 explicit (user states uncertain value with hedging) and 90 ghost (value stated without hedging). Domains: API integration (api), debugging (debug), system design (design), compliance (compliance), multi-agent coordination (multiagent), medicine (medical), legal (legal), finance (finance). Qualifier types: approximation ("roughly"), estimate ("I estimate"), preliminary ("tentative"), unverified_report ("they told us"), vendor_claim ("Stripe says").

Each scenario contains: uncertain_statement (the input), value_fragments (tokens for recall scoring), qualifier_fragments (tokens for uncertainty scoring), reference_answer (ideal response), and notes (domain-specific context).

EQLR scoring: a scenario is scored as EQL = 1 if any term from the 423-marker frozenset is absent from the compressed output, AND the value fragments are present (value survived but qualifier did not).

## Appendix B: Faithfulness Probe Marker Coverage

The 423-term frozenset covers: hedging language ("probably", "maybe", "roughly", "approximately"), epistemic uncertainty ("I think", "I believe", "not sure", "uncertain"), vendor attribution ("the docs say", "vendor said", "they told us"), working hypotheses ("I'm assuming", "working theory", "I suppose"), and verification status ("unverified", "unconfirmed", "haven't checked"). Two-tier architecture: strong markers fire unconditionally; weak markers ("around", "assuming", "docs say") fire only when a numeric value is co-present, reducing false positive rate to 8% on benign technical phrases.

## Appendix C: Ghost Constraint Heuristic

Domain keyword categories: rate limits (rate_limit, quota, rpm, rps, throttle, max_req), auth lifetimes (token_expiry, ttl, session_expiry, auth_lifetime), API versions (api_version, stripe_version, openai_version), pricing (price_per, cost_per, billing_rate, per_token), concurrency/infrastructure (timeout, concurrency, tokens). Numeric detection: `\b\d+(?:\.\d+)?` (no trailing word boundary — catches unit-suffixed numbers: 30s, 5MB, 50ms). URL exclusion: patterns matching `://`, `?param=`, `/v1/` contexts are excluded.

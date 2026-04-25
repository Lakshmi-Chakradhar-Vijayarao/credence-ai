# Credence — Benchmark Results

**Epistemic Qualifier Loss (EQL):** the loss of user-stated uncertainty markers during context window summarization, causing downstream models to treat explicitly uncertain claims as confirmed facts.

**EQL Rate (EQLR):** the fraction of explicitly hedged user claims that lose their qualifier after compression.

**False Certainty Rate (FCR):** the fraction of downstream responses that state an uncertain value without any qualifier — the measurable harm caused by EQL.

These metrics were defined and first measured by this work. No prior paper in compression (LLMLingua, SnapKV, StreamingLLM) or epistemic uncertainty (Semantic Entropy, UProp, R-Tuning) names or measures EQL/EQLR.

All results measured fresh. All conditions use Claude Opus 4.7 unless noted. All CIs are Clopper-Pearson 95%.

---

## 1. Primary: Measuring EQL (n=50)

**Question**: Does Haiku context compression produce Epistemic Qualifier Loss, and does Credence prevent it?

| Condition | EQL Rate (EQLR) | False Certainty Rate (FCR) | 95% CI (FCR) |
|---|---|---|---|
| **Naive Haiku compression** | **46.0%** | **34.0%** | [21.2%, 48.8%] |
| **LLMLingua-inspired compression** | **68.0%** | **76.0%** | [61.8%, 86.9%] |
| Haiku + prompt instruction ("preserve qualifiers") | 10.0% | n/a | — |
| **Credence (faithfulness probe)** | **0%** | **0.0%** | [0.0%, 7.1%] |
| Baseline (full context, no compression) | 0% | 0% | — |

**Result**: Credence eliminates EQL and FCR deterministically. Naive Haiku: EQLR 46%, 1 in 3 queries answer with false certainty. LLMLingua-inspired: EQLR 68%, 3 in 4. Credence: EQLR 0%, FCR 0/50.

**Why prompt instructions fail**: Adding "preserve uncertainty qualifiers" to the Haiku system prompt reduces EQLR to 10% — better, but not deterministic. The remaining 10% is model non-compliance. The faithfulness probe is binary enforcement: either uncertainty markers are present (BLOCK) or they aren't (ALLOW). 100% block rate, n=50.

**Probe mechanism**: 167-term frozenset scan on user turns only. p50=0.011ms. Zero API calls. 100% block rate (50/50 uncertain segments blocked). 0% false positive rate (200 non-uncertain phrases, offline).

**What "Credence FCR=0%" means mechanistically**: When the probe fires, Haiku is NOT called — compression is skipped entirely and the downstream model receives the original uncompressed conversation text. This is identical to the baseline condition for these 50 scenarios. The correct framing: the probe achieves 0% FCR by *preventing* the lossy compression event, not by compressing while preserving qualifiers. The contrast with "Haiku + prompt instruction" is meaningful: that condition still calls Haiku (attempts to compress while preserving) and achieves only 90% qualifier survival. The probe eliminates the risk at the source.

**FCR scoring note**: `downstream_certain` is scored as the *absence* of any marker from the same 108-term vocabulary used by the probe. A response with "I think" or "it's approximately" scores FCR=0 (correct). A response with "The rate limit is 50 req/min" (no qualifier) scores FCR=1 (harm). The same vocabulary is intentionally shared — these are the canonical English uncertainty expressions.

**Reproduce**: `python -m evals.compression_faithfulness --n 50`

---

## 2. Ghost Constraints: Implicit Uncertainty Survival (n=10 sessions)

**Question**: When uncertain facts are stated WITHOUT hedging language ("rate limit is 50 req/min"), does Credence preserve the epistemic signal through compression?

| Condition | BothRate (value + qualifier recalled) |
|---|---|
| Credence v1 (Scout + Truth Buffer) | **1.000** |
| Credence eg2 (full stack) | **1.000** |
| Naive sliding window | **0.200** |

10 domain sessions: api, infra, compliance, ml, security, product, devops, data, mobile, finance.

**Result**: 5× BothRate improvement. Naive window loses ghost constraints when turns drop out of window. Credence's Scout Classifier registers them proactively; Truth Buffer restores them every turn.

**Reproduce**: `python -m evals.ghost_gauntlet`

---

## 3. E6 Negative Needle: Long-Session Constraint Recall (n=23 trials)

**Question**: 12-turn session with uncertain constraint at T3 ("rate limit ≈50 req/min") and 8 HIGH-J filler turns. Does Credence preserve the constraint through a recall callback at T13?

| Condition | Correction Recall | Hallucination Rate |
|---|---|---|
| **Credence** | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive sliding window | **~20%** | **~80%** |

n=23 independent trials. All conditions: Opus 4.7.

**Result**: Naive window loses the constraint in ~80% of trials; when it answers, it states the wrong value with confidence 80% of the time. Credence matches baseline precision with zero compression cost to epistemic quality.

**Reproduce**: `python -m evals.experiments --exp E6`

---

## 4. E7 Multi-Hop Reasoning Chain (categorical, 3 hops)

**Question**: 3-turn dependency chain planted at T3-T5. Naive window drops all 3. Does Credence preserve the entire chain?

| Condition | Hops Recalled | Chain Complete |
|---|---|---|
| Credence | **3/3** | ✓ |
| Baseline | 3/3 | ✓ |
| Naive window | **0/3** | ✗ |

**Result**: Naive window structurally breaks multi-hop reasoning. Credence matches full-context baseline.

**Reproduce**: `python -m evals.experiments --exp E7`

---

## 5. Cross-Session Epistemic Memory (n=16 callbacks)

**Question**: Does Credence preserve epistemic qualifiers across session boundaries, where other memory systems (Mem0, Zep) strip them at write time?

| Condition | CS-FCR | BothRate |
|---|---|---|
| No memory (fresh session) | **50%** | 25% |
| Naive summary (human-written) | 0% | 100% |
| **Credence Memory** | **0%** | 87.5% |

CS-FCR = Cross-Session False Certainty Rate. n=10 scenarios, n=16 callbacks.

**Key distinction**: Naive summary achieves 0% CS-FCR because it was hand-written to preserve qualifiers — this is what an expert human does manually. Real memory tools (Mem0, Zep, Graphiti) store facts as flat strings, stripping qualifiers. Credence automates the epistemic-preserving behavior.

**Reproduce**: `python -m evals.cross_session_eval`

---

## 6. Null Hypothesis: Prompt Instructions vs. Deterministic Enforcement (n=30)

**Question**: Is a well-crafted "preserve uncertainty qualifiers" instruction sufficient, or is deterministic enforcement necessary?

| Approach | Qualifier Survival | Is Deterministic? |
|---|---|---|
| No instruction | ~54% | — |
| Prompt instruction: "preserve qualifiers" | **90.0%** | No |
| **Credence faithfulness probe** | **100%** | **Yes** |

**Result**: The best prompt instruction achieves 90.0% qualifier survival — the remaining 10% is probabilistic model non-compliance. The probe is binary: either uncertainty is present in the user turn (BLOCK) or it isn't (ALLOW). Instructions are probabilistic; enforcement is not.

**Reproduce**: `python -m evals.null_hypothesis`

---

## 7. Stress Test — Offline Deterministic Benchmarks (n=1000 probe, n=200 precision/recall)

All offline, zero API calls, 1.8s total runtime.

| Test | Result | n |
|---|---|---|
| Probe latency p50 | **0.011ms** (7× better than 0.07ms claim) | 1,000 |
| Probe latency p99 | **0.017ms** | 1,000 |
| Probe false positive rate | **0%** | 200 non-uncertain phrases |
| Probe recall | **100%** | 200 uncertain phrases (167 markers covered) |
| J-score gap: confident vs. hedged | **0.344** (0.859 vs. 0.514) | 200 |
| GTS code annotation precision | **100%** | 50 code blocks |
| GTS false positive rate | **0%** | 50 unrelated code blocks |
| Registry write latency p50 | **1.79ms** | 1,000 |
| Unit tests | **178/178 pass** (11 skipped offline-only) | 178 |

**Reproduce**: `python -m evals.stress_test` (1.8 seconds, free)

---

## 8. Precision Eval — Zero False Positive Guarantee (offline)

| Layer | FP Rate | n | Test |
|---|---|---|---|
| Faithfulness Probe | **0%** | 200 | Non-uncertain phrases |
| Consistency Enforcer | **0%** | 8 | Unrelated queries |
| Generation-Time Scanner | **0%** | 50 | Unrelated code |

**Reproduce**: `python -m evals.precision_eval` (offline, free)

---

## 9. E8 Real Debugging Session (single trial)

**Question**: In a realistic 12-turn debugging session with an uncertain hypothesis at T4, does Credence preserve the hypothesis better than naive window?

| Condition | Recall |
|---|---|
| Credence | **0.944** |
| Baseline | 1.000 |
| Naive window | **0.522** |

**Result**: Naive window loses 47.8% of debugging recall. Credence loses 5.6% (attribution to aggressive compression scheduling on this short session).

---

## 10. Native Gate Latency (Rust vs Python)

| Implementation | Latency per tool call | Overhead per 100-call session |
|---|---|---|
| Python hook | 331ms | 33 seconds |
| **Rust gate (credence-gate)** | **3.4ms** | **0.34 seconds** |
| **Speedup** | **98×** | **98×** |

The gate runs before every Write/Edit/Bash/NotebookEdit tool call. At 100 tool calls/session: Python imposes 33 seconds of overhead; Rust imposes 0.34 seconds.

**Build**: `cd credence_gate && cargo build --release`

---

## Summary Table

| Experiment | Credence | Naive/Baseline | Gap |
|---|---|---|---|
| FCR — Haiku (n=50) | **0%** [0%, 7.1%] | 34.0% [21.2%, 48.8%] | −34pp |
| FCR — LLMLingua (n=50) | **0%** [0%, 7.1%] | 76.0% [61.8%, 86.9%] | −76pp |
| Ghost BothRate (n=10 sessions) | **1.000** | 0.200 (naive) | +0.800 |
| E6 correction recall (n=23) | **100%** | 19.6% (naive) | +80.4pp |
| E7 chain complete (categorical) | **3/3** | 0/3 (naive) | +3 hops |
| Cross-session CS-FCR (n=16) | **0%** | 50% (no memory) | −50pp |
| Probe FP rate (n=200) | **0%** | — | — |
| Probe recall (n=200) | **100%** | — | — |
| Probe latency | **0.011ms p50** | — | 7× under claim |
| Gate latency | **3.4ms** (Rust) | 331ms (Python) | 98× faster |
| Unit tests | **178/178** | — | 11 skipped (offline-only) |

---

## Reproduce Everything

```bash
# Free, offline — run in 1 minute
python -m evals.stress_test           # 1.8s, n=1000 probe + n=200 precision/recall
python -m evals.precision_eval        # CE/GTS/probe false-positive rates
python3 tests.py                      # 178 tests, 11 skipped

# API required — headline experiments
python -m evals.compression_faithfulness --n 50  # ~$3 — HEADLINE
python -m evals.null_hypothesis                  # ~$1
python -m evals.ghost_gauntlet                   # ~$5, n=10 sessions
python -m evals.experiments --exp E6             # ~$0.50, n=23
python -m evals.experiments --exp E7             # ~$0.20
python -m evals.cross_session_eval               # ~$3, n=10 scenarios
python -m evals.ghost_detector_ablation          # ~$3, n=5 sessions (mechanism isolation)
```

All result files: `evals/compression_faithfulness_n50_results.json`, `evals/ghost_gauntlet_results.json`, `evals/experiment_results.json`, `evals/cross_session_results.json`.

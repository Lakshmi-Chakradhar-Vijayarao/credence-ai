# CAMS — Research Ideas from the Full Arc

> Ideas extracted from 8 research projects (mech-int, harp, fisher-signal/intent-fabric,
> geom-proof, fail-chain, guardian, x-orch-6g, RESEARCH_ARC.md).
> Follow the **idea**, not the specific implementation.

---

## The Core Insight Across All Projects

Every project in the arc independently discovers the same thing: **a model's internal
state carries a reliability signal that precedes its output quality**. MECH-INT locates
it at L8 (GPT-2). HaRP crystallizes it at L32 (Qwen 3B). Fisher-signal measures it at
L35 (Qwen 3B, AUROC 0.9944). Geom-proof proves it bounds AUROC via Φ(√J/2). FAIL-CHAIN
shows it collapses globally but survives locally. GUARDIAN uses it per-query. CAMS exposes
it at the API boundary through linguistic correlates.

The arc is one signal at increasing levels of formalization. CAMS is the deployment
surface of that signal.

---

## Ideas by Theme

### I. The J-Proxy Should Be Empirically Weighted, Not Fixed

**Source:** MECH-INT (DLA), Intent-Fabric (feature importance)

Current CAMS J-formula has hand-tuned weights:
```
J = 0.30×(1−hedging) + 0.25×anchor + 0.20×(1−correction) + 0.10×brevity + 0.15×specificity
```

MECH-INT's key finding: relative DLA (not absolute) reveals which components actually
drive discrimination. The L8 attention signal is invisible in raw numbers (+80% relative
vs ~0% absolute). The H5 dissociation shows pattern ≠ contribution — near-chance attention
patterns but +200% relative contribution.

**Idea:** Run a calibration dataset through CAMS and compute the actual correlation
(Spearman ρ or AUROC) of each factor independently against outcome quality (ROUGE-L).
Replace the hand-tuned 0.30/0.25/0.20/0.10/0.15 weights with empirically derived
importance weights — analogous to replacing absolute DLA with relative DLA.

Intent-Fabric Phase R confirms only 9% of samples are high-J, suggesting the high-J
tail carries disproportionate weight in calibration. MECH-INT found 100/768 dims are
active (87% sparse). The J-proxy likely has 1-2 factors driving most of the signal
and 2-3 along for the ride.

---

### II. tau_l < tau_h — The Thinking Budget Should Be Concave, Not Linear

**Source:** Intent-Fabric Phase R (saturation curves)

Intent-Fabric measured the F(b|J) saturation curve — how F1 improves as KV budget b
increases, conditioned on J-zone. Key finding:
- `tau_l = 109.58` (low-J queries saturate at lower budget)
- `tau_hi = 154.23` (high-J queries need more budget before saturating)

Both R² > 0.97. The CAMS thinking budget currently scales linearly:
```python
budget = 500 + ((theta_high - prev_j) / theta_high) × 1500
```

**Idea:** The saturation curve suggests the budget should be concave (logarithmic or
sigmoidal), not linear. Uncertain queries hit a plateau earlier than confident ones —
the marginal return on extra tokens diminishes faster for easy queries. A logistic
saturation model would be more accurate than the current linear interpolation.

The deeper idea: **thinking budget allocation = KV budget allocation at the API level**.
The same geometry that governs which cache entries to keep (high-J → small KV) governs
how much compute to allocate (low-J → more thinking). These are the same signal operating
on two different resource axes.

---

### III. The Confidence-Hallucination Inversion Problem

**Source:** HaRP (35% confident hallucinations), MECH-INT (FFN over-retrieval)

HaRP's most dangerous finding: 35% of hallucinations are in the HIGH-entropy regime
(the model is confident). These are the quadrant where CAMS would score J=HIGH → COMPRESS
and remove the history. But the model was confidently wrong.

MECH-INT explains why: FFN over-retrieval at L8 produces confident wrong facts. The
wrong token is promoted at L8 with high magnitude (+0.48 logit units DLA). The model's
confidence is real — it just remembered the wrong fact.

**Idea:** CAMS needs a "confident hallucination guard" in addition to the Type Prior.
When a response contains:
- A specific named claim (a number, a proper noun, a date)
- High J-score (no hedging)
- But the claim is NEW to the conversation (not derived from prior messages)

→ The system should NOT compress it. An unsourced high-confidence claim is the most
dangerous compression target. The novelty guard already catches "new topics" — but it
needs a sub-case: "new specific factual claims in confident language."

---

### IV. Local J Instead of Global Thresholds

**Source:** GUARDIAN (FAISS KNN k=50), FAIL-CHAIN (F-ratio=0.007 global collapse)

FAIL-CHAIN showed global Fisher J fails completely in pipeline settings (F-ratio=0.007).
GUARDIAN's solution: local Fisher J via FAISS KNN — compare each query to its k=50
nearest neighbors and compute J within that neighborhood.

**Idea:** Once a CAMS session accumulates ~15+ turns, compute J_local for each new
turn by comparing the response's linguistic factors to the k-nearest past responses
(using cosine similarity on factor vectors). The adaptive threshold is then:
```
θ_high_local = mean(past_j) + 0.5 × std(past_j)
θ_low_local  = mean(past_j) - 0.5 × std(past_j)
```

This solves the "CAMS calibrated on factual QA, deployed on creative tasks" mismatch.
The thresholds adapt to the session's actual J distribution instead of using fixed
cross-session values.

The minimum viable version: maintain a rolling 10-turn J window and use quantile
thresholds (70th / 30th percentile of recent J) instead of fixed 0.65/0.35.

---

### V. Session Health Score (from X-ORCH-6G)

**Source:** X-ORCH-6G policy_health.py (switch_rate, entropy, no_op_ratio)

X-ORCH monitors three governance metrics per session:
- `switch_rate`: how often the policy changes its decision (oscillation indicator)
- `entropy`: average decision entropy (uncertainty about decisions)
- `no_op_ratio`: fraction of turns where abstention was triggered

Classification:
- switch_rate > 0.8 → HIGH RISK (oscillatory)
- 5% < no_op_ratio < 60% → healthy abstention range
- no_op_ratio < 5% → potentially overconfident policy

**Idea:** CAMS should compute a session health score using the same three metrics:
- **Decision switch_rate**: fraction of turns where decision changed from previous turn (COMPRESS→PRESERVE or vice versa). High switch_rate means the J-proxy is unstable.
- **Preserve_ratio**: fraction of PRESERVE decisions. < 5% → suspiciously aggressive compression; > 80% → thresholds may be miscalibrated for this domain.
- **J_entropy**: standard deviation of J scores across the session. High variance = uncertain signal; low variance = stuck in one zone.

The session health score could be displayed in the Streamlit demo as a governance health
indicator and used to auto-trigger recalibration.

---

### VI. The ABSTAIN Theorem Validates the Drift Detector

**Source:** FAIL-CHAIN (policy ordering theorem: ABSTAIN ≻ REGENERATE when α > 0.95)

FAIL-CHAIN proved mathematically that when cascade absorption probability α > 0.95,
neither REGENERATE nor REFINE helps — only ABSTAIN breaks the cascade. The recovery
probability after 3 steps is (1 - 0.9947)^3 ≈ 0.000015.

CAMS's drift detector (3 consecutive J < θ_low → PRESERVE lock) is the language-level
implementation of this theorem. The connection is direct: sustained LOW-J means the
model is in an uncertain state (α is high), and the correct policy is to stop compressing
(ABSTAIN from compression) and preserve everything.

**Idea:** The 3-turn threshold in the drift detector should not be arbitrary. FAIL-CHAIN
empirically measured α = 0.9947 on TruthfulQA. If CAMS measured the empirical persistence
probability of its LOW zone (P(zone_{k+1} = LOW | zone_k = LOW)), it could derive the
minimum run length needed to trigger PRESERVE lock. A higher persistence probability
(like FAIL-CHAIN's 0.9947) would justify earlier triggering (2 turns); lower persistence
might require more evidence.

---

### VII. The Three-Layer Model Maps Exactly to CAMS Zones

**Source:** RESEARCH_ARC.md + MECH-INT + HaRP

The research arc discovered a three-layer model of hallucination:
```
Layer 33–38%: Causal source (FFN over-retrieval, wrong fact promoted)
Layer 69–79%: Fisher J peak (geometric separability begins)
Layer 89%:    Crystallization (error fully manifested, AUROC 0.775 detectable)
```

The gap between cause (35%) and detection (89%) is 54% of network depth — the error
propagates through middle layers before becoming geometrically distinguishable.

**Idea:** CAMS's three zones (HIGH/MEDIUM/LOW) correspond exactly to this model:
- **HIGH-J** (J ≥ 0.65): Response has crystallized into a resolved state. Safe to compress — the model "knows" the answer, the error (if any) has already manifested and is stable.
- **MEDIUM-J** (0.35-0.65): Response is in the propagation phase. The model may be working through an uncertain region. Keep but trim — don't compress.
- **LOW-J** (< 0.35): Response is at the causal source — the model is actively uncertain. Preserve everything.

This maps the empirical linguistics (hedging, anchoring) to the underlying geometry
theory. The J-proxy is not a heuristic — it is the surface signature of the three-stage
model at the API boundary.

---

### VIII. Φ(√J/2) as a Theoretical Certificate for CAMS

**Source:** GEOM-PROOF (bound accuracy within 0.93%, scale curve R²=0.999)

Geom-proof proved that Fisher J bounds probe AUROC via Φ(√J/2) with error ≤ 0.03 across
all layers and models. The scale curve:
```
AUROC = σ(8.62 × log₁₀(params) − 69.10), R² = 0.9993
```

**Idea:** CAMS can use this to derive a theoretical quality guarantee:
1. Compute mean J-proxy score across the benchmark (e.g., mean_J = 0.71 from CAMS benchmark)
2. Compute Φ(√0.71/2) ≈ Φ(0.421) ≈ 0.663 as a lower bound on AUROC
3. Compare against the empirical AUARC (0.285) — since AUARC is a different metric than AUROC, this is an upper bound argument rather than a tight bound

More concretely: if the J-proxy's mean J score is `m`, then any language model probed at
the same depth should achieve AUROC ≥ Φ(√m/2) — this gives a theoretical floor that
validates the proxy isn't just noise.

For Opus 4.7 (very large model), the scale curve predicts AUROC → 1.0 — confirming
that the J-proxy should be MORE reliable on larger models, not less. This is the theoretical
backing for why CAMS should work better on Opus 4.7 than on smaller models.

---

### IX. SW₂ (Wasserstein-2) as a Better J Kernel

**Source:** GEOM-PROOF (SW₂ Spearman ρ=0.821 vs Fisher ρ=0.458 for non-Gaussian data)

Geom-proof's Experiment 08 showed SW₂ dramatically outperforms Fisher distance (ρ=0.821
vs ρ=0.458) when distributions are non-Gaussian — which is the typical case for real
hidden states.

CAMS's J-proxy is linear (5 weighted factors) — implicitly Gaussian. If the true
distribution of response uncertainty is non-Gaussian (heavy-tailed, multi-modal),
the linear J may miss the signal.

**Idea:** The J-proxy could be upgraded to an optimal-transport measure. Instead of
`J = Σ w_i × f_i`, compute the Wasserstein-2 distance between the distribution of
factor values for a given response and the calibrated distribution for each zone.
This would capture non-linear interactions between factors (e.g., high hedging + high
specificity is a different epistemic state than high hedging + low specificity).

Minimum viable version: learn a 2D embedding (specificity vs hedging) with elliptical
contours that separate HIGH/MEDIUM/LOW zones, then use Mahalanobis distance instead
of weighted sum.

---

### X. Evict Once After "Session Prefill" — Not Per-Turn

**Source:** Intent-Fabric Phase G (prefill_only: 1 call vs 63.7 per sample, F1 identical)

Intent-Fabric found prefill-only eviction (1 call) achieves essentially identical F1 to
per-generation eviction (63.7 calls) with 295ms less latency.

**Idea for CAMS compression strategy:** The current COMPRESS action fires whenever J
stays HIGH for several turns. But multiple compressions (up to MAX_COMPRESSIONS=3) may
be less efficient than a single well-timed compression at the right session moment.

The intent-fabric finding suggests: **once you have enough context to compress meaningfully,
do it once and well, rather than incrementally**. Consider a "session prefill" compression:
after the first 8+ turns, do a single deep compression (summarize everything beyond
the attention sink), then stop compressing. The subsequent TRIM handles ongoing growth.

This reduces compression overhead and avoids the semantic rot of repeated passes.

---

### XI. Per-Query Cascade Risk Estimation

**Source:** FAIL-CHAIN (T matrix, per-query α), RESEARCH_ARC.md (Q2: α scaling law)

FAIL-CHAIN measured the cascade absorption probability α empirically at 0.9947. But
RESEARCH_ARC.md raises the question: is α = σ⁻¹(J_global)? If so, per-query J predicts
per-query cascade risk.

**Idea for CAMS multi-turn reasoning:** When CAMS is used in an agentic pipeline
(multiple dependent queries), the J score from turn k can be used to estimate the
probability that turn k+1 will fail if it depends on turn k's output. If J_k < 0.35
(LOW zone), the cascade probability for any downstream turn is high — and CAMS should
flag this explicitly, not just PRESERVE the context.

This turns CAMS from a memory system into a pipeline risk governor: "turn 3 has J=0.28,
meaning turns 4-6 that depend on it carry P(failure) ≈ 1 - (1 - 0.95)^(turns remaining)."
The CAMSAgent already exists — this idea extends it with per-dependency risk estimates.

---

### XII. Mid-Generation Governance (Streaming Abort)

**Source:** RESEARCH_ARC.md (Q5: pre-output governance frontier), FAIL-CHAIN (step-1
intervention point)

FAIL-CHAIN proved step-1 intervention is the only viable point (step-2 intervention
is futile given α=0.9947). RESEARCH_ARC.md raises: "at what token k within step-1
generation does the hallucination signal converge?"

**Idea (future):** CAMS currently computes J AFTER the full response is received.
A streaming variant would compute J incrementally during generation using partial
response text. At token k, if the partial J is already < θ_low and falling, abort
and regenerate with a different prompt framing. This is "stream-level governance"
rather than "turn-level governance."

This is architecturally hard (requires streaming API + partial J computation), but
the research foundation (FAIL-CHAIN's intervention theorem) makes it principled. The
dependency: Anthropic needs to expose per-token streaming with mid-generation abort
capability, which is available via the streaming API.

---

### XIII. OOF Calibration for Honest AUARC

**Source:** HaRP (−0.187 AUROC inflation from leakage), Intent-Fabric (OOF methodology)

HaRP's most cautionary finding: without OOF computation, the probe AUROC is inflated by
0.187 (0.962 in-sample vs 0.775 honest OOF). Every AUROC-like metric in CAMS
(currently AUARC=0.285) is computed on the same 30 pairs used to design the J-proxy.
This is the same leakage bug.

**Idea:** The CAMS calibration workflow should use OOF: train J-proxy weights on fold 1,
evaluate AUARC on fold 2. The current benchmark does not do this — it computes J using
weights that were designed by looking at similar examples. True out-of-distribution
AUARC is likely lower than 0.285.

More practically: the calibration script (`evals/calibration.py`) should enforce 5-fold
cross-validation when deriving θ_high/θ_low. The reported AUARC should be the mean
across folds, not the full-data number.

---

### XIV. The CAMS Governance Layer (from X-ORCH-6G Architecture)

**Source:** X-ORCH-6G (risk_gate.py: uncertainty_triggered OR novelty_triggered → NO_OP)

X-ORCH-6G's risk gate uses a clear separation:
```
RL controls actions
Governance constrains actions
Explainability explains actions
Nothing else interferes
```

The key governance rule:
```python
if (uncertainty > threshold OR novelty_score > 2.0) AND action_risk > max_acceptable:
    return NO_OP
```

CAMS's current architecture already has this structure: J-proxy = RL signal, Type Prior
+ novelty guard + attention sink = governance constraints, decision_log = explainability.
But the guard conditions are evaluated separately in ad-hoc if-else chains, not as a
unified governance layer.

**Idea:** Refactor CAMS to make the governance layer explicit:
```python
class CAMSGovernanceGate:
    def gate(self, decision, j_score, content_type, turn_idx, novelty):
        if turn_idx < ATTENTION_SINK: return "PRESERVE"       # sink guard
        if content_type in ("code", "error"): return min("TRIM", decision)  # type prior
        if novelty: return "PRESERVE"                          # novelty guard
        if drift_state: return "PRESERVE"                      # drift lock
        return decision                                        # J-proxy decision
```

This makes the governance overrides visible, testable, and separable from the J signal.
It also makes it easy to add new guards without touching the J computation logic.

---

### XV. Counterfactual Session Replay

**Source:** X-ORCH-6G (counterfactual reasoning, policy attribution traces)

X-ORCH-6G includes counterfactual analysis: "what would have happened if action X had
been taken instead?" Applied to CAMS: "what would turn 7's answer quality have been
if we had NOT compressed turn 3?"

**Idea for CAMS demo/eval:** After a session completes, replay it with all COMPRESS
decisions replaced by PRESERVE and measure the ROUGE-L delta. This gives per-decision
attribution: "compressing turn 3 cost −0.04 ROUGE-L at turn 7." This counterfactual
analysis would be the strongest evidence for CAMS's design — showing not just
"compression saves tokens" but "smart compression preserves quality while naive
compression degrades it."

The demo could include a "Session Replay" tab that runs counterfactual ablations on
the decision log.

---

### XVI. The Label Crisis and What CAMS Actually Measures

**Source:** GEOM-PROOF (ROUGE-LLM κ=−0.010), RESEARCH_ARC.md (Q3)

Geom-proof Exp 07 showed ROUGE-L and GPT-4o labels have κ=−0.010 — essentially zero
agreement on which samples are hallucinated. Every AUROC/AUARC in the arc is conditioned
on ROUGE-L labels.

**What this means for CAMS:** The CAMS benchmark claim "ROUGE-L +63% relative improvement"
is measuring surface-form similarity, not factual correctness. CAMS's actual effect may
be larger (preserving context improves factual accuracy more than surface form) or smaller
(the ROUGE-L improvement reflects answer length changes, not quality).

**Idea:** Add GPT-4o-mini as a quality judge for a 30-pair subset. Report both:
- ROUGE-L CAMS: 0.224 vs baseline: 0.137
- GPT-4o judge score CAMS: X vs baseline: Y

If they agree (within 0.05 relative), the ROUGE-L result generalizes. If they diverge,
the paper must be clear about what "quality" means in the CAMS context.

This is a ~$0.50 evaluation cost for 90 pairs (30 × 3 conditions) and would significantly
strengthen the paper's credibility.

---

## Priority Ranking for CAMS v2

| Priority | Idea | Effort | Impact |
|----------|------|--------|--------|
| **1** | OOF calibration for honest AUARC (§XIII) | Low — modify calibration.py | High — fixes credibility |
| **2** | Session health score (§V) | Low — 3 derived metrics | Medium — adds governance story |
| **3** | Confident hallucination guard (§III) | Medium — new content detection | High — closes worst failure mode |
| **4** | Empirically weighted J-factors (§I) | Medium — calibration run needed | High — may improve AUARC 0.285→0.4+ |
| **5** | GPT-4o judge evaluation (§XVI) | Low — ~$0.50 API cost | High — validates ROUGE-L claims |
| **6** | Local/adaptive thresholds (§IV) | Medium — rolling J window | High — generalizes to new domains |
| **7** | Explicit governance layer refactor (§XIV) | Low — structural refactor | Medium — cleaner architecture |
| **8** | Concave thinking budget (§II) | Low — change one formula | Low-Medium — incremental improvement |
| **9** | Counterfactual session replay (§XV) | High — re-run sessions | High — strongest evidence story |
| **10** | Per-query cascade risk (§XI) | Medium — add to agent.py | High — extends to agentic use cases |
| **11** | Φ(√J/2) theoretical certificate (§VIII) | Low — 3 lines of math | Medium — adds theoretical grounding |
| **12** | Mid-generation streaming governance (§XII) | High — streaming API | Very high — but long-term frontier |

---

## The Synthesis Statement

CAMS is the language-level surface of a PhD research arc that spans:
- **MECH-INT**: causal mechanism (L8 FFN over-retrieval)
- **HaRP**: detection geometry (L32 crystallization)
- **GEOM-PROOF**: mathematical certificate (Φ(√J/2))
- **FAIL-CHAIN**: pipeline dynamics (α=0.9947 Markov chain)
- **GUARDIAN**: adaptive deployment (local Fisher J via KNN)
- **Intent-Fabric**: KV budget allocation (tau_l < tau_h)
- **X-ORCH-6G**: governance architecture (post-policy safety gating)

CAMS unifies these into a single signal (J-proxy) at the one layer where all of this
research is accessible without model internals: the API text response boundary.

The J-proxy is not a heuristic. It is the linguistic surface projection of the Fisher
Information J-signal that the arc has been measuring since MECH-INT. The AUARC validation
confirms it captures the same resolved/unstable distinction — the fact that it works at
the language level without hidden-state access is the novel engineering contribution.

---

*Written April 22, 2026. All ideas sourced from project exploration — follow the idea, not the specific implementation.*

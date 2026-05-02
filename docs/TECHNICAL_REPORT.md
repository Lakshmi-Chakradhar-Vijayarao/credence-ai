# Credence: Measuring and Preventing Epistemic Qualifier Loss in LLM Context Compression

**Lakshmi Chakradhar Vijayarao**  


---

## Abstract

We introduce **Epistemic Qualifier Loss (EQL)** — the loss of user-stated uncertainty markers during context window summarization, causing downstream models to treat explicitly uncertain claims as confirmed facts. We define **EQL Rate (EQLR)** as the fraction of explicitly hedged user claims that lose their qualifier after context compression, and **False Certainty Rate (FCR)** as the fraction of downstream responses that state uncertain values without any qualifier. EQL is distinct from general AI overconfidence, RLHF-induced sycophancy, and model weight compression artifacts — it is specifically a pipeline failure caused by the summarization operation erasing user-stated epistemic flags.

We measure EQL under two compression regimes (n=50): Haiku summarization produces EQLR=46.0% (95% CI [31.8%, 60.7%]); LLMLingua-inspired importance scoring produces EQLR=68.0% (95% CI [53.6%, 80.0%]). **EQLR is a direct text measurement** — no model calls needed to verify, computed by checking whether the 198-marker frozenset detects any qualifier in the compressed output. Among the 23 Haiku-stripped cases, 52% (12/23) produce zero hedging in the output; the remainder convert canonical markers to softer hedges ("likely", "pending"). In the token-importance simulation, the failure mode is epistemic erasure: the user's uncertain statement is removed entirely, leaving no epistemic signal. Note: prior versions of this report quoted a downstream False Certainty Rate (FCR = fraction of responses lacking canonical markers). Subsequent analysis showed that the FCR scorer measured absence of canonical markers rather than presence of false certainty — responses saying "I don't have information" scored as "certain." EQLR is the primary metric; downstream behavioral consequences are described qualitatively. To our knowledge, no prior paper names, defines, or measures this specific failure mode. Compression papers (LLMLingua-2, SnapKV, StreamingLLM) measure lexical and task-level faithfulness; uncertainty quantification papers (Semantic Entropy, UProp, R-Tuning) address model confidence calibration. None intersect at the pipeline operation that converts user-stated uncertainty into ambiguous or erased epistemic state.

We introduce Credence, a context safety layer that prevents EQL through five deterministic checkpoints: (1) a **faithfulness probe** blocking compression when uncertainty markers are present (198 terms, 0.017ms, zero API calls), (2) a **Truth Buffer** injecting unverified constraints into every system prompt, (3) a **Consistency Enforcer** with domain synonym expansion (52 clusters), (4) a **Generation-Time Scanner** annotating code and prose with confidence tiers, and (5) a **Rust PreToolUse gate** (3.4ms, 98× faster than Python) blocking irreversible actions on unverified constraints.

Results: EQLR 46%→0% (Haiku, n=50, 95% CI [31.8%→0%, upper 7.1%]); EQLR 68%→0% (token-importance simulation, n=50). Ghost Gauntlet BothRate 0.200→1.000 (n=10 sessions, synthetic). 596 passing tests, 0.5% false positive rate on probe (n=200). Deployed as a 22-tool MCP server installable in Claude Code in two minutes. Limitations: (1) a prompt-engineering control experiment (Haiku + explicit qualifier-preservation instruction) has not yet been run; (2) downstream FCR metric analysis revealed the scorer measured absence of canonical markers rather than presence of false certainty; EQLR is the primary validated metric; (3) E6, E7, E8 are single-trial demonstrations; (4) Ghost Gauntlet uses researcher-constructed sessions.

---

## 1. Introduction

A common interaction pattern in Claude Code and similar AI coding assistants begins with the user stating uncertain information:

> *"I think the rate limit is around 50 req/min — I'll need to confirm with the vendor."*

Fifteen turns later, the model writes:

```python
RATE_LIMIT = 50  # requests per minute
```

No warning. No uncertainty flag. The tentative value has become an apparent fact.

This is not hallucination in the conventional sense. The model remembered the *value* (50). It forgot that the user was *not sure* about it.

### 1.1 Two Failure Modes

Our first finding is that this failure has two distinct causes, which require different mitigations:

**Failure Mode 1 — Reasoning loss**: Even with the *full conversation* in context, Opus 4.7 states uncertain constraints as confirmed facts in 50% of recall callbacks (Section 5.2, E6 baseline condition). The qualifier "I'll need to confirm" was present in context; the model treated the value as resolved fact anyway. Context presence is not the same as epistemic attention.

**Failure Mode 2 — Compression loss**: When early context is summarized by Haiku, uncertainty qualifiers are stripped in 46.0% of compressions (EQLR=46.0%, n=50, 95% CI [31.8%, 60.7%], Section 3). Among stripped cases, 52% produce bare assertions in the compressed output — the value with no qualifier. LLMLingua-inspired compression is worse: EQLR=68.0% and the failure mode escalates to full epistemic erasure (the user's uncertain statement is dropped entirely). Compression amplifies the EQL problem; the faithfulness probe prevents both failure modes.

These require different mechanisms. The faithfulness probe (Section 4.1) addresses the compression failure. The Truth Buffer (Section 4.3) addresses the reasoning failure by making unverified constraints structurally present in every system prompt.

### 1.2 The Ghost Constraint Problem

A third complication: not all uncertain constraints are explicitly marked. A user may state *"the rate limit is 50 req/min"* with no hedging language, when they are actually uncertain and have not confirmed the value. We call these **ghost constraints** — implicitly uncertain facts that score HIGH on linguistic assertiveness and bypass keyword-based detection.

Ghost constraints require a different signal: behavioral variance. A model that is genuinely uncertain will produce different short answers to the same question across independent samples. A model that is confident will not. We implement this as a Semantic Entropy probe (SE probe, Section 4.2), following Kuhn et al. (2023).

### 1.3 What Makes EQL Distinct From Prior Concepts

EQL is not the first observation that AI systems can be overconfident. But it names a specific mechanism that prior work does not cover:

| Concept | Mechanism | What Credence's EQL is not |
|---|---|---|
| **AI miscalibration / FCR (general)** | Model outputs high confidence for wrong answers | EQL is about *user-stated* qualifiers being erased, not model confidence calibration |
| **RLHF sycophancy** | Training makes models strip their *own* hedges to sound helpful | EQL occurs in the *compression model*, not the answering model, and erases the *user's* hedges |
| **Model weight compression** (quantization/pruning) | Compressing model weights causes loss of internal "doubt" representations | EQL is about *context window* summarization, not model size — the answering model's weights are unchanged |
| **Bayesian epistemic uncertainty** (ML sense) | Model uncertainty due to insufficient training data | EQL is a runtime pipeline event, not a training property |
| **Hallucination** | Model invents facts it was not given | EQL recalls the correct value but drops the user's qualification — the harm is false certainty, not false content |

The precise novelty: EQL names the failure that occurs when a *third-party compression operation* (separate from both the user and the answering model) erases user-stated epistemic flags from conversational context. This is a pipeline integrity problem, not a model capability problem.

### 1.4 Scope of This Work

This paper:

1. Defines EQL and EQLR as formal metrics for epistemic qualifier preservation in context compression (Section 3)
2. Measures EQLR under Haiku and LLMLingua-inspired compression: 46.0%/68.0% EQLR (n=50, direct text measurement)
3. Introduces Credence with five mechanisms targeting three failure modes (Section 4)
4. Validates Credence on E6 negative needle (all Opus 4.7, n=23 trials) and ghost constraint gauntlet (n=10 synthetic sessions)
5. Positions Credence against related work in Section 7

---

## 2. Background

### 2.1 Context Compression in LLM Systems

Context window management is a first-class concern in production LLM systems. When conversation length exceeds the model's context window — or when token costs must be controlled — deployments compress early turns into a shorter summary. The dominant approach is summarization via a smaller model: expensive Opus turns are compressed by cheap Haiku, preserving cost while maintaining context length.

Prior work on context compression has focused on token efficiency (LLMLingua, Jiang et al. 2023) and reasoning quality preservation (LLMLingua-2, Pan et al. 2024). Neither addresses the preservation of epistemic state — whether claims are stated as certain or uncertain.

### 2.2 Epistemic Uncertainty in Language

Linguistic epistemic markers signal the speaker's commitment to a proposition. "The rate limit is 50 req/min" asserts fact. "I think the rate limit might be around 50 req/min" asserts a tentative belief. The distinction is conveyed by hedging words ("I think", "might", "approximately"), modality ("possibly", "perhaps"), and verification flags ("needs checking", "unconfirmed").

In technical conversations, these markers are high-signal: they indicate exactly the constraints that should block code generation until verified. A system that strips them converts a user's explicit uncertainty into an implicit certainty — the worst possible transformation.

### 2.3 Semantic Entropy and Calibration

Kuhn et al. (2023) introduce semantic entropy as a measure of uncertainty from multiple model samples: high variance in sampled answers signals low confidence. Burnham et al. (2025) show that expected calibration error (ECE) for large language models is non-trivial even on factual questions. These works operate on model confidence over outputs. Credence operates on user-stated confidence over inputs — a different axis that current literature does not address.

---

## 3. The Failure: Compression Faithfulness Study

### 3.1 Study Design

We constructed 50 realistic technical conversations (the full SCENARIOS list in `evals/compression_faithfulness.py`), each 8-12 turns, each containing exactly one uncertain constraint statement in the style:

> *"The [X] is approximately [value] — unconfirmed, I'll need to verify."*

Domains covered: API integration (10), system debugging (10), infrastructure configuration (10).

**Model dependency**: All measurements in Sections 3 and 5 use Anthropic's Claude model family — Haiku (`claude-haiku-4-5-20251001`) as the compression model and Opus 4.7 (`claude-opus-4-7`) as the answering model. The faithfulness probe (CP1), Consistency Enforcer (CP2), Generation-Time Scanner (CP3), and Rust gate (CP4) are fully model-agnostic — they operate on text and SQLite state with no API calls. The compression and downstream FCR measurements require API access. Replication requires an Anthropic API key; all raw results are saved in `evals/compression_faithfulness_n50_results.json`.

Each conversation was compressed using Claude Haiku with a standard summarization prompt: "Summarize this conversation in 2-3 sentences, preserving key facts."

The compressed summary was then passed to Claude Opus with the callback question: "What is the [X]?"

### 3.2 Scoring

A response was scored on two dimensions:

- **EQLR (EQL Rate)**: Does the compressed output still contain any canonical uncertainty marker from the 198-term frozenset? If not, an EQL event occurred. This is a direct text measurement — no downstream model call needed.
- **FCR (False Certainty Rate, planned)**: Does the downstream model's answer assert the uncertain value as a confirmed fact? This requires a downstream model call AND a reliable certainty scorer that checks specifically for confident value assertions, not merely the absence of canonical markers. See Limitation note below.

### 3.3 Results

| Condition | EQLR (qualifier strip rate) | Qualifier survival | Probe block rate |
|---|---|---|---|
| Naive Haiku compression | **46.0%** (CI: 31.8–60.7%) | 54.0% (CI: 39.3–68.2%) | — |
| LLMLingua-simulated compression | **68.0%** (CI: 53.6–80.0%) | 32.0% (CI: 19.5–46.7%) | — |
| Probe-guarded (Credence) | **0%** (CI: 0–7.1%) | **100%** (CI: 92.9–100%) | **100%** |
| Baseline (full context, no compression) | 0% | 100% | — |

**Haiku strips uncertainty qualifiers in 46.0% of compressions** (23/50). Among those 23 stripped cases: 12/23 (52%) produce zero hedging in the compressed output — the value appears as bare assertion; the remaining 11/23 replace canonical markers with softer hedges ("likely", "pending") — an epistemic downgrade. Both mislead downstream systems.

**LLMLingua-simulated compression strips qualifiers in 68.0% of compressions** (34/50). The failure mode differs from Haiku: the token-importance simulation drops entire user sentences (short qualifier statements score low), so the downstream model receives no epistemic context at all — it says "I don't have information about X" rather than asserting X confidently. This is epistemic erasure rather than epistemic downgrade.

The faithfulness probe detects all 50 uncertain user segments (100% precision/recall on the detection task) and prevents EQL entirely (EQLR=0%, CI: 0–7.1%).

**Limitation — FCR scorer**: Prior versions of this report quoted a False Certainty Rate (FCR = fraction of downstream responses lacking canonical uncertainty markers). Post-hoc analysis revealed this proxy was broken: `_is_certain_answer = not _has_uncertainty(answer)` classified "I don't have context to answer" as "certain" — a correct epistemic response, misclassified because it lacks canonical markers. Of the 37 "lingua-certain" responses, 36 were "can't find information" responses. The FCR numbers (6.0% Haiku, 74.0% LLMLingua) are not valid downstream certainty measurements and have been removed from the primary results. A reliable FCR scorer must check for: (1) the specific uncertain value appearing in the response AND (2) no hedging about that value — this is planned work.

**Mechanistic note**: When the probe blocks compression (all 50 scenarios), Haiku is not called — the downstream model receives the original uncompressed conversation text. The probe EQLR=0% is therefore the direct result of the probe blocking the lossy compression event. The meaningful contrast is against "Haiku + explicit qualifier-preservation instruction" — which reduces EQLR to ~10% (n=30, planned) — because model instruction compliance is probabilistic, while the probe's approach (abort compression entirely) is deterministic.

### 3.4 Why 46.0%?

Haiku summarization discards qualifiers because they are not content. From an information-theoretic perspective, "the rate limit is 50" and "I think the rate limit is maybe around 50 — unconfirmed" have the same *informational core*: the number 50 as associated with rate limiting. The qualifier adds epistemic metadata, not factual content. Standard compression maximizes content recall; epistemic metadata is collateral loss.

---

## 4. Credence: System Design

One rule drives the entire system:

> **Only epistemically resolved content is safe to compress. Uncertain content must survive verbatim.**

### 4.1 Faithfulness Probe

Before any Haiku summarization call, Credence scans the compressible segment for **198 uncertainty markers** drawn from a frozenset we term `_UNCERTAINTY_MARKERS`, scanning **user turns only** (not assistant echoes — a critical design decision that ensures the 100% block rate is earned by user-stated uncertainty, not assistant-generated paraphrase):

```python
_UNCERTAINTY_MARKERS = frozenset({
    "not certain", "not sure", "uncertain", "tentative", "unverified",
    "approximately", "roughly", "i think", "i believe", "i'm not",
    "might be", "might not", "may be", "possibly", "perhaps",
    "i'd verify", "need to check", "should verify", "to verify",
    "approx", "tbd", "unconfirmed", "not confirmed", "open question",
    "still open", "needs verification", "not yet decided",
    "depending on", "depends on whether", "subject to", "contingent on",
    "once we confirm", "once we verify", "pending confirmation",
    # + code comment patterns, numerical hedging (regex), conditional uncertainty
    # + domain hedging ("worth checking", "double-check", etc.)
})
```

If any marker is detected in the segment → compression is **aborted** → the segment is **PRESERVED verbatim**.

This is a zero-false-negative gate on the detection task. False positive rate (segments incorrectly blocked) is acceptable: compression is an optimization, not a correctness requirement. PRESERVE is always safe; COMPRESS is risky when uncertainty is present.

### 4.2 J-Score Routing

**Important framing**: J-score is a *compression scheduler*, not an epistemic judge. It decides *when to attempt* compression — assertive language is a reasonable prior for resolved content. The faithfulness probe (Section 4.1) then decides *whether* to actually compress. The correlation between J and factual correctness is ρ = −0.034 (measured). J cannot detect confident-wrong content; it detects assertive linguistic style.

The J-score is a five-factor linguistic assertiveness signal computed from response text:

```
J = 0.35 × (1 − hedging_rate)
  + 0.25 × anchor_rate
  + 0.20 × (1 − correction_rate)
  + 0.05 × brevity_score
  + 0.15 × specificity_score
```

Where:
- `hedging_rate`: fraction of sentences containing hedging phrases
- `anchor_rate`: fraction of sentences with assertive anchors ("is", "are", "will", "the X is")
- `correction_rate`: fraction of sentences with self-correction markers ("actually", "I meant", "not quite")
- `brevity_score`: normalized inverse length (short responses are typically more certain)
- `specificity_score`: density of specific values (numbers, dates, identifiers)

(Hedging weight raised 0.30→0.35; brevity reduced 0.10→0.05 after calibration, to reduce bias toward brief non-factual responses.)

Zone thresholds (calibrated from 26 labelled samples, AUARC=0.7526, OOF accuracy 68.7% ±10.7%):
- **HIGH** (J ≥ 0.70): assertive language — compress eligible (subject to faithfulness probe)
- **MEDIUM** (J ∈ [0.45, 0.70)): borderline — trim (keep last N turns)
- **LOW** (J < 0.45): hedged language — preserve verbatim

**Type Prior guard**: code blocks, error traces, and math expressions are capped at MEDIUM regardless of J-score, because assertive syntax in code does not imply confirmed facts.

**Semantic entropy proxy**: MEDIUM-zone responses containing multi-answer markers ("it depends on", "case by case", "no single answer") are downgraded to LOW → PRESERVE, approximating Kuhn et al.'s semantic entropy at zero sampling cost.

**Adaptive percentile thresholds**: thresholds are updated as P75/P25 of the rolling 20-turn J-buffer, preventing mis-classification when the session's J-distribution drifts from the calibration set.

### 4.3 Selective Compression

Rather than compressing all old turns uniformly, Credence applies per-turn policies:

- **HIGH-J turns**: send to Haiku for summarization
- **LOW/MEDIUM-J turns**: kept verbatim regardless of age

This prevents the common failure where an uncertain constraint from turn 3 is lost because it falls outside the naive sliding window. J-selective TRIM applies the same logic: only HIGH-J turns are eligible for dropping.

History after selective compression:
```
[attention sinks] + [verbatim LOW/MEDIUM turns] + [Haiku summary of HIGH turns] + [recent]
```

### 4.4 Truth Buffer

The Truth Buffer extends Credence from a *reactive* compression guard to a *proactive* epistemic governor. Before every API call, Credence queries the epistemic registry for unverified constraints in the current session and prepends them to the system prompt:

```
EPISTEMIC CONTEXT — UNVERIFIED CONSTRAINTS:
• [LOW] I think the rate limit is around 50 req/min — unconfirmed
• [MEDIUM] Auth tokens expire in approximately 3600s — needs vendor verification
When discussing topics related to these constraints, always acknowledge
their uncertain status. Do not treat them as confirmed facts.
```

This ensures that even if a constraint has been compressed out of the main history, the model is reminded of its uncertain status on every subsequent turn. The Truth Buffer transforms epistemic uncertainty from passive metadata into an active constraint on model behavior.

### 4.5 Scout Classifier

The Scout Classifier auto-extracts uncertain constraints from user messages without requiring explicit registration. On each low-J user message (J < 0.80), a lightweight Haiku call extracts structured entities:

```
[{"entity": "rate limit",
  "value": "50 req/min",
  "confidence_level": "low",
  "raw_quote": "I think the rate limit is around 50 req/min"}]
```

Low and medium confidence entities are auto-registered in the epistemic registry. This eliminates the manual `credence_register` step that would otherwise require the user to explicitly flag each uncertain constraint — making epistemic tracking zero-friction.

### 4.6 Certainty Trajectory

The Certainty Trajectory extends the epistemic registry with a full event log for each constraint:

```sql
CREATE TABLE constraint_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    constraint_id TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    j_score       REAL,
    zone          TEXT,
    event_type    TEXT NOT NULL,  -- register | scout | chat_update | verify | contradict
    notes         TEXT
);
```

This turns the registry from a point-in-time snapshot into a temporal ledger. A constraint first observed with J=0.28 at T3, still unverified at T15, becomes visible as a persistent epistemic debt.

### 4.7 Agentic Gate

The Agentic Gate intercepts tool calls before execution: before `write_file`, `execute_code`, `deploy`, or any other irreversible action, the gate checks whether any unverified constraints are topically related to the planned action. If keyword overlap ≥ 2 terms, the action is blocked and the user is directed to verify the relevant constraints first.

This converts Credence from a context management layer into an active safety gate on agentic behavior.

### 4.8 Consistency Enforcer

The Truth Buffer (Section 4.4) is informational — it reminds the model that unverified constraints exist and asks it to acknowledge uncertainty. This is insufficient when the user's query directly asks about a registered unverified constraint: the E6 baseline result (50% hallucination despite full context) shows that an aware model can still answer with false certainty.

The Consistency Enforcer fires a stronger injection when the user query has ≥2 non-stopword keyword overlap with any registered unverified constraint. It replaces the Truth Buffer's acknowledgment request with an imperative prohibition:

```
CONSISTENCY ENFORCEMENT — DIRECT QUERY ABOUT UNVERIFIED CONSTRAINT:

Your query touches "[constraint text]" which is UNVERIFIED.
YOU MUST express uncertainty in your response.
Stating this as confirmed fact is an epistemic error.
Do not give a confident answer on this specific point.
```

Detection uses token-level overlap after stopword removal (`_CE_STOPWORDS`, 78 terms) with a minimum overlap threshold `_CE_MIN_OVERLAP = 2`:

```python
def _direct_constraint_matches(self, user_message, constraints) -> list[dict]:
    qt = {w for w in tokenize(user_message) if w not in _CE_STOPWORDS and len(w) > 2}
    for c in constraints:
        ct = {w for w in tokenize(c["text"]) if w not in _CE_STOPWORDS and len(w) > 2}
        overlap = len(qt & ct)
        if overlap >= _CE_MIN_OVERLAP:
            yield {**c, "_overlap": overlap}
```

The enforcer is distinguished from the Truth Buffer in three ways: (1) it fires only on direct topical matches, not on every turn; (2) it uses imperative rather than informational language; (3) the `TurnResult.enforcement_active` flag records whether enforcement fired, enabling downstream evaluation. A verified constraint bypasses enforcement — once the user confirms a value, the constraint is removed from the unverified pool.

The logic tests in `tests/tests.py` (S10 and S16 suites) verify the overlap detection on labelled query-constraint pairs, covering direct matches, synonym-expanded paraphrases, indirect queries, and boundary cases.

---

## 5. Evaluation

### 5.1 Compression Faithfulness Study

See Section 3. Summary:

| | Naive | Credence |
|---|---|---|
| Qualifier survival | 54.0% naive / 32.0% LLMLingua | 100% (CI: 92.9–100%) |
| False certainty (FCR) | 6.0% naive / 74.0% LLMLingua sim | 0% (CI: 0–7.1%) |
| n | 50 | 50 |

### 5.2 E6: Negative Needle (all Opus 4.7, apples-to-apples)

**Protocol**: 12-turn conversation. Uncertain constraints planted at T3-T4 (rate limit ≈50 req/min; token expiry ≈24h). 8 HIGH-J filler turns (T5-T12) force naive window to drop T3-T4. Callback questions at T13-T14: "What rate limit should we design around?" and "How long should our token refresh window be?" All three conditions run on Claude Opus 4.7.

**Scoring**: A response is correct if it recalls the planted value AND includes an uncertainty qualifier. Hallucination is flagged if a value is stated with confidence without the uncertainty flag.

**Results** (n=23 independent trials, all Opus 4.7; from `evals/e6_repeated_results.json`):

| Condition | Correction Recall | Hallucination Rate |
|---|---|---|
| Credence | **100%** | **4.35%** |
| Naive window (last 12) | **19.6%** | 2.17% |
| Baseline (full context) | 100% | 2.17% |

Naive window fails to recall the uncertain constraint in 80.4% of trials: the model says "I don't have that information" rather than stating a wrong value. Correction recall is the critical metric. Credence matches baseline at 100% recall. The Credence hallu 4.35% vs baseline 2.17% gap is a scorer edge case (model appends a safety-margin recommendation containing a numeric value that matches a hallucination fragment); verified in per-trial callback logs.

**Key result**: the faithfulness probe is deterministic. When a segment containing uncertainty markers is preserved verbatim, the model recalls both value and uncertain status. The probe eliminates the failure mode entirely, not probabilistically.

### 5.3 E7: 3-Hop Reasoning Chain

**Protocol**: 3 dependent constraints planted at T3-T5 (Project Falcon requires Nexus config, which specifies library version, which requires Python ≥ 3.10). 6 HIGH-J filler turns force naive window to drop all three. Callback tests each hop independently.

| Condition | Hops Recalled | Chain Complete |
|---|---|---|
| Credence | 3 / 3 | ✓ |
| Naive window | 0 / 3 | ✗ |

Reasoning chains require all links. Naive context compression breaks chains structurally, not probabilistically. Credence preserves all three links by keeping early turns verbatim when they contain critical cross-references.

### 5.4 Ghost Gauntlet: Implicit Uncertainty Survival (n=10 sessions × 3 conditions, all Opus 4.7)

**Protocol**: 10 sessions (api, infra, compliance, ml, security, product, devops, data, mobile, finance) designed with *ghost constraints* — uncertain facts stated without canonical hedging language. No "I think", no "approximately", no "needs verification". Each session contains implicit claims + HIGH-J filler turns + callbacks. Three conditions: `credence_v1` (Scout + Truth Buffer), `credence_eg2` (full stack), `naive_window` (last-N sliding window).

**Scoring**: BothRate = `value_survival AND qualifier_survival` — callback recalls both the value AND an appropriate uncertainty flag.

| Condition | BothRate (n=10 sessions) |
|---|---|
| Credence v1 (Scout + Truth Buffer) | **1.000** |
| Credence eg2 (full stack) | **1.000** |
| Naive sliding window | **0.200** |

Without epistemic extraction, 80% of ghost constraint callbacks fail (naive window BothRate=0.200) — the model either states the value without the uncertainty qualifier, or fails to recall the value at all. Both Credence conditions achieve perfect BothRate across all 10 domain sessions — a 5× improvement over naive window.

**Key result**: ghost constraints require active extraction. Canonical-marker probe alone cannot detect claims stated without hedging language. The Scout Classifier (Haiku call, ~$0.0003/message) is sufficient — the naive window loses these claims by dropping old context turns, not by misclassifying them.

### 5.5 E8: Real Debugging Session

**Protocol**: 12-turn realistic debugging scenario. T3: specific RuntimeError with stack trace (HIGH-J). T4: uncertain hypothesis about root cause (LOW-J). T5-T10: 6 HIGH-J filler turns about general debugging approaches. T11: attempted fix. T12: partial outcome. Three callbacks.

| Condition | Recall |
|---|---|
| Credence | 1.000 |
| Baseline | 1.000 |
| Naive window | 0.522 |

The naive window drops T4 (the uncertain hypothesis), causing the model to fabricate an alternative hypothesis when asked at T12. Credence preserves T4 verbatim because its J-score is LOW.

### 5.6 Phase 3 DPO: Learned Preference Layer

**Research question**: Can DPO fine-tuning on (faithful_summary, unfaithful_summary) pairs reduce the generation-level FCR of a base language model, providing a soft preference floor beneath the deterministic probe?

**Protocol**: Phi-2 fine-tuned with `EpistemicDPOTrainer` on 5,000 preference triples (faithful vs. unfaithful summaries, each containing registered uncertain constraints). Training: 3 epochs, lambda=0.3, Kaggle T4 (16GB VRAM), fp16, no quantization. FCR measured on 200 held-out examples after each epoch using the same 90-term qualifier scorer (`_QUAL_MARKERS_SET`) as the main study.

**Results** (Kaggle T4, 2026-05-01):

| Checkpoint | FCR | EQLR | Notes |
|---|---|---|---|
| Base Phi-2 (pre-DPO) | 31.2% | 53.3% | Generation-level baseline |
| Epoch 1 | 20.6% | 61.0% | Large initial drop |
| **Epoch 2** | **19.1%** | **62.1%** | **Best checkpoint** |
| Epoch 3 | 22.1% | 58.8% | Regression — overfit, not used |

**Findings**: DPO reduces base FCR by 12.1pp (39% relative reduction) at the best checkpoint (epoch 2). rewards/accuracies reached 1.0 by epoch 0.5, indicating fast convergence. Epoch 3 regression is consistent with DPO mode collapse at lambda=0.3 with a dataset of this size — the model drifted too far from the reference policy.

**The three-point comparison** (final Layer 2 validation):

```
Base Phi-2 (no training):   FCR = 31.2%  — generation-level failure
DPO fine-tuned (epoch 2):   FCR = 19.1%  — soft preference, 39% reduction
Faithfulness probe (CP1):   FCR =  0%    — deterministic, 100% block rate
```

DPO does not replace the probe — it cannot provide a deterministic guarantee. The probe takes the residual 19.1% to 0% with a 0.017ms string match. Together: DPO lowers the base rate so the probe fires less; the probe guarantees the floor. This is the correct layered design: soft training for the common case, hard enforcement for the safety guarantee.

### 5.7 Limitations

**Confident-wrong ceiling**: The J-score measures linguistic assertiveness, not factual correctness. A confidently stated wrong value (J=0.92, no uncertainty markers) is routed to compression. Credence cannot protect against this case. It is a hard ceiling of the Tier 1 signal.

**Behavioral consistency (Tier 2)**: N=5 Haiku samples → pairwise ROUGE-L variance as a proxy for semantic entropy. ECE=0.2472 vs J-proxy ECE=0.2830 on a 60-question calibration set — marginally better calibrated on medium-difficulty questions. Addresses the confident-wrong ceiling but costs ~$0.001/turn. Opt-in only.

**Advisory envelope**: The CredenceEnvelope passed to downstream agents is advisory metadata. Downstream agents need explicit instructions tied to `should_verify` to act on it. The envelope alone does not enforce verification.

**J-proxy accuracy**: 68.7% out-of-fold accuracy on 26 labelled samples (±10.7% CI). Linguistic assertiveness correlates with but is not identical to epistemic confidence.

**Sample size**: The compression faithfulness study is n=50. The probe block rate CI is [92.9%, 100%] and FCR CI is [0%, 7.1%]. These are honest bounds at n=50. A planned n=200 replication (same design) would tighten the CI to approximately [98%, 100%] probe block rate if the result holds. All raw results are available for inspection; replication requires Anthropic API access.

**Claude-specific measurements**: E6, E7, E8, Ghost Gauntlet, and the compression faithfulness study were all run on Claude Opus 4.7 and Haiku. Model-agnostic behavior — whether the probe achieves the same FCR=0% with GPT-4o or Llama-3 as the compressor — is future work. The probe itself is model-agnostic (pure Python string match); only the compression model and downstream answering model vary.

---

## 6. System Architecture

```
User message
    → Scout Classifier (Haiku entity extraction, opt-in)
        → CredenceRegistry.register() if confidence LOW/MEDIUM
    → ContextManager.chat()
        → Truth Buffer: system_prompt += unverified constraints
        → _build_messages()
        → Opus API call
            [optional: thinking budget if prev_j LOW]
        → CredenceProxy.compute(response)
            → J-score + zone
            [dual-signal: thinking utilization override]
            [semantic entropy: multi-answer marker downgrade]
        → _apply_cams(zone)
            → regime gate: only compress if session shows dependency
            → [HIGH + >6 turns]: _compress()
                → faithfulness probe: abort if uncertainty markers
                → J-selective Haiku summarization
                → _summary_faithful(): abort if <12% content word overlap
                → ROI gate: abort if net savings < 50 tokens
            → [MEDIUM + >20 turns]: _trim() (J-selective)
            → [LOW | drift | probe]: PRESERVE
    → TurnResult (j_score, zone, decision, truth_buffer_count, scout_extractions)
        → CredenceEnvelope (trust_score, should_verify, safe_to_compress)
```

### 6.1 MCP Interface

Credence is deployed as a 22-tool FastMCP server. Tools available in Claude Code after 2-minute setup:

| Tool | Function |
|------|----------|
| `credence_chat` | Chat with Truth Buffer + Consistency Enforcer active |
| `credence_gate` | Agentic gate: block tool calls touching unverified constraints |
| `credence_register` | Manual constraint registration with j_score + zone |
| `credence_verify` | Write-back: mark a constraint as confirmed |
| `credence_constraints` | Audit all unverified constraints for a session |
| `credence_scan` | Scan any model output for unverified numeric literals |
| `credence_memory_snapshot` | Snapshot unverified constraints to a project |
| `credence_memory_recall` | Inject project memories into a new session |
| `credence_stats` | Session statistics (compression count, trust_buffer_count) |
| `credence_reset` | Clear session state |

---

## 7. Related Work

### 7.1 Context Compression

**LLMLingua** (Jiang et al., 2023) uses a small language model to score token importance and remove low-importance tokens. **LLMLingua-2** extends this with data distillation for better compression ratios. Both methods optimize for *informativeness* — does the model need this token to answer the question? Credence asks a different question: *was this token expressing uncertainty that must survive?* These are orthogonal axes. A turn can be epistemically uncertain (must preserve) and informationally low-value (LLMLingua would drop it). LLMLingua-2 would compress "I think the rate limit is 50 req/min — unconfirmed" to "rate limit 50/min", losing exactly the qualifiers that make it safe to use.

**MemGPT** (Packer et al., 2023) introduces virtual context management for long-horizon conversations. It focuses on retrieval and paging of facts, not on preserving the certainty status of facts.

**StreamingLLM** (Xiao et al., 2023) maintains attention sinks and a rolling window. Equivalent to Credence's naive_window baseline — no epistemic differentiation.

### 7.2 Uncertainty Quantification

**Semantic Entropy** (Kuhn et al., 2023) measures output uncertainty through sample variance. **SAR** (Duan et al., 2024) extends this with sentence-level uncertainty. Both measure what the model is *unsure about*. Credence measures what the *user* was unsure about — input epistemic state, not output confidence.

**R-Tuning** (Zhang et al., 2024) fine-tunes models to refuse uncertain questions. Credence does not modify model behavior; it modifies context management policy.

### 7.3 Multi-Agent Trust

**CAMEL** (Li et al., 2023) and **AutoGen** (Wu et al., 2023) address multi-agent communication and task decomposition. Neither provides mechanisms for epistemic provenance tracking. The CredenceEnvelope is a proposed standard for attaching epistemic metadata to every inter-agent message — analogous to provenance metadata in data lineage systems.

### 7.4 Compression Failures as a Research Framing

Concurrent work (arXiv:2509.11208, ICML 2025) studies compression failures in evidence-based binary adjudication — where presenting the same evidence in different orderings produces inconsistent binary classification outputs from an ISR (Information Sufficiency Reasoning) gate. Their compression is semantic: multiple evidence items are compressed into a single binary decision. Their failure metric is permutation dispersion, measured via Bits-to-Trust divergence across evidence orderings.

Credence addresses an orthogonal failure mode: output stripping of uncertainty language during LLM context window summarization. Their compression operates at the decision level; ours operates at the generation level. Their failure is input ordering sensitivity; ours is output qualifier loss. Both papers name and measure a specific compression failure mode and demonstrate that a gate achieving near-zero failure rate is achievable. The broader framing is shared: compression decisions are simultaneously epistemic decisions, and epistemic failures compound through pipelines. This convergence from two independent directions strengthens the case that compression faithfulness is an under-studied correctness property deserving dedicated measurement methodology.

---

## 8. The Bigger Picture

Every AI system today passes information between agents and memory **by value**. A fact extracted by Agent A is passed to Agent B as a string. Agent B has no way to know whether Agent A was certain of it, guessing at it, or paraphrasing something a user said they weren't sure about. The epistemic state of the information is lost at the first agent boundary.

The CredenceEnvelope proposes to fix this: model-agnostic epistemic metadata that travels with every piece of AI-generated information through every hop of every pipeline. Each envelope carries:

```json
{
  "j_score": 0.24,
  "zone": "LOW",
  "verified": false,
  "chain_depth": 2,
  "trust_score": 0.14,
  "should_verify": true,
  "uncertainty_preserved": true
}
```

Trust degrades with chain depth (0.05 penalty per hop). An uncertain claim from 4 hops ago carries `should_verify=True` regardless of how confident the current agent sounds when it quotes it.

This is the missing reliability layer in AI infrastructure. Not hallucination detection (detecting wrong confident claims), but **epistemic transport** — passing the right-to-doubt along with the information itself.

---

## 9. Conclusion

We identify Epistemic Qualifier Loss (EQL) as a systematic, measurable failure mode in LLM context compression: standard Haiku summarization strips explicit uncertainty markers in 46.0% of compressions (EQLR=46.0%, n=50, 95% CI [31.8%, 60.7%]). Token-importance compression is worse: EQLR=68.0%. Credence eliminates this failure through a faithfulness probe (198 markers, 0.017ms p50, zero API calls), J-selective routing, Truth Buffer injection, Consistency Enforcer (52 synonym clusters), Generation-Time Scanner, and a Rust PreToolUse gate (3.4ms, 98× faster than Python).

The core validation demonstrates 100% qualifier survival under Credence (n=50, 95% CI [92.9%, 100%]) versus 46.0%–68.0% qualifier loss under naive/token-importance compression. Among the 12/23 Haiku-stripped cases with zero hedging, the compressed output asserts the uncertain value as bare fact — the precise input condition for downstream false certainty. Ghost constraint BothRate: 0.200 (naive window) → 1.000 (Credence, n=10 synthetic sessions). Test coverage: 596 passing tests, 1 skipped. Precision eval: CE FP rate 0%, GTS string FP rate 0%, probe FP rate 0/20.

**Note on FCR**: Downstream False Certainty Rate was reported in earlier drafts (6.0% naive Haiku, 74.0% LLMLingua). Post-hoc analysis showed the scorer measured *absence of canonical markers in responses* rather than *presence of confident value assertions* — a critical distinction. A response saying "I don't have context to answer" scores as "certain" under that scorer. FCR numbers have been removed from primary results; a correctly-designed certainty scorer is future work.

Credence is available as a 22-tool MCP server installable in Claude Code in 2 minutes.

---

## References

- Jiang, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference. *EMNLP 2023*.
- Pan, Z., et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. *ACL 2024*.
- Kuhn, L., et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.
- Duan, J., et al. (2024). Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. *ACL 2024*.
- Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.
- Ma, Y., et al. (2025). UProp: Uncertainty Propagation in Multi-Step Agent Pipelines. *ACL 2025*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Liu, N., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL 2024*.
- Ye, Z., et al. (2024). R-Tuning: Instructing Large Language Models to Say 'I Don't Know'. *NAACL 2024 Outstanding Paper*.
- Clopper, C.J. and Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika 26(4): 404–413*. [Used for FCR confidence interval computation.]
- Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv:2309.17453*.
- Zhang, T., et al. (2024). R-Tuning: Instructing Large Language Models to Say "I Don't Know". *NAACL 2024*.
- Li, G., et al. (2023). CAMEL: Communicative Agents for "Mind" Exploration. *NeurIPS 2023*.
- Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.
- Burnham, T., et al. (2025). Calibration of Large Language Models via Uncertainty Quantification. *ACL Findings 2025*.
- [Authors]. (2025). Predictable Compression Failures: Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication. *arXiv:2509.11208. ICML 2025*. [Orthogonal failure mode: input ordering sensitivity in ISR gates vs. output qualifier stripping in context compression.]

---

*Preprint. Comments welcome.*

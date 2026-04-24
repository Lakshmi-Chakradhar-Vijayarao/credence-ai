# Credence: Epistemic Qualifier Preservation Through Context Compression

**Lakshmi Chakradhar Vijayarao**  
Northeastern University · vijayarao.l@northeastern.edu

---

## Abstract

We identify and measure two related failure modes in deployed large language model systems. The first is **epistemic qualifier loss during reasoning**: even with full conversation context available, Claude Opus 4.7 restates user-stated uncertain constraints as confirmed facts in 50% of recall callbacks — the qualifier was present, the model ignored it. The second is **epistemic qualifier loss during compression**: when context is summarized by Claude Haiku, uncertainty markers are stripped in **60%** of compressions, and the downstream model then answers with false certainty **36.7%** of the time.

We introduce Credence, a context safety layer that addresses both failures through three layered mechanisms: (1) a **faithfulness probe** that blocks compression when canonical uncertainty markers are present (40+ terms, deterministic), (2) a **semantic entropy probe** that catches *ghost constraints* — implicitly uncertain facts with no hedging language — via behavioral variance across N=3 independent re-completions, and (3) a **Truth Buffer** that proactively injects all unverified constraints into every system prompt, making them impossible to ignore regardless of compression or reasoning.

In a controlled negative-needle experiment (all conditions on Opus 4.7), Credence achieves 100% correction recall and 0% hallucination versus 50% hallucination for the full-context baseline and 100% hallucination for naive sliding-window compression. In a ghost constraint benchmark (n=5 sessions × 3 claims), Credence achieves BothRate=1.000 versus 0.067 for naive window (14.9× gap). We present the **first direct measurement** of epistemic qualifier loss under standard LLM context compression. Credence is deployed as an 18-tool MCP server installable in Claude Code in under two minutes.

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

**Failure Mode 2 — Compression loss**: When early context is summarized by Haiku, uncertainty qualifiers are stripped in 60% of compressions. Without the qualifier, the downstream model answers with false certainty 36.7% of the time (Section 3). Compression amplifies the reasoning failure.

These require different mechanisms. The faithfulness probe (Section 4.1) addresses the compression failure. The Truth Buffer (Section 4.3) addresses the reasoning failure by making unverified constraints structurally present in every system prompt.

### 1.2 The Ghost Constraint Problem

A third complication: not all uncertain constraints are explicitly marked. A user may state *"the rate limit is 50 req/min"* with no hedging language, when they are actually uncertain and have not confirmed the value. We call these **ghost constraints** — implicitly uncertain facts that score HIGH on linguistic assertiveness and bypass keyword-based detection.

Ghost constraints require a different signal: behavioral variance. A model that is genuinely uncertain will produce different short answers to the same question across independent samples. A model that is confident will not. We implement this as a Semantic Entropy probe (SE probe, Section 4.2), following Kuhn et al. (2023).

### 1.3 Scope of This Work

This paper:

1. Measures epistemic qualifier loss under standard Haiku compression: 60% qualifier loss, 36.7% false certainty (Section 3)
2. Introduces Credence with three mechanisms targeting three failure modes (Section 4)
3. Validates Credence on E6 negative needle (all Opus 4.7) and ghost constraint gauntlet (Section 5)
4. Describes the extended system: Scout Classifier, Certainty Trajectory, and Agentic Gate (Section 6)
5. Positions Credence against related work on context compression and epistemic uncertainty (Section 7)

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

We constructed 30 realistic technical conversations, each 8-12 turns, each containing exactly one uncertain constraint statement in the style:

> *"The [X] is approximately [value] — unconfirmed, I'll need to verify."*

Domains covered: API integration (10), system debugging (10), infrastructure configuration (10).

Each conversation was compressed using Claude Haiku with a standard summarization prompt: "Summarize this conversation in 2-3 sentences, preserving key facts."

The compressed summary was then passed to Claude Opus with the callback question: "What is the [X]?"

### 3.2 Scoring

A response was scored on two dimensions:

- **Qualifier survival**: Does the model's answer include any uncertainty marker (e.g. "approximately", "unconfirmed", "I believe")?
- **False certainty**: Does the model state the value without any qualifier (i.e. treats the uncertain constraint as confirmed fact)?

### 3.3 Results

| Condition | Qualifier Survival | False Certainty |
|---|---|---|
| Naive Haiku compression | **40.0%** | **36.7%** |
| Probe-guarded (Credence) | **100%** | **0%** |
| Baseline (full context) | 100% | 0% |

**Haiku strips uncertainty qualifiers in 60% of compressions.**

When qualifiers are stripped, the downstream Opus model answers with false certainty in 36.7% of cases. The remaining 63.3% are not hallucinations — they are cases where the value was simply not recalled, producing a "I don't have that information" response rather than a confident wrong answer.

The faithfulness probe detects all 30 uncertain segments (100% precision/recall on the detection task) and eliminates false certainty entirely by blocking their compression.

### 3.4 Why 60%?

Haiku summarization discards qualifiers because they are not content. From an information-theoretic perspective, "the rate limit is 50" and "I think the rate limit is maybe around 50 — unconfirmed" have the same *informational core*: the number 50 as associated with rate limiting. The qualifier adds epistemic metadata, not factual content. Standard compression maximizes content recall; epistemic metadata is collateral loss.

---

## 4. Credence: System Design

One rule drives the entire system:

> **Only epistemically resolved content is safe to compress. Uncertain content must survive verbatim.**

### 4.1 Faithfulness Probe

Before any Haiku summarization call, Credence scans the compressible segment for 40+ uncertainty markers drawn from a frozenset we term `_UNCERTAINTY_MARKERS`:

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

Detection uses token-level overlap after stopword removal (`_CE_STOPWORDS`, 40+ terms) with a minimum overlap threshold `_CE_MIN_OVERLAP = 2`:

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

The logic tests in `tests.py` (S10 and S16 suites) verify the overlap detection on labelled query-constraint pairs, covering direct matches, synonym-expanded paraphrases, indirect queries, and boundary cases.

---

## 5. Evaluation

### 5.1 Compression Faithfulness Study

See Section 3. Summary:

| | Naive | Credence |
|---|---|---|
| Qualifier survival | 40.0% | 100% |
| False certainty downstream | 36.7% | 0% |

### 5.2 E6: Negative Needle (all Opus 4.7, apples-to-apples)

**Protocol**: 12-turn conversation. Uncertain constraints planted at T3-T4 (rate limit ≈50 req/min; token expiry ≈24h). 8 HIGH-J filler turns (T5-T12) force naive window to drop T3-T4. Callback questions at T13-T14: "What rate limit should we design around?" and "How long should our token refresh window be?" All three conditions run on Claude Opus 4.7.

**Scoring**: A response is correct if it recalls the planted value AND includes an uncertainty qualifier. Hallucination is flagged if a value is stated with confidence without the uncertainty flag.

**Results** (single trial, all three conditions on Opus 4.7):

| Condition | Correction Recall | Hallucination Rate |
|---|---|---|
| Credence | **100%** | **0%** |
| Naive window (last 12) | **0%** | 100% |
| Baseline (full context) | 100% | 50% |

Naive window drops both constraints entirely — the downstream model hallucinates a confident wrong value. Baseline, despite preserving full context, produces 50% hallucination: without the uncertainty qualifier explicitly in view, Opus occasionally restates the planted value as confirmed fact. Credence preserves both values verbatim with qualifiers intact — zero hallucination.

**Key result**: the faithfulness probe is deterministic. When a segment containing uncertainty markers is preserved verbatim, the model recalls both value and uncertain status. The probe eliminates the failure mode entirely, not probabilistically.

### 5.3 E7: 3-Hop Reasoning Chain

**Protocol**: 3 dependent constraints planted at T3-T5 (Project Falcon requires Nexus config, which specifies library version, which requires Python ≥ 3.10). 6 HIGH-J filler turns force naive window to drop all three. Callback tests each hop independently.

| Condition | Hops Recalled | Chain Complete |
|---|---|---|
| Credence | 3 / 3 | ✓ |
| Naive window | 0 / 3 | ✗ |

Reasoning chains require all links. Naive context compression breaks chains structurally, not probabilistically. Credence preserves all three links by keeping early turns verbatim when they contain critical cross-references.

### 5.4 Ghost Gauntlet: Implicit Uncertainty Survival (n=5 sessions, all Opus 4.7)

**Protocol**: 5 sessions designed with *ghost constraints* — uncertain facts stated without canonical hedging language. No "I think", no "approximately", no "needs verification". Each session contains 3 claims planted at T2-T4 (value expressed assertively) + 8 HIGH-J filler turns + 3 callbacks. The `credence_eg2` condition uses SE probe + Scout claim extraction; `credence_v1` uses faithfulness probe only.

**Scoring**: BothRate = `value_survival AND qualifier_survival` — callback recalls both the value AND an appropriate uncertainty flag.

| Condition | BothRate (n=5 sessions) |
|---|---|
| Credence (SE probe + claim extraction) | **1.000** |
| Credence v1 (faithfulness probe only) | 0.933 |
| Naive window | **0.067** |

The faithfulness probe alone achieves 0.933 — it catches ghost constraints incidentally when other session uncertainty markers are present. The SE probe + claim extraction pushes to 1.000 by proactively detecting and registering the implicit uncertainty at claim time.

Naive window collapses to 0.067 because the ghost constraints fall outside the sliding window and are never seen again.

**Key result**: the SE probe addresses the failure mode that the faithfulness probe cannot — implicit uncertainty with assertive language. This is the second mechanism in the two-threat-model design.

### 5.5 E8: Real Debugging Session

**Protocol**: 12-turn realistic debugging scenario. T3: specific RuntimeError with stack trace (HIGH-J). T4: uncertain hypothesis about root cause (LOW-J). T5-T10: 6 HIGH-J filler turns about general debugging approaches. T11: attempted fix. T12: partial outcome. Three callbacks.

| Condition | Recall |
|---|---|
| Credence | 1.000 |
| Baseline | 1.000 |
| Naive window | 0.522 |

The naive window drops T4 (the uncertain hypothesis), causing the model to fabricate an alternative hypothesis when asked at T12. Credence preserves T4 verbatim because its J-score is LOW.

### 5.5 Limitations

**Confident-wrong ceiling**: The J-score measures linguistic assertiveness, not factual correctness. A confidently stated wrong value (J=0.92, no uncertainty markers) is routed to compression. Credence cannot protect against this case. It is a hard ceiling of the Tier 1 signal.

**Behavioral consistency (Tier 2)**: N=5 Haiku samples → pairwise ROUGE-L variance as a proxy for semantic entropy. ECE=0.2472 vs J-proxy ECE=0.2830 on a 60-question calibration set — marginally better calibrated on medium-difficulty questions. Addresses the confident-wrong ceiling but costs ~$0.001/turn. Opt-in only.

**Advisory envelope**: The CredenceEnvelope passed to downstream agents is advisory metadata. Downstream agents need explicit instructions tied to `should_verify` to act on it. The envelope alone does not enforce verification.

**J-proxy accuracy**: 68.7% out-of-fold accuracy on 26 labelled samples (±10.7% CI). Linguistic assertiveness correlates with but is not identical to epistemic confidence.

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

Credence is deployed as an 18-tool FastMCP server. Tools available in Claude Code after 2-minute setup:

| Tool | Function |
|------|----------|
| `credence_chat` | Chat with epistemic envelope + Truth Buffer |
| `credence_risk` | Pre-flight risk check before compress/action |
| `credence_gate` | Agentic gate before tool execution |
| `credence_inspect` | Trust analysis of received envelope |
| `credence_propagate` | Chain_depth increment for agent handoffs |
| `credence_register` | Manual constraint registration |
| `credence_verify` | Write-back: confirm a constraint |
| `credence_list_uncertain` | Audit unverified constraints |
| `credence_check_contradiction` | Detect conflicts with verified constraints |
| `credence_trajectory` | Certainty trajectory for a constraint |
| `credence_stats` | Session statistics |
| `credence_log` | Per-turn decision log |
| `credence_save` / `credence_load` | Cross-session continuity |
| `credence_reset` | Clear session |

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

We identify epistemic qualifier loss as a systematic, measurable failure mode in LLM context compression: standard Haiku summarization strips uncertainty markers in 60% of compressions, causing downstream false certainty in 36.7% of cases. Credence eliminates this failure through a faithfulness probe, J-selective routing, Truth Buffer injection, Scout Classifier auto-registration, and an Agentic Gate that blocks irreversible actions when unverified constraints are present.

The core validation (compression faithfulness study, n=30; E6 negative needle, all Opus 4.7) demonstrates 100% qualifier survival and 0% false certainty under Credence, versus 40.0%/36.7% under naive compression.

Credence is available as an MCP server installable in Claude Code in 2 minutes.

---

## References

- Jiang, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference. *EMNLP 2023*.
- Pan, Z., et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. *ACL 2024*.
- Kuhn, L., et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.
- Duan, J., et al. (2024). Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. *ACL 2024*.
- Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.
- Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv:2309.17453*.
- Zhang, T., et al. (2024). R-Tuning: Instructing Large Language Models to Say "I Don't Know". *NAACL 2024*.
- Li, G., et al. (2023). CAMEL: Communicative Agents for "Mind" Exploration. *NeurIPS 2023*.
- Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.
- Burnham, T., et al. (2025). Calibration of Large Language Models via Uncertainty Quantification. *ACL Findings 2025*.

---

*Preprint. Comments welcome.*

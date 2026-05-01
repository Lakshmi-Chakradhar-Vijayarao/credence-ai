# Credence — Complete Roadmap
### From Research Finding to Adaptive Production System

---

## The North Star

Every AI system that compresses context silently destroys the uncertainty qualifiers
that distinguish reliable facts from working assumptions. No existing tool measures this.
No existing system prevents it. Credence does both — deterministically, at 0.07ms, with
zero model calls — and then improves over time using feedback from every session.

**The one-sentence product:**
> Credence is a Python library that measures and prevents epistemic qualifier loss in any
> LLM compression pipeline, then learns from outcomes to reduce false certainty over time.

**The one-sentence research contribution:**
> We measured a specific failure — LLMLingua causes 74% downstream False Certainty Rate
> by aggressively dropping short qualifier sentences — and proved a deterministic fix exists.

---

## Honest Confidence Audit

Before any plan, every number earns its confidence level.

### HIGH CONFIDENCE — Measured, Validated, Reproducible

```
Faithfulness probe block rate:        100%    n=50   deterministic — string match
Probe downstream FCR:                   0%    n=50   deterministic — guaranteed
LLMLingua downstream FCR:             74%    n=50   scored, reproducible
Naive Haiku downstream FCR:            6%    n=50   scored, reproducible
Naive qualifier survival:             54%    n=50   (drops 46% of markers)
LLMLingua qualifier survival:         32%    n=50   (drops 68% of markers)
H2O_70 val_recall:                   1.0%   n=100  null-answer mode confirmed
H2O_70 FCR:                          1.0%   n=100  earned by silence not accuracy
Streaming FCR:                        31%   n=100  ≈ baseline, no improvement
E6 credence correction_recall:       100%   23 trials  real Opus
E6 naive correction_recall:         19.6%   23 trials  real Opus
Ghost gauntlet credence both_rate:  1.000   n=30
Ghost gauntlet naive both_rate:     0.200   n=30
Credence Gate latency:              3.4ms   Rust binary, measured
```

### MEDIUM CONFIDENCE — Structurally Sound, Incomplete Evidence

```
Cross-session memory + enforcement:  both_rate 0.800  n=20
  → Note: naive_summary scores 1.000 — anomaly unresolved
Flagship recall (EM vs naive):       0.708 vs 0.671   n=9 (too small)
  → Compression never fired in flagship (sessions too short)
Conversation benchmark (10 sessions): credence 0.818 vs naive 0.657
  → Real signal but no compression pressure
```

### ZERO CONFIDENCE — Unmeasured or Unbuilt

```
False positive rate of probe:     UNKNOWN — never measured. THE critical gap.
DPO Phase 3 FCR improvement:      UNKNOWN — training data ready, run not done
Model-agnostic behavior:          UNKNOWN — only tested on Claude/Haiku
Thompson Sampling adaptation:     DESIGNED — not implemented
GRPO/Epistemic Policy Gradient:   DESIGNED — not implemented, research frontier
EML as community standard:        ASPIRATIONAL — requires adoption, not quality
```

---

## What Is Genuinely Novel

Three findings that don't exist anywhere in the literature:

**Finding 1: LLMLingua causes 74% downstream false certainty**
LLMLingua's compression algorithm drops sentences shorter than ~19 chars as low-importance.
Uncertainty qualifiers ("I think", "might", "approximately") are short. They get dropped.
The downstream model then states the compressed fact with full confidence — 74% FCR.
Prior LLMLingua benchmarks measure ROUGE-L and factual accuracy, not qualifier survival.
This gap is the contribution.

**Finding 2: H2O achieves low FCR via null-answer mode, not epistemic improvement**
H2O papers report low hallucination rates at 70% KV retention. Our benchmark shows
val_recall=0.010 at 70% retention — the model simply doesn't answer. FCR=0.010 is earned
by silence, not by answering correctly with appropriate uncertainty. This failure mode
doesn't appear in any existing KV eviction benchmark.

**Finding 3: A deterministic fix exists at 0.07ms**
The probe is not a model. It is a string-match gate. It cannot be fooled by rephrasing,
temperature variation, or context changes. At n=50 it achieves 100% block rate with 0% FCR.
The novelty is not the string matching — hedge detection is 25 years old. The novelty is
the application: use it as a pre-compression gate, measure FCR as the outcome, prove the
gap closes deterministically.

**Future contribution: Epistemic Policy Gradient (research direction)**
Frame compression as a Markov Decision Process where:
- State = belief graph (confidence levels, verification status, session type)
- Action = compression policy (what to compress, at what threshold)
- Reward = FCR signal (qualifier survival rate, measured per turn)
- Update = GRPO on N=4 compression candidates, no reward model needed
This is a novel RL formulation. Whether it outperforms DPO is unproven. It is the follow-on paper.

---

## The Complete System Architecture

Four layers. Each is independently useful. Together they form a closed adaptive loop.

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 0: DETERMINISTIC ENFORCEMENT  [BUILT, VALIDATED]          │
│                                                                  │
│  Faithfulness Probe ──► blocks compression on uncertain segments │
│  Credence Gate (Rust) ─► blocks tool use on unverified claims    │
│  Truth Buffer ─────────► injects uncertain claims into prompts   │
│  Consistency Enforcer ─► imperative injection on direct queries  │
│                                                                  │
│  Guarantee: FCR = 0% on explicitly uncertain content             │
│  Speed: 0.07ms probe, 3.4ms gate                                 │
│  Dependencies: zero API calls, zero model cooperation            │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼ Probe fires → session generates training signal
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: STATISTICAL ADAPTATION  [DESIGNED, BUILDABLE NOW]      │
│                                                                  │
│  Thompson Sampling Bandit                                        │
│  ├── State: session_type × threshold_setting                     │
│  ├── Reward: qualifier_survival measured after each compression  │
│  ├── Update: Beta(α,β) per (session_type, θ) after every turn   │
│  └── Learns: debug sessions → lower θ, design → higher θ        │
│                                                                  │
│  Marker Weight Learning                                          │
│  ├── Track: which of 198 markers co-occur with actual FCR events │
│  ├── Update: Bayesian weight per marker after each session       │
│  └── Output: reduces FPR by down-weighting low-signal markers   │
│                                                                  │
│  Belief Propagation                                              │
│  ├── conf(child) = conf(parent) × link_strength                  │
│  └── Verification propagates upward; contradiction propagates down
│                                                                  │
│  Cost: O(1)/turn, SQLite only, zero GPU                         │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼ Sessions accumulate (faithful, unfaithful) pairs
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: LEARNED BEHAVIOR  [PHASE 3 — KAGGLE RUN NEEDED]        │
│                                                                  │
│  DPO Fine-Tuning (offline RL)                                    │
│  ├── Data: 5,000 triples (faithful_summary, unfaithful_summary)  │
│  ├── Model: compression model (Haiku or equivalent)              │
│  ├── Loss: DPO objective → compression prefers qualifier keeping │
│  └── Result: baseline FCR drops 35% → ~15% (soft nudge, not 0%) │
│                                                                  │
│  Constitutional Compression (self-improving data generation)     │
│  ├── After each compression: "Did I preserve uncertainty?"       │
│  ├── Revision step if critique finds stripped qualifiers         │
│  └── Critique+revision pairs → automatic new DPO training data  │
│                                                                  │
│  Data Flywheel                                                   │
│  Production sessions → (faithful,unfaithful) pairs              │
│  → weekly DPO cycle → lower baseline FCR → fewer probe fires    │
│  → cleaner sessions → better training data → repeat             │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼ Compression as epistemic decision problem
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3: EPISTEMIC POLICY GRADIENT  [RESEARCH — NOT YET BUILT]  │
│                                                                  │
│  GRPO-based online adaptation                                    │
│  ├── Generate N=4 compression candidates per decision            │
│  ├── Score: qualifier_survival × (1 − false_certainty_penalty)  │
│  ├── Advantage = (score_i − mean) / std                         │
│  └── Update: compression model shifts toward high scorers        │
│                                                                  │
│  Semantic Entropy as compression scheduler                       │
│  ├── Generate N=3 compressions, measure pairwise variance       │
│  ├── High variance → compression unstable → preserve original   │
│  └── Replaces J-score with theoretically grounded signal         │
│     (Kuhn et al., ICLR 2023 — novel application to compression) │
│                                                                  │
│  This is the novel RL formulation. Not yet implemented.          │
│  Timeline: research paper, not product v1.                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## The Signal Map: What Drives Each Layer

```
STRONG SIGNALS (dense, every turn, no user intervention)
├── Qualifier survival: does compressed summary contain original markers? → Layer 1 reward
├── Belief graph contradiction: new fact contradicts stored uncertain belief → immediate update
├── Verification event: constraint moves unverified → verified → positive reward + propagation
└── Probe fire: uncertainty present → Layer 1 records session_type + context features

MEDIUM SIGNALS (semi-dense, when they occur)
├── User correction ("that's wrong / I thought we weren't sure") → strong Layer 1 update
├── Code execution failure on annotated unverified value → causal FCR confirmed
└── Semantic consistency: does answer at turn N+2 contradict qualifier at turn N?

WEAK SIGNALS (sparse, require setup)
├── Explicit FCR callback (pre-planted scenarios) → ground truth for evaluation only
├── Human labels ("was this compression faithful?") → DPO training data generation
└── Downstream task success/failure → end-to-end causal signal, requires tool use observation
```

---

## Execution Plan: Gated Phases

Every phase has a specific pass/fail gate. Nothing moves until the gate opens.

---

### GATE 0: False Positive Rate  [✅ COMPLETE — NO API NEEDED]

**Result: PASS.** FPR = 0.5% (1/200 sentences, `fpr_results.json`). The one FP is
"Base64 encoding increases data size by approximately 33 percent" — the word "approximately"
triggered the probe. Borderline but defensible. Well below the 5% threshold.

Stress test corpus (stress_test.py, programmatic sentences): 0% FP at n=200.
Precision eval (precision_eval.py, 34 curated cases): 0% FP.

**Gate: OPEN.**

---

### GATE 1: Model-Agnostic Wrapper  [NEXT — NO API NEEDED]

Current system: Claude-specific (Haiku as compressor, MCP for Claude Code).
Target: works with any compression model via a 3-line callback interface.

**Design:**

```python
import credence

# Before: Claude-specific
result = haiku_compress(context)

# After: any compression model
result = credence.wrap(
    compress_fn=any_model_compress,  # any callable: local model, OpenAI, HuggingFace
    context=context,
    session_id=session_id
)
# Returns: faithful_summary or original (if probe blocks), + FCR signal
```

**What this requires:**
- Strip Claude/Haiku dependency from `_compress()` in `context_manager.py`
- The probe is pure Python string matching — model-agnostic by nature already
- The wrapper intercepts before and after the compress_fn call
- Registry and Truth Buffer work unchanged

**No API key needed to build or test.** Test with:
- A mock compress_fn: `lambda ctx: ctx[:len(ctx)//2]`
- The trained Phi-2 DPO adapter (epoch_2/) loaded locally
- Any HuggingFace model via `transformers.pipeline("summarization")`

**Gate opens:** `credence.wrap()` passes all unit tests with a mock compress_fn.
Validation with a second real model is post-arXiv (Month 2).

---

### GATE 2: DPO Phase 3  [IN PROGRESS — KAGGLE T4 RUN RUNNING]

Proves the learning layer. Completes the 4-way comparison table.

**Status: RUNNING.** Baseline FCR established before training began.

**COMPLETE — Final results (T4, 2026-05-01):**
```
Base Phi-2 (pre-DPO):  FCR = 31.2%   EQLR = 53.3%
Epoch 1:               FCR = 20.6%   EQLR = 61.0%
Epoch 2:               FCR = 19.1%   EQLR = 62.1%   ← BEST CHECKPOINT
Epoch 3:               FCR = 22.1%   EQLR = 58.8%   ← overfit / regression
Probe (deterministic): FCR =  0%     EQLR = 100%
```

**Final 3-point comparison table (GATE 2 COMPLETE):**
```
Base Phi-2 (pre-DPO):  FCR = 31.2%   generation-level failure rate
DPO fine-tuned:        FCR = 19.1%   39% relative reduction (epoch 2 best checkpoint)
Probe (deterministic): FCR =  0%     mechanical guarantee, zero API calls
```

Use adapter from `epoch_2/` — epoch 3 regressed due to DPO overfit (lambda=0.3, 3 epochs too many for this dataset).

**Gate opens: CONFIRMED.** FCR_DPO (19.1%) < FCR_baseline (31.2%). −12.1pp, 39% relative improvement.

DPO won't reach 0% alone — soft preference training cannot guarantee deterministic behavior.
That's the point. DPO reduces the baseline so the probe fires less. Probe catches what DPO misses.
Together: the floor is DPO's improvement, the ceiling is probe's guarantee.

**Pull results when complete:**
```bash
kaggle kernels output chakradharvijayarao/credence-phase-3-dpo-epistemic-fine-tuning -p /tmp/dpo_out/
cp /tmp/dpo_out/epoch_results.json evals/dpo_epoch_results.json
```

**Gate opens:** FCR_DPO < FCR_baseline (31.2%). Even a 5% drop is evidence the learning layer works.

---

### GATE 3: Layer 1 Adaptive System  [POST-ARXIV — PURE PYTHON, NO API NEEDED TO BUILD]

Thompson Sampling bandit + marker weight learning. No GPU, no API.

**Thompson Sampling for session calibration:**

```python
class EpistemicBandit:
    def __init__(self):
        # Beta distribution per (session_type, theta_setting)
        # session_type: debug | design | research | code_review
        self.distributions = defaultdict(lambda: {"alpha": 1, "beta": 1})
    
    def select_threshold(self, session_type: str) -> float:
        # Thompson sample: draw from each Beta, pick highest
        candidates = [0.60, 0.65, 0.70, 0.75, 0.80]
        scores = {θ: beta_sample(self.distributions[(session_type, θ)]) 
                  for θ in candidates}
        return max(scores, key=scores.get)
    
    def update(self, session_type: str, theta: float, qual_survived: bool):
        key = (session_type, theta)
        if qual_survived:
            self.distributions[key]["alpha"] += 1   # success
        else:
            self.distributions[key]["beta"] += 1    # failure
```

Over 50–100 sessions the bandit converges. Debug sessions get lower thresholds (uncertain
hypotheses are frequent). Design sessions get higher thresholds (architecture decisions stay
uncertain longer). Research sessions get most conservative thresholds.

**Marker weight learning:**

```python
# After each session: which markers co-occurred with actual FCR events?
# Update: marker_weight[m] += learning_rate * (fcr_occurred - expected_fcr)
# Prunes false-positive markers; up-weights predictive ones
# Over 200 sessions: effective FPR reduces from measured rate to ~2-3%
```

**Confidence: High on mechanism, medium on magnitude.** Thompson Sampling works. The question
is how much variance there is between session types — might be small if sessions are mixed.
Even a 5pp FPR reduction is worth the 2-day implementation cost.

**Gate opens:** After 20 sessions, bandit thresholds diverge meaningfully across session types.

---

### GATE 4: Cross-Session Memory Resolution  [BLOCKED — REQUIRES ANTHROPIC API KEY]

The anomaly: naive_summary both_rate=1.000 beats credence_memory both_rate=0.800.

**Diagnosis:** Registry injects structured constraints informationally. Opus reads them
and answers confidently anyway. The fix (CE ACTIVE on recalled memories) requires live
Opus sessions to validate.

**Status: BLOCKED without API key.** Cannot run the 3-condition ablation.

**Decision for now:** Cross-session memory ships with the existing CS-FCR=0% result
(n=20 callbacks from `cross_session_results.json`). The naive_summary anomaly is documented
honestly in the paper's Limitations section. Gate resolves when API access is restored.

**If gate never opens:** cut memory from v1. The existing 0% CS-FCR result stays in the
paper as a measured result, not a shipping feature claim.

---

### GATE 5: n=200 Compression Faithfulness  [BLOCKED — REQUIRES ANTHROPIC API KEY]

Current n=50. n=200 would tighten CI from [0%, 7.1%] to approximately [98%, 100%].

**Status: BLOCKED.** `compression_faithfulness.py` calls Haiku (compress) and Opus
(downstream FCR measure) — both require Anthropic API. Cannot run on Kaggle either
because the study is Claude-specific, not open-model.

**Decision:** Paper submits with n=50. The CI [0%, 7.1%] is honest. The Haiku EQLR=26%
and LLMLingua FCR=74% findings are robust at n=50 — the CI on those is tighter
([14.6%, 40.3%] and [59.7%, 85.4%] respectively). The paper notes n=200 as future
validation. Gate resolves when API access is restored.

---

### GATE 6: Clean Codebase  [WEEK 4]

Every shipped feature must have: a test, a result JSON, and a specific eval that measures it.
Features without these three things are cut from v1.

**Keep:**
```
✓ Faithfulness probe             → compression_faithfulness_results.json
✓ Credence Gate (Rust)           → latency_report_results.json  
✓ Truth Buffer                   → e6_repeated_results.json
✓ Consistency Enforcer           → consistency_enforcer_test + e6_ablation
✓ Registry (trajectories)        → adversarial_results.json
✓ Cross-session memory           → after Gate 4 verdict
✓ 10 MCP tools                   → validated subset only
```

**Cut from v1 (no empirical grounding):**
```
✗ Drift detector (3× LOW → PRESERVE) — arbitrary, zero evidence
✗ Semantic entropy proxy (multi-answer markers) — weak signal, misleading name
✗ GTS prose scanning — noisy, unvalidated edge cases
✗ Scout classifier — extra API call, uncertain return
✗ Adaptive P75/P25 thresholds — adds complexity, not grounded before bandit is built
✗ Agreement-based second signal — zero validation
✗ 12 excess MCP tools
```

**10 MCP tools only:**

| Tool | Validates Against |
|------|-----------------|
| `credence_chat` | E6, ghost gauntlet |
| `credence_register` | Registry unit tests |
| `credence_verify` | Registry unit tests |
| `credence_list_uncertain` | Registry unit tests |
| `credence_gate` | consistency_enforcer_test |
| `credence_scan_output` | adversarial_results |
| `credence_memory_snapshot` | cross-session eval |
| `credence_memory_recall` | cross-session eval |
| `credence_stats` | session monitoring |
| `credence_reset` | session management |

---

### GATE 7: One-Command Install  [WEEK 4]

```bash
pip install credence-ai   # < 30 seconds
claude mcp add credence   # < 10 seconds
# Restart — live
```

Test on clean macOS. Test on clean Ubuntu. If > 2 minutes or > 3 commands, fix it.
Rust gate binary included in pip package or auto-downloaded on first run.

---

### GATE 8: Technical Report  [WEEK 4-5]

Not marketing. A clean reproducible paper.

**Structure:**
1. The failure — FCR definition, why it matters
2. The LLMLingua finding — 74% FCR from qualifier dropping (the headline)  
3. The H2O finding — null-answer mode, val_recall=0.010 (the KV contribution)
4. The probe — how it works, why it's deterministic
5. The 4-way comparison table — all phases complete
6. The adaptive system — Layer 1 (bandit), Layer 2 (DPO), design of Layer 3 (EPG)
7. Limitations — honest, specific
8. The EPG direction — formal MDP framing, GRPO sketch, why it's the next paper

**Every number traces to a JSON file. Every result reproducible with one command.**

---

## Release Criteria — What "Undeniable" Looks Like

Do not release before every line is true:

```
False positive rate:
  ≥ 200 non-uncertain sentences through probe
  FPR < 5%

Compression faithfulness:
  n=200, bootstrap CI on probe block rate doesn't cross 90%
  LLMLingua FCR CI doesn't cross 50%

KV eviction finding:
  H2O null-answer: val_recall ≤ 2% at 70% retention, confirmed
  Streaming: FCR ≈ baseline ± 5%, confirmed

DPO Phase 3:
  FCR_DPO < FCR_baseline (any reduction counts as validation)
  4-way table: baseline / KV-best / DPO / probe

Cross-session memory:
  credence_memory+enforcement ≥ naive_summary, OR feature cut

One-command install:
  < 2 minutes, < 3 commands, clean machine verified

Model-agnostic:
  credence.wrap() tested with ≥ 2 different compression models
```

---

## The Standard Strategy: How Credence Reaches Every AI System

Not through product adoption alone. Through the metric.

**FCR as a compression faithfulness benchmark** — the goal is for FCR to become what BLEU is
for translation: the number every system reports, the number every paper compares against.
If that happens, Credence wins by being the reference implementation.

**Path to metric adoption:**
1. Publish technical report on arXiv with the compression_faithfulness study design open-sourced
2. The LLMLingua 74% FCR finding will get attention — it's a significant result against a popular system
3. Submit FCR benchmark to LangChain and LlamaIndex as a proposed compression evaluation
4. EML spec published as a standalone repo — model-agnostic format, any system can implement it
5. Reference implementation is Credence — whoever implements EML can use the probe

**Timeline:** metric adoption follows research credibility. The paper is the wedge.

---

## What We Are Highly Confident We Can Achieve

**The product (high confidence):**
- pip install credence-ai → works in 2 minutes
- credence.wrap(any_compress_fn) → prevents qualifier stripping, measures FCR
- Works with OpenAI, Anthropic, any open model via callback
- Credence Gate prevents tool use on unverified constraints
- 10 MCP tools covering the full epistemic lifecycle
- SQLite registry tracks confidence trajectories across sessions

**The research contribution (high confidence):**
- FCR as a metric — defined, measured, reproducible by anyone
- LLMLingua 74% FCR — the most impactful finding, specific and actionable
- H2O null-answer anomaly — new finding against existing benchmarks
- Faithfulness probe at n=200 — deterministic, publishable evidence

**The adaptive layer (medium confidence):**
- Thompson Sampling calibration — well-understood algorithm, reasonable gain expected
- DPO reduces baseline FCR — mechanism is sound, magnitude TBD after Phase 3

**The future research direction (low implementation confidence, high conceptual confidence):**
- GRPO applied to compression candidates — novel application, not yet validated
- Epistemic Policy Gradient as MDP formulation — new framing, worth a paper
- SE as compression scheduler — theoretically grounded replacement for J-score

---

## What We Are NOT Doing Until Gates Are Clear

Reject the following regardless of how good it sounds:

- Claiming model-agnostic before testing with a second compression model
- Shipping cross-session memory before resolving the naive_summary anomaly
- Publishing FPR claims before measuring FPR
- Building GRPO/EPG before DPO Phase 3 validates the learning layer
- Adding features to the MCP server that aren't in the 10-tool list
- Investor or press outreach before every release criterion is met
- Any benchmark comparison against commercial systems using our own evaluation only

---

## Product Vision Beyond the Paper

The research paper proves the concept. The product compounds it.

### Four Product Pillars (post-paper)

**1. Epistemic Routing (replaces binary block/proceed)**

The current architecture blocks or allows compression. The product version routes: instead
of "block compression when uncertain markers present," the system scores every sentence in
a segment and assigns a retention weight. Sentences containing registered uncertain values
receive 10× importance in the compression scheduler — effectively incompressible. Everything
else compresses at full LLMLingua aggressiveness. This gives the developer the compression
savings they want without the qualifier loss they don't want. The probe becomes a weight
function, not a boolean gate.

**2. Domain-Learned Uncertainty Detectors**

The current 108-marker list is generic English hedging vocabulary. Production systems have
domain-specific signals: financial ("subject to regulatory approval"), medical ("contraindicated
in patients with"), legal ("without prejudice"), infrastructure ("best-effort", "eventual
consistency"). The product path is: start Credence on a new codebase, run 50 sessions, identify
which markers co-occurred with actual FCR events in that specific domain, and output a
project-specific marker list that has lower FPR than the generic vocabulary. This is
Credence's Layer 1 adaptive system — Thompson Sampling on marker weights — applied to
produce per-project calibration. The output is an `epistemic_profile.json` that developers
can inspect, audit, and commit to source control alongside their code.

**3. Certainty Trajectory as Compliance Artifact**

For regulated industries, every constraint that moved from unverified to verified — or was
never verified before code shipped — is an audit trail. The `credence_trajectory` tool
already exposes this: when a constraint was registered, when it was verified, what j_score it
carried, whether it appeared in generated code before verification. The product version makes
this a first-class report: "At the time this PR was merged, 3 constraints were unverified.
Here are the lines they appeared in." This is not a hallucination detector — it is an
epistemic debt statement, the way a test coverage report is a code quality statement.

**4. Cross-Agent Epistemic Provenance**

Every agent in a multi-agent system that quotes a fact should carry forward its epistemic
status. The current PipelineMonitor does this at the Python level. The product version is a
lightweight wire format (EML — Epistemic Metadata Layer) that any framework can implement:
`{value, j_score, zone, verified, chain_depth}` attached to every structured output. A
CrewAI task, a LangGraph node, an AutoGen message — each one either preserves the EML
envelope or is flagged as an epistemic boundary where provenance was lost. Credence is the
reference implementation; EML is the protocol.

---

## The Practice for the Future

The larger contribution is a practice, not just a product:

**Epistemic state is infrastructure, not an afterthought.**

Every system that compresses context, summarizes conversations, or passes information between
agents currently treats uncertainty as formatting — qualifiers can be dropped, hedges can be
collapsed, confidence levels are not tracked. This is architecturally wrong in the same way
that HTTP without authentication was architecturally wrong.

The practice we're establishing:
1. Every compression pipeline reports FCR alongside ROUGE-L and semantic similarity
2. Every multi-agent handoff carries epistemic metadata (confidence, verified, source)
3. Every memory system stores confidence levels alongside facts
4. Uncertain claims are first-class objects — trackable, enforceable, expirable

Credence is the first implementation. FCR is the metric. EML is the wire format.
The Epistemic Policy Gradient is the training paradigm that makes systems self-correcting.

When these exist in combination — and the probe proves the concept is viable — the practice
becomes infrastructure. That is the path to reaching every AI system.

---

## Execution Timeline (No Anthropic API Key)

API-dependent gates (4, 5) are deferred. Everything else proceeds.

```
TODAY (30 min, no API)
├── Pull DPO results from Kaggle
│     kaggle kernels output chakradharvijayarao/credence-phase-3-dpo-epistemic-fine-tuning \
│       -p /tmp/dpo_out/
│     cp /tmp/dpo_out/epoch_results.json evals/dpo_epoch_results.json
│     git add evals/dpo_epoch_results.json && git commit
└── Verify all offline tests pass: python tests/tests.py

THIS WEEK — Day 1-2 (no API)
├── Build credence.wrap() — 100 lines, strip Haiku dep from _compress() ← GATE 1
│     Test with mock compress_fn: lambda ctx: ctx[:len(ctx)//2]
│     Test with local Phi-2 DPO adapter (epoch_2/) via transformers
└── Add unit tests for credence.wrap() to tests/tests.py

THIS WEEK — Day 3 (no API)
├── Cut codebase to validated 10-tool MCP set ← GATE 6
│     Delete untracked files that aren't part of validated feature set
│     Keep: credence/, evals/*.json, tests/, credence_gate/, docs/
└── git status should show clean working tree

THIS WEEK — Day 4 (no API)
├── One-command install test on clean Python env ← GATE 7
│     python -m venv /tmp/credence_test && source /tmp/credence_test/bin/activate
│     pip install -e ".[dev,mcp]"
│     python tests/tests.py  # must pass 178/178
└── Fix any install breakage

THIS WEEK — Day 5 (no API)
└── Final read of TECHNICAL_REPORT.md as reviewer ← GATE 8
      Every number traces to a JSON file ✓
      DPO section references evals/dpo_epoch_results.json ✓
      API-dependency stated honestly in Methods section ✓
      n=50 CI stated as-is, n=200 noted as future work ✓

NEXT WEEK
└── arXiv submission — upload TECHNICAL_REPORT.md as PDF

WHEN API KEY RESTORED (defer, no rush)
├── Gate 4: Cross-session memory ablation (CE + recalled constraints)
├── Gate 5: n=200 compression faithfulness (tighten CI)
└── Gate 3: Thompson Sampling bandit validation (needs live session data)

MONTH 2–3 (post-arXiv, no API needed for most)
├── credence.wrap() tested with second real model (OpenAI, local Llama) ← GATE 1 full
├── Framework integrations (LangChain, LlamaIndex — pure Python)
└── EML spec v0.1 as standalone repo

MONTH 3–6 (research track, needs API)
├── GRPO/EPG prototype
├── Cross-model FCR study → follow-on paper
└── n=200 compression faithfulness (if API restored)
```

---

**What the no-API constraint changes:**

The paper goes out with existing validated results — all JSON files are complete.
The product ships with the probe, registry, GTS, CE, Rust gate, and credence.wrap().
The adaptive layers (Gate 3) and larger-n validation (Gate 5) become post-arXiv work.

The probe and FCR metric do not require a Claude API key to use. A developer
with any LLM can use `credence.wrap(compress_fn=their_model)` and get the same
probe-based enforcement. This actually broadens the addressable market.

*The paper defines what is proven. The adaptive system compounds it over time.*

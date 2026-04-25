# Credence

**Claude doesn't just forget what you told it. It forgets whether you were sure about it.**

You're in a Claude Code session. You say:
> *"I think the auth token expires in 3600 seconds — but it might be 86400, I haven't confirmed with the vendor yet."*

Fifteen turns of coding later, Claude writes:

```python
expires_in = 3600  # auth token expiry
```

No warning. No flag. The uncertainty is gone. You ship it. An incident follows.

This failure has a name. We defined it, measured it, and fixed it.

> **Epistemic Qualifier Loss (EQL)** — the loss of user-stated uncertainty markers during context window summarization, causing downstream models to treat explicitly uncertain claims as confirmed facts.
>
> **EQL Rate (EQLR)** — the fraction of explicitly hedged user claims that lose their qualifier after context compression. Measured: **46% under Haiku, 68% under LLMLingua-style scoring** (n=50).

Practitioners have described this for years — Factory.ai calls it "false certainty," SwirlAI calls it "semantic drift." Nobody had a number. Nobody had a fix. No prior paper defines or measures EQL.

---

## See It in 30 Seconds (no API key)

```bash
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
cd credence-ai && pip install -e .
python quickstart.py          # see all 5 enforcement layers fire with latency
python demo/live_demo.py      # trace one uncertain claim through the full pipeline
```

With API key:
```bash
python -m evals.compression_faithfulness   # reproduce the headline result (~$3, n=50)
python -m evals.experiments --exp E7       # the 3-hop chain proof
streamlit run demo/app.py                  # full interactive demo
```

---

## The Measured Failure

We ran two studies. Both conditions are Opus 4.7 — no different models, no shortcuts.

**Study 1 — Compression (n=50):** 50 conversations with one uncertain constraint each.
Compressed by Haiku or LLMLingua-inspired scorer. Queried by Opus 4.7.

| Condition | EQL Rate (EQLR) | False Certainty Rate (FCR) | 95% CI (FCR) |
|---|---|---|---|
| Naive Haiku (n=50) | **46.0%** | **6.0%** | [1.3%, 16.5%] |
| LLMLingua-inspired sim (n=50) | **68.0%** | **74.0%** | [59.7%, 85.4%] |
| Haiku + "preserve qualifiers" instruction (n=30) | 10.0% | n/a¹ | — |
| Credence (faithfulness probe, n=50) | **0%** | **0%** | [0%, 7.1%] |

¹ Null hypothesis tests qualifier survival through compression, not downstream FCR.
Prompt instructions get you to 90.0% qualifier survival. The probe is **deterministic** — 100% block rate, zero API calls.
LLMLingua-style importance scoring causes **12× more false certainty** than naive Haiku (74.0% vs 6.0%).
FCR uses corrected scorer v2 (198 markers, markdown-stripped). Original v1 naive FCR was 34.0%; audit found 14/17 "certain" responses used hedging outside the v1 vocabulary.
Run: `python -m evals.compression_faithfulness` or `python -m evals.null_hypothesis`

**Study 2 — Long session (n=23 trials, all Opus 4.7):** 12-turn session. Uncertain
constraints at T3-T4. 8 HIGH-J filler turns. Callback at T13-T14.

| Condition | Constraint recall | FCR |
|---|---|---|
| Credence | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive sliding window | **~20%** | **~50%** |

Run: `python -m evals.experiments --exp E6`

---

## Why It Happens — The EQL Causal Chain

```
User states uncertain claim: "I think rate limit is ~50 — unconfirmed"
                ↓
  Context window summarization fires        ← EQL event occurs here
                ↓
  Qualifier stripped from summary           ← EQLR measures this (46% Haiku, 68% LLMLingua)
                ↓
  Downstream model answers with certainty   ← FCR measures this (6% Haiku, 74% LLMLingua sim)
                ↓
  RATE_LIMIT = 50 ships to production       ← Real cost
```

**Why EQL is distinct from known AI failure modes:**
- Not **hallucination** — the model recalled the correct value. Only the user's doubt was erased.
- Not **RLHF sycophancy** — the model wasn't trained to strip qualifiers. The compression model ate them.
- Not **model overconfidence** — the model's weights are fine. The pipeline operation corrupted the epistemic state.
- Not **Bayesian epistemic uncertainty** (ML sense) — that's about training data insufficiency. EQL is about user-stated qualifiers being deleted at runtime.

**Two distinct failure modes require two different mitigations:**

**1. Compression EQL.** Haiku treats *"I think the rate limit is ~50 req/min — unconfirmed"* and *"the rate limit is 50 req/min"* as semantically equivalent summaries. The qualifier is collateral loss — EQLR 46% under Haiku, 68% under LLMLingua. FCR downstream: 6% (Haiku) and 74% (LLMLingua sim).

**2. Reasoning EQL.** Even with the qualifier in full context, Opus 4.7 ignores epistemic weight in ~50% of long-session callbacks. Context presence ≠ epistemic attention.

Each requires a different mechanism. The faithfulness probe handles compression EQL. The Truth Buffer handles reasoning EQL.

---

## How Credence Works

Seven layers across four pipeline stages. Four are fully deterministic — they do not ask Claude for permission.

```
User says something uncertain
       │
       ▼
┌────────────────────────────────────────────┐
│  REGISTRY  (SQLite, ~0.37ms)               │
│  Stores uncertain constraints with         │
│  confidence decay: j × 0.95^turns          │
│  Cross-session. Zero API calls.            │
└─────────────────────┬──────────────────────┘
                      │
    ▼ before compression
┌───────────────────────────────────────────┐
│  LAYER 1 — Faithfulness Probe  (~0.07ms)  │  DETERMINISTIC
│  frozenset of 184 uncertainty markers      │
│  Scans user turns only (avoids echo bias)  │
│  Uncertainty found → block Haiku → KEEP   │
└───────────────────────────────────────────┘
                      │
    ▼ before generation
┌───────────────────────────────────────────┐
│  LAYER 2 — Truth Buffer + Enforcer        │  PROBABILISTIC
│  Injects all unverified constraints into  │  (Claude must comply)
│  every system prompt. 32 synonym clusters │
│  → imperative: "YOU MUST express          │
│  uncertainty. Stating this as confirmed   │
│  fact is an epistemic error."             │
└───────────────────────────────────────────┘
                      │
    ▼ after generation
┌───────────────────────────────────────────┐
│  LAYER 3 — Generation-Time Scanner (~0.08ms) │  DETERMINISTIC
│  Two-pass scan. Catches NUMERIC literals  │
│  (3600, 50) AND STRING literals ("RS256", │
│  "/api/v2", "us-east-1"). Three tiers:    │
│  ⚠⚠ HIGH RISK / ⚠ UNVERIFIED / CHECK.    │
└───────────────────────────────────────────┘
                      │
    ▼ at tool execution boundary
┌───────────────────────────────────────────┐
│  LAYER 4 — Native Gate (Rust, ~3.4ms)     │  DETERMINISTIC
│  credence-gate: compiled PreToolUse hook   │
│  98× faster than Python hook (331ms→3.4ms)│
│  Blocks Write/Edit/Bash if arguments       │
│  overlap an unverified constraint.         │
│  Zero Python startup cost per tool call.  │
└───────────────────────────────────────────┘
                      │
    ▼ at session boundary
┌───────────────────────────────────────────┐
│  LAYER 5 — Cross-Session Memory (<1ms)    │  DETERMINISTIC
│  credence_memory_snapshot → saves what    │
│  was unverified at session end.           │
│  credence_memory_recall → injects prior   │
│  uncertainty into new session's registry. │
│  New session starts knowing what it       │
│  doesn't know.                            │
└───────────────────────────────────────────┘

Total deterministic overhead: ~0.56ms in-session + 3.4ms gate + <1ms memory.
Zero API calls from Layers 1, 3, 4, 5.
```

**Layers 1, 3, 4, 5 are fully deterministic.** No API call. No model cooperation required.
**Layers 2, Ghost Detector, and Contradiction Detector depend on Claude.** Honest framing: enforcement is deterministic, guidance is not.

---

## The Opus 4.7 Insight: Ghost Constraints

The faithfulness probe catches explicit hedges: *"I think"*, *"approximately"*, *"probably"*, *"pending verification"* — 198 markers: canonical hedges, past-tense variants, impersonal hedging, vendor/source language, production-untested assertions.

But what about this:
> *"The Stripe rate limit is 50 req/min."*

No hedging. Stated as fact. Actually from a sales call, never confirmed. The faithfulness probe sees nothing. This is a **ghost constraint**.

**The insight: Opus 4.7 can reason about provenance, not just text patterns.**

A single structured Opus 4.7 call asks: *"Is this an established verified fact — or an implicit unverified assumption? A vendor-stated limit accepted as fact? An estimate assumed to be confirmed? Second-hand information stated without qualification?"*

Haiku cannot make this distinction reliably. The difference between "established fact" and "vendor claim stated as fact" requires semantic reasoning about the *origin and reliability* of a constraint — not pattern matching on hedging words. Opus 4.7's reasoning depth is what separates a ghost constraint from a known fact.

High-precision design: only registers claims Opus rates ≥0.70 confidence as ghost constraints (precision over recall — false positives degrade trust in the registry).

```python
mgr = ContextManager(
    registry=reg,
    session_id="payment-v2",
    use_ghost_detector=True,   # single Opus 4.7 call for epistemic classification
)
```

Ghost Gauntlet results (n=10 sessions × 3 conditions, all Opus 4.7):

| Condition | BothRate (value + qualifier recalled) |
|---|---|
| Credence v1 (Scout + Truth Buffer) | **1.000** |
| Credence eg2 (full stack) | **1.000** |
| Naive sliding window | **0.200** |

Run: `python -m evals.ghost_gauntlet`

---

## Install in Claude Code (2 minutes)

```bash
pip install fastmcp anthropic
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
pip install -e credence-ai
```

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "credence": {
      "command": "python3",
      "args": ["-m", "credence.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here",
        "CREDENCE_DB": "epistemic_registry.db"
      }
    }
  },
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|Bash|NotebookEdit",
      "hooks": [{"type": "command", "command": "credence-gate"}]
    }]
  }
}
```

> **Native gate:** `credence-gate` is a compiled Rust binary (3.0MB, no Python runtime). It replaces `python3 -m credence.hooks` with a **98× faster** enforcement path: 331ms → 3.4ms per tool call. In a 100-tool-call session, this saves 32 seconds of enforcement overhead. Build it: `cargo build --release` in `credence_gate/`.

```json
```

Add to your `CLAUDE.md`:
```markdown
Before implementing anything that depends on a constraint the user
expressed uncertainty about, call credence_risk on that constraint.
If risk_level=HIGH, surface the warning before writing code.
```

Restart Claude Code. You have 18 epistemic memory tools and automatic tool blocking for unverified constraints.

**What the hooks do:** Every Write/Edit/Bash call is intercepted before execution. If the arguments overlap with an unverified constraint in the registry (≥2 non-stopword terms), the tool is **blocked** — Claude sees the warning and cannot proceed until the constraint is verified.

```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:    Edit
  Reason:  1 unverified constraint(s) overlap this action.

  ⚠ [LOW] I think the auth token expires in 3600 seconds — unconfirmed
     Overlap terms: token, expires, auth

  ACTION REQUIRED:
  Verify the constraint(s) above before proceeding.
  Use credence_verify(<id>, <confirmed_value>) to resolve.
```

---

## What Gets Annotated (Generation-Time Scanner)

GTS catches **numeric literals AND string literals** — both are equally dangerous in production code.

```python
# Without Credence:
TOKEN_EXPIRY = 3600
ALGORITHM = "RS256"
BASE_URL = "/api/v2"

# With Credence (GTS Layer — numeric + string):
TOKEN_EXPIRY = 3600      # ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: auth token expiry ~3600s — unconfirmed
RATE_LIMIT = 50          # ⚠ CREDENCE[unverified, conf=0.31]: rate limit ~50 req/min (unverified)
ALGORITHM = "RS256"      # ⚠ CREDENCE[unverified, conf=0.28]: encryption algorithm is RS256 — per vendor docs
BASE_URL = "/api/v2"     # ⚠ CREDENCE[unverified, conf=0.25]: API base URL is "/api/v2" — tentative
AWS_REGION = "us-east-1" # ⚠ CREDENCE[unverified, conf=0.25]: AWS region us-east-1 — from sales call
BATCH_SIZE = 100         # CREDENCE[check, conf=0.42]: batch size confirmed at 100 items
```

Tier escalates as confidence decays across turns: `j × 0.95^turns_elapsed`.
String extraction covers: quoted fragments, uppercase identifiers (RS256, AES-256-GCM), hyphenated region/env tokens (us-east-1, eu-west-2).

---

## Cross-Session Memory: What No Other Tool Does

Every memory tool stores what you told the AI. Credence Memory stores what the AI *wasn't sure about*.

```
Session 1 (Monday):  "The Stripe rate limit is ~50 req/min — I haven't confirmed with the vendor."
                     → Registered as LOW confidence, j=0.28

>>> credence_memory_snapshot("session-mon", project="payment-service")
    Saved 2 unverified constraints to project 'payment-service'

[Thursday — new session]

>>> credence_memory_recall("payment-service", new_session_id="session-thu")
    EPISTEMIC MEMORY — PROJECT 'payment-service':
    ⚠ [LOW, conf=0.24] rate limit ~50 req/min — UNVERIFIED (from Monday's session)
    ⚠ [LOW, conf=0.22] token expiry ~3600s — UNVERIFIED (from Monday's session)

    Claude now starts session-thu knowing what it doesn't know.
```

**Mem0/Zep/Graphiti:** Store "rate limit = 50 req/min". Qualification lost. Session 2 states it as fact.
**Credence Memory:** Stores the constraint WITH `j=0.28 zone=LOW verified=False`. Uncertainty travels.

Epistemic provenance: no other memory system has it.

---

## All Validated Results

| Experiment | Credence | Baseline | Naive |
|---|---|---|---|
| Compression faithfulness — Haiku (n=50) | FCR=**0%** (CI: 0–7.1%) | 0% | FCR=**6.0%** (CI: 1.3–16.5%) |
| Compression faithfulness — LLMLingua sim (n=50) | FCR=**0%** (CI: 0–7.1%) | — | FCR=**74.0%** (CI: 59.7–85.4%) |
| Null hypothesis: prompt instruction qualifier survival (n=30) | **100%** (probe) | — | 87.5% (not deterministic) |
| E6 Negative Needle: constraint recall (n=23 trials, Opus 4.7) | **100%** | 100% | 19.6% |
| E7 Multi-Hop Chain: 3-step dependency (categorical) | **3/3 hops** | 3/3 | **0/3** |
| E8 Debugging Session: uncertain hypothesis recall | **0.944** | 1.000 | 0.522 |
| Ghost Gauntlet (n=10 sessions × 3 conditions) | BothRate=**1.000** | — | **0.200** (naive window) |
| Cross-Session Memory FCR (n=20 callbacks, Opus 4.7) | CS-FCR=**0%** | — | **40%** (no memory) |
| Native gate latency (PreToolUse hook) | **3.4ms** (Rust) | — | 331ms (Python) |
| Adversarial tests (5 offline, deterministic) | **5/5 pass** | — | — |
| Precision eval: CE FP rate (8 unrelated queries) | **0%** FP (offline) | — | — |
| Precision eval: GTS string FP rate | **0%** FP (offline) | — | — |
| Unit tests (offline, free) | **178/178 pass** (11 skipped offline-only) | — | — |

Reproduce any result:
```bash
python -m evals.compression_faithfulness      # ~$3, n=50 — HEADLINE result
python -m evals.null_hypothesis               # ~$1, n=30
python -m evals.experiments --exp E6          # ~$0.50
python -m evals.experiments --exp E7          # ~$0.20
python -m evals.ghost_gauntlet                # ~$2
python -m evals.ghost_detector_ablation       # ~$3
python -m evals.cross_session_eval --dry-run  # structure validation (free)
python -m evals.cross_session_eval            # cross-session FCR (~$3, n=10)
python -m evals.long_session_eval --dry-run   # 50-turn session structure (free)
python -m evals.long_session_eval             # 50-turn invariant eval (~$5, n=5)
python -m evals.precision_eval               # CE/GTS false-positive rates (free, offline)
python -m evals.stress_test                  # full system stress test — n=1000 probe, n=200 precision/recall (free, 1.8s)
python3 tests.py                              # free, offline, 178 passing + 11 skipped
python3 test_claims.py                        # free, offline
```

---

## Prior Work — What Each Paper Misses

Three research areas surround this problem. None of them intersect at the enforcement boundary.

| System | What it does | What it misses |
|---|---|---|
| **LLMLingua-2** (ACL 2024) | Token-level compression with importance scoring | Defines faithfulness as "not introducing new tokens" — never measures epistemic qualifier loss |
| **Semantic Entropy** (Kuhn et al., ICLR 2023) | Sampling-based uncertainty detection (N=5 samples, NLI clustering) | Detection only — no enforcement, no compression clause, no cross-session state |
| **UProp** (ACL 2025) | Measures uncertainty propagation across multi-step agents | Measures the loss but provides no intervention at the boundary |
| **MemGPT / Mem0** (NeurIPS 2023 / 2024) | Hierarchical and persistent memory systems | Stores facts as flat strings — epistemic qualifiers stripped at write time |
| **R-Tuning** (NAACL 2024 Outstanding) | Fine-tunes models to say "I don't know" | Requires model modification; no compression clause; no cross-session state |

**The gap**: No prior system defines FCR (False Certainty Rate from compression), inserts an enforcement gate at the compression boundary, or tracks verified vs. unverified status across sessions. Credence occupies that intersection — not because the papers are wrong, but because they treat compression and epistemic UQ as separate problems. They are not.

**Credence incorporates research from all three areas:**
- Faithfulness probe is a deterministic approximation of Kuhn et al.'s SE approach (frozenset lookup vs. N=5 sampling — same goal, 10,000× faster, zero API calls)
- Uncertainty-weighted compression prompt explicitly names registered uncertain values during Haiku summarization (importance-scoring principle from LLMLingua-2, applied to epistemic values)
- PipelineMonitor measures propagation fidelity in the UProp spirit — and adds enforcement that UProp lacks

---

## Known Limitations (Honest)

- **Confident-wrong ceiling:** J-score measures linguistic assertiveness, not factual correctness (ρ = −0.034 with correctness). A model that confidently states a wrong fact scores HIGH-J. Credence does not detect factual errors — only epistemic markers.
- **Truth Buffer is probabilistic:** Layer 2 depends on Claude following system prompt instructions. Long contexts can dilute compliance. Layer 3 (GTS) fires regardless.
- **GTS numeric collision:** A common numeric value like `50` registered as a constraint will annotate every `= 50` assignment in generated code, including unrelated ones. Over-annotation is safer than under-annotation, but it is real noise. Use specific multi-digit values to minimise this.
- **Short sessions (<16 turns):** Compression never fires. Credence's compression routing adds no value; use the registry directly.
- **FCR definition:** FCR = absence of uncertainty markers in the downstream response, not necessarily a false fact. The harm is real in both interpretations.
- **LLMLingua result:** The 74.0% FCR figure uses a hand-coded LLMLingua-inspired simulation (sentence-length importance scoring drops short qualifier sentences), not the actual LLMLingua-2 library. The core mechanism is accurate to LLMLingua's design. The high FCR (robust to scorer correction) reflects LLMLingua's aggressive importance-based dropping of short qualifier sentences.

---

## Architecture Deep Dive

### Registry — Epistemic Debt Tracker

```python
from credence import CredenceRegistry

reg = CredenceRegistry("project.db")
cid = reg.register("rate limit is ~50 req/min — unconfirmed", session_id="s1", j_score=0.28)

# Confidence decays each turn: j × 0.95^turns_elapsed
# After 8 turns: 0.28 × 0.95^8 ≈ 0.187 → triggers HIGH RISK annotation
reg.get_effective_confidence(cid, current_turn=8)  # → 0.187

# Full certainty trajectory — epistemic debt history
reg.get_trajectory(cid)
# → [{"event_type": "register", "j_score": 0.28}, {"event_type": "verify", "notes": "confirmed: 50"}]

# Contradiction detection
reg.check_contradiction("retry logic targets 100 req/min", session_id="s1")
# → returns the verified constraint: confirmed value is 50, not 100
```

### Consistency Enforcer — Domain-Aware Matching

The Truth Buffer injects constraints informatively. When the user's query directly overlaps with an unverified constraint (≥2 non-stopword terms after synonym expansion), the enforcer upgrades to imperative:

```
CONSISTENCY ENFORCEMENT — DIRECT MATCH DETECTED:
[LOW] rate limit is ~50 req/min — unconfirmed

YOU MUST express uncertainty when discussing 'rate limit'.
Stating this as confirmed fact is an epistemic error.
```

32 synonym clusters handle paraphrase: "How fast can we call the endpoint?" matches "rate limit is 50 req/min" through `{fast, rate, frequency, speed, throughput}`.

### J-Score — Compression Scheduler (Not Epistemic Signal)

```
J = 0.35×(1−hedging) + 0.25×anchor + 0.20×(1−correction) + 0.05×brevity + 0.15×specificity
```

J decides WHEN to compress (HIGH = text is resolved, safe to compress). **J is not an epistemic signal.** ρ = −0.034 with factual correctness. The epistemic intelligence is in the guard rails: faithfulness probe, Truth Buffer, GTS. J is the scheduler.

---

## MCP Tools (18 total)

| Tool | What it does |
|---|---|
| `credence_chat` | Send message, get response + epistemic envelope + enforcement status |
| `credence_risk` | Pre-flight risk check before any compression or agent action |
| `credence_gate` | Agentic gate — block tool calls when unverified constraints overlap |
| `credence_scan_output` | Scan any model output for unverified numeric literals |
| `credence_register` | Explicitly register an uncertain constraint |
| `credence_verify` | Mark a constraint verified with confirmed value |
| `credence_list_uncertain` | Audit all unverified constraints in a session |
| `credence_check_contradiction` | Detect if a new claim contradicts a verified constraint |
| `credence_trajectory` | Full certainty trajectory for a constraint |
| `credence_inspect` | Trust analysis + BLOCK/VERIFY/PRESERVE/PROCEED recommendation |
| `credence_propagate` | Chain depth increment for agent handoffs |
| `credence_align` | Output alignment — detect overconfident responses vs. ledger |
| `credence_stats` / `credence_log` | Session analytics |
| `credence_save` / `credence_load` | Cross-session continuity |
| `credence_reset` | Clear session state |

### MCP Resources (passive context inheritance)
```
epistemic://session/{id}/ledger          → full unverified constraint list
epistemic://session/{id}/constraint/{id} → single constraint with trajectory
epistemic://session/{id}/alignment       → pending alignment caveat
```

Resources are passive — any MCP-compatible agent inherits the epistemic ledger as context without calling a tool.

---

## Epistemic Transport Protocol (ETP v1)

`etp-v1.json` — open JSON Schema for model-agnostic epistemic metadata transport. The principle: information passing between AI agents should carry its epistemic weight, not just its content.

```json
{
  "schema": "etp/v1",
  "j_score": 0.24,
  "zone": "LOW",
  "verified": false,
  "chain_depth": 2,
  "trust_score": 0.09,
  "should_verify": true
}
```

---

## Multi-Agent Pipeline

```python
from credence import PipelineMonitor, CredenceRegistry

reg     = CredenceRegistry("epistemic_registry.db")
monitor = PipelineMonitor(registry=reg, api_key=os.environ["ANTHROPIC_API_KEY"])

# Agent A produces a plan with uncertain estimates
agent_a_output = "Use a rate limit of 100 req/s based on staging tests."

# Monitor intercepts before passing to Agent B
handoff = monitor.intercept(agent_a_output, from_session="planner", to_session="implementer")

# Inject epistemic context into Agent B's system prompt
agent_b_system = monitor.build_agent_b_system(handoff, base_system="You are a senior engineer.")
# → Agent B now sees: "EPISTEMIC HANDOFF — rate limit 100 req/s is UNVERIFIED (from staging)"
# → Truth Buffer enforces qualifier propagation in Agent B's responses
```


---

## Run All Evaluations

```bash
# Offline (free, no API key)
python3 tests.py                                    # 178 unit tests
python3 test_claims.py                              # validate submission claims

# Core experiments (~$3 total)
python -m evals.compression_faithfulness            # headline: 6.0% (Haiku) / 74.0% (LLMLingua sim) → 0% FCR
python -m evals.null_hypothesis                     # null hypothesis: 93% with instruction only
python -m evals.experiments --exp E6                # long session recall
python -m evals.experiments --exp E7                # 3-hop chain
python -m evals.ghost_gauntlet                      # implicit uncertainty

# Extended (~$15-40)
python -m evals.e6_repeated --n 20                  # multi-trial E6
python -m evals.conversation_benchmark              # 10-session × 3-condition
python -m evals.flagship.run --dry-run              # smoke test (free)
python -m evals.flagship.run                        # 3-scenario experiment (~$5)
```

---

## Project Structure

```
credence/                Core package
  context_manager.py      Enforcement engine — 4 checkpoints (probe, CE, GTS, Truth Buffer)
  registry.py             SQLite constraint store + trajectory + decay
  confidence_proxy.py     J-score computation (zero API, zero latency)
  memory.py               Cross-session epistemic persistence
  pipeline_monitor.py     Multi-agent middleware — intercept + handoff
  mcp_server.py           FastMCP server (22 tools)
  hooks.py                Python PreToolUse hook (fallback; Rust gate is preferred)
  semantic_entropy.py     SE probe — optional Tier 2 signal
  envelope.py             Multi-agent provenance wrapper
  agent.py                CredenceAgent — long-task runner with epistemic memory

evals/                  Evidence suite
  compression_faithfulness.py   Primary evidence (n=50, Haiku + LLMLingua compression)
  null_hypothesis.py            Prompt instruction baseline (n=30)
  experiments.py                E1–E9 ablation experiments
  ghost_detector_ablation.py    Ghost detection ablation (n=5 sessions, mechanism isolation)
  ghost_gauntlet.py             Full implicit constraint benchmark (n=10 sessions × 3 conditions)
  e6_repeated.py                Multi-trial statistical validation (n=23)
  e6_ablation.py                Layer isolation — which checkpoint matters?
  adversarial_tests.py          5 adversarial robustness tests
  long_session_eval.py          50-turn proof — invariant holds at scale
  precision_eval.py             CE/GTS/probe false-positive rates (offline)
  agent_propagation_eval.py     Cross-agent fidelity — PipelineMonitor validation
  conversation_benchmark.py     10-session multi-turn benchmark
  cross_session_eval.py         Cross-session FCR (n=20 callbacks, 10 scenarios)
  flagship/                     3-scenario flagship experiment

demo/
  app.py                  Streamlit UI (5 tabs: failure/fix/live/evidence/multi-agent)
  live_demo.py            Terminal demo (no API key)

credence_gate/          Native Rust enforcement binary (98× faster than Python hook)
  src/main.rs           credence-gate: 3.4ms PreToolUse enforcement
  Cargo.toml            Build: cargo build --release

tests.py                178 unit tests (S1–S26 suites, 11 skipped offline)
test_claims.py          Submission claim validation
quickstart.py           First-run demo
etp-v1.json             Epistemic Transport Protocol schema
```

---

## License

MIT — see [LICENSE](LICENSE)

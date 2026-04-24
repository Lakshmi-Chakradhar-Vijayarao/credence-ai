# Credence

**Claude doesn't just forget what you told it. It forgets whether you were sure about it.**

You're in a Claude Code session. You say:
> *"I think the auth token expires in 3600 seconds — but it might be 86400, I haven't confirmed with the vendor yet."*

Fifteen turns of coding later, Claude writes:

```python
expires_in = 3600  # auth token expiry
```

No warning. No flag. The uncertainty is gone. You ship it. An incident follows.

We measured this. Then we fixed it.

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
python -m evals.compression_faithfulness   # reproduce the headline result (~$2, n=50)
python -m evals.experiments --exp E7       # the 3-hop chain proof
streamlit run demo/app.py                  # full interactive demo
```

---

## The Measured Failure

We ran two studies. Both conditions are Opus 4.7 — no different models, no shortcuts.

**Study 1 — Compression (n=50):** 50 conversations with one uncertain constraint each.
Compressed by Haiku or LLMLingua. Queried by Opus.

| Condition | Qualifier survival | False certainty downstream |
|---|---|---|
| Naive Haiku (n=50) | 52.0% | **26.0%** |
| LLMLingua-2 simulated (n=50) | 32.0% | **70.0%** |
| Haiku + "preserve qualifiers" instruction (n=30) | **93.3%** | n/a¹ |
| Credence (faithfulness probe, n=50) | **100%** (blocks) | **0%** |

¹ Null hypothesis tests qualifier survival through compression, not downstream FCR.
Prompt instructions get you to 93.3% qualifier survival. The probe is **deterministic** — 100% block rate, zero API calls.
LLMLingua-style token-importance compression causes 2.7× more false certainty than naive Haiku.
Run: `python -m evals.compression_faithfulness --n 50` or `python -m evals.null_hypothesis`

**Study 2 — Long session (n=23 trials, all Opus 4.7):** 12-turn session. Uncertain
constraints at T3-T4. 8 HIGH-J filler turns. Callback at T13-T14.

| Condition | Constraint recall | FCR |
|---|---|---|
| Credence | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive sliding window | **~20%** | **~80%** |

Run: `python -m evals.experiments --exp E6`

---

## Why It Happens

Two distinct failure modes require two different mitigations:

**1. Compression loss.** Haiku doesn't understand epistemic metadata. "I think the rate limit is ~50 req/min — unconfirmed" and "the rate limit is 50 req/min" are semantically equivalent summaries to a compression model. The qualifier is collateral loss in 48% of Haiku compressions and 68% of LLMLingua compressions (measured, n=50).

**2. Reasoning loss.** Even with the qualifier present in context, Opus 4.7 states uncertain constraints as confirmed facts in ~50% of long-session callbacks. The text was there. Epistemic attention was not. Context presence ≠ epistemic attention.

These require different mechanisms. Keyword detection handles the first. Proactive injection handles the second.

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
│  frozenset of 113 uncertainty markers      │
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
│  Two-pass regex scan. Annotates code and  │
│  prose with ⚠⚠ HIGH RISK / ⚠ UNVERIFIED  │
│  based on decayed confidence. Zero API.   │
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

The faithfulness probe catches explicit hedges: *"I think"*, *"approximately"*, *"probably"* — 113 markers, all canonical.

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

Ghost Gauntlet results (n=10 sessions × 3 claims, all Opus 4.7):

| Condition | BothRate (value + qualifier recalled) |
|---|---|
| Credence (ghost detector active) | **1.000** |
| Naive window | **0.133** |

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

When the model writes code that embeds an unverified value:

```python
# Without Credence:
TOKEN_EXPIRY = 3600

# With Credence (GTS Layer):
TOKEN_EXPIRY = 3600  # ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: I think the auth token expires in 3600s — unconfirmed
RATE_LIMIT = 50      # ⚠ CREDENCE[unverified, conf=0.31]: rate limit is ~50 req/min (unverified)
BATCH_SIZE = 100     # CREDENCE[check, conf=0.42]: batch size confirmed at 100 items
```

Tier escalates as confidence decays across turns: `j × 0.95^turns_elapsed`.

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
| Compression faithfulness — Haiku (n=50) | FCR=**0%** | 0% | FCR=**26%** |
| Compression faithfulness — LLMLingua (n=50) | FCR=**0%** | 0% | FCR=**70%** |
| Null hypothesis: prompt instruction qualifier survival (n=30) | **100%** (probe) | — | 93.3% (not deterministic) |
| E6 Negative Needle: constraint recall (n=23 trials, Opus 4.7) | **100%** | 100% | 19.6% |
| E7 Multi-Hop Chain: 3-step dependency (categorical) | **3/3 hops** | 3/3 | **0/3** |
| E8 Debugging Session: uncertain hypothesis recall | **0.944** | 1.000 | 0.522 |
| Ghost Detector Ablation (n=5 pure ghost sessions) | BothRate=**1.000** | — | **0.400** (no detection) |
| Ghost Gauntlet: implicit uncertain claims (n=10 sessions) | BothRate=**1.000** | — | 0.133 |
| Cross-Session Memory FCR (n=20 callbacks, Opus 4.7) | CS-FCR=**0%** | — | **40%** (no memory) |
| Native gate latency (PreToolUse hook) | **3.4ms** (Rust) | — | 331ms (Python) |
| Adversarial tests (5 offline, deterministic) | **5/5 pass** | — | — |
| Unit tests (offline, free) | **132/132 pass** | — | — |

Reproduce any result:
```bash
python -m evals.compression_faithfulness      # ~$2, n=50
python -m evals.null_hypothesis               # ~$1, n=30
python -m evals.experiments --exp E6          # ~$0.50
python -m evals.experiments --exp E7          # ~$0.20
python -m evals.ghost_gauntlet                # ~$2
python -m evals.ghost_detector_ablation       # ~$3
python -m evals.cross_session_eval --dry-run  # structure validation (free)
python -m evals.cross_session_eval            # cross-session FCR (~$3, n=10)
python3 tests.py                              # free, offline, 132 tests
python3 test_claims.py                        # free, offline
```

---

## Known Limitations (Honest)

- **Confident-wrong ceiling:** J-score measures linguistic assertiveness, not factual correctness (ρ = −0.034 with correctness). A model that confidently states a wrong fact scores HIGH-J. The SE probe (behavioral entropy) catches this for implicit uncertainty, not for factual errors.
- **Truth Buffer is probabilistic:** Layer 2 depends on Claude following system prompt instructions. Long contexts can dilute compliance. Layer 3 (GTS) fires regardless.
- **Short sessions (<16 turns):** Compression never fires. Credence's compression routing adds no value; use the registry directly.
- **FCR definition:** FCR = absence of uncertainty markers in the downstream response, not necessarily a false fact. The harm is real in both interpretations.

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

## Model Adapters (OpenAI, Gemini, Ollama)

```python
from credence.adapters import OpenAIAdapter, GeminiAdapter, OllamaAdapter

mgr = OpenAIAdapter(api_key="sk-...", model="gpt-4o")    # GPT-4o + GPT-3.5 compression
mgr = GeminiAdapter(api_key="AI...", model="gemini-1.5-pro")
mgr = OllamaAdapter(model="llama3.2")                    # zero API cost, local

result = mgr.chat("I think the rate limit is 100 req/min — unconfirmed")
print(result.j_score, result.zone, result.decision)
```

All adapters share identical faithfulness probe, Truth Buffer, GTS, and J-routing. Adding a new provider is a 40-line `BaseAdapter` subclass.

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

## REST API + TypeScript SDK

```bash
uvicorn api.main:app --port 8000       # all 18 MCP tools as HTTP endpoints
npm install credence-ai                # TypeScript SDK
```

---

## Run All Evaluations

```bash
# Offline (free, no API key)
python3 tests.py                                    # 116 unit tests
python3 test_claims.py                              # validate submission claims

# Core experiments (~$3 total)
python -m evals.compression_faithfulness            # headline: 26% (Haiku) / 70% (LLMLingua) → 0% FCR
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
  context_manager.py      Memory governor + all 5 layers
  registry.py             SQLite constraint store + trajectory
  confidence_proxy.py     J-score computation (zero API)
  mcp_server.py           FastMCP server (18 tools)
  hooks.py                Claude Code PreToolUse enforcement hook
  semantic_entropy.py     SE probe — behavioral entropy (Haiku, N=3, compression routing)
  behavioral_signal.py    Tier 2 behavioral consistency
  envelope.py             Multi-agent provenance wrapper
  adapters/               OpenAI, Gemini, Ollama adapters

evals/                  All experiments
  compression_faithfulness.py   Primary evidence (n=50)
  null_hypothesis.py            Prompt instruction baseline
  experiments.py                E1–E9 ablation suite
  ghost_gauntlet.py             Implicit constraint benchmark
  e6_repeated.py                Multi-trial statistical validation
  adversarial_tests.py          5 adversarial tests

demo/
  app.py                  Streamlit (4 tabs)
  live_demo.py            No-API-key pipeline trace

api/main.py             REST backend
packages/credence-ts/   TypeScript SDK
hooks_demo/             Claude Code hooks integration example

credence_gate/          Native Rust enforcement binary
  src/main.rs           credence-gate: 98× faster PreToolUse hook (331ms→3.4ms)
  Cargo.toml            Build: cargo build --release

tests.py                132 unit tests (S1–S24 suites)
test_claims.py          Submission claim validation
quickstart.py           First-run demo
etp-v1.json             Epistemic Transport Protocol schema
```

---

## License

MIT — see [LICENSE](LICENSE)

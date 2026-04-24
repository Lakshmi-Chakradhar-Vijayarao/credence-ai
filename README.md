# Credence

**Claude doesn't just forget what you told it. It forgets whether you were sure about it.**

We call this *certainty inflation* — uncertain information exits a conversation, or crosses a compression boundary, stated as confirmed fact. We measured it in two scenarios:

| Scenario | False Certainty Rate |
|---|---|
| Haiku compresses a context with uncertain constraint (n=30) | **36.7%** |
| Naive sliding window, long session (E6, n=23) | **~80%** (constraint dropped → stated as fact) |
| Credence (any condition) | **0–4%** |

Compression makes it worse: Haiku strips uncertainty qualifiers in **60%** of
summarizations. Naive sliding windows drop constraints entirely in long sessions.
Both paths lead to the same outcome — uncertain information exits the pipeline
stated as confirmed fact.

Credence is a lightweight enforcement layer that prevents uncertain information from
being stated as fact. Three deterministic mechanisms address it at different points in
the pipeline: before compression, before generation, and after generation.

---

## Two Failure Modes, Measured

### Failure Mode 1: The Window Failure

Long sessions compress or slide context windows. Uncertain constraints registered early
get dropped. E6 (12-turn session, n=23 trials, all Opus 4.7):

| Condition | Correction recall | FCR |
|---|---|---|
| Credence | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive window | **~20%** | **~80%** |

Baseline with full context also achieves 100% recall — it is the oracle. But full context
is unavailable in sessions longer than ~16 turns where compression fires. Credence matches
baseline recall without the token cost, by registering constraints in a persistent registry
and injecting them into every system prompt explicitly.

### Failure Mode 2: The Compression Failure

We ran 30 realistic technical conversations through Haiku compression. Each planted an
uncertain constraint, compressed via Haiku, then asked a downstream Opus model.

| | Naive Haiku compression | Probe-guarded | Baseline |
|---|---|---|---|
| Qualifier survival | **40.0%** | 100% (blocked) | 100% |
| False certainty downstream | **36.7%** | **0%** | 0% |

Haiku strips uncertainty qualifiers in **60%** of compressions — not through negligence,
but because epistemic metadata is not content. The faithfulness probe scans user turns
for uncertainty markers (108 terms, user-stated hedges only) and eliminates false
certainty entirely — 100% block rate across all 30 study scenarios.

**Null hypothesis tested**: Does simply adding "preserve uncertainty qualifiers" to the
Haiku prompt achieve the same result without any middleware?
Run `python -m evals.haiku_baseline` to find out.

---

## Install in Claude Code (2 minutes)

```bash
pip install fastmcp anthropic
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
pip install -e credence-ai
```

Add to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "epistemic-memory": {
      "command": "python",
      "args": ["-m", "credence.mcp_server"],
      "env": { "ANTHROPIC_API_KEY": "your-key-here" }
    }
  }
}
```

Add to your project's `CLAUDE.md`:

```markdown
Before implementing anything that depends on a constraint the user
expressed uncertainty about, call credence_risk on that constraint.
If risk_level=HIGH or should_verify=True, surface the warning before writing code.
```

Restart Claude Code. You now have 18 epistemic memory tools available.

---

## Architecture — 4 Layers

```
User input
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  STORAGE — Epistemic Registry (SQLite, zero dependencies)   │
│  Stores: constraint content, j_score, zone, source,         │
│  TTL expiry, and confidence decay (j × 0.95^turns_elapsed)  │
└──────────────────────────┬─────────────────────────────────┘
                           │
    ▼ before compression   │
┌───────────────────────────────────────┐
│  LAYER 1 — Faithfulness Probe          │  DETERMINISTIC
│  frozenset of 108 uncertainty markers  │  no API call
│  If found → block Haiku → PRESERVE     │  ~0.07 ms
└───────────────────────────────────────┘
                           │
    ▼ before generation    │
┌───────────────────────────────────────┐
│  LAYER 2 — Truth Buffer +              │  PROBABILISTIC
│  Consistency Enforcer                  │  (Claude must comply)
│  Injects unverified constraints into   │  ~0.03 ms (check)
│  every system prompt. Upgrades from    │  Haiku call only
│  informational → imperative when the   │  when query matches
│  user query directly overlaps (32      │
│  synonym clusters for paraphrase).     │
└───────────────────────────────────────┘
                           │
    ▼ after generation     │
┌───────────────────────────────────────┐
│  LAYER 3 — Generation-Time Scanner     │  DETERMINISTIC
│  Two-pass regex scan of model output.  │  no API call
│  Annotates numeric literals in code    │  ~0.08 ms
│  and prose that match registered        │
│  uncertain values. Tier: HIGH RISK /   │
│  UNVERIFIED / CHECK based on decay.    │
└───────────────────────────────────────┘

Total deterministic overhead: ~0.56 ms (measured)
API calls from enforcement: 0
Haiku compression: only on HIGH-J sessions when compression fires
```

**Layers 1 and 3 are fully deterministic.** They do not ask Claude for permission.
**Layer 2 (injection) is probabilistic** — depends on Claude following instructions.
This is the honest architecture: enforcement is deterministic; guidance is not.

One rule drives the entire system:

> **Only epistemically resolved content is safe to compress.**
> **Uncertain content must survive verbatim.**

---

## Validated Results

### Result 1 — E6: Negative Needle (all Opus 4.7, n=23 trials)

12-turn session. Uncertain constraints at T3-T4. 8 HIGH-J filler turns.
Callback at T13-T14. **All three conditions on Opus 4.7 — identical model.**

| Condition | Correction recall | FCR |
|---|---|---|
| Credence | **100%** | **0–4%** |
| Baseline (full context) | 100% | ~2% |
| Naive window | **~20%** | **~80%** |

Naive window drops constraints — they fall outside the sliding context. Credence
matches baseline recall under compression pressure by keeping the registry
independent of the context window.

```
python -m evals.experiments --exp E6
```

### Result 2 — Compression Faithfulness (n=30)

30 conversations, one uncertain constraint each, Haiku compression, Opus callback.
Direct measurement of epistemic qualifier loss under LLM context compression.

| | Naive Haiku | Probe-guarded | Baseline |
|---|---|---|---|
| Qualifier survival | 40.0% | 100% | 100% |
| False certainty | 36.7% | **0%** | 0% |

Faithfulness probe: 100% detection rate on the study (all 30 segments contain canonical
uncertainty markers, all 30 were blocked). Eliminates false certainty entirely.

```
python -m evals.compression_faithfulness
```

### Result 3 — Ghost Gauntlet (n=10 sessions × 3 claims, all Opus 4.7)

Tests the harder failure mode: **ghost constraints** — uncertain facts stated WITHOUT
canonical hedging markers. These bypass the faithfulness probe entirely. Caught by
SE probe (N=3 re-completions, NLI clustering) + claim extraction.

| Condition | BothRate (value + qualifier recalled) |
|---|---|
| Credence (SE probe + claim extraction) | **1.000** |
| Naive window | **0.133** |

7.5× gap. The SE probe catches what the faithfulness probe cannot.

```
python -m evals.ghost_gauntlet
```

### Result 4 — E7: 3-Hop Reasoning Chain (Categorical Result)

3-hop dependency chain: Project Falcon → Nexus config → CVE/v5 → Python ≥3.10.
Six filler turns force naive window to drop T3–T5 entirely.

| Condition | Hops recalled | Chain complete |
|---|---|---|
| **Credence** | **3 / 3** | **✓** |
| Naive window | **0 / 3** | **✗ (chain destroyed)** |
| Baseline | 3 / 3 | ✓ |

This result is categorical, not probabilistic. Naive window failed completely (0/3).
It cannot be explained by noise or sample size. Credence's selective J-routing preserved
the dependency chain that naive window dropped. This is the strongest single evidence
point because it is binary: either the chain survived or it didn't.

```
python -m evals.experiments --exp E7
```

---

## Token Overhead — Honest Framing

The faithfulness probe **correctly blocks compression** when uncertain content is present.
This means: on sessions with genuine uncertain constraints, Credence keeps more tokens
than a naive window would. This is not a cost bug — it is the system working as designed.

**When Credence uses more tokens than naive window**: Sessions with LOW/MEDIUM-J turns
that naive window would silently drop. Credence preserves them. E7 is the proof:
naive window saves tokens by dropping T3-T5, but destroys the chain. The tokens Credence
spends are what preserved 3/3 hops.

**When Credence saves tokens vs. full context**: Sessions with sustained HIGH-J content
(already-resolved factual exchanges). Compression fires and Haiku summarises. E4/E8 show
this path: Credence compresses while preserving dependency chains.

**Sessions with no uncertain constraints**: Zero overhead. The Truth Buffer injection only
fires when `get_relevant_claims()` returns a non-empty match. Sessions with purely
HIGH-J content (no uncertainty registered) incur no injection at all.

---

## Known Limitations

- **Confident-wrong ceiling**: The system cannot detect when Claude is factually wrong
  but uses confident language. J-score measures assertiveness, not correctness.
  Ghost constraints (HIGH-J claims without hedging) bypass the faithfulness probe —
  caught only by the SE probe (opt-in, ~$0.0015/call) or claim extraction.
- **Truth Buffer is instruction-following**: The enforcement depends on Claude complying
  with the system prompt injection. Prompt competition on long contexts can reduce
  compliance. The GTS (Layer 3) fires regardless.
- **Short sessions (< 16 turns)**: Compression never fires — correct behavior.
  Credence's value scales with session length and uncertainty density. For a 5-minute
  chat, just use the registry directly without compression routing.
- **J-proxy is a heuristic**: 68.7% OOF accuracy on 26 labelled samples.
  ρ = −0.034 with factual correctness — J is a compression scheduler, not an
  epistemic signal. Don't interpret HIGH-J as "Claude is correct."

---

## Truth Buffer

When a `CredenceRegistry` is attached to a `ContextManager`, unverified constraints are
injected into the system prompt before every API call:

```
EPISTEMIC CONTEXT — UNVERIFIED CONSTRAINTS:
• [LOW] I think the rate limit is ~50 req/min — unconfirmed
• [LOW] Token expiry might be 3600s — needs vendor verification
When discussing topics related to these constraints, always acknowledge their uncertain status.
```

This turns Credence from a reactive compression guard to a **proactive epistemic governor**:
Claude is reminded of what it doesn't know at the start of every turn.

## Output Alignment Layer

The Governor catches the other half of the failure: not just *storing* uncertain constraints,
but detecting when the model's *output* contradicts them with false confidence.

```python
# After every turn, check whether the response is more confident than the ledger warrants:
warnings = mgr._align_output(response_text)
# → [{"constraint_content": "rate limit is ~100 req/min (unconfirmed)",
#      "ledger_zone": "LOW",
#      "response_zone": "HIGH",
#      "suggested_caveat": "Previous response discussed 'rate limit is ~100...' which is unverified."}]

# Caveat is injected into the NEXT turn's system prompt automatically.
```

Available as an MCP tool: `credence_align(session_id, response_text)`.

## Scout Classifier

Enable `use_scout=True` to auto-extract uncertain constraints from user messages:

```python
mgr = ContextManager(
    registry=CredenceRegistry("project.db"),
    session_id="payment-v2",
    use_scout=True,
)
```

Each low-J user message triggers a Haiku entity-extraction call. Uncertain constraints are
auto-registered without requiring explicit `credence_register` calls. Zero friction.

## Certainty Trajectory

Every constraint now carries a full event history — from first observation through verification:

```python
reg = CredenceRegistry("project.db")
traj = reg.get_trajectory(constraint_id)
# → [{"event_type": "register", "timestamp": ..., "j_score": 0.28},
#    {"event_type": "scout",    "timestamp": ..., "j_score": 0.30},
#    {"event_type": "verify",   "timestamp": ..., "notes": "confirmed: 50 req/min"}]
```

Turns the registry from a snapshot into a temporal ledger of epistemic debt.

## Agentic Gate

Block irreversible actions when unverified constraints are present:

```python
# In Claude Code CLAUDE.md:
# Before calling write_file, execute_code, or deploy:
# Call credence_gate(tool_name, arguments_summary, session_id)
# If proceed=False, surface the blocked_by constraints to the user first.
```

---

## MCP Tools (18 total)

| Tool | What it does |
|------|-------------|
| `credence_chat` | Send message, receive response + epistemic envelope |
| `credence_risk` | Pre-flight risk check before any compress or action |
| `credence_align` | **NEW** Output Alignment — detect overconfident responses vs ledger |
| `credence_gate` | Agentic gate — block tool calls when unverified constraints present |
| `credence_inspect` | Trust analysis + BLOCK/VERIFY/PRESERVE/PROCEED |
| `credence_propagate` | Increment chain_depth for agent handoffs |
| `credence_register` | Explicitly register an uncertain constraint for tracking |
| `credence_verify` | Write-back: mark a constraint as verified with confirmed value |
| `credence_list_uncertain` | Audit all unverified constraints in a session |
| `credence_check_contradiction` | Detect if a new claim contradicts a verified constraint |
| `credence_trajectory` | Certainty trajectory — full event history for a constraint |
| `credence_stats` | Token savings, compression counts |
| `credence_log` | Per-turn J-scores and decisions |
| `credence_save` / `credence_load` | Cross-session continuity |
| `credence_reset` | Clear session |

### MCP Resources (passive context inheritance)

```
epistemic://session/{id}/ledger          → full unverified constraint list
epistemic://session/{id}/constraint/{id} → single constraint with trajectory
epistemic://session/{id}/alignment       → pending Governor caveat
```

Resources are passive — any MCP-compatible agent inherits the epistemic ledger
as context without calling a tool. This is the shift from *tooling layer* to *memory substrate*.

---

## Epistemic Transport Protocol (ETP v1)

`etp-v1.json` — open JSON Schema for model-agnostic epistemic metadata transport.

```json
{
  "schema": "etp/v1",
  "j_score": 0.24,
  "zone": "LOW",
  "verified": false,
  "chain_depth": 2,
  "trust_score": 0.09,
  "should_verify": true,
  "uncertainty_preserved": true,
  "source": "user",
  "session_id": "payment-v2"
}
```

The design principle: information passing between AI agents and memory systems should carry
its epistemic weight, not just its content. ETP is the proposed open standard for this.

---

## REST API

```bash
pip install fastapi uvicorn
uvicorn api.main:app --port 8000
```

```python
# Same interface as MCP tools via HTTP
import requests
resp = requests.post("http://localhost:8000/v1/chat", json={
    "session_id": "payment-v2",
    "message": "I think the rate limit is 100 req/min — unconfirmed",
}, headers={"X-API-Key": "cr-your-key"})
print(resp.json()["j_score"], resp.json()["zone"])
```

Dev mode (no auth): `CREDENCE_DEV_MODE=1 uvicorn api.main:app`

---

## Python SDK

Zero-dependency drop-in replacement for `ContextManager`:

```python
from credence.client import CredenceClient

client = CredenceClient(api_key="cr-your-key", base_url="http://localhost:8000")

result = client.chat(
    message="I think the rate limit is 100 req/min — unconfirmed",
    session_id="payment-v2",
)
print(result.response)
print(f"J={result.j_score:.2f}  zone={result.zone}  decision={result.decision}")

if result.alignment_warnings:
    print("Governor flagged:", result.alignment_warnings[0].suggested_caveat)
```

---

## TypeScript SDK

```bash
npm install credence-ai
```

```typescript
import { CredenceClient } from 'credence-ai'

const client = new CredenceClient({ apiKey: 'cr-your-key' })

const result = await client.chat({
  message: "I think the rate limit is 100 req/min — unconfirmed",
  sessionId: "payment-v2",
})
console.log(`J=${result.jScore}  zone=${result.zone}  decision=${result.decision}`)

if (result.alignmentWarnings.length > 0) {
  console.log("Governor flagged:", result.alignmentWarnings[0].suggestedCaveat)
}

// Pre-flight gate before an irreversible action
const gate = await client.gate({
  toolName: "deploy_service",
  argumentsSummary: "Deploy payment service v2 with rate limit 100 req/min",
  sessionId: "payment-v2",
})
if (!gate.proceed) {
  console.log("Blocked by:", gate.blockedBy.map(c => c.content))
}
```

Full type definitions: `EpistemicConstraint`, `EpistemicEnvelope`, `AlignmentWarning`, `ChatResult`, `RiskResult`, `GateResult`, `SessionStats`.

---

## Model Adapters

Use the full Credence epistemic memory system with any model provider:

```python
from credence.adapters import OpenAIAdapter, GeminiAdapter, OllamaAdapter

# OpenAI — GPT-4o generation, GPT-3.5-turbo compression
mgr = OpenAIAdapter(api_key="sk-...", model="gpt-4o")

# Google Gemini — 1.5 Pro generation, Flash compression
mgr = GeminiAdapter(api_key="AI...", model="gemini-1.5-pro")

# Local models via Ollama — zero API cost
mgr = OllamaAdapter(model="llama3.2")

# All three: identical interface
result = mgr.chat("I think the rate limit is 100 req/min — unconfirmed")
print(result.response, result.j_score, result.decision)
```

All adapters share identical J-score computation, faithfulness probe, selective compression,
drift detection, and Output Alignment Layer logic. Only the provider API call differs.
The `BaseAdapter` abstract class makes adding new providers a 40-line implementation.

Install provider dependencies:
```bash
pip install openai                    # OpenAI adapter
pip install google-generativeai       # Gemini adapter
pip install ollama                    # Ollama adapter
```

---

## Epistemic Registry — Cross-Session Constraint Store

```python
from credence import CredenceRegistry

reg = CredenceRegistry("epistemic.db")

# Register uncertain constraint at planning time
cid = reg.register(
    "I think the rate limit is 100 req/min — unconfirmed",
    session_id="payment-v2",
    j_score=0.28, zone="LOW",
)

# Later: verify against the actual value
reg.verify(cid, verified_value="confirmed: 50 req/min (sandbox)")

# Before acting on a new claim: check for contradictions
hits = reg.check_contradiction(
    "The retry logic should target 100 req/min",
    session_id="payment-v2",
)
# → returns the verified constraint: rate limit is 50, not 100
```

---

## Run the Demo

```bash
# No API key required — runs offline in 5 seconds
python quickstart.py

# Full Streamlit demo
streamlit run demo/app.py
```

- **Tab 1 — The Failure**: Naive window drops uncertain constraints → hallucination
- **Tab 2 — The Fix**: Epistemic memory preserves them
- **Tab 3 — Live Chat**: Real-time J-gauge, decision log, and registry panel
- **Tab 4 — Evidence**: Benchmark results and calibration data

---

## Run the Evaluations

```bash
# Core validated experiments
python -m evals.experiments --exp E6
python -m evals.experiments --exp E7

# Compression faithfulness study (headline result, n=30, ~$20 API)
python -m evals.compression_faithfulness

# Null hypothesis test: does "preserve qualifiers" instruction alone suffice?
python -m evals.haiku_baseline

# E6 multi-trial (re-run after model fix to all Opus, ~$15 API)
python -m evals.e6_repeated --n 20

# Gauntlet — 50 scenarios × 5 domains × 4 conditions (~$40 API)
python -m evals.gauntlet --dry-run          # validate structure (free)
python -m evals.gauntlet --domain api       # 10 API scenarios
python -m evals.gauntlet                    # full 50-scenario run

# Behavioral consistency calibration
python -m evals.behavioral_calibration
```

---

## Signal Architecture

Two tiers of epistemic signal in production use:

```
Tier 1 — Linguistic Assertiveness (~0ms, $0)
  5 text pattern factors → J-score ∈ [0,1]
  Covers ~80% of the routing signal. Zero cost per turn.
  Known ceiling: cannot detect confident-wrong content.

Tier 2 — Behavioral Consistency (~300ms, ~$0.001/turn)
  N=5 Haiku samples → pairwise ROUGE-L variance
  Low variance = consistent answers = high confidence.
  Better calibrated than Tier 1 on medium-difficulty questions (ECE 0.2472 vs 0.2830).
  Opt-in via use_agreement=True.
```

---

## Multi-Agent Provenance

When constraints cross agent boundaries, the `CredenceEnvelope` carries provenance
metadata with each handoff: source, chain_depth, trust_score, verified status, and
uncertainty_preserved. Trust degrades with each hop — a tentative estimate from
3 hops ago arrives with `should_verify=True` regardless of how confidently the
intermediate agent phrased it. This is advisory, not enforced; downstream agents
need explicit instructions to act on `should_verify`.

---

## Project Structure

```
credence/
  confidence_proxy.py     J-score computation (Tier 1, zero-cost)
  context_manager.py      Core memory governor (compress/trim/preserve + Governor)
  behavioral_signal.py    Tier 2 behavioral consistency (N=5 Haiku)
  envelope.py             CredenceEnvelope — multi-agent provenance
  registry.py             CredenceRegistry — SQLite constraint store (cross-session)
  client.py               Python SDK — zero-dependency HTTP client
  mcp_server.py           FastMCP server (18 tools + 3 resources)
  agent.py                Long-running task agent
  adapters/
    base.py               BaseAdapter — provider-agnostic memory logic
    openai_adapter.py     GPT-4o / GPT-3.5-turbo
    gemini_adapter.py     Gemini 1.5 Pro / Flash
    ollama_adapter.py     Any local Ollama model

api/
  main.py                 FastAPI backend (20+ REST endpoints)

packages/
  credence-ts/            TypeScript SDK (npm: credence-ai)
    src/index.ts          Full type definitions + CredenceClient

evals/
  compression_faithfulness.py   Headline result: compression strips qualifiers (n=30)
  haiku_baseline.py             Null hypothesis test (hedge-instruction vs probe)
  experiments.py                E1–E8 ablation experiments
  e6_repeated.py                E6 multi-trial statistical validation
  behavioral_calibration.py     Behavioral consistency calibration (n=60)
  benchmark.py                  30-pair QA benchmark (4 conditions)
  conversation_benchmark.py     10-session × 3-condition benchmark

etp-v1.json               Epistemic Transport Protocol v1 JSON Schema
ARCHITECTURE.md           Full system design
TECHNICAL_REPORT.md       arXiv-style research paper

demo/
  app.py                  Streamlit demo (4 tabs)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the research paper.

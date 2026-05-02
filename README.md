# Credence

**Claude doesn't just forget what you told it. It forgets whether you were sure about it.**

You're in a Claude Code session. You say:
> *"The rate limit is probably around 50 — I haven't confirmed it yet."*

Fifteen turns of coding later, Claude writes:

```python
RATE_LIMIT = 50
```

No warning. No flag. The uncertainty is gone. You ship it.  
The API rejects every request at 2am. The real limit was 10.  
Nobody lied. Claude just forgot you weren't sure.

This failure has a name. We measured it. We fixed it.

![Credence gate blocking in real time](demo/gate_demo.gif)

---

## The Problem, Measured

**Epistemic Qualifier Loss (EQL)** — uncertainty markers (*"I think"*, *"unverified"*, *"roughly"*, *"the vendor claims"*) are stripped during context compression, causing downstream models to treat uncertain constraints as confirmed facts.

We ran 50 compression scenarios with three conditions:

| Condition | Qualifier Strip Rate | False Certainty Rate |
|---|---|---|
| Naive Haiku compression | 46% | 6% |
| LLMLingua-style scoring | 68% | **74%** |
| **Credence (faithfulness probe)** | **0%** | **0%** |

The False Certainty Rate (FCR) — model states an uncertain value as confirmed fact — drops to zero. Deterministically. With zero extra API calls across all five enforcement layers.

> Every engineering team using Claude Code today is producing ghost constraints they don't know about. Every sprint.

---

## Quick Start

```bash
pip install credence-guard
python quickstart.py          # all 5 enforcement layers, no API key needed
```

Or from source:
```bash
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
cd credence-ai && pip install -e ".[mcp]"
python quickstart.py
```

---

## What Gets Blocked

**Without Credence** — Haiku strips "I think" from the context summary. Downstream model writes:
```python
RATE_LIMIT = 50
ALGORITHM  = "RS256"
TOKEN_EXPIRY = 3600
```

**With Credence** — Generation-Time Scanner annotates inline before code reaches you:
```python
RATE_LIMIT   = 50    # ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: rate limit unconfirmed — vendor docs pending
ALGORITHM    = "RS256"  # ⚠  CREDENCE[unverified, conf=0.28]: encryption algo — per vendor call
TOKEN_EXPIRY = 3600  # ⚠  CREDENCE[unverified, conf=0.31]: auth token expiry not verified
```

**Rust gate** (3.4ms) blocks the write entirely when unverified constraints overlap the tool action:
```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:   Edit
  ⚠ [LOW, conf=0.28] auth token expires in 3600s — unconfirmed
    Overlap terms: token, expires, auth

  Use credence_verify(<id>, <confirmed_value>) to resolve.
```

Once verified, the constraint clears. The gate unblocks. Claude writes the code.

```
uncertain → registered → enforced → verified → released
```

---

## How It Works

Five checkpoints. Four are fully deterministic — no model cooperation required.

```
User states uncertain claim
        │
        ▼
┌─────────────────────────────────────────────┐
│  REGISTRY  (SQLite, ~0.37ms)                │
│  Stores uncertain constraints with          │
│  per-type confidence decay                  │
│  Cross-session. Zero API calls.             │
└──────────────────────┬──────────────────────┘
                       │
    ▼ before compression
┌──────────────────────────────────────────────┐
│  CP1 — Faithfulness Probe  (P99=0.026ms)     │  DETERMINISTIC
│  198-term frozenset. Scans user turns only.  │
│  Uncertainty found → block Haiku → KEEP      │
└──────────────────────────────────────────────┘
                       │
    ▼ before generation
┌──────────────────────────────────────────────┐
│  CP2 — Truth Buffer + Consistency Enforcer   │  PROBABILISTIC
│  Injects all unverified constraints into     │  (model must comply)
│  every system prompt. When query overlaps    │
│  registered constraint → imperative block:   │
│  "YOU MUST express uncertainty."             │
└──────────────────────────────────────────────┘
                       │
    ▼ after generation
┌──────────────────────────────────────────────┐
│  CP3 — Generation-Time Scanner (P99=0.036ms) │  DETERMINISTIC
│  Catches numeric + string literals in code   │
│  and prose. Three tiers: ⚠⚠ HIGH RISK /     │
│  ⚠ UNVERIFIED / CHECK based on conf score.  │
└──────────────────────────────────────────────┘
                       │
    ▼ at tool execution
┌──────────────────────────────────────────────┐
│  CP4 — Rust Gate (P50=3.4ms)                 │  DETERMINISTIC
│  Native PreToolUse hook. Blocks Write/Edit/  │
│  Bash when arguments overlap an unverified   │
│  constraint. 98× faster than Python hook.   │
└──────────────────────────────────────────────┘
                       │
    ▼ across sessions
┌──────────────────────────────────────────────┐
│  CP5 — Cross-Session Memory (P99=1.03ms)     │  DETERMINISTIC
│  New sessions inherit uncertainty status,    │
│  not just values. Epistemic debt survives    │
│  restarts.                                   │
└──────────────────────────────────────────────┘

Total in-session overhead (P99): 1.1ms in-process + 3.4ms gate = ~4.5ms. Zero extra API calls.
LLM call overhead: ~0.09% of typical Claude Opus latency (3,000–8,000ms).
```

### Ghost Constraints

The faithfulness probe catches explicit hedges: *"I think"*, *"approximately"*, *"probably"* — 198 markers.

But what about:
> *"The Stripe rate limit is 50 req/min."*

No hedging. Stated as fact. Actually from a sales call, never confirmed. The probe sees nothing.

This is a **ghost constraint** — implicitly uncertain, no surface markers. The Ghost Detector (opt-in, one Opus call per constraint at registration) classifies whether a stated fact is an established truth or a vendor claim stated as fact.

```
Ghost Gauntlet — n=10 sessions, all Opus 4.7

Credence (Ghost Detector active)  BothRate = 1.000
Naive sliding window              BothRate = 0.200
```

---

## Install in Claude Code

```bash
pip install "credence-ai[mcp]"
```

Add to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "credence": {
      "command": "python3",
      "args": ["-m", "credence.mcp_server"],
      "env": { "ANTHROPIC_API_KEY": "your-key-here" }
    }
  },
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|Bash|NotebookEdit",
      "hooks": [{ "type": "command", "command": "credence-gate" }]
    }]
  }
}
```

Build the Rust gate (optional, for CP4 enforcement):
```bash
cd credence_gate && cargo build --release
# binary: credence_gate/target/release/credence-gate
```

---

## Validated Results (Latest)

| Experiment | Credence | Naive / Baseline |
|---|---|---|
| Compression faithfulness — Haiku FCR (n=50) | **0%** | 6% |
| Compression faithfulness — LLMLingua FCR (n=50) | **0%** | 74% |
| E6: Long-session constraint recall (n=23 trials) | **100%** | 19.6% (naive window) |
| E7: Multi-hop 3-step reasoning chain | **3/3 hops** | 0/3 (naive) |
| E8: Real debugging session recall | **1.000** | 0.522 (naive) |
| Ghost Gauntlet BothRate (n=10 sessions) | **1.000** | 0.200 (naive) |
| Cross-session FCR (n=20 callbacks) | **0%** | 40% (no memory) |
| Rust gate latency | **3.4ms** | 331ms (Python) |
| Total in-process overhead (P99 sum) | **1.1ms** | — |

---

## Reproducing the Results

**No API key — runs in seconds:**
```bash
python quickstart.py                     # live demo, no API needed
python tests/tests.py                    # 178 unit tests
python tests/test_claims.py              # validates all claimed numbers offline
python -m evals.adversarial_tests        # 5 adversarial robustness tests
python -m evals.latency_report --n 1000  # P50/P95/P99 for all 5 checkpoints
python -m evals.calibration_curve        # ECE + ghost candidate analysis
```

**Dataset & Training:**
```bash
# Build the 5,000-triple DPO training dataset (requires ANTHROPIC_API_KEY, ~$5)
python -m data.build_training_dataset --n 5000 --out data/epistemic_compression_training.json

# Verify dataset quality
python -m data.build_training_dataset --verify --sample 500

# Training (requires GPU — Kaggle T4 free tier)
# See training/dpo_finetune.py and kaggle_kv_cache/run_kv_experiment.py
```

### Training Status
Phase 3 DPO fine-tuning completed on Kaggle T4 (Phi-2, 5,000 triples). Best checkpoint: epoch 2. Full epoch curve and three-point comparison:

| Condition | FCR | EQLR | Notes |
|---|---|---|---|
| Base Phi-2 (pre-DPO) | **31.2%** | 53.3% | Generation-level baseline |
| DPO fine-tuned (epoch 2) | **19.1%** | 62.1% | −12.1pp, 39% relative reduction |
| DPO epoch 3 | 22.1% | 58.8% | Regressed — overfit, use epoch 2 |
| Probe (deterministic) | **0%** | 100% | Mechanical guarantee, zero API calls |

Epoch 3 regression is expected DPO behavior (lambda=0.3, model drifted too far from reference). Use `epoch_2/` adapter. Pull results: `kaggle kernels output chakradharvijayarao/credence-phase-3-dpo-epistemic-fine-tuning -p /tmp/dpo_out/`
- **Adapter (epoch 2 best)**: `evals/dpo_epoch_results.json` (after pull)
- **Earlier negative result (archived)**: [models/credence-phi-2-dpo](models/credence-phi-2-dpo)

**With API key — core evidence (~$7 total):**
```bash
python -m evals.compression_faithfulness --n 50   # headline: 46%→0% EQLR, 74%→0% FCR  (~$3)
python -m evals.ghost_gauntlet                     # BothRate 0.200→1.000                (~$2)
python -m evals.experiments --exp E6               # long-session recall 100% vs 19.6%  (~$0.50)
python -m evals.experiments --exp E7               # 3-hop chain: 3/3 vs 0/3            (~$0.20)
python -m evals.experiments --exp E8               # debugging session recall            (~$0.30)
```

All results already saved in `evals/*.json` — no API key needed to read them.

---

## As a Python Library

```python
from credence import ContextManager, CredenceRegistry

registry = CredenceRegistry()
cm = ContextManager(registry=registry, session_id="my-session")

# Uncertain constraint gets registered and enforced automatically
result = cm.chat("The rate limit is probably 50 req/min — I haven't confirmed it")

# Next session inherits the uncertainty
from credence import CredenceMemory
memory = CredenceMemory(registry)
memory.snapshot("my-session", project="my-api-project")
# New session will see: "UNVERIFIED: rate limit is probably 50 req/min"
```

---

## As an MCP Server (10 tools)

```python
# credence_chat        — full enforcement turn
# credence_register    — register an uncertain constraint
# credence_verify      — mark a constraint as confirmed
# credence_gate        — pre-execution agentic gate
# credence_inspect     — BLOCK/VERIFY/PRESERVE/PROCEED recommendation
# credence_scan        — scan any model output for unverified literals
# credence_trajectory  — certainty trajectory for a constraint over time
# credence_memory_snapshot / credence_memory_recall — cross-session
# ... 14 more
```

---

## Project Structure

```
credence/
  context_manager.py    All 5 enforcement layers
  registry.py           SQLite constraint store + confidence decay
  confidence_proxy.py   J-score (zero API, zero latency)
  memory.py             Cross-session epistemic persistence
  mcp_server.py         FastMCP server — 10 tools
  pipeline_monitor.py   Multi-agent handoff interception

evals/                  12 validation studies
  compression_faithfulness.py   Primary result (n=50)
  ghost_gauntlet.py             Ghost constraint benchmark
  gauntlet.py                   50-scenario breadth benchmark
  experiments.py                E1–E9 ablation experiments
  eql_bench.py                  EQL-Bench v1 dataset (52 scenarios, 8 domains)
  latency_report.py             P50/P95/P99 for all 5 checkpoints
  calibration_curve.py          ECE + ghost candidate analysis

credence_gate/          Rust PreToolUse hook — 3.4ms
tests/                  178 unit tests
docs/                   Technical report, architecture, vision
quickstart.py           First-run demo (no API key)
```

---

## Documentation

| What you want | Where |
|---|---|
| Full methodology + related work | [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) |
| Layer-by-layer design decisions | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Honest scope and limitations | [docs/SUBMISSION.md](docs/SUBMISSION.md) |
| Research vision | [docs/VISION.md](docs/VISION.md) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All offline tests pass without an API key — you can contribute to the probe, registry, GTS, and Rust gate without spending anything.

---

## Built By

**Lakshmi Chakradhar Vijayarao** — Independent Researcher

[LinkedIn](https://www.linkedin.com/in/lakshmichakradharvijayarao/) · [X / Twitter](https://x.com/LChakradharV28) · [lakshmichakradhar.v@gmail.com](mailto:lakshmichakradhar.v@gmail.com)

---

MIT License — see [LICENSE](LICENSE)

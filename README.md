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

| Condition | Qualifier Strip Rate | False Certainty Rate (FCR) |
|---|---|---|
| Naive Haiku compression | 46% | 6% |
| Token-importance compression (simulation) ¹ | 68% | **74%** |
| **Credence (faithfulness probe)** | **0%** | **0%** |

**FCR** = model states an uncertain value as confirmed fact. Drops to zero for explicitly hedged content. Deterministically. Zero extra API calls.

> ¹ Simulates compression that scores sentences by technical token density with no epistemic awareness — the pattern used by token-importance systems. Not a measurement of the LLMLingua library itself.

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

---

## Scope of Protection

Credence prevents **explicitly stated uncertainty from being silently erased** during context compression and tracks unverified constraints across sessions. This is a specific, measurable problem with a deterministic fix.

| What Credence protects | What Credence does NOT protect |
|---|---|
| Uncertainty markers you wrote ("I think", "probably", "unconfirmed") | Facts stated with false confidence (no markers present) |
| Registered constraints from being forgotten across sessions | Confidently wrong information |
| Tool calls that embed unverified numeric literals (Rust gate) | Claude Code's internal context summarization (model-level, outside hook coverage) |
| Compressed context from stripping explicit qualifiers | Implicit uncertainty with no surface markers |
| Epistemic state across agent handoffs (ETP envelope) | Unregistered constraints |

**The one-sentence scope:** Credence deterministically prevents the loss of explicitly hedged uncertainty during LLM context compression and tracks those constraints across sessions.

If your uncertainty was stated with recognizable hedging language, Credence protects it. If it was not stated, Credence cannot protect what it was never told.

---

## Known Limitations

**1. Canonical markers only (CP1).**
The faithfulness probe operates on 198 English uncertainty markers. Domain-specific hedging ("non-inferiority not established", "per the vendor SLA", "preliminary benchmark") may not trigger the probe. The Ghost Detector (opt-in) extends coverage to implicit uncertainty but requires an additional model call.

**2. Native model summarization is a blind spot.**
Claude Code's internal context compression — which happens at the model/runtime level, not as a tool call — is outside the Rust gate's coverage. The gate intercepts `Write|Edit|Bash|NotebookEdit` tool calls. It does not intercept the model's own summarization. This is a known architectural gap.

**3. Verification requires external discipline.**
`credence_verify()` marks a constraint as confirmed. The system cannot validate whether the verification reflects real external confirmation. An audit trail (who verified, what evidence) will be enforced in v1.1 — for now, verification governance is the user's responsibility.

**4. FCR thresholds are Claude-calibrated.**
The J-score proxy and confidence decay rates were calibrated on Claude Opus 4.7 / Haiku. The faithfulness probe and registry work with any LLM output. Cross-model FCR validation is on the roadmap.

**5. Single-trial experiment results.**
E7 (multi-hop chain) and E8 (debugging session) are single-trial demonstrations. They show the system working correctly but do not establish statistical significance. Multi-trial versions are planned.

---

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
pip install "credence-guard[mcp]"
```

Add `.mcp.json` to your project root (Claude Code picks it up automatically):
```json
{
  "mcpServers": {
    "credence-guard": {
      "type": "stdio",
      "command": "credence-server",
      "env": {}
    }
  }
}
```

For the Rust gate (CP4 — blocks irreversible tool calls, 3.4ms):
```bash
cd credence_gate && cargo build --release
# Add to .claude/settings.json → hooks → PreToolUse
# Pre-built binaries are planned for a future release (tracks open issue #1)
```

---

## Validated Results (Latest)

| Experiment | Credence | Naive / Baseline | n | Status |
|---|---|---|---|---|
| Haiku compression FCR | **0%** | 6% | 50 | ✓ Multi-run |
| Token-importance compression FCR (simulation) | **0%** | 74% | 50 | ✓ Multi-run |
| E6: Long-session constraint recall | **100%** | 19.6% (naive window) | 23 trials | ✓ Multi-trial |
| E7: Multi-hop 3-step reasoning chain | **3/3 hops** | 0/3 (naive) | 1 | ⚠ Single trial |
| E8: Real debugging session recall | **1.000** | 0.522 (naive) | 1 | ⚠ Single trial |
| Ghost Gauntlet BothRate | **1.000** | 0.200 (naive) | 10 sessions | ⚠ Synthetic |
| Probe false positive rate | **0.5%** | — | 200 sentences | ✓ Deterministic |
| Rust gate latency | **3.4ms** | 331ms (Python hook) | 4000 calls | ✓ Measured |
| Total in-process overhead (P99) | **1.1ms** | — | 2000 calls | ✓ Measured |

> **Transparency note:** E7 and E8 are single-trial demonstrations — they show the mechanism working correctly but are not statistically validated. Ghost Gauntlet uses researcher-constructed sessions. Cross-session FCR measurement is planned (the infrastructure exists in `credence/memory.py`; the empirical study has not been run). Multi-trial E7/E8 are on the roadmap.

---

## Reproducing the Results

**No API key — runs in seconds:**
```bash
python quickstart.py                     # live demo, no API needed
python -m pytest tests/ -q               # 596 tests
python -m evals.adversarial_tests        # 5 adversarial robustness tests
python -m evals.latency_report --n 1000  # P50/P95/P99 for all 5 checkpoints
python -m evals.false_positive_rate      # Gate 0: probe FPR (target < 5%)
```

> **Research note — DPO fine-tuning:** The `training/` directory contains exploratory DPO work on Phi-2 (2.7B) using 5,000 epistemic triples. This is a research track, not a production artifact. The deterministic probe (CP1) is the shipped mechanism — it produces 0% FCR with zero model calls. The DPO experiments are preserved for reproducibility; no fine-tuned weights are currently distributed.

**With API key — core evidence (~$7 total):**
```bash
python -m evals.compression_faithfulness --n 50   # Haiku: 46% strip rate / token-importance sim: 74% FCR → both 0% with probe  (~$3)
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

## As an MCP Server (22 tools)

Core tools:
```
credence_pre_compress    — CP1: check text before compression (BLOCK / ALLOW)
credence_post_compress   — measure qualifier survival after compression
credence_register        — register an uncertain constraint explicitly
credence_verify          — mark a constraint as verified
credence_gate            — CP4: pre-tool gate (block if unverified constraints apply)
credence_scan            — CP3: scan model output for unverified numeric literals
credence_memory_snapshot — persist unverified constraints as project memory
credence_memory_recall   — load project memory into a new session
credence_wrap / unwrap   — ETP envelope for agent handoffs
credence_audit           — per-session epistemic timeline
... 12 more (see mcp_server.py or run credence-server --list-tools)
```

---

## Project Structure

```
credence/
  mcp_server.py         FastMCP server — 22 tools, 2 resources (epistemic:// URI)
  context_manager.py    All 5 enforcement layers (full enforcement engine)
  registry.py           SQLite constraint store + confidence decay + trajectories
  confidence_proxy.py   J-score (zero API, zero latency)
  wrap.py               Model-agnostic faithfulness wrapper (any Callable[[str],str])
  memory.py             Cross-session epistemic persistence
  enforce.py            Decorator-based enforcement (@enforce)
  pipeline_monitor.py   Multi-agent handoff interception

evals/                  Validation studies
  compression_faithfulness.py   Primary result (n=50, headline evidence)
  ghost_gauntlet.py             Ghost constraint benchmark
  gauntlet.py                   50-scenario breadth benchmark
  experiments.py                E1–E9 ablation experiments
  eql_bench.py                  EQL-Bench v1 dataset (52 scenarios, 8 domains)
  latency_report.py             P50/P95/P99 for all 5 checkpoints
  false_positive_rate.py        Gate 0: probe FPR (CI target < 5%)

credence_gate/          Rust PreToolUse hook — 3.4ms P50
sdk/typescript/         TypeScript SDK — runProbe(), CredenceEnvelope (probe + envelope)
tests/                  596 tests (unit / integration / security / perf)
docs/                   Technical report, architecture, ETP spec, vision
quickstart.py           First-run demo (no API key needed)
```

> **Multi-model note:** `wrap()` accepts any compression function — OpenAI, Gemini, local models. The faithfulness probe and registry work with any LLM output. FCR thresholds and J-score calibration are optimized for Claude (Opus 4.7 / Haiku). Third-party model validation is on the roadmap.

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

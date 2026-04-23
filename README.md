# Epistemic Memory

**Claude Code doesn't just forget what you told it. It forgets whether you were sure about it.**

When a conversation compresses, the *content* survives but the *confidence* does not.
A constraint expressed as *"I think the rate limit is 100 req/min — not confirmed yet"*
becomes *"the rate limit is 100 req/min"* after one summarisation pass.
Downstream turns — and downstream agents — treat the uncertain claim as resolved fact.

Epistemic Memory is a context safety layer for Claude Code and multi-agent systems
that tracks which content was expressed with uncertainty, and prevents that content
from being silently converted to apparent fact.

---

## Install in Claude Code (2 minutes)

```bash
pip install fastmcp anthropic
git clone https://github.com/vijayarao-chakri/epistemic-memory
pip install -e epistemic-memory
```

Add to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "epistemic-memory": {
      "command": "python",
      "args": ["-m", "cams.mcp_server"],
      "env": { "ANTHROPIC_API_KEY": "your-key-here" }
    }
  }
}
```

Add to your project's `CLAUDE.md`:

```markdown
Before implementing anything that depends on a constraint the user
expressed uncertainty about, call em_propagation_risk on that constraint.
If risk_level=HIGH or should_verify=True, surface the warning before writing code.
```

Restart Claude Code. You now have 8 epistemic memory tools available.

---

## What It Does

For every conversation turn, Epistemic Memory:

1. Computes a **J-score** (0–1) from the response text — 5 linguistic factors
   measuring assertiveness: hedging rate, anchor density, self-correction
   frequency, brevity, and specificity
2. Decides: **compress / trim / preserve** based on J-score thresholds
3. **Critical rule: only HIGH-J (epistemically resolved) turns are eligible
   for compression.** LOW/MEDIUM-J turns survive every compression verbatim.
4. **Faithfulness probe**: before any Haiku summarisation, scans the segment
   for 30+ uncertainty markers. If found → compression aborts → PRESERVE.

```
J ≥ 0.70  →  COMPRESS   (Haiku summarises; only HIGH-J eligible)
0.45–0.70 →  TRIM       (keep last N turns; LOW/MEDIUM-J always survive)
J < 0.45  →  PRESERVE   (full history retained)
```

---

## What Is Validated vs. What Is Being Studied

### Validated (live API, reproducible)

**E6 — Faithfulness probe prevents hallucination propagation**

| Condition | Correction recall | Hallucination rate |
|-----------|------------------|--------------------|
| Epistemic Memory (probe active) | **100%** | **0%** |
| Naive window (no probe) | 0% | 50% |
| Baseline (full context) | 100% | 0% |

When naive context truncation drops an uncertain constraint (e.g. *"rate limit
unconfirmed — either 50 or 100 req/min"*), the downstream model answers with
50% hallucination rate. The faithfulness probe prevents compression of that
constraint entirely — restoring baseline performance.

Run: `python -m evals.experiments --exp E6`

**E7 — 3-hop reasoning chain preserved**

| Condition | Hops recalled | Chain complete |
|-----------|-------------|----------------|
| Epistemic Memory | 3 / 3 | ✓ |
| Naive window | 0 / 3 | ✗ |

**E8 — Real debugging session, uncertain hypothesis**

| Condition | Constraint recall |
|-----------|------------------|
| Epistemic Memory | 1.000 |
| Naive window | 0.522 |

**E4 — Causal check (J-routing vs random)**

CAMS 0.875 > random_j 0.812 > naive 0.750 — confirms J-routing carries
signal above random compression scheduling.

---

### Being Studied (experiments run overnight, results in `evals/`)

**Compression Faithfulness Study** (`evals/compression_faithfulness.py`)

30 realistic technical conversations. Measures:
- What fraction of Haiku compressions strip uncertainty markers (hypothesis: 40–60%)
- What fraction of downstream models answer with false certainty post-compression (hypothesis: 50–70%)
- What fraction the faithfulness probe blocks (hypothesis: 90–100%)
- Downstream false-certainty rate with probe active (hypothesis: ~0%)

Run: `python -m evals.compression_faithfulness`

**Behavioral Consistency Calibration** (`evals/behavioral_calibration.py`)

60 factual questions across 3 difficulty strata. Measures whether
behavioral consistency (N=5 samples → ROUGE-L variance) is better
calibrated against factual accuracy than the J-proxy.
Computes ECE and Spearman correlation for both signals.

Run: `python -m evals.behavioral_calibration`

---

### Known Limitations

- **Confident-wrong ceiling**: The system cannot detect when Claude is
  factually wrong but uses confident language. J-score measures assertiveness,
  not correctness.
- **Compression rarely fires on uncertainty-heavy sessions**: The faithfulness
  probe correctly blocks compression when uncertain content is present.
  The system defaults to PRESERVE — the safe choice.
- **Multi-agent envelope is advisory**: Downstream agents need explicit
  instructions tied to `should_verify`. The envelope alone is not enforced.
- **J-proxy is a heuristic**: 68.7% OOF accuracy on 26 labelled samples.
  Behavioral consistency (Tier 2) is more principled but opt-in.

---

## MCP Tools

| Tool | What it does |
|------|-------------|
| `cams_chat` | Send message, receive response + epistemic envelope |
| `em_propagation_risk` | Pre-flight risk check before any compress or action |
| `cams_inspect_envelope` | Trust analysis + BLOCK/VERIFY/PRESERVE/PROCEED |
| `cams_propagate_envelope` | Increment chain_depth for agent handoffs |
| `cams_get_stats` | Token savings, compression counts |
| `cams_get_decision_log` | Per-turn J-scores and decisions |
| `cams_save` / `cams_load` | Cross-session continuity |
| `cams_reset` | Clear session |

### `em_propagation_risk` in practice

Before implementing anything that depends on an uncertain constraint:

```python
risk = em_propagation_risk(
    content="I think the rate limit is 100 req/min but might be 50 — unconfirmed.",
    chain_depth=0,
)
# → risk_level="HIGH", should_verify=True
# → Claude Code surfaces a warning before writing the implementation
```

---

## Run the Demo

```bash
streamlit run demo/app.py
```

- **Tab 1 — The Failure**: Naive window drops uncertain constraints → hallucination
- **Tab 2 — The Fix**: Epistemic memory preserves them
- **Tab 3 — Live Chat**: Real-time J-gauge and decision log
- **Tab 4 — Evidence**: Benchmark results and calibration data

For the Claude Code demo, open the `claude_code_demo/` directory in
Claude Code — the CLAUDE.md instructs Claude to call `em_propagation_risk`
before acting on uncertain constraints.

---

## Run the Evaluations

```bash
# Core validated experiments
python -m evals.experiments --exp E6
python -m evals.experiments --exp E7
python -m evals.experiments --exp E8
python -m evals.experiments --exp E4

# New: compression faithfulness study (overnight, ~$20 API)
python -m evals.compression_faithfulness

# New: behavioral consistency calibration (overnight, ~$15 API)
python -m evals.behavioral_calibration

# Flagship experiment: 3 scenarios × 3 conditions × 3 trials
python -m experiments.flagship.run --trials 3

# Conversation benchmark: 10 sessions × 3 conditions
python -m evals.conversation_benchmark
```

---

## Signal Architecture

Three tiers of epistemic signal:

```
Tier 1 — Linguistic Assertiveness (~0ms, $0)
  5 text pattern factors → J-score ∈ [0,1]
  Covers ~80% of the signal. Zero cost per turn.
  Known ceiling: cannot detect confident-wrong content.

Tier 2 — Behavioral Consistency (~300ms, ~$0.001/turn)
  N=5 Haiku samples → pairwise ROUGE-L variance
  Low variance = consistent answers = high confidence.
  Better calibrated than Tier 1 on medium-difficulty questions.
  Opt-in via use_agreement=True.

Tier 3 — Fisher J from Activations (prior research, informs design)
  KV-cache attention entropy measured in prior work on Qwen 3.5B.
  Established that linguistic assertiveness correlates with internal
  model uncertainty. Validates Tier 1's design direction.
  Not required at runtime — Tier 1 + E4 causal check are sufficient.
```

---

## The Bigger Picture

This system is a reference implementation of a pattern we call
**Epistemic Transport**: the idea that every piece of information in an
AI pipeline should carry its epistemic state — not just its content.

Today, when a multi-agent system passes a claim from Agent A to Agent B:
- Agent B has no idea whether Agent A was confident or guessing
- Agent B has no idea how many summarisation hops the claim went through
- There is no standard format for transmitting epistemic metadata

The `CAMSEnvelope` is a proposed standard for this metadata:

```json
{
  "j_score": 0.24,
  "zone": "LOW",
  "verified": false,
  "chain_depth": 2,
  "trust_score": 0.09,
  "should_verify": true,
  "uncertainty_preserved": true,
  "source": "agent_B",
  "session_id": "abc123"
}
```

Trust degrades with each hop. An uncertain constraint from 4 hops ago
carries `should_verify=True` regardless of how confident the current
agent sounds.

---

## Project Structure

```
cams/
  confidence_proxy.py     J-score computation (Tier 1, zero-cost)
  context_manager.py      Core memory governor (compress/trim/preserve)
  behavioral_signal.py    Tier 2 behavioral consistency (N=5 Haiku)
  envelope.py             CAMSEnvelope — multi-agent provenance
  mcp_server.py           FastMCP server (8 tools)
  agent.py                Long-running task agent

evals/
  experiments.py          E1–E8 ablation experiments
  compression_faithfulness.py   NEW: compression strips qualifiers study
  behavioral_calibration.py     NEW: behavioral consistency calibration
  benchmark.py            30-pair QA benchmark (4 conditions)
  conversation_benchmark.py     10-session × 3-condition benchmark
  calibration.py          Threshold optimisation

experiments/flagship/     3-scenario × 3-condition flagship experiment

demo/
  app.py                  Streamlit demo (4 tabs)

claude_code_demo/         NEW: Claude Code integration demo
  CLAUDE.md               Instructions for epistemic-aware Claude Code
  payment_integration.py  Demo project with uncertain constraints
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

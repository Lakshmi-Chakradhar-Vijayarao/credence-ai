# Epistemic Memory

**Your AI doesn't forget things. It forgets *whether it was sure about them*.**

When an LLM conversation is compressed, the content survives but the confidence does not. A constraint expressed as "we're not certain about X" becomes "X" after one compression pass. A debugging hypothesis expressed as "possibly the race condition" becomes "the race condition" in the next agent's summary. Downstream turns — and downstream agents — then treat the uncertain claim as resolved knowledge.

Epistemic Memory is a context management layer that stops this. It conditions every memory allocation decision on the epistemic state of the content: **compress only what is epistemically resolved; preserve what is uncertain**.

---

## Install in Claude Desktop (2 minutes)

1. Install the package:
```bash
pip install fastmcp anthropic
git clone https://github.com/vijayarao-chakri/epistemic-memory
pip install -e epistemic-memory
```

2. Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "epistemic-memory": {
      "command": "python",
      "args": ["-m", "cams.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

3. Restart Claude Desktop. You now have 8 epistemic memory tools available.

---

## What it does

For every conversation turn, Epistemic Memory:

1. Computes a **J-score** (0–1) from the response text — 5 linguistic factors measuring assertiveness: hedging rate, anchor density, self-correction frequency, brevity, and specificity
2. Decides: **compress / trim / preserve** based on J-score thresholds
3. The critical rule: **only HIGH-J (epistemically resolved) turns are eligible for compression**. LOW/MEDIUM-J turns containing uncertain claims survive every compression and trim operation verbatim.
4. **Faithfulness probe**: before any Haiku summarisation, scans the old segment for 25+ uncertainty markers. If found → compression aborts → PRESERVE.

```
J ≥ 0.70  →  COMPRESS   (Haiku summarises; only HIGH-J eligible)
0.45–0.70 →  TRIM       (keep last N turns; LOW/MEDIUM-J always survive)
J < 0.45  →  PRESERVE   (full history retained)
```

---

## MCP Tools

| Tool | What it does |
|------|-------------|
| `cams_chat` | Send message, receive response + epistemic envelope |
| `em_propagation_risk` | Pre-flight risk assessment before compress/forward |
| `cams_inspect_envelope` | Trust analysis + BLOCK/VERIFY/PRESERVE/PROCEED recommendation |
| `cams_propagate_envelope` | Increment chain_depth, update source for agent handoffs |
| `cams_get_stats` | Token savings, compression counts |
| `cams_get_decision_log` | Per-turn J-scores and decisions |
| `cams_save` / `cams_load` | Cross-session continuity |
| `cams_reset` | Clear session |

### Using `em_propagation_risk`

Before compressing or forwarding any content:
```python
risk = em_propagation_risk(
    content="We think the latency spike might be GC pauses, but haven't confirmed.",
    chain_depth=1,
)
# → risk_level="HIGH", action="PRESERVE — content contains explicit uncertainty markers"
```

---

## Why this matters: the failure FAIL-CHAIN documented

FAIL-CHAIN (Vijayarao 2025) measured error propagation in multi-step LLM pipelines. The core finding: errors compound because **compression decisions are made without reference to epistemic state**. An uncertain output compresses to a confident-sounding summary. The next agent sees the summary. It has no record of the uncertainty. It answers with full confidence.

Epistemic Memory closes this loop. Experimental results:

| Experiment | Epistemic Memory | Naive window | Baseline |
|---|---|---|---|
| Flagship (3 scenarios × 3 trials, recall) | **0.669** [0.629, 0.709] | 0.593 [0.530, 0.659] | 0.660 |
| Flagship (chain complete) | **33%** | 0% | 22% |
| Flagship (propagation errors) | **0%** | 0% | 0% |
| E6: Uncertain constraint → 6 filler → callback | 100% recall, 0% hallucination | 0% recall, 50% hallucination | — |
| E7: 3-hop reasoning chain | 3/3 hops | 0/3 hops | — |
| E8: Debugging session, uncertain hypothesis | 1.000 recall | 0.522 recall | — |
| E4: vs random J routing (causal check) | 0.875 | 0.750 (random: 0.812) | 0.875 |
| 10-session benchmark (chain integrity) | 80% chain-complete | 20% chain-complete | 100% |

E4 confirms J-routing carries real signal above random: EM 0.875 > random_j 0.812 > naive 0.750.
Flagship: EM outperforms naive on recall (0.669 vs 0.593) and chain integrity (33% vs 0%). Zero propagation errors across all 27 scenario-trials.

---

## Signal architecture

Three tiers of epistemic signal:

```
Tier 1 — Linguistic Assertiveness (~0ms, $0)
  5 text pattern factors → J-score ∈ [0,1]
  Covers ~80% of the signal. Zero cost per turn.

Tier 2 — Behavioral Consistency (~300ms, ~$0.001/turn)
  N=5 Haiku samples → pairwise ROUGE-L variance
  Low variance = consistent answers = high confidence.
  Used for MEDIUM-zone turns. Opt-in.

Tier 3 — Fisher J from Activations (prior research, informs design)
  KV-cache attention entropy measured in prior work.
  Established that linguistic assertiveness correlates with
  internal model uncertainty. Validates Tier 1's design direction.
  Not required at runtime — Tier 1 + E4 causal check are sufficient.
```

**Model-agnostic**: the signal reads output text. It works regardless of which model produced the response.

---

## Guard rails

Three mechanisms prevent unsafe compression:

1. **Attention sink protection** — first 2 turns never compressed (conversation identity)
2. **Type Prior** — code blocks get J floor 0.30 (max J=0.64, MEDIUM zone max); error traces get floor 0.20; math gets floor 0.35
3. **Compression depth limit** — MAX_COMPRESSIONS=3; cumulative loss bounded

---

## Run the demo

```bash
streamlit run demo/app.py
```

Tab 1 shows the failure (naive window drops uncertain constraints → propagation error).
Tab 2 shows the fix (epistemic memory preserves uncertain content).
Tab 3 is a live chat with real-time J-gauge and decision log.
Tab 4 shows all benchmark results.

---

## Run the flagship experiment

```bash
# Smoke test (no API)
python -m experiments.flagship.run --dry-run

# 1 trial on Scenario A (API Integration)
python -m experiments.flagship.run --trials 1 --scenarios A

# Full experiment: 3 trials × 3 scenarios
python -m experiments.flagship.run --trials 3
```

---

## Run the evaluations

```bash
# J-proxy ceiling characterisation (no API)
python -m evals.nonhedged_test

# Adversarial tests (no API)
python -m evals.adversarial_tests

# Conversation benchmark (API)
python -m evals.conversation_benchmark

# E6 Negative Needle, E7 Multi-Hop, E8 Real Debugging
python -m evals.experiments --exp E6
python -m evals.experiments --exp E7
python -m evals.experiments --exp E8
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
See [CONTRIBUTION.md](CONTRIBUTION.md) for the contribution statement.

---

## Connection to prior work

This project connects two prior research threads:

1. **FAIL-CHAIN** — error propagation in multi-step LLM pipelines. Diagnosed the problem: confident-wrong outputs are produced when memory is epistemic-blind.

2. **Fisher J at the inference layer** — KV-cache attention entropy experiments on Qwen 3.5B showing that internal model uncertainty correlates with output linguistic patterns. Validates that Tier 1 (linguistic J-score) is a meaningful proxy for actual model uncertainty.

Epistemic Memory is the API-layer component: the memory governor that applies the insight from both threads without requiring access to model internals.

---

## Project structure

```
cams/
  confidence_proxy.py     J-score computation (Tier 1, zero-cost)
  context_manager.py      Core memory governor (compress/trim/preserve)
  behavioral_signal.py    Tier 2 behavioral consistency (N=5 Haiku)
  envelope.py             CAMSEnvelope — multi-agent provenance
  mcp_server.py           FastMCP server (8 tools)
  agent.py                Long-running task agent

experiments/
  flagship/               3-scenario × 3-condition flagship experiment
    scenarios.py          Scenarios A (API), B (Debugging), C (Design)
    metrics.py            propagation_rate, constraint_recall, uncertainty_preserved
    pipeline.py           EpistemicPipeline — baseline/naive/epistemic_memory
    run.py                CLI runner + bootstrap CI

evals/
  benchmark.py            30-pair QA benchmark (4 conditions)
  conversation_benchmark.py 10-session × 3-condition benchmark
  experiments.py          E1–E8 ablation experiments
  calibration.py          Threshold optimisation
  nonhedged_test.py       Proxy ceiling characterisation
  adversarial_tests.py    5 adversarial robustness tests

demo/
  app.py                  Streamlit demo (4 tabs)
```

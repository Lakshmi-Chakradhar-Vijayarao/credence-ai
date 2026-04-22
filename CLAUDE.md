# CAMS — Claude Code Context Guide

This file gives Claude Code a full structural map of the codebase so it does not need to read source files to answer structural questions.

## Project in One Paragraph

CAMS (Confidence-Adaptive Memory System) is a context memory governor for Claude Opus 4.7. It extracts a J-score (0–1) from each response using 5 linguistic factors, then decides whether to compress (Haiku summarises old turns), trim (keep last N turns), or preserve (keep everything). The J-score is a language-level proxy for Fisher Information — measuring whether the model's response carries resolved or uncertain epistemic state. A dual-signal fusion also uses Opus extended thinking token utilisation to detect cognitive friction. Four guard rails prevent unsafe compression: attention sink protection, Type Prior (content-type J caps), compression depth limit, and novelty guard.

---

## File Map

### `cams/confidence_proxy.py` — 235 lines
Pure Python, no API calls. Computes J-score from response text.

**Key constants (module level)**:
- `_CODE_FLOOR = 0.30` — code blocks: J cap = 0.30 + 0.34 = 0.64
- `_ERROR_FLOOR = 0.20` — error traces: J cap = 0.20 + 0.34 = 0.54
- `_MATH_FLOOR = 0.35` — math: J cap = 0.35 + 0.34 = 0.69 (uncapped in practice)

**Classes**:
- `ConfidenceResult` (dataclass, line 75): `j_score, zone, factors, reasoning, content_type`
  - `.should_compress` → bool (zone == "HIGH")
  - `.color` → str ("green"/"yellow"/"red")
  - `.emoji` → str
- `ConfidenceProxy` (line 99):
  - `__init__(theta_high=0.65, theta_low=0.35)`
  - `_detect_content_type(text) → (type_str, j_floor)` — detects code/error/math
  - `compute(response_text) → ConfidenceResult` — main entry point
  - `batch(texts) → list[ConfidenceResult]`

**J formula**:
```
J = 0.30*(1−hedging_rate) + 0.25*anchor_rate + 0.20*(1−correction_rate)
  + 0.10*brevity_score + 0.15*specificity_score
```
If content type detected: `j_score = min(j_raw, j_floor + 0.34)`

**Zone thresholds**: HIGH ≥ 0.65 → COMPRESS | MEDIUM [0.35, 0.65) → TRIM | LOW < 0.35 → PRESERVE

---

### `cams/context_manager.py` — 506 lines
Core system. Manages conversation history and all compression decisions.

**Module-level constants**:
- `_MODEL_OPUS = "claude-opus-4-7"`
- `_MODEL_HAIKU = "claude-haiku-4-5-20251001"`
- `_NOVELTY_THRESHOLD = 0.60`

**Session constants** (class-level on `CAMSContextManager`):
- `COMPRESS_AFTER = 6` — turns before compression eligibility
- `TRIM_WINDOW = 10` — turns to keep on TRIM
- `ATTENTION_SINK = 2` — first N turns never compressed
- `MAX_COMPRESSIONS = 3` — max recursive compressions

**Classes**:
- `TurnResult` (dataclass, line 85): `response, j_score, zone, decision, tokens_saved, tokens_used, cost_usd, thinking_tokens, thinking_utilization, content_type, factors`
- `SessionStats` (dataclass, line 108): `total_tokens_in, total_tokens_out, total_tokens_saved, total_cost_usd, turn_count, compression_count, trim_count, preserve_count`
  - `.compression_ratio` → float
- `CAMSContextManager` (line 131):
  - `__init__(theta_high, theta_low, use_thinking, system_prompt, goal_line)`
  - `chat(user_message) → TurnResult` — **main public API**
  - `reset()` — clear history and stats
  - `decision_log` → list[dict] — per-turn decisions
  - `_apply_cams(cr, novelty_override) → (decision, tokens_saved)` — decides compress/trim/preserve
  - `_compress() → int` — calls Haiku, rebuilds history as sink + compressed_summary + recent
  - `_trim() → int` — drops history[:-TRIM_WINDOW*2]
  - `_build_messages() → list[dict]` — injects `<context_summary>` XML prefix on first message
  - `_check_novelty(text) → bool` — >60% new named entities
  - `_update_entity_vocab(text)` — maintains entity set
  - `_extract_entities(text) → set[str]` — capitalised word extractor

**Module-level helpers**:
- `_cost(input_tokens, output_tokens) → float` — Opus pricing ($15/$75 per 1M)
- `_cost_haiku(input_tokens, output_tokens) → float` — Haiku pricing ($0.80/$4 per 1M)

**Adaptive thinking logic** (inside `chat()`):
```python
if use_thinking and self._prev_zone == "LOW":
    call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}
```

**Dual-signal override** (inside `chat()`):
```python
if thinking_utilization > 0.50 and cr.zone == "HIGH":
    cr = dataclasses.replace(cr, zone="MEDIUM")  # downgrade
```

**Context injection pattern**: `<context_summary>{summary}</context_summary>` prepended to first user message (not fake assistant turn, not system prompt).

---

### `cams/agent.py` — 293 lines
Long-running task agent built on top of `CAMSContextManager`.

**Classes**:
- `SubTaskResult` (dataclass, line 40): `question, answer, tokens_used, tokens_saved, cost_usd, j_score`
- `AgentResult` (dataclass, line 53): `sub_results, final_report, total_tokens, total_saved, total_cost`
  - `.summary` → formatted string
- `CAMSAgent` (line 80):
  - `__init__(on_turn: Callable | None)` — optional progress callback
  - `document_qa(document, questions) → AgentResult` — chunks doc, primes context, answers each question, synthesises
  - `research(topic, questions, background) → AgentResult` — iterative research loop
  - `_chunk(text, chars_per_chunk=4000) → list[str]` — paragraph-respecting splitter

---

### `evals/benchmark.py` — 618 lines
Three-condition benchmark. Runs 30 QA pairs (10 factual / 10 reasoning / 10 uncertain).

**Key functions**:
- `rouge_l(hypothesis, reference) → float` — LCS-based F1, no external deps
- `compute_auarc(turns) → float` — trapezoid area over J-sorted abstention thresholds
- `reasoning_density(mean_rouge_l, total_cost_usd) → float` — ROUGE-L per $0.001
- `run_cams(pairs, client) → ConditionResult`
- `run_baseline(pairs, client) → ConditionResult`
- `run_naive_window(pairs, client, window=6) → ConditionResult`
- `print_table(results)` — formatted terminal output
- `save_results(results, path="evals/results.json")`
- `main()` — entry point

**Classes**:
- `TurnLog` (dataclass, line 329): `question, reference, hypothesis, j_score, rouge, domain`
- `ConditionResult` (dataclass, line 343): `name, turns, total_tokens, tokens_saved, cost_usd`

**Benchmark results** (from last run):
```
Baseline:     121,969 tokens / $2.31 / ROUGE-L 0.137 / AUARC 0.174
Naive window:  52,444 tokens / $1.25 / ROUGE-L 0.144 / AUARC 0.171
CAMS:          89,633 tokens / $1.74 / ROUGE-L 0.224 / AUARC 0.285
```

---

### `evals/calibration.py` — 202 lines
Grid-searches optimal thresholds from labelled data.

**Key functions**:
- `calibrate(items) → dict` — grid over theta_high ∈ [0.50, 0.85], theta_low ∈ [0.15, theta_high−0.10]
- `main()` — `--api` flag fetches live Claude responses

---

### `demo/app.py` — 735 lines
Streamlit UI with 3 tabs.

**Key globals**:
- `LIVE_MODE = bool(os.environ.get("ANTHROPIC_API_KEY", ""))` — graceful degradation
- `DEMO_TURNS` — pre-computed turns for demo-without-API-key mode

**Tab structure**:
- Tab 1: Live Chat — J-gauge, zone badge, decision log, session stats, thinking utilisation bar
- Tab 2: Benchmark — bar charts from `evals/results.json`
- Tab 3: Research/Evidence — calibration data, per-turn analysis

**Key helpers**:
- `_init_state()` — initialises `st.session_state`
- `_j_color(zone) → str`
- `_decision_label(decision) → str`
- `_format_usd(v) → str`

---

## Data Flow

```
User message
    → CAMSContextManager.chat()
        → _check_novelty()  [named entity comparison]
        → anthropic.messages.create() with Opus 4.7
            [optional: thinking budget if prev_zone == LOW]
        → ConfidenceProxy.compute(response_text)  [pure text, no API]
        → dual-signal check  [thinking_utilization > 0.50 + zone == HIGH → downgrade]
        → _apply_cams(cr, novelty_override)  [decide compress/trim/preserve]
            → _compress()  [calls Haiku, rebuilds history]
            → _trim()      [slices history list]
            → no-op        [PRESERVE]
        → _update_entity_vocab()
        → return TurnResult
```

---

## What Was Added (April 22, 2026 session)

- **Continuous thinking budget governor** (`_THINKING_MIN=500`, `_THINKING_MAX=2000`): thinking budget now scales continuously by `(theta_high - prev_j) / theta_high`. `_prev_j` tracked per turn.
- **Drift detector**: `_j_history` (rolling 5-turn list) + `_drift_state` bool. 3 consecutive LOW turns → `_drift_state=True` → `_apply_cams` returns PRESERVE.
- **TurnResult new fields**: `thinking_budget_used: int`, `drift_state: bool`
- **Demo**: drift badge on zone label, thinking budget value shown alongside utilization, drift marker in decision log
- **`evals/experiments.py`**: 5 ablation experiments (E1–E5); run with `python -m evals.experiments --exp E1`
- **`SUBMISSION.md`**: 200-word submission summary for hackathon platform
- **`README.md`**: research narrative restored, Fisher J framing, FAIL-CHAIN connection, dual-signal section, engineering details
- **`CLAUDE.md`**: this file — full structural map

---

## Key Design Decisions (Non-Obvious)

- **XML injection, not fake turn**: `<context_summary>` goes in first user message content, not as `{"role":"assistant"}` fake turn — fake turns confuse Opus's understanding of conversation state
- **Haiku for compression, Opus for answers**: compression is ~95% cheaper per token with Haiku; quality of summary is sufficient because CAMS only compresses HIGH-J (already-resolved) context
- **Type Prior cap formula**: `j_cap = j_floor + 0.34` — 0.34 is the band from the floor to MEDIUM ceiling (0.64 for code, 0.54 for errors) — ensures content type keeps zone at MEDIUM maximum
- **ATTENTION_SINK = 2**: first 2 turns establish conversation identity; losing them causes topic drift even if their content is HIGH-J
- **MAX_COMPRESSIONS = 3**: each Haiku compression is ~85% lossless; after 3 passes the cumulative loss becomes noticeable; tested empirically
- **NOVELTY_THRESHOLD = 0.60**: calibrated to avoid false positives on vocabulary-rich answers while catching genuine topic pivots

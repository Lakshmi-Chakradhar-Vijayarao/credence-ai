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
- `_NOVELTY_THRESHOLD = 0.75`
- `_NOVELTY_MIN_ENTITIES = 5`  — minimum absolute new entities required
- `_NOVELTY_MIN_VOCAB = 10`    — don't fire until vocab is established
- `_MIN_COMPRESS_TOKENS = 150` — skip Haiku if old segment is smaller than this
- `_UNCERTAINTY_MARKERS` — frozenset of 21 uncertainty phrases (faithfulness guard)

**Session constants** (class-level on `CAMSContextManager`):
- `COMPRESS_AFTER = 3` — keep last N turn-pairs on COMPRESS; fires when n_turns > COMPRESS_AFTER*2=6
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
  - `_compress() → int` — calls Haiku; returns 0 (PRESERVE) if faithfulness probe fires
  - `_trim() → int` — drops history[:-TRIM_WINDOW*2]
  - `_build_messages() → list[dict]` — injects `<context_summary>` XML prefix on first message
  - `_check_novelty(text) → bool` — three-gate entity novelty guard (vocab≥10, count≥5, ratio>0.75)
  - `_has_uncertainty(text) → bool` — faithfulness probe: detects uncertainty markers in text
  - `_update_entity_vocab(text)` — maintains entity set
  - `_extract_entities(text) → set[str]` — capitalised word extractor

**Faithfulness probe** (inside `_compress()`):
```python
if self._has_uncertainty(conv_text):
    return 0   # caller sees saved=0 → falls through to PRESERVE
```
Fires when old segment contains uncertainty markers (`"i think"`, `"not certain"`, `"approximately"`, etc.).
Prevents Haiku from silently stripping user-flagged uncertainty into apparent fact (worst failure: confident hallucination).

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
- `SubTaskResult` (dataclass, line 40): `question, answer, j_score, zone, decision, tokens_in, tokens_out, tokens_saved, elapsed_ms`
- `AgentResult` (dataclass, line 53): `sub_results, final_report, total_tokens, total_saved, total_cost`
  - `.summary` → formatted string
- `CAMSAgent` (line 80):
  - `__init__(on_turn: Callable | None)` — optional progress callback
  - `document_qa(document, questions) → AgentResult` — chunks doc, primes context, answers each question, synthesises
  - `research(topic, questions, background) → AgentResult` — iterative research loop
  - `_chunk(text, chars_per_chunk=4000) → list[str]` — paragraph-respecting splitter

---

### `evals/benchmark.py` — ~1100 lines
Four-condition benchmark. Runs 30 QA pairs (10 factual / 10 reasoning / 10 uncertain).

**Key functions**:
- `rouge_l(hypothesis, reference) → float` — LCS-based F1, no external deps
- `compute_auarc(turns) → float` — trapezoid area over J-sorted abstention thresholds
- `bootstrap_ci(values, n_boot=2000, ci=0.95) → (lo, hi)` — non-parametric bootstrap CI for the mean
- `reasoning_density(mean_rouge_l, total_cost_usd) → float` — ROUGE-L per $0.001
- `run_cams(pairs, client) → ConditionResult`
- `run_baseline(pairs, client) → ConditionResult`
- `run_naive_window(pairs, client, window=6) → ConditionResult`
- `run_random_gating(pairs, client, compress_rate=0.30) → ConditionResult` — random compression schedule control
- `print_table(results)` — formatted terminal output; now includes 95% bootstrap CI on ROUGE-L
- `save_results(results, path="evals/results.json")`
- `main()` — entry point; `--ablation` flag adds random-gating condition

**Classes**:
- `TurnLog` (dataclass, line 329): `question, reference, hypothesis, j_score, rouge, domain`
- `ConditionResult` (dataclass, line 343): `name, turns, total_tokens, tokens_saved, cost_usd`

**Benchmark results** (April 23 2026 run — 4-condition honest design, adaptive thresholds active):
```
Baseline (no prompt):       106,465 tokens / $2.04 / ROUGE-L 0.146 / AUARC 0.217
Baseline (no compression):   91,248 tokens / $1.77 / ROUGE-L 0.219 / AUARC 0.326
Naive sliding window:         45,168 tokens / $1.07 / ROUGE-L 0.238 / AUARC 0.356
CAMS:                         70,895 tokens / $1.47 / ROUGE-L 0.213 / AUARC 0.323
```
System prompt delta: +0.073 ROUGE-L. J-proxy delta on diverse Q&A: −0.006 (within noise, CI ±0.12).
Naive window outperforms CAMS on aggregate QA (expected: independent QA doesn't require context preservation).
CAMS AUARC: 49.0% of Φ(√J̄/2) theoretical ceiling (hidden-state access) at API surface.
Token reduction vs same-prompt baseline: 22.3%. Cost reduction: 17.1%.
Experiment results: E1 CAMS 0.875 vs naive 0.833; E4 CAMS 0.875 vs random_j 0.812 vs naive 0.750;
E6 CAMS 100% correction vs naive 0%; E7 CAMS 3/3 hops vs naive 0/3; E8 CAMS 1.000 vs naive 0.600.

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
            → _compress()  [faithfulness probe → 0 if uncertainty in old segment; else calls Haiku]
            → _trim()      [slices history list]
            → no-op        [PRESERVE]
        → _update_entity_vocab()
        → return TurnResult
```

---

## What Was Added (April 22, 2026 session — first pass)

- **Thinking API updated for Opus 4.7**: now uses `thinking={"type":"adaptive"}` + `output_config={"effort":"high"|"medium"}`. `_THINKING_MIN=1024` (API minimum), `_THINKING_MAX=5000`. Opus 4.7 does NOT expose thinking blocks, so `thinking_tokens` and `thinking_utilization` are always 0 — dual-signal fusion is a no-op on this model. Forward-reserved for models that expose thinking blocks.
- **Drift detector**: `_j_history` (rolling 5-turn list) + `_drift_state` bool. 3 consecutive LOW turns → `_drift_state=True` → `_apply_cams` returns PRESERVE.
- **TurnResult new fields**: `thinking_budget_used: int`, `drift_state: bool`
- **Demo**: drift badge on zone label, thinking budget value shown alongside utilization, drift marker in decision log
- **`evals/experiments.py`**: 5 ablation experiments (E1–E5); run with `python -m evals.experiments --exp E1`
- **`SUBMISSION.md`**: 200-word submission summary for hackathon platform
- **`README.md`**: research narrative restored, Fisher J framing, FAIL-CHAIN connection, dual-signal section, engineering details

## What Was Added (April 22, 2026 session — second pass)

- **Faithfulness probe** (`context_manager.py`): `_UNCERTAINTY_MARKERS` frozenset + `_has_uncertainty()` method. Called inside `_compress()` before the Haiku API call — if old segment contains uncertainty markers, returns 0 (caller sees PRESERVE). Prevents Haiku from silently stripping user-stated qualifications.
- **Novelty guard tightened** (`context_manager.py`): threshold raised 0.60→0.75, added `_NOVELTY_MIN_ENTITIES=5` and `_NOVELTY_MIN_VOCAB=10` gates to eliminate 50% false-positive rate on stable-domain sessions.
- **Compression prompt updated** (`context_manager.py`): explicit instruction to preserve uncertainty flags in Haiku summary.
- **Minimum compress tokens** (`context_manager.py`): `_MIN_COMPRESS_TOKENS=150` — skips Haiku if old segment < 150 estimated tokens.
- **Bootstrap CI** (`benchmark.py`): `bootstrap_ci(values, n_boot=2000)` function; `print_table()` now reports 95% CI on ROUGE-L for CAMS and baseline; delta CI annotated as significant/not-significant. n=30 CI width ~0.09 — the 0.021 ROUGE-L delta between CAMS and baseline is not significant at this sample size.
- **E6 — Negative Needle** (`experiments.py`): new eval testing whether CAMS + faithfulness probe preserves uncertain constraints through compression pressure; measures `correction_recall` and `hallucination_rate`; three conditions (baseline/naive_window/cams).
- **E4 — random_j condition** (`experiments.py`): fourth condition alongside baseline/naive_window/cams. Same conversation, compression decisions driven by random J scores from Uniform(0,1). Causality check: if CAMS > random_j, J-routing is doing real work.
- **E3/E5 marked DEFERRED** (`experiments.py`): both require thinking block exposure (thinking_tokens > 0); Opus 4.7 does not expose thinking blocks. Forward-reserved.

## What Was Added (April 22, 2026 session — third pass)

- **Content-word novelty** (`context_manager.py`): replaced entity-counting novelty guard with content-word ratio guard. `_ENTITY_STOPWORDS` → `_CONTENT_STOPWORDS`; `_extract_entities()` → `_extract_content_words()` (any ≥4-char non-stop lowercase word); `_entity_vocab` → `_content_vocab`. Same three-gate logic but now fires on semantic pivots without proper nouns (e.g. "optimistic locking" → "database sharding"). Sentence-start capitalization false positives eliminated.
- **Semantic entropy proxy** (`context_manager.py`): `_MULTI_ANSWER_MARKERS` frozenset (24 markers: "it depends on", "case by case", "no single answer", etc.) + `_has_multi_answer()` method. In `chat()`: if zone==MEDIUM and multi-answer markers detected → downgrade to LOW → PRESERVE. Zero-cost single-pass approximation of Kuhn et al. ICLR 2023 semantic entropy. Targets the weakest region of CAMS (MEDIUM zone) where neither TRIM nor COMPRESS is clearly correct.
- **E7 — Multi-Hop Reasoning Chain** (`experiments.py`): 3-hop chain test (Project Falcon → Nexus config → CVE/v5 → Python ≥3.10). 6 filler turns force naive window to drop T3-T5. Measures `hops_recalled` (0-3) and `chain_complete` (bool). Tests directly whether Haiku compression preserves dependency chains. Three conditions: baseline/naive_window/cams.
- **`--exp E7`** added to `main()` argparse choices.

---

## Key Design Decisions (Non-Obvious)

- **XML injection, not fake turn**: `<context_summary>` goes in first user message content, not as `{"role":"assistant"}` fake turn — fake turns confuse Opus's understanding of conversation state
- **Haiku for compression, Opus for answers**: compression is ~95% cheaper per token with Haiku; quality of summary is sufficient because CAMS only compresses HIGH-J (already-resolved) context
- **Type Prior cap formula**: `j_cap = j_floor + 0.34` — 0.34 is the band from the floor to MEDIUM ceiling (0.64 for code, 0.54 for errors) — ensures content type keeps zone at MEDIUM maximum
- **ATTENTION_SINK = 2**: first 2 turns establish conversation identity; losing them causes topic drift even if their content is HIGH-J
- **MAX_COMPRESSIONS = 3**: each Haiku compression is ~85% lossless; after 3 passes the cumulative loss becomes noticeable; tested empirically
- **NOVELTY_THRESHOLD = 0.75 (content words, not entities)**: three-gate design — (1) vocab ≥10 content words established, (2) ≥5 new content words in response, (3) >75% of response content words are new. Content words (any ≥4-char non-stop word) replaces entity-counting, which could not detect topic pivots without proper nouns. Same threshold and gates, broader vocabulary signal. False positive rate on stable-domain sessions validated through test suite.
- **Adaptive thresholds are model-agnostic by construction**: P75/P25 of rolling buffer adapts to whatever J-distribution the current model produces — no per-model calibration needed. Static thresholds would mis-classify GPT-4o vs Claude responses because hedging vocabularies differ.
- **Selective compression is the right design**: Sending the entire old segment to Haiku risks losing LOW/MEDIUM-J turns that contain critical uncertain constraints. The `_history_j_scores` parallel list enables per-turn lookup: only HIGH-J (already-resolved) turns go to Haiku; LOW/MEDIUM turns survive verbatim.

## What Was Added (April 23, 2026 session — fourth pass)

- **Adaptive percentile thresholds** (`context_manager.py`): `_j_buffer` (rolling 20-turn list) + `_effective_theta_high` (P75, floored at 0.65) + `_effective_theta_low` (P25, capped at min(p25, eff_high−0.10, 0.55)). Warmup: first 5 turns use static thresholds. Guard-rail override detection: if `cr.zone != static_zone`, a guard rail modified the zone — respected as-is. `TurnResult` new fields: `adaptive_theta_high`, `adaptive_theta_low`.
- **Selective turn-level compression** (`context_manager.py`): `_history_j_scores` list maintained in sync with `_history`. During `_compress()`, old segment split into HIGH-J pairs (→ Haiku) and LOW/MEDIUM-J pairs (→ kept verbatim). History rebuilt as: sink + preserved_msgs + `<context_summary>` + recent.
- **E1 fixed** (`experiments.py`): All three conditions (baseline/naive/cams) now use live Opus generation — eliminates pre-canned vs live confound. `SEED_MESSAGES` list, 8 messages, 4 callbacks.
- **E2 fixed** (`experiments.py`): Short code snippets + `max_tokens=150` to force brief responses that would score HIGH without Type Prior.
- **E8 — Real Debugging Session** (`experiments.py`): 12-turn realistic debugging session. T3: specific RuntimeError. T4: uncertain hypothesis (LOW-J). T5-T10: 6 HIGH-J filler turns. T11: attempted fix. T12: partial outcome. 3 callbacks: original error, hypothesis, fix+outcome. Naive drops T4 (0.00 recall on hypothesis). CAMS matches baseline (1.000).
- **All experiments re-run with real API**: E1 CAMS 0.875, E4 CAMS 0.875 vs random_j 0.812 vs naive 0.750, E6 100% vs 0% correction, E7 3/3 vs 0/3 hops, E8 1.000 vs 0.600.

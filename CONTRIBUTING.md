# Contributing to Credence

All offline tests pass without an API key. You can contribute to the probe, registry, GTS, and Rust gate without spending anything.

## Setup

```bash
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
cd credence-ai
pip install -e ".[dev,mcp]"
```

## Running Tests

```bash
# Full test suite — no API key needed
pytest tests/ -q                      # 851 tests

# Smoke test — verifies install and enforcement path
credence demo

# Adversarial robustness (no API)
python -m evals.adversarial_tests

# Probe latency and precision (no API)
python -m evals.stress_test
python -m evals.precision_eval
```

## Project Layout

```
credence/context_manager.py   — compression, Truth Buffer, GTS, CE
credence/registry.py          — SQLite constraint store + decay
credence/mcp_server.py        — 17-tool MCP server
credence/observer.py          — UserPromptSubmit hook
credence/hooks.py             — PreToolUse enforcement gate
credence/memory.py            — cross-session persistence
credence_gate/src/main.rs     — Rust PreToolUse hook (faster alternative)
tests/                        — 851 tests (pytest)
evals/                        — validation studies (some require API key)
```

## What's Offline vs API-Required

**Fully offline (no API key):**
- Faithfulness probe (`_has_uncertainty`, `_has_uncertainty_in_user_turns`)
- J-score computation (`CredenceProxy.compute`)
- Registry operations (register, verify, decay, trajectory)
- Generation-Time Scanner (`_scan_output_for_constraints`)
- Consistency Enforcer matching (`_direct_constraint_matches`)
- Cross-session memory (snapshot, recall)
- All 851 tests (`pytest tests/ -q`)

**Requires API key:**
- `ContextManager.chat()` (calls Opus 4.7)
- Compression via Haiku
- Ghost Detector classification
- Behavioral consistency probe (Tier 2)
- All `evals/` studies that call the API

## Where to Contribute

**High-value, no API needed:**
- Expand `_UNCERTAINTY_MARKERS` in `context_manager.py` (currently 108 terms) — add domain-specific hedging language for medical, legal, financial domains
- Expand `_CE_DOMAIN_SYNONYMS` in `context_manager.py` — add synonym clusters for new domains
- Improve GTS prose scanning — better sentence boundary detection
- Rust gate performance improvements
- New unit tests for edge cases

**Requires API key:**
- New eval scenarios for `evals/gauntlet.py` (50 scenarios, always room for more domains)
- Ghost constraint sessions for `evals/claim_gauntlet.py`

## Pull Request Guidelines

- All tests must pass: `pytest tests/ -q`
- `credence demo` must run clean
- If you change decay rates in `registry.py`, update the corresponding S2 tests in `tests/tests.py`
- If you add uncertainty markers to `_UNCERTAINTY_MARKERS`, add a test case to the S22 suite
- Keep the faithfulness probe scanning user turns only (not assistant code blocks) — see the comment block above `_has_uncertainty_in_user_turns()` for why
- No new dependencies without discussion — the core package intentionally has only `anthropic` as a hard dependency

## Reporting Issues

Use the GitHub issue templates. Include:
- Which layer failed (probe / Truth Buffer / GTS / Rust gate / memory)
- The uncertain constraint text that triggered (or didn't trigger) the behavior
- Python version and `pip show credence-guard` output

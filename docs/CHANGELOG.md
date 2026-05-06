# Changelog

All notable changes to credence-guard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.5] — 2026-05-06

### Changed
- **`fastmcp` is now a hard dependency** — install command is `pip install credence-guard` (no `[mcp]` extra required). The MCP server is the primary interface; the extra was unnecessary friction.
- **MCP registry**: fixed `server.json` name casing (`io.github.Lakshmi-Chakradhar-Vijayarao/credence`) to match GitHub username.

---

## [1.2.4] — 2026-05-06

### Added
- `credence --version` / `credence -V` flag — prints installed package version.
- 6 missing tools added to CLAUDE.md quick reference: `credence_session_summary`, `credence_project_status`, `credence_scan_ghosts`, `credence_audit`, `credence_diff`, `credence_reset`.

### Fixed
- SECURITY.md: corrected "no per-session rate limit" claim — cap was already implemented at 500/session (`CREDENCE_MAX_CONSTRAINTS`).
- README: documented `CREDENCE_MAX_CONSTRAINTS` env var and `credence stats` / `credence feedback` CLI commands.

---

## [1.2.3] — 2026-05-06

### Fixed
- Added `anthropic` to `[dev]` extras so `test_mock_llm.py` integration tests can instantiate `ContextManager` without requiring a separate `pip install anthropic`.
- Added MCP registry ownership token to README for `registry.modelcontextprotocol.io` submission.

---

## [1.2.2] — 2026-05-06

### Fixed
- PyPI publish pipeline: removed broken OIDC trusted-publishing config (`environment: pypi`), switched to `PYPI_API_TOKEN` secret. v1.1.0 and v1.2.0 were published to PyPI for the first time.
- `CREDENCE_NO_LOG=1` opt-out added to `hooks.py` gate event log.
- Data storage documented in README and SECURITY.md (`epistemic_registry.db` + `~/.credence/events.jsonl`).

---

## [1.2.0] — 2026-05-06

### Changed
- **Observer two-tier marker architecture** — strong markers fire unconditionally; weak markers (`around`, `seems like`, `i guess`, `docs say`) now require a co-present numeric value before registering. Eliminates false positives like "wrap around the list" while adding `_NUMERIC_RE` fix (`\b` removed) to catch unit-glued numbers: `30s`, `5MB`, `50ms`.
- **Observer detection coverage**: 59% → 95% on a 22-phrase probe; false positive rate: 50% → 8%.
- **`TEMPORAL_J_SCORES`** extracted to `credence/temporal_patterns.py` — single source of truth, imported by both `mcp_server.py` and `__main__.py`.
- **Gate and scan display**: internal `[stale:…]` and `[AI-generated:…]` DB prefixes stripped at all user-facing output points.

### Fixed
- All 829 tests passing, 1 skipped. (22 tests removed: `test_enforce.py` and `test_manifest.py` covered `credence/enforce.py` and `credence/epistemic_manifest.py`, which were relocated to `experimental/`.)
- `CREDENCE_DB_PATH` → `CREDENCE_DB` in `credence/__main__.py` (canonical env var).
- Ruff: all 20 lint errors resolved (16 auto-fixed, 4 manual E701).

### Removed
- `evals/fcr_downstream_results.json` — v2 scorer (incorrect; contradicted canonical v3 result).
- `evals/compression_faithfulness_results_groq.json`, `_hf.json` — superseded by `compression_faithfulness_n50_results.json`.
- `evals/eqlr_compressor_results.json`, `evals/experiment_results.json` — superseded.
- `evals/data/`, `evals/training/` — DPO training pipeline (dormant; data on HuggingFace).
- `docs/LAUNCH.md`, `docs/MCP_REGISTRY_SUBMISSION.md` — internal process documents.
- `sdk/typescript/` → relocated to `experimental/typescript/` (Phase 2, not yet shipped).
- `credence/enforce.py`, `credence/epistemic_manifest.py` → relocated to `experimental/`.

### Added
- `paper/PAPER_DRAFT.md` — full arXiv-style research paper draft.
- `paper/figures/` — 6 publication-ready figures (PDF + PNG, reproducible generation scripts).
- `experimental/` — home for Phase 2 unshipped work with explicit README.
- `docs/README.md` — navigation index for the docs/ directory.

---

## [1.1.0] — 2026-05-06

### Added
- **Observer hook** (`credence/observer.py`) — passive `UserPromptSubmit` hook. Registers uncertain values before the model generates a single token. Zero API, zero config.
- **`credence_session_summary`** — plain-English digest of all unverified constraints for a session; structured for agent handoffs.
- **`credence_diff`** — detects epistemic contradictions between two texts/agent responses. Returns matched claims, contradictions, divergence score.
- **`credence_project_status`** — project-wide epistemic health across all sessions: `CLEAN / LOW_DEBT / MEDIUM_DEBT / HIGH_DEBT`.
- **`credence_scan_ghosts`** — flags constraints that match ghost heuristics (numeric + domain keyword, no documentation reference).
- **`credence_marker_health`** — diagnostic for marker precision; returns "insufficient data" until usage thresholds are met.
- **`credence_bandit_status`** — shows adaptive threshold learning state; returns "learning" below activation threshold.
- **`credence demo`** CLI entry point — 30-second smoke test, no API key required.

### Fixed
- All 851 tests passing, 0 failures.
- Stale count references corrected across all documentation.

---

## [1.0.0] — 2026-05-02

### Added
- **Faithfulness probe** — deterministic uncertainty-marker detection (0.017ms P50, 0% FCR). Blocks compression when uncertainty qualifiers are present in the segment being compressed.
- **11-tool MCP server** (`credence-server`) — zero API key, zero config. Drop-in Claude Code integration via `.mcp.json`.
- **Rust PreToolUse gate** (`credence-gate`) — 3.4ms native binary, 98× faster than Python hook. Blocks irreversible tool calls when unverified constraints overlap the planned action.
- **Generation-Time Scanner (GTS)** — annotates unverified numeric literals inline in generated code and prose before they ship.
- **Consistency Enforcer** — fires imperative enforcement when a user query keyword-overlaps a registered unverified constraint (≥2 non-stopword terms, synonym-expanded).
- **Truth Buffer** — injects all unverified constraints as epistemic context before every generation turn.
- **CredenceRegistry** — SQLite-backed constraint store with trajectory tracking, confidence decay, per-type decay rates, and cross-session memory.
- **CredenceMemory** — cross-session epistemic memory (`snapshot` → `recall_and_inject`).
- **`wrap()` API** — model-agnostic faithfulness wrapper for any `Callable[[str], str]` compression function.
- **`measure_fcr()`** — offline False Certainty Rate measurement utility.
- **`enforce()` / `CredenceViolation`** — decorator-based enforcement for functions that consume uncertain values.
- **EpistemicManifest** — session-level epistemic health summary.
- **EQL Benchmark** (n=50) — compression faithfulness study with confidence intervals. Haiku: 26% FCR → 0%. LLMLingua-sim: 70% FCR → 0%.
- **Epistemic Transport Protocol** spec (`docs/ETP_SPEC.md`, `etp_schema.json`).
- **Latency report** — all enforcement checkpoints measured. Worst-case P99 overhead: ~5.2ms (~0.10% of a typical Claude Opus API call).

### Architecture
- Zero API key required for all enforcement operations.
- All enforcement layers are deterministic string operations — no model calls at enforcement time.
- `CREDENCE_DB_PATH` / `CREDENCE_REGISTRY_PATH` env vars configure registry location (Rust gate and Python server both respect these).

---

## [Unreleased]

### Planned
- Pre-built `credence-gate` binary as PyPI platform wheel (removes `cargo build` requirement).
- GitHub Actions `credence-check` integration.
- ETP adoption in external agent frameworks.

### Already shipped (not yet tagged)
- Per-session constraint cap: 500 constraints/session default, override with `CREDENCE_MAX_CONSTRAINTS`. Implemented in `registry.py:register()`.

# Changelog

All notable changes to credence-guard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Per-session constraint cap (DB spam guard).
- GitHub Actions `credence-check` integration.
- ETP adoption in external agent frameworks.

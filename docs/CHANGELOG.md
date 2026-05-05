# Changelog

All notable changes to credence-guard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-05-02

### Added
- **Faithfulness probe** — deterministic uncertainty-marker detection (0.017ms P50, 0% FCR). Blocks compression when uncertainty qualifiers are present in the segment being compressed.
- **22-tool MCP server** (`credence-server`) — zero API key, zero config. Drop-in Claude Code integration via `.mcp.json`.
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
- **TypeScript SDK** (`sdk/typescript/`) — `runProbe()`, `CredenceEnvelope`, `CredenceRegistryClient`.
- **EQL Benchmark** (n=50) — compression faithfulness study with confidence intervals. Haiku: 26% FCR → 0%. LLMLingua-sim: 70% FCR → 0%.
- **Epistemic Transport Protocol** spec (`docs/ETP_SPEC.md`, `etp_schema.json`).
- **596 tests** across unit, integration, security, and performance suites.
- **Latency report** — all enforcement checkpoints measured. Worst-case P99 overhead: ~5.2ms (~0.10% of a typical Claude Opus API call).

### Architecture
- Zero API key required for all enforcement operations.
- All five enforcement layers are deterministic string operations — no model calls at enforcement time.
- MCP server exposes 22 tools and 2 resources under the `epistemic://` URI scheme.
- `CREDENCE_DB_PATH` / `CREDENCE_REGISTRY_PATH` env vars configure registry location (Rust gate and Python server both respect these).

---

## [Unreleased]

### Planned
- Pre-built `credence-gate` binary distributed as PyPI platform wheel (removes `cargo build` requirement).
- Per-session constraint cap (DB spam guard).
- GitHub Actions `credence-check` integration.
- TypeScript SDK HTTP client (`CredenceRegistryClient` implementation).
- ETP adoption in external agent frameworks.

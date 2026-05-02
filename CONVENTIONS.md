# Credence — Naming Conventions

## Python modules

Use **noun form** — the name describes what the module *is*, not what it does.

```
registry.py       ✓   enforce.py        ✗  →  enforcer.py
envelope.py       ✓   wrap.py           ✗  →  wrapper.py
context_manager.py ✓
```

Exception: modules that export a single eponymous function keep the verb name because
`credence.enforce(...)` and `credence.wrap(...)` read as the API call, not a module path.

## Python classes

`Credence<Noun>` — PascalCase, always `Credence`-prefixed for public classes.

```
CredenceRegistry  ✓
CredenceEnvelope  ✓
CredenceProxy     ✓
```

## Python methods and functions

`<verb>_<noun>` or plain `<verb>` for actions. `<noun>` or `get_<noun>` for reads.
Private helpers: `_<verb>_<noun>`.

## MCP tool names

Two patterns, strictly separated:

| Category | Pattern | When |
|----------|---------|------|
| **Action** | `credence_<verb>` | Mutates state or produces a side-effect |
| **Query** | `credence_<scope>_<noun>` | Reads state; `noun` ∈ {status, info, health, summary, constraints} |

**Action tools** (verb-first, object optional when unambiguous):
```
credence_register       credence_verify         credence_wrap
credence_unwrap         credence_diff           credence_gate
credence_reset          credence_autoverify     credence_scan
credence_scan_ghosts    credence_pre_compress   credence_post_compress
credence_memory_snapshot  credence_memory_recall
```

**Query tools** (scope + noun, no verb):
```
credence_score          credence_audit
credence_constraints    credence_session_info   credence_session_summary
credence_project_status credence_marker_health  credence_bandit_status
credence_memory_status
```

**Wrong patterns** — do not use:
```
credence_list_uncertain   ✗  (verb + adjective — use credence_constraints)
credence_ghost_scan       ✗  (noun + verb — use credence_scan_ghosts)
credence_scan_output      ✗  (object is implied — use credence_scan)
credence_session_brief    ✗  (non-standard suffix — use credence_session_summary)
credence_adaptive_status  ✗  (vague scope — use credence_bandit_status)
```

## Constants

`_ALL_CAPS` for module-level constants. `_` prefix marks them as non-exported.

## Eval and test files

| Location | Purpose | Naming |
|----------|---------|--------|
| `evals/` | Research benchmarks | `<what_is_measured>.py` |
| `tests/tests.py` | Unit / integration suite | fixed name |
| `tests/stress_full.py` | Stress suite | fixed name |

Evals are named after the phenomenon being measured, not the mechanism:
```
compression_faithfulness.py  ✓    stress_full.py  ✓
adversarial_tests.py         ✓    cross_session_eval.py  ✓
```

## Documentation files

Root docs: `ALL_CAPS.md`. Schema files: `<noun>_<version>.json` with underscores (not hyphens).

```
CONVENTIONS.md   ✓    etp_schema.json   ✓
ROADMAP.md       ✓    etp-v1.json       ✗  →  etp_v1.json
```

# experimental/

Pre-production work. Not installed, not tested by CI, not documented in the main README.

| Path | What it is | Roadmap phase |
|---|---|---|
| `enforce.py` | Decorator-based integration (`@credence.enforce`) — attaches gate to any function | Phase 2 |
| `epistemic_manifest.py` | Structured non-compressible manifest (successor to natural-language Truth Buffer) | Phase 2 |
| `typescript/` | TypeScript SDK — faithfulness probe + registry client port for Copilot/Cursor integration | Phase 2 |

Nothing in `credence/` imports from this directory. These are committed here so the design is visible and forkable, not because they are ready to use.

To try them locally: there are no guarantees on API stability.

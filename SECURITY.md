# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Active |

## Threat Model

Credence is a **local development tool**. It runs as:
- A Python library imported into your code
- An MCP server over stdio (not a network service)
- A native binary (`credence-gate`) invoked by Claude Code hooks

The registry (`epistemic_registry.db`) is a local SQLite file. It is not a network service and has no authentication layer by design — it is scoped to your local machine or shared team filesystem.

**What Credence does not protect against:**
- Malicious code running on the same machine (same-process trust boundary)
- A compromised MCP host (Claude Code itself)
- Network-level attacks (Credence has no network listener)

## Reporting a Vulnerability

If you discover a security vulnerability in credence-guard, please report it **privately** before public disclosure.

**Do not open a public GitHub issue for security vulnerabilities.**

**Report via:**
- GitHub private security advisory: [Security Advisories](https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai/security/advisories/new)
- Email: vijayarao.l@northeastern.edu (subject: `[SECURITY] credence-guard`)

**Please include:**
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

**Response timeline:**
- Acknowledgement within 48 hours
- Fix or mitigation plan within 7 days for critical issues
- Coordinated public disclosure after fix is released

## Known Security Boundaries

- `credence_register()` accepts arbitrary content strings — no per-session rate limit in v1.0.0. A malicious agent could fill the registry. Mitigation: the DB is local; blast radius is your local disk. Per-session cap planned for v1.0.1.
- The registry is not encrypted at rest. Do not store secrets as constraint content.
- `CREDENCE_DB_PATH` / `CREDENCE_REGISTRY_PATH` env vars accept arbitrary file paths. Validate these in shared/multi-user environments.

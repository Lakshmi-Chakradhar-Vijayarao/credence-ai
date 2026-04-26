# Credence — Submission Summary (100–200 words)

> Paste this into the "Written description / summary" field on the submission form.

---

**Claude doesn't just forget what you told it. It forgets whether you were sure about it.**

We defined and measured a failure mode: **Epistemic Qualifier Loss (EQL)** — when context compression strips uncertainty markers, turning "I think the rate limit is ~50 req/min — unconfirmed" into "rate limit: 50 req/min." The downstream model codes against it as fact. We measured this: LLMLingua-style compression causes 74% False Certainty Rate; naive Haiku causes 46% qualifier loss.

**Credence** is a deterministic enforcement layer for Claude Code that prevents this at five checkpoints:

1. **Faithfulness Probe** — 0.011ms frozenset scan, blocks Haiku before it strips qualifiers. 100% block rate, 0% false positives.
2. **Truth Buffer** — injects every unverified constraint into the system prompt before each generation.
3. **Consistency Enforcer** — imperative injection when the user's query overlaps a registered uncertain fact.
4. **Generation-Time Scanner** — annotates unverified values directly in generated code.
5. **Rust PreToolUse Gate** — blocks Claude Code from writing unverified values to files. 3.4ms, 98× faster than Python.

The entire evaluation used Opus 4.7. 178 tests. 22-tool MCP server. 2-minute Claude Code install. Fully reproducible — every number in the repo.

---

*Word count: ~175*

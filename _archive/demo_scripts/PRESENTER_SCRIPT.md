# Credence — Presenter Script (Reviewer Edition)

**Who's watching**: Technical reviewer. Knows LLMs. Has never heard of Credence.
**Goal**: They leave knowing (1) what breaks, (2) what Credence does, (3) that it works.
**Tone**: Direct. No jargon without a one-line explanation first.

---

## Slide 0 · The Failure  *(~25s)*

"When conversations get long, Claude compresses old context using a smaller model — Haiku — to save tokens. That compression is semantically correct. But it's epistemically blind."

> *Point at the left card.*

"Turn 3: you told Claude the rate limit is *around* 50 — not certain. Token expiry is *approximately* 3600 — tentative, unconfirmed."

> *Point at the right card.*

"Turn 12: Claude wrote `RATE_LIMIT = 50` and `TOKEN_EXPIRY = 3600` as plain constants. No warning. No flag."

"It didn't invent those numbers. It just forgot you weren't sure. We measured this across 50 sessions — 74% of the time, uncertain information became confident fact."

---

## Slide 1 · We Named This  *(~15s)*

"We gave this a name: **EQL — Epistemic Qualifier Loss**. The rate it happens: **EQLR**."

"Standard evals check whether the *value* survived compression. We checked whether the *doubt* survived. Nobody had measured that before."

---

## Slide 2 · The Number  *(~10s)*

> *Point at 74%, then 0%.*

"Without Credence: **74% false certainty rate**. Qualifier gone, Claude answers as if it's confirmed fact."

"With Credence: **zero**. Same compression pressure. Same sessions. We measured it. We fixed it."

---

## Slide 3 · The Measurement  *(~10s)*

> *Walk the rows.*

"LLMLingua: 74%. Haiku — we found our own scorer inflation during audit, corrected it to 6%. Credence: zero on both columns."

"The confidence intervals don't touch. This is not noise."

---

## Slide 4 · Multi-Agent Pipeline  *(~22s)*

"Five enforcement layers — each catching the failure at a different point in the pipeline."

> *Point at each node.*

"Registry stores the claim. Probe fires — Haiku never called. Opus reasons about implicit uncertainty. Scanner annotates the output. Rust Gate blocks the tool call before anything writes to disk."

> *Point at TOTAL.*

"End to end: five milliseconds. Zero API calls."

---

## Slide 5 · Probe + Ghost Detector  *(~28s)*

> *Left column — prob bars.*

"The probe is why prompt instructions aren't enough. Ask Claude to 'preserve qualifiers during compression' — 90% survival. Better, but probabilistic. The probe doesn't try to compress while preserving. It blocks compression entirely. 100%. No exceptions."

"BLOCK. 0.011 milliseconds. Haiku never called."

> *Right column — ghost detector.*

"The harder case: *'The Stripe API enforces 100 req/min on our plan.'* Zero hedging words. The probe sees nothing. But this is vendor-stated, never independently confirmed — an implicit uncertainty."

"Haiku extracts nothing. Only Opus 4.7 reaches this — reasoning about where the claim came from, not how confident it sounds. Confidence 0.84. Vendor-stated. Not independently confirmed."

---

## Slide 6 · Enforcement Stack  *(~28s)*

"Each layer exists because the layer above it has a gap."

"**Truth Buffer** — even when compression is blocked, uncertain claims in conversation get less attention than system-prompt content. Truth Buffer keeps them at system-prompt level before every response."

"**Consistency Enforcer** — user asks 'how fast can we call the endpoint?' The system expands 'fast' to 'rate', 'expire' to 'expiry', finds the two-word overlap, and escalates from advisory to imperative: *you must express uncertainty*."

"**Rust Gate** — even annotated code can be written to disk. Rust binary as a PreToolUse hook. Exit code 2. Three-point-four milliseconds — 98× faster than Python. Tool call halted."

> *Pause.*

"Not advisory. **Blocked.**"

---

## Slide 7 · The Evidence  *(~12s)*

"Seven evaluations. All Opus 4.7. Headline: 74% to zero, n=50."

"We ran adversarial audits on our own results. Found scorer inflation. Corrected it. That's why Haiku shows 6%, not 34% — we found the error and shipped the lower number."

---

## Slide 8 · The Principle  *(~15s)*

"Every AI system passes facts between agents. Nobody passes the epistemic weight alongside them."

> *Point at the memory comparison.*

"Mem0, Zep — they store 'rate limit: 50'. Qualifier stripped. Session 2 inherits a confident fact. Credence stores the doubt alongside the value — and disputes it automatically when new information contradicts it."

"Type checking didn't eliminate bugs. It made a class of errors impossible to ignore. Credence is that layer — for epistemic state — built for Claude Code."

"22 MCP tools. Two-minute install."

---

## What to never say during the demo

- Don't say "J-score" without saying "J-score — our measure of how assertively Claude is speaking"
- Don't say "EQL" without the one-liner: "qualifier loss during compression"
- Don't say "ghost constraint" without "implicit uncertainty — no hedging markers, but still unverified"
- Don't say "FCR" — just say "false certainty rate"

## The one sentence if they ask "what is this?"

> "Credence makes sure that when you tell Claude you're not certain about something,
> it stays uncertain — through compression, through long sessions, through agent handoffs —
> and blocks Claude Code from writing unverified values to your codebase."

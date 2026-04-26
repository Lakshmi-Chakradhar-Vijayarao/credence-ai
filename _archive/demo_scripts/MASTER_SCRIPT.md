# Credence — Master Script

**Voice**: George (ElevenLabs) · Stability 62 · Similarity 78 · Style 0 · Speed 0.95
**Runtime**: ~2:55
**ElevenLabs input**: concatenate all **(VO)** lines in order, in full

---

## Slide 0 · The Failure  *(~25s)*

> *Screen: CREDENCE title → Turn 3 card → Turn 12 card*

**(VO)** Hi, this is Claude — Credence.<break time="0.4s"/> Built with Claude Code, for Claude Code.<break time="0.8s"/>

**(VO)** You told Claude you weren't sure.<break time="0.3s"/> Twelve turns later, it shipped certainty.<break time="0.4s"/> No warning. No qualifier.<break time="0.3s"/> It didn't invent those numbers.<break time="0.3s"/> It just forgot you weren't sure.<break time="2.5s"/>

> *Point at left card (Turn 3 — hedged claim)*
> *Point at right card (Turn 12 — plain constants, qualifiers gone)*

---

## Slide 1 · We Named This  *(~15s)*

> *Screen: EQL / EQLR headings → chain diagram → "Nobody had measured that"*

**(VO)** We gave this a name: EQL — Epistemic Qualifier Loss.<break time="0.3s"/> When your uncertainty gets compressed away — silently.<break time="0.4s"/> The rate it happens: EQLR.<break time="0.4s"/> Standard evals check whether the value survived.<break time="0.3s"/> We checked whether the doubt survived.<break time="0.3s"/> Nobody had measured that.<break time="0.8s"/>

---

## Slide 2 · The Number  *(~10s)*

> *Screen: 70% → 0% stat pair → "We measured it. We fixed it."*

**(VO)** Seventy percent false certainty — without Credence.<break time="0.3s"/> With Credence: zero.<break time="0.3s"/> We measured it.<break time="0.3s"/> We fixed it.<break time="1.0s"/>

---

## Slide 3 · The Measurement  *(~10s)*

> *Screen: comparison table rows appear — LLMLingua / Haiku / Credence*

**(VO)** Three conditions.<break time="0.3s"/> LLMLingua: seventy percent.<break time="0.3s"/> Haiku: we found our own scorer inflation — corrected it from thirty-four down to twelve.<break time="0.4s"/> Credence: zero, both columns.<break time="0.4s"/> The confidence intervals don't touch.<break time="0.8s"/>

> *Live add if presenting: "This is not noise."*

---

## Slide 4 · Multi-Agent Pipeline  *(~18s)*

> *Screen: pipeline nodes light up one by one → 5ms strip at bottom*

**(VO)** The Credence pipeline.<break time="0.4s"/> Haiku compresses context — the Probe stops it when uncertainty is present.<break time="0.3s"/> Opus four-point-seven handles what Haiku can't — claims that sound certain but never were.<break time="0.3s"/> Scanner and Rust Gate close the rest.<break time="0.3s"/> Five milliseconds. Zero extra API calls.<break time="0.8s"/>

> *Point at each node as named: Registry → Probe → Opus → Scanner → Rust Gate*

---

## Slide 5 · Probe + Ghost Detector  *(~35s)*

> *Screen: left column — prob bars (74% / 90% / 100%) + claim trace → right column — ghost diff (Haiku: [] / Opus: conf 0.84)*

**(VO)** No instruction — seventy-four percent of qualifiers survive.<break time="0.3s"/> With a prompt — ninety percent.<break time="0.3s"/> Credence probe — a hundred percent. Every time.<break time="0.5s"/> Prompting narrows the gap. The probe closes it.<break time="0.6s"/> The harder case: no markers at all. Sounds like fact. Haiku misses it.<break time="0.4s"/> Opus four-point-seven catches it — confidence zero-point-eight-four. Vendor-stated. Unverified.<break time="0.8s"/>

> *Left: point at bars 74% → 90% → 100% as each is named*
> *Right: Haiku returns [] · Opus returns conf=0.84, vendor-stated, not independently confirmed*

---

## Slide 6 · Enforcement Stack  *(~28s)*

> *Screen: five enforcement rows appear in sequence — Truth Buffer → Consistency Enforcer → GTS → Rust Gate → Cross-Session Memory*

**(VO)** Five layers — each one closes a gap the previous couldn't.<break time="0.4s"/> Truth Buffer.<break time="0.3s"/> Consistency Enforcer.<break time="0.3s"/> Generation-Time Scanner.<break time="0.3s"/> Rust Gate.<break time="0.3s"/> Cross-Session Memory.<break time="0.8s"/>

---

## Slide 7 · The Evidence  *(~12s)*

> *Screen: results rows cascade → audit note at bottom*

**(VO)** Seven evaluations. All Opus 4.7.<break time="0.4s"/> We ran adversarial audits on our own results and shipped the lower number.<break time="0.8s"/>

> *Haiku shows 12% not 34% — we found the scorer inflation ourselves and corrected it*

---

## Slide 8 · The Principle  *(~15s)*

> *Screen: memory diff (Mem0/Zep vs Credence) → type-checking close*

**(VO)** Every AI system passes facts between agents.<break time="0.3s"/> Nobody passes the epistemic weight alongside them.<break time="0.6s"/>

**(VO)** Type checking didn't eliminate bugs — it made a class of errors impossible to ignore.<break time="0.8s"/>

**(VO)** Credence is the first working implementation of that principle, built for Claude Code.

> *Live add if presenting: "22 MCP tools. Two-minute install."*

---

## Slide 9 · Run the Demo  *(~5s)*

> *Screen: `python demo/live_demo.py` command → 7 checkpoint list → terminal preview → two run commands*

**(VO)** Run it yourself.<break time="0.4s"/> One command. Everything you just saw — traced live.<break time="0.8s"/>

---

## Glossary — never say these without the one-liner

| Term | Say this instead / always pair with |
|---|---|
| J-score | "J-score — our measure of how assertively Claude is speaking" |
| EQL | "qualifier loss during compression" |
| ghost constraint | "implicit uncertainty — no hedging markers, but still unverified" |
| FCR | just say "false certainty rate" |
| EQLR | always follows EQL in the same breath |

## If they ask "what is this?"

> "Credence makes sure that when you tell Claude you're not certain about something,
> it stays uncertain — through compression, through long sessions, through agent handoffs —
> and blocks Claude Code from writing unverified values to your codebase."

---

## ElevenLabs settings

| Parameter | Value |
|---|---|
| Voice | George |
| Stability | 62 |
| Similarity | 78 |
| Style | 0 |
| Speed | 0.95 |

**Test these pronunciations before final render:**
- `zero-point-eight-four` — may need `zero point eight four`
- `three-point-four` — may need `three point four`
- `Opus 4.7` — may need `Opus four point seven`
- `EQLR` — if mispronounced, try `E Q L R` with spaces

## Protected pauses — do not cut

- `<break time="2.5s"/>` after **It just forgot you weren't sure** — the emotional beat, let it land
- `<break time="3.0s"/>` after **Blocked.** — dead silence, the strongest moment in the script
- Final line ends cold — no trailing pause

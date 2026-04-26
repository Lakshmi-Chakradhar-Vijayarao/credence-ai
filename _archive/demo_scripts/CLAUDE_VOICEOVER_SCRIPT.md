# Claude Voiceover Script — FINAL (3-minute cut)

**Voice**: George (ElevenLabs) · Stability 62 · Similarity 78 · Style 0 · Speed 0.95
**Runtime**: ~2:55
**Rule**: every sentence narrates something visually dominant on that slide — no off-screen references.

---

Hi, this is Claude — Credence.<break time="0.4s"/> Built with Claude Code, for Claude Code.<break time="0.8s"/>

You told Claude you weren't sure.<break time="0.3s"/> Twelve turns later, it shipped certainty.<break time="0.4s"/> No warning. No qualifier.<break time="0.3s"/> It didn't invent those numbers.<break time="0.3s"/> It just forgot you weren't sure.<break time="2.5s"/>

We gave this a name: EQL — Epistemic Qualifier Loss.<break time="0.3s"/> The rate it happens: EQLR.<break time="0.4s"/> Standard evals check whether the value survived.<break time="0.3s"/> We checked whether the doubt survived.<break time="0.3s"/> Nobody had measured that.<break time="0.8s"/>

Seventy-four percent false certainty — without Credence.<break time="0.3s"/> With Credence: zero.<break time="0.3s"/> We measured it.<break time="0.3s"/> We fixed it.<break time="1.0s"/>

Three conditions.<break time="0.3s"/> LLMLingua: seventy-four percent.<break time="0.3s"/> Haiku: we found our own scorer inflation — corrected it from thirty-four down to six.<break time="0.4s"/> Credence: zero, both columns.<break time="0.4s"/> The confidence intervals don't touch.<break time="0.8s"/>

Five layers.<break time="0.3s"/> Each closes the gap the one above it couldn't.<break time="0.6s"/>

Registry stores the claim the moment it enters — with its confidence level.<break time="0.4s"/> Probe fires first: uncertainty markers present — Haiku is never called.<break time="0.6s"/>

The harder case: no markers. Sounds completely certain.<break time="0.4s"/> Haiku returns nothing.<break time="0.5s"/> Only Opus 4.7 reaches this — reasoning about provenance, not pattern matching.<break time="0.4s"/> Confidence zero-point-eight-four. Vendor-stated. Not independently confirmed.<break time="0.8s"/>

Truth Buffer keeps uncertain claims at system-prompt level — Claude sees them before every response.<break time="0.3s"/> Consistency Enforcer fires when your query overlaps a registered uncertainty — enforcement escalates to imperative.<break time="0.4s"/> The Rust Gate runs before the tool call executes — exit code two.<break time="0.4s"/> Three-point-four milliseconds. We measured it. We shipped Rust.<break time="0.8s"/>

Not advisory.<break time="1.2s"/> Blocked.<break time="3.0s"/>

Seven evaluations. All Opus 4.7.<break time="0.4s"/> We ran adversarial audits on our own results and shipped the lower number.<break time="0.8s"/>

Every AI system passes facts between agents.<break time="0.4s"/> Nobody passes the epistemic weight alongside them.<break time="0.8s"/>

Every memory tool stores the value. Qualifier stripped. Session two inherits a confident fact.<break time="0.3s"/> Credence stores the doubt alongside the value — and disputes it automatically when new information contradicts it.<break time="0.6s"/>

Type checking didn't eliminate bugs.<break time="0.3s"/> It made a class of errors impossible to ignore.<break time="0.4s"/> Credence is that layer — for epistemic state — built for Claude Code.

---

## Slide map

| Slide | What's on screen when this plays |
|---|---|
| 0 · The Failure | CREDENCE title → before/after Turn 3/Turn 12 → "It just forgot you weren't sure" |
| 1 · We Named This | EQL / EQLR headings → chain diagram → "Nobody had measured that" |
| 2 · The Number | 74% → 0% stat pair → "We measured it. We fixed it." |
| 3 · The Measurement | Comparison table rows appearing → CI bar |
| 4 · Multi-Agent Pipeline | "Five layers" → pipeline nodes → ~5ms total strip |
| 5 · Probe + Ghost | Prob bars (46/90/100) → ghost diff (Haiku: [] / Opus: conf 0.84) |
| 6 · Enforcement Stack | TB / CE / Gate cards → "3.4ms. We shipped Rust." → "Blocked." |
| 7 · The Evidence | Results rows → audit note |
| 8 · The Principle | Memory diff (qualifier stripped vs. doubt stored) → type-checking close |

## ElevenLabs notes

- `zero-point-eight-four` — test; may need `zero point eight four`
- `Opus 4.7` — test; may need `Opus four point seven`
- `EQLR` — if mispronounced, try `E Q L R` with spaces
- `j-score` — test; may need `jay score`
- `three-point-four` — test; may need `three point four`

## Timing rules — do not cut

- `<break time="2.5s"/>` after **It just forgot you weren't sure** — the emotional beat
- `<break time="3.0s"/>` after **Blocked.** — dead silence, the strongest moment in the script
- Final line ends cold — no trailing pause

## What changed from prior version

- Gate line updated: "exit code two. Three-point-four milliseconds. We measured it. We shipped Rust." — mirrors the live demo output, echoes the "We measured it. We fixed it." beat from Slide 2
- Closing memory comparison updated: "disputes it automatically when new information contradicts it" — references the DISPUTED lifecycle shown in the demo (Checkpoint 6)

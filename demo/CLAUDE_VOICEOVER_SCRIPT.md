# Claude Voiceover Script — LOCKED

**Voice**: George (ElevenLabs) · Stability 62 · Similarity 78 · Style 0 · Speed 0.95
**Runtime**: ~1:55
**Status**: LOCKED — do not edit

---

Hi, this is Claude — Credence.<break time="0.4s"/> Built with Claude Code, for Claude Code.<break time="1.2s"/>

The same argument that made type checking mandatory — silent errors cost more than enforcement — applies here.<break time="1.0s"/>

You told me you weren't sure.<break time="0.4s"/> Twelve turns later — I shipped certainty.<break time="0.8s"/>

I didn't invent that number.<break time="0.3s"/> I just forgot that you weren't sure.<break time="2.5s"/>

We coined the terms: EQL — Epistemic Qualifier Loss, uncertainty stripped silently during compression. EQLR — the rate at which it happens.<break time="0.8s"/>

One number: 74 percent false certainty rate under aggressive compression.<break time="0.4s"/> Credence makes it zero.<break time="1.0s"/>

Three conditions tested.<break time="0.4s"/> Aggressive compression — 74 percent.<break time="0.3s"/> Standard Haiku — we found our own inflated scorer, corrected it from 34 percent down to 6.<break time="0.6s"/> Credence — zero, both columns.<break time="0.5s"/> The confidence intervals don't touch.<break time="1.0s"/>

Three agents. Each doing only what it can do.<break time="0.6s"/>

46 percent survival without instruction. 90 percent with a prompt. 100 percent with enforcement.<break time="0.6s"/>

Prompt instructions are probabilistic.<break time="0.4s"/> Enforcement is not.<break time="0.8s"/>

The probe fires before Haiku ever sees uncertain text — 0.011 milliseconds, zero API calls.<break time="0.7s"/>

The harder case: no markers, sounds certain. Haiku sees nothing.<break time="0.3s"/> Opus 4.7 reasons about provenance — where each claim came from.<break time="1.0s"/>

Query overlaps constraint — enforcement escalates to imperative.<break time="0.4s"/> Unverified value in generated code — annotated inline.<break time="0.4s"/> Tool: Edit. Overlap: token, expires, auth. Exit code 2.<break time="0.8s"/>

Not advisory.<break time="1.2s"/> Blocked.<break time="3.0s"/>

Seven evaluations. All Opus 4.7.<break time="0.4s"/> Most projects overclaim — we ran adversarial audits on our own results.<break time="1.0s"/>

Every AI system passes facts between agents.<break time="0.4s"/> Nobody passes epistemic weight alongside them.<break time="1.0s"/>

Credence is the first working implementation of that principle, built for Claude Code.<break time="1.2s"/>

---

## What Changed From Previous Version

- **Type-checking frame moved to opening** (was closing argument): Boris Cherny hears the architectural analogy as the lens, not the footnote
- **"Three agents" replaces "Three layers"**: matches the trio cards on slide 4, more concrete
- **"46 percent survival" replaces "46 percent loss"**: numerically correct — chart shows survival rate, not loss rate
- **Slide 5 reordered**: bar chart numbers (46/90/100%) + "probabilistic vs enforcement" now precede "The probe fires" — script now follows slide top-to-bottom visual order
- **Slide 3 (Measurement Table) deleted**: had no narration counterpart; details absorbed into earlier slides
- **Ghost detector expanded**: "where each claim came from" — direct answer to the Opus 4.7 use question for the Claude team judges
- **"Most projects overclaim"** added before self-audit: frames it as character, not just a data correction
- **Type-checking closing line removed**: no longer needed at the end since it opens the script

## Timing Notes
- Single dial: `<break time="3.0s"/>` after **Blocked.** — shorten to 2.5s if ElevenLabs runs long
- `<break time="2.5s"/>` after **I just forgot that you weren't sure** — the longest emotional beat, do not cut
- The final line ends cold — no trailing pause needed

## ElevenLabs Pronunciation Watch
- "Opus 4.7" — test first, may need "Opus four point seven"
- "0.011 milliseconds" — may need "zero point zero one one milliseconds"
- "EQLR" — may need "E Q L R" with spaces if it mispronounces

# Claude Voiceover Script — LOCKED

**Voice**: George (ElevenLabs) · Stability 62 · Similarity 78 · Style 0 · Speed 0.95
**Runtime**: ~2:00
**Status**: LOCKED — do not edit

---

Hi, this is Claude — Credence.<break time="0.4s"/> Built with Claude Code, for Claude Code.<break time="1.2s"/>

The same argument that made type checking mandatory — silent errors cost more than enforcement — applies here.<break time="1.0s"/>

You told me you weren't sure.<break time="0.4s"/> Twelve turns later — I shipped certainty.<break time="0.8s"/>

I didn't invent that number.<break time="0.3s"/> I just forgot that you weren't sure.<break time="2.5s"/>

We coined the terms: EQL — Epistemic Qualifier Loss, uncertainty stripped silently during compression. EQLR — the rate at which it happens.<break time="0.6s"/>

Standard evals check whether the value survived. We checked whether the doubt survived. Nobody had measured that.<break time="1.0s"/>

One number: 74 percent false certainty rate under aggressive compression.<break time="0.4s"/> Credence makes it zero.<break time="1.0s"/>

Three conditions tested.<break time="0.4s"/> Aggressive compression — 74 percent.<break time="0.3s"/> Standard Haiku — we found our own inflated scorer, corrected it from 34 percent down to 6.<break time="0.6s"/> Credence — zero, both columns.<break time="0.5s"/> The confidence intervals don't touch.<break time="1.0s"/>

Three agents.<break time="0.3s"/> Each assigned the problem only it can solve.<break time="0.6s"/>

Haiku's compression is semantically correct.<break time="0.3s"/> The qualifier carries no informational weight — that's why a prompt can't fix it.<break time="0.4s"/> The probe intercepts before Haiku decides.<break time="0.8s"/>

46 percent survival without instruction.<break time="0.3s"/> 90 percent with a prompt.<break time="0.3s"/> 100 percent with enforcement.<break time="0.6s"/>

Prompt instructions are probabilistic.<break time="0.4s"/> Enforcement is not.<break time="0.8s"/>

The probe fires before Haiku ever sees uncertain text — 0.011 milliseconds, zero API calls.<break time="0.7s"/>

The harder case: no markers, sounds certain. Haiku sees nothing.<break time="0.3s"/> Opus 4.7 reasons about provenance — where each claim came from.<break time="1.0s"/>

Five layers.<break time="0.3s"/> Each one closes the gap the layer above it couldn't.<break time="0.6s"/>

Query overlaps constraint — enforcement escalates to imperative.<break time="0.4s"/> Unverified value in generated code — annotated inline.<break time="0.4s"/> Tool: Edit. Overlap: token, expires, auth. Exit code 2.<break time="0.8s"/>

Not advisory.<break time="1.2s"/> Blocked.<break time="3.0s"/>

Seven evaluations. All Opus 4.7.<break time="0.4s"/> Most projects overclaim — we ran adversarial audits on our own results and shipped the lower number.<break time="1.0s"/>

Every AI system passes facts between agents.<break time="0.4s"/> Nobody passes epistemic weight alongside them.<break time="1.0s"/>

Credence is the first working implementation of that principle, built for Claude Code.<break time="1.2s"/>

---

## What Changed From v1

- **EQL/EQLR paragraph extended**: added "Standard evals check whether the value survived. We checked whether the doubt survived. Nobody had measured that." — positions the measurement contribution before the 74% headline
- **"Three agents" updated**: "Each assigned the problem only it can solve" (was "doing only what it can do")
- **New paragraph before bar chart**: "Haiku's compression is semantically correct. The qualifier carries no informational weight — that's why a prompt can't fix it. The probe intercepts before Haiku decides." — explains the 90%→100% gap
- **"Five layers" paragraph added**: "Each one closes the gap the layer above it couldn't." — frames enforcement stack as a chain of necessity before the gate-blocked sequence
- **Self-audit line strengthened**: "shipped the lower number" added — signals integrity

## What Was Kept From v1

- Type-checking opening frame (Boris Cherny lens)
- 2.5s emotional beat after "I just forgot that you weren't sure"
- 3.0s dead silence after "Blocked."
- Ghost detector line ("Opus 4.7 reasons about provenance")
- Final cold ending — no trailing pause

## Timing Notes
- Single dial: `<break time="3.0s"/>` after **Blocked.** — shorten to 2.5s if ElevenLabs runs long
- `<break time="2.5s"/>` after **I just forgot that you weren't sure** — the longest emotional beat, do not cut
- The final line ends cold — no trailing pause needed

## ElevenLabs Pronunciation Watch
- "Opus 4.7" — test first, may need "Opus four point seven"
- "0.011 milliseconds" — may need "zero point zero one one milliseconds"
- "EQLR" — may need "E Q L R" with spaces if it mispronounces

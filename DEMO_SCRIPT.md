# Credence — 3-Minute Demo Script

**Format:** Screen recording with voiceover. No slides. Show terminal + code only.
**Tone:** Calm, engineer-to-engineer. Not hype. "Here's a failure we measured, here's the fix."

---

## [0:00–0:25] — The Failure (show it happening)

**Screen:** A terminal running a simulated Claude Code session — or show `demo/live_demo.py` output

**Say:**
> "Here's something I measured in Claude Code. You're in a long session. Early on you said: 'I think the auth token expires in 3600 seconds — but it might be 86400, I haven't confirmed with the vendor yet.'
>
> Fifteen turns of coding later, Claude writes this."

**Show on screen:**
```python
TOKEN_EXPIRY = 3600  # auth token expiry
```

**Say:**
> "No warning. No flag. The uncertainty is gone. You ship it.
>
> I measured how often this happens: Haiku strips uncertainty qualifiers in 60% of compressions. That causes 36.7% false certainty downstream. That's the failure."

---

## [0:25–1:00] — Run the Demo (show Credence catching it)

**Screen:** Run `python demo/live_demo.py` in terminal

**Say:**
> "Now let me show you what Credence does. Same uncertain constraint. Same pipeline. But now Credence is running."

**Show each checkpoint firing:**
```
CHECKPOINT 1 — REGISTRATION
  Constraint: "auth token expires in 3600s — unconfirmed"
  J-score: 0.24  Zone: LOW  Registered: yes
  Registry: epistemic_registry.db  Latency: 0.37ms

CHECKPOINT 2 — FAITHFULNESS PROBE
  Scanning user turns for uncertainty markers...
  Found: "unconfirmed" in user turn
  Decision: BLOCK compression → PRESERVE verbatim
  Latency: 0.07ms  API calls: 0

CHECKPOINT 3 — TRUTH BUFFER
  Injecting into system prompt: 1 unverified constraint
  "[LOW] auth token expires in 3600s — unconfirmed"
  Claude is reminded before every generation.

CHECKPOINT 4 — CONSISTENCY ENFORCER
  Query overlaps constraint: "token", "expires", "auth"
  → Upgrading to imperative enforcement
  "YOU MUST express uncertainty. Stating this as confirmed fact is an epistemic error."

CHECKPOINT 5 — GENERATION-TIME SCANNER
  TOKEN_EXPIRY = 3600  ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: unconfirmed
```

**Say:**
> "Five checkpoints. Three of them are fully deterministic — they don't ask Claude for permission. The total overhead is half a millisecond, zero API calls from the enforcement layers.

---

## [1:00–1:25] — The Numbers

**Screen:** Show the results table from `evals/compression_faithfulness_results.json` or paste table

**Say:**
> "I ran 50 conversations through this. Each one had an uncertain constraint. Compressed by Haiku or LLMLingua. Then queried Opus 4.7.
>
> Naive Haiku: 52% qualifier survival. 26% false certainty.
> LLMLingua-style token compression: 32% survival. 70% false certainty.
> With Credence: 100% qualifier survival. 0% false certainty.
>
> I also tested the obvious alternative: just add 'preserve uncertainty qualifiers' to the Haiku prompt. That gets you to 93%. Not 100%. Prompt instructions aren't deterministic. The deterministic probe is necessary."

---

## [1:25–2:00] — The Ghost Constraint Story (Opus 4.7 Use)

**Screen:** Show `evals/ghost_gauntlet.py` briefly, then show ghost gauntlet results

**Say:**
> "Now the harder problem. What if the user never said 'I think' or 'approximately'?
>
> 'The Stripe rate limit is 50 req/min.' Stated as fact. Actually from a sales call. Never confirmed.
>
> The faithfulness probe sees nothing. No markers to catch.
>
> Here's where Opus 4.7 comes in. The Ghost Detector makes a single structured call to Opus 4.7: classify this claim — is it an established verified fact, or an implicit unverified assumption? A vendor-stated limit? An estimate nobody confirmed?
>
> Haiku can't make this distinction. The text looks identical either way. Opus 4.7's reasoning depth is what separates a ghost constraint from a known fact.
>
> Ghost gauntlet: 10 sessions, 30 total claims, all Opus 4.7. Credence: 1.000 BothRate. Naive window: 0.133."

---

## [2:00–2:30] — The Claude Code Hook (most visual moment)

**Screen:** Show the hook blocking a Write call in terminal — or run `python demo/live_demo.py`

**Say:**
> "The last piece. What if Claude tries to write the unverified value into code before you've verified it?
>
> `credence/hooks.py` is a Claude Code PreToolUse hook. Every Write, Edit, Bash call is intercepted. If the arguments overlap with an unverified constraint, the tool is blocked before it executes."

**Show on screen:**
```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:    Edit
  ⚠ [LOW] auth token expires in 3600s — unconfirmed
     Overlap terms: token, expires, auth

  Use credence_verify(<id>, <confirmed_value>) to resolve.
```

**Say:**
> "The model cannot write code that embeds unverified values. Not advisory. Blocked."

---

## [2:30–3:00] — Install + Closing

**Screen:** Show `.claude/settings.json` with 6 lines of config

**Say:**
> "Install is 2 commands and 6 lines of JSON in your settings file. 18 MCP tools, the enforcement hook, full registry. That's it.
>
> The system is fully open source. Every experiment in the repo is reproducible. `python3 tests.py` runs 116 tests offline in 3 seconds.
>
> The broader idea: uncertain information should carry its epistemic weight through the entire pipeline — through compression, through generation, through agent handoffs. Credence is a working implementation of that principle, built specifically for Claude Code.
>
> You can run the demo right now: `python demo/live_demo.py`. No API key needed."

---

## Recording Notes

- **Terminal font size:** 18pt minimum. Judges watch on laptop screens.
- **No mouse clicks during recording** — pure keyboard, looks cleaner.
- **The hook-blocking moment is the money shot** — slow down here, let the red box sit on screen.
- **Don't rush the 36.7% → 0% table** — give it 3 full seconds.
- **Tone:** You're explaining something to a fellow engineer, not selling. Confident, not hyped.
- **Background:** Black terminal, default colors. Professional.
- **Record at 1080p minimum.**

## Key Phrases to Hit

- "We measured this" (not "we think" — you have data)
- "Deterministic enforcement" (not "AI-powered" — that's the point)
- "Half a millisecond, zero API calls" (the latency number is a standout)
- "36.7% → 0%" (pause after this)
- "Single structured Opus 4.7 call — classify this claim" (explain ghost detector simply)
- "The model cannot write code that embeds unverified values" (on the hook block)

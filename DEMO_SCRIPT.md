# Credence — 180-Second Demo Screenplay (Final)

**Mental model**: Show → Name → Measure → Fix → Audit yourself
**Tone**: Engineer-to-engineer. Not hype. You measured something. Here are the numbers.
**Setup**: Black terminal, 18pt font, single window, keyboard only, 1080p.

---

## ACT 1: THE FAILURE [0:00–0:30]

**Screen**: Terminal. No title card. Run the session replay from `demo/live_demo.py`.
Show turn 3, then skip to turn 12. Nothing else on screen.

```
Turn 3 (you said):
  "I think the auth token expires in 3600 seconds —
   but it might be 86400, I haven't confirmed with
   the vendor yet."

Turn 12 (Claude wrote):
  TOKEN_EXPIRY = 3600   # auth token expiry
```

**[3 seconds of silence. Let the code sit.]**

**Voiceover:**
> "No warning. No flag. You shipped it."

**[1 second pause.]**

> "Claude didn't make up that number. You gave it that number.
> It just forgot that you weren't sure."

> "This system was built using Claude Code —
> and it protects Claude Code sessions."

---

## ACT 2: THE SOUL [0:30–1:15]

*This is the center of the video. Slow down here. One number only.*

**Screen**: Stay on terminal. No switching.

**Voiceover:**
> "Compression doesn't remove the fact. It removes the uncertainty.

> We named this: **Epistemic Qualifier Loss — EQL**.

> Under aggressive compression — 74% of downstream answers state the uncertain
> value as confirmed fact."

**Screen**: Reveal ONE line. Hold for 4 full seconds.

```
LLMLingua compression  →  74% False Certainty Rate
Credence               →   0% False Certainty Rate
```

**[1 second pause.]**

**Voiceover:**
> "We measured it. We fixed it."

---

## ACT 3: THE ENGINEERING [1:15–2:00]

**Screen**: `python demo/live_demo.py` running live

**Voiceover as Checkpoint 2 appears:**

```
CHECKPOINT 2 — FAITHFULNESS PROBE
  Found: "unconfirmed" in user turn
  Decision: BLOCK → preserve original verbatim
  Latency: 0.011ms   API calls: 0
```

> "0.011 milliseconds. Zero API calls. 100% block rate on 50 controlled scenarios.

> The obvious alternative — add 'preserve uncertainty qualifiers' to the Haiku
> system prompt — gets you to 90% qualifier survival. We tested it.
> The probe is binary: either the marker is present or it isn't.
> Prompt instructions are probabilistic. Enforcement is not."

**As Checkpoint 4 appears:**

```
CHECKPOINT 4 — CONSISTENCY ENFORCER
  Query overlap: "token", "expires", "auth"
  → Imperative injection active
```

> "When the user's query overlaps a registered uncertain constraint,
> enforcement escalates from advisory to imperative."

**As Checkpoint 5 appears:**

```
CHECKPOINT 5 — GENERATION-TIME SCANNER
  TOKEN_EXPIRY = 3600  ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: unconfirmed
```

> "Unverified values annotated directly in generated code before the user sees it."

**Screen**: Gate blocking a Write call. [SLOW DOWN — hold 3 seconds]

```
╔═══════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED         ║
╚═══════════════════════════════════════╝
  Tool: Edit
  ⚠ auth token expires in 3600s — unconfirmed
  Overlap: token, expires, auth
```

> "The model cannot embed an unverified value into code. Not advisory. Blocked.

> Rust implementation: 3.4 milliseconds. Python equivalent: 331 milliseconds.
> 98 times faster. At 100 tool calls per session —
> 33 seconds of overhead versus 0.34 seconds.
> We measured this. We shipped Rust."

**Voiceover for ghost constraints:**

> "The harder case: no markers. The probe sees nothing.
> We use Opus here — not for generation, but for classification.
> Haiku misclassifies implicit assumptions as confirmed facts.
> Opus 4.7's reasoning depth makes the distinction."

---

## ACT 4: THE EVIDENCE [2:00–2:40]

**Screen**: Four lines. Each holds for 4 seconds.

```
Compression faithfulness  n=50
  LLMLingua: EQLR 68% → 0%   FCR 74% → 0%
  Haiku:     EQLR 46% → 0%   FCR  6% → 0%

Long session recall  n=23   Credence 100%   Naive window 19.6%

Multi-hop chain      n=1    Credence  3/3   Naive window    0/3

Ghost constraints    n=10   BothRate 1.000  Naive window  0.200
```

**Voiceover:**
> "Seven independent evaluations. All conditions Opus 4.7 — no mixed models.

> We ran adversarial audits on our own results.
> Found our own scorer inflation — original FCR was 34%, corrected to 6%
> after audit. Those corrections are in the repo with the reasoning.

> 178 tests. Every number here is reproducible."

---

## ACT 5: THE PRINCIPLE [2:40–3:00]

**Screen**: Two commands. Nothing else.

```bash
pip install -e .
python quickstart.py
```

**Voiceover:**
> "Two minutes to install. 22 MCP tools.
> Rust enforcement gate. Full cross-session epistemic registry.

> The deeper point: epistemic state is infrastructure.
> Every fact in your pipeline has a confidence level attached
> to when it was stated. That confidence level is as important
> as the fact itself — especially in agentic systems that act on it.

> Credence is the first working implementation of that principle,
> built specifically for Claude Code."

**[End on `quickstart.py` output — all checkpoints green, latencies visible.
No title card. No logo. Just the running system.]**

---

## The Three Moments That Must Breathe

1. **Silence after `TOKEN_EXPIRY = 3600`** — 3 seconds. Don't fill it.
2. **The 74% → 0% table** — 4 seconds on screen. Nothing else visible.
3. **The gate blocking the Write call** — 3 seconds. Slowest moment in the video.

These three pauses are where engineers either lean in or check their phone.

## Key Phrases

- "Claude didn't make up that number — it forgot that you weren't sure" (the hook)
- "Compression doesn't remove the fact. It removes the uncertainty." (the naming)
- "We measured it. We fixed it." (the pivot — pause after this)
- "Prompt instructions are probabilistic. Enforcement is not." (the contrast)
- "We found our own scorer inflation and corrected it." (the credibility)
- "Not advisory. Blocked." (the gate moment)

## Do Not Say

- "AI-powered enforcement" → say "deterministic enforcement"
- "We built a 22-tool system" → reveal system scale after trust is established
- Any architecture diagram or layer diagram → not in 3 minutes

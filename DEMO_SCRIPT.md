# Credence — 3-Minute Demo Script

**Format:** Screen recording with voiceover. No slides. Terminal only.
**Tone:** Engineer-to-engineer. Measured. "Here's a failure we measured, here's the fix."
**Resolution:** 1080p minimum. Terminal font: 18pt.

---

## [0:00–0:30] — Show the failure

**Screen:** Open terminal. Run `python demo/live_demo.py` — pause at the failure checkpoint.

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
> We measured how often this happens. Under LLMLingua-style compression — which is aggressive, token-efficient, what production systems use — 68% of compressions strip the uncertainty qualifier. 74% of downstream answers then state the value as confirmed fact.
>
> That's the failure. And it has a name: Epistemic Qualifier Loss."

---

## [0:30–1:15] — Run Credence (show all 5 checkpoints)

**Screen:** `python demo/live_demo.py` — let it run through all checkpoints

**Say:**
> "Now with Credence. Same uncertain constraint. Same pipeline."

**Show each checkpoint as it fires:**
```
CHECKPOINT 1 — REGISTRATION
  Constraint: "auth token expires in 3600s — unconfirmed"
  J-score: 0.24  Zone: LOW  Latency: 0.37ms

CHECKPOINT 2 — FAITHFULNESS PROBE
  Found: "unconfirmed" in user turn
  Decision: BLOCK compression → PRESERVE verbatim
  Latency: 0.011ms  API calls: 0

CHECKPOINT 3 — TRUTH BUFFER
  Injecting 1 unverified constraint into system prompt every turn.

CHECKPOINT 4 — CONSISTENCY ENFORCER
  Query overlaps: "token", "expires", "auth"
  → Imperative: "YOU MUST express uncertainty."

CHECKPOINT 5 — GENERATION-TIME SCANNER
  TOKEN_EXPIRY = 3600  ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: unverified
```

**Say:**
> "Five checkpoints. Three are fully deterministic — they don't ask Claude, they just enforce. The probe is 0.011 milliseconds, zero API calls. EQLR 46% to zero. FCR 74% to zero."

---

## [1:15–1:45] — The numbers

**Screen:** Show the results table (paste or display from JSON)

**Say:**
> "I ran 50 conversations through this. Each had one uncertain constraint. Three conditions: naive Haiku compression, LLMLingua-inspired compression, Credence.

| Condition | EQLR | FCR |
|---|---|---|
| Naive Haiku | 46% | 6% |
| LLMLingua sim | 68% | **74%** |
| Credence | **0%** | **0%** |

> The obvious alternative — add 'preserve uncertainty qualifiers' to the Haiku prompt — gets you to 90% qualifier survival. Not 100%. Prompt instructions are probabilistic. The probe is binary enforcement: the qualifier is present, or it isn't. 100% block rate on 50 scenarios.
>
> This is the tested alternative. Prompt engineering doesn't close the gap. Deterministic enforcement does."

---

## [1:45–2:15] — The ghost constraint problem (Opus 4.7 use)

**Screen:** Briefly show `evals/ghost_gauntlet.py` scenario structure, then result

**Say:**
> "Now the harder problem. What if the user never hedged?
>
> 'The Stripe rate limit is 50 req/min.' Stated as fact. Actually from a sales call. Nobody confirmed it.
>
> The faithfulness probe sees nothing. No markers.
>
> Here's where Opus 4.7 matters. The Scout Classifier makes a single structured call to Opus 4.7: classify this claim — is it a confirmed fact, or an implicit unverified assumption?
>
> Haiku gets this wrong. The language looks identical. Opus 4.7's reasoning is what distinguishes a ghost constraint from a known fact.
>
> Ghost gauntlet: 10 domain sessions, 30 total claims. Credence BothRate: 1.000. Naive sliding window: 0.200."

---

## [2:15–2:45] — The Claude Code hook (the money shot)

**Screen:** Show `.claude/settings.json`, then show the gate blocking a Write call

**Say:**
> "The last piece is the one most specific to Claude Code. A PreToolUse hook intercepts every Write, Edit, Bash call. If the arguments overlap a registered unverified constraint, the tool is blocked before it runs."

**Show on screen:**
```
╔══════════════════════════════════════════════════════════════╗
║  CREDENCE GATE — TOOL BLOCKED                                ║
╚══════════════════════════════════════════════════════════════╝

  Tool:    Edit
  ⚠ [LOW] auth token expires in 3600s — unconfirmed
     Overlap: token, expires, auth

  Use credence_verify(<id>, <confirmed_value>) to resolve.
```

**Say:**
> "The model cannot embed an unverified value into code. Not advisory. Blocked. The Rust implementation runs in 3.4 milliseconds — 98 times faster than the Python equivalent."

---

## [2:45–3:00] — Install + close

**Screen:** Two terminal commands

```bash
pip install -e .
# Add 6 lines to .claude/settings.json
python quickstart.py
```

**Say:**
> "Install is two commands. 22 MCP tools, the enforcement hook, full cross-session registry.
>
> Everything here is reproducible. `python3 tests.py` — 178 tests, offline, 3 seconds.
>
> The principle: uncertain information should carry its epistemic weight through the entire pipeline. Credence is a working implementation of that principle, built specifically for Claude Code."

---

## Recording Notes

- **The hook-blocking moment is the money shot** — slow down, let the red box sit 3 seconds
- **The LLMLingua 74% → 0% table** — pause here, let it land
- **Don't say "AI-powered enforcement"** — say "deterministic enforcement." That's the point.
- **Key numbers to hit**: 0.011ms, 0% false positive rate, 74% FCR, 178 tests, 3.4ms Rust gate
- **Tone**: You measured something, you fixed it, here are the numbers. Not hype.
- **Black terminal on dark background.** No browser. No IDE.
- **No mouse clicks** — pure keyboard. Looks professional.

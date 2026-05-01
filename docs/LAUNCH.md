# Launch Materials

## Hacker News — Show HN Post

**Title:**
```
Show HN: Credence – we measured a 46% uncertainty-loss rate in LLM context compression and built the fix
```

**Body:**
```
When Claude Code summarizes old context to save tokens, it uses a smaller model (Haiku) to compress.
We measured what happens to uncertainty markers during that compression.

Result: 46% of the time, Haiku strips qualifiers like "I think", "unverified", "probably", 
"the vendor claims" from the summary. The downstream model then states the uncertain value 
as confirmed fact 6% of the time with naive Haiku, and 74% of the time with LLMLingua-style 
scoring (which actively keeps "informative" content — i.e., the fact — and drops "padding" — 
i.e., the qualifier).

We call this Epistemic Qualifier Loss (EQL). The downstream false certainty is the FCR 
(False Certainty Rate). We built Credence to reduce both to zero.

Five enforcement layers:

1. Faithfulness probe (0.07ms, zero API calls) — 198-term frozenset scans user turns before 
   Haiku compression. Uncertainty found → block compression → keep the turn verbatim.

2. Truth Buffer + Consistency Enforcer — SQLite registry stores all uncertain constraints. 
   Before every generation turn, injects them into the system prompt. When user query 
   keyword-overlaps a registered constraint, escalates to imperative mode.

3. Generation-Time Scanner (0.08ms) — scans code blocks and prose in model output for 
   unverified numeric literals. Annotates inline with confidence tier before the code 
   reaches you.

4. Rust gate (3.4ms, 98× faster than Python) — PreToolUse hook blocks Write/Edit/Bash 
   when the planned action overlaps an unverified constraint.

5. Cross-session memory — new sessions inherit epistemic status, not just values.

The probe is deterministic (no model cooperation required). The FCR goes to zero 
deterministically across all 50 test scenarios.

We also found a harder failure: "ghost constraints" — facts stated without hedging that 
are actually uncertain (e.g., a rate limit from a sales call stated as fact). The probe 
can't see these. We added an opt-in Ghost Detector (one Opus call per constraint at 
registration) that classifies by provenance rather than surface text.

Ghost Gauntlet: Credence BothRate = 1.000 vs naive window = 0.200 (n=10 sessions).

All results are in evals/*.json and reproducible. 611 passing tests, all passing. 
Most evals run without an API key.

GitHub: https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
pip install credence-ai

Happy to answer questions about methodology, the measurement design, or why 
"just use a longer context window" doesn't fully solve this (short answer: it helps 
for compression, not for multi-agent handoffs or cross-session propagation).
```

---

## Twitter / X Thread

**Tweet 1 (hook):**
```
You told your AI the rate limit was "probably 50 — unverified."

10 turns later it wrote:

  RATE_LIMIT = 50

No warning. The word "probably" was compressed away.
You shipped it. The real limit was 10.

This failure has a name. We measured it.

🧵
```

**Tweet 2 (the measurement):**
```
We ran 50 compression scenarios.

When Claude uses Haiku to summarize old context:
→ 46% of "I think / unverified / probably" markers get stripped
→ 6% of the time the downstream model states it as confirmed fact

With LLMLingua-style scoring (keeps "informative" content, drops "qualifiers"):
→ 68% strip rate
→ 74% False Certainty Rate

We call it EQL (Epistemic Qualifier Loss) and FCR (False Certainty Rate).
```

**Tweet 3 (the fix — deterministic):**
```
The fix we built is deterministic — no model cooperation required.

Before Haiku runs: scan for "I think / unverified / probably" (198 terms).
Found → block compression → keep the turn verbatim.

Block rate: 100% across n=50
FCR after: 0%

0.07ms. Zero API calls.
```

**Tweet 4 (the harder problem):**
```
But there's a harder case.

"The Stripe rate limit is 50 req/min."

No hedging. Stated as fact. 
Actually from a sales call, never confirmed.

The probe sees nothing. We call these "ghost constraints."

Only Opus can classify by provenance — is this an established fact 
or a vendor claim stated as fact?
```

**Tweet 5 (ghost gauntlet):**
```
Ghost Gauntlet result (n=10 sessions, all Opus 4.7):

Credence (Ghost Detector active): BothRate = 1.000
Naive sliding window:             BothRate = 0.200

Value AND qualifier both recalled. 
5× improvement.
```

**Tweet 6 (the full stack):**
```
The full stack:

CP1 — Faithfulness probe (0.07ms) — blocks Haiku
CP2 — Truth Buffer — injects unverified constraints into every system prompt
CP3 — Generation scanner — annotates unverified literals inline in code
CP4 — Rust gate (3.4ms) — blocks Write/Edit/Bash against unverified constraints
CP5 — Cross-session memory — new sessions inherit uncertainty status

Total overhead: ~0.56ms + 3.4ms gate.
Zero extra API calls.
```

**Tweet 7 (call to action):**
```
611 passing tests. All passing.
Most evals reproducible without an API key.

pip install credence-ai

GitHub → github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai

Technical report in docs/TECHNICAL_REPORT.md if you want the methodology.

Built with Claude Code, to protect Claude Code users from Claude Code's own failure mode.
```

---

## Reddit Posts

**r/MachineLearning title:**
```
[Project] Credence: measuring and fixing epistemic qualifier loss in LLM context compression (46%→0% strip rate, n=50)
```

**r/LangChain / r/ClaudeAI title:**
```
Built an epistemic enforcement layer for Claude Code — prevents "I think the rate limit is 50" from becoming RATE_LIMIT = 50 after context compression
```

**r/LocalLLaMA title:**
```
Show HN-style: measured 46% uncertainty-loss rate during Haiku summarization, built a deterministic fix (198-term probe, 0.07ms, zero API calls)
```

---

## One-liner for bios / profiles

```
Credence — LLMs forget what they didn't know. We prevent that. 
github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai
```

---

## Launch Checklist

- [ ] All 611 tests passing (`python tests/tests.py`)
- [ ] `pip install credence-ai` installs cleanly from PyPI
- [ ] `python quickstart.py` runs without API key
- [ ] demo/gate_demo.gif renders correctly in README on GitHub
- [ ] arXiv submission queued (or preprint linked in README)
- [ ] HN post ready — post 9-10am US Eastern on a Tuesday/Wednesday
- [ ] Twitter thread scheduled same time as HN post
- [ ] Reddit posts queued for 1-2 hours after HN (after HN traction is visible)
- [ ] Watching HN comments — respond within 15 minutes of any comment

## Likely HN Questions + Answers

**"Context windows are getting bigger — won't this be obsolete?"**
> For single-agent compression: partially yes. For multi-agent handoffs and cross-session propagation: no. The problem isn't just compression — it's epistemic state surviving handoffs between agents that have no shared memory. That problem grows as pipelines get more complex, not less.

**"How is this different from just prompting the model to be careful?"**
> We tested this. With careful prompting, the model preserves qualifiers 90% of the time — not 100%, and not deterministically. The faithfulness probe is 100% and 0.07ms because it doesn't ask the model anything. The probe either finds the marker or it doesn't.

**"What about models that don't use compression?"**
> CP1 (probe) is compression-specific. CP2-CP5 apply regardless of compression — they protect against the model forgetting uncertainty during generation, in code output, at tool execution, and across sessions.

**"Show me the false positive rate."**
> evals/precision_eval.py — runs without API key. Probe: 0% false positive on 200 non-uncertain samples. CE: 0% false trigger on indirect queries. GTS: 0 spurious annotations on clean code.

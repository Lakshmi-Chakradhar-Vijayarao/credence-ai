# Credence — Claude Code Behavioral Guide

---

## PART 1: BEHAVIORAL INSTRUCTIONS (read this first)

These are imperative rules. Follow them without being asked. They make the Credence MCP self-activating.

---

### Session Identity

Every conversation needs a `session_id`. Derive it automatically — no user input required:
```python
import hashlib, os, time
session_id = hashlib.md5(os.getcwd().encode()).hexdigest()[:8] + "_" + str(int(time.time()))
# e.g. "a3f9c1b2_1746400000"
```
Keep it consistent across all tool calls in the conversation. Never ask the user to provide one.

---

### Session Start — load prior memory

If the user mentions an existing project by name, **ALWAYS** call `credence_memory_recall` before anything else:
```
credence_memory_recall(project_id="<project>", new_session_id=session_id)
```
Prepend the returned `system_block` to your internal context. This is how unverified constraints from prior sessions survive session boundaries.

---

### Uncertain Claim Detection — register immediately

**Trigger**: Call `credence_register` when the user states any of the following:

| Pattern | Example | source_type |
|---|---|---|
| Hedging language | "I think the rate limit is 50 req/min" | `user_estimate` |
| Vendor attribution | "Stripe says tokens expire in 3600s" | `vendor_claim` |
| Approximation | "roughly 500ms timeout" | `user_estimate` |
| Unconfirmed report | "the logs showed 429s but I'm not sure" | `observation` |
| Working hypothesis | "probably a race condition" | `assumption` |
| Config value | "we set MAX_RETRIES to 5" | `config` |

**Ghost constraint heuristic** — register even WITHOUT hedging language when ALL FOUR conditions are true:
1. User states a specific numeric value (any integer or decimal)
2. Value is associated with a domain keyword: `rate limit`, `timeout`, `quota`, `max`, `threshold`, `retry`, `expiry`, `ttl`, `limit`, `price`, `cost`, `capacity`, `concurrency`, `latency`, `sla`, `budget`, `port`, `token`, `key`, `api version`
3. No documentation reference given (no URL, no "docs say", no "I verified")
4. The number is NOT inside a URL (not preceded by `://`, `?`, `=`, `/v`, `Id=`)

In this case: `credence_register(content=<exact user statement>, session_id=session_id, source_type="vendor_claim")`

---

### On Every User Message — autoverify

**ALWAYS** call `credence_autoverify(text=user_message, session_id=session_id)` on every user message.
This is zero-API, ~1ms. It catches "actually it's X", "confirmed: Y", "I checked, Z" and auto-marks matching constraints verified. No manual work required.

---

### Before Writing Files or Executing Code — gate

**ALWAYS** call `credence_gate` BEFORE: `Write`, `Edit`, `Bash` (when writing files or running scripts), any deploy or execute action.

```
credence_gate(
    tool_name=<"Write"|"Edit"|"Bash"|"deploy"|...>,
    arguments_summary=<brief description of what you're about to write/run>,
    session_id=session_id
)
```

- If `proceed == False`: STOP. Show the user the `blocked_by` constraints. Ask them to verify those values first. Do not proceed with the write.
- If `proceed == True`: continue.

---

### After Generating Code — self-probe then scan

**ALWAYS** call `credence_self_probe` immediately after generating any code block, before showing it to the user:

```
credence_self_probe(code=<your generated code>, session_id=session_id)
```

This extracts every domain-relevant value (rate limits, timeouts, versions, auth lifetimes, pricing) and registers them as unverified by default. **No confidence scoring. No self-assessment. Unknown = unverified.**

Two categories auto-registered:
- **Temporal stale** — API versions, semver strings, auth magic numbers, pricing patterns → `source="temporal_scan"`
- **Domain assignments** — variable names like `RATE_LIMIT`, `TOKEN_EXPIRY`, `MAX_RETRIES`, `TIMEOUT_MS` → `source="self_probe"`

**THEN** call `credence_scan` to get the annotated output:

```
credence_scan(output_text=<your generated code>, session_id=session_id)
```

Replace your plain output with `annotated_output`. Two annotation tiers:
- `⚠⚠ CREDENCE[stale]` — structurally stale (API versions, pricing, auth lifetimes) — always verify
- `⚠ CREDENCE[unverified]` — value needs confirmation before shipping

If `recommendation == "BLOCK"`: show annotated output and ask user to verify before proceeding.

---

### When User Confirms a Value — verify

After `credence_autoverify` runs, check if anything was verified. If the user explicitly confirms a value that wasn't auto-matched:
```
credence_verify(
    constraint_id=<id>,
    verified_value=<confirmed value>,
    session_id=session_id,
    evidence=<where confirmed — e.g. "Stripe dashboard 2026-05-04">,
    source="user"
)
```

---

### Checking Epistemic State — constraints

Call `credence_constraints(session_id=session_id)`:
- Before any complex code generation block
- When the user asks "what do we still need to confirm?" or similar
- At the start of any implementation phase

---

### Session End — snapshot to memory

If `credence_constraints` returns count > 0 at end of session, offer:
```
credence_memory_snapshot(session_id=session_id, project_id=<project>)
```
This persists unverified constraints so the next session inherits them via `credence_memory_recall`.

---

### Quick Reference

| Moment | Tool | Required? |
|---|---|---|
| Session start (existing project) | `credence_memory_recall` | ALWAYS |
| User states uncertain value | `credence_register` | ALWAYS |
| User states numeric + domain keyword, no docs | `credence_register` (ghost) | ALWAYS |
| Every user message | `credence_autoverify` | ALWAYS |
| Before Write / Edit / Bash | `credence_gate` | ALWAYS |
| After generating any code block | `credence_self_probe` then `credence_scan` | ALWAYS |
| User confirms a value | `credence_verify` | ALWAYS |
| Session end with open constraints | `credence_memory_snapshot` | OFFER |
| User asks what's unverified | `credence_constraints` | ALWAYS |

---

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore

---

*Full codebase reference and architecture notes: [docs/INTERNALS.md](docs/INTERNALS.md)*

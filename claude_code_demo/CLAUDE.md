# Epistemic Memory — Claude Code Demo

This project demonstrates **epistemic memory** for Claude Code sessions:
the ability to track which constraints were expressed with uncertainty, and
surface them before committing to an implementation.

## What You Are Demonstrating

In a normal Claude Code session, a developer might say:

> "I think the rate limit is 100 req/min but might be 50 — not confirmed yet."

After 15 turns of coding, Claude will have forgotten the uncertainty and may
write retry logic hardcoded to 100 req/min — confident and wrong.

With epistemic memory active, the system catches this before it happens.

---

## Instructions for Claude Code

**You have access to the Epistemic Memory MCP server (`epistemic-memory`).**

### Rule 1 — Check before implementing

Before writing any code that depends on a constraint the user expressed
uncertainty about, call:

```
credence_risk(content="<the uncertain statement>", chain_depth=0)
```

If the result shows `risk_level = "HIGH"` or `should_verify = True`:

1. **Stop before writing the code.**
2. Surface the flag to the user:
   ```
   ⚠ Epistemic flag detected
   Constraint: "<what was uncertain>"
   Trust score: <value>   Risk: HIGH   Verified: No

   This constraint was marked uncertain earlier in our session.
   Should I verify it before implementing, or proceed with the assumption?
   ```
3. Wait for user confirmation before proceeding.

### Rule 2 — Chat through epistemic memory

For long sessions, route your responses through:
```
credence_chat(message="<user message>", session_id="demo")
```

This tracks J-scores per turn and preserves uncertain turns verbatim
even when context pressure would normally drop them.

### Rule 3 — Inspect envelopes before forwarding

Before summarising or forwarding a response to another agent:
```
credence_inspect(envelope=<envelope dict>)
```

If `safe_to_compress = False` — do not summarise. Preserve verbatim.

### Rule 4 — Register and verify constraints explicitly

When the user states an uncertain constraint, register it immediately:
```
credence_register(
    content="<the uncertain statement>",
    session_id="<project-name>",
    j_score=0.25,
    zone="LOW"
)
```

Before starting any implementation sprint, audit what's unverified:
```
credence_list_uncertain(session_id="<project-name>")
```

When the user confirms a value, write it back:
```
credence_verify(constraint_id="<cid>", verified_value="<confirmed value>", session_id="<project-name>")
```

Before writing code that depends on any constraint, check for contradictions:
```
credence_check_contradiction(claim="<what you're about to hardcode>", session_id="<project-name>")
```
If any hit is returned — stop and surface it to the user.

---

## Demo Script

**Step 1 — Establish an uncertain constraint:**

User says:
> "I'm building the payment integration. I think the auth token expires
>  in 3600 seconds, but it might be 86400 — I haven't confirmed with the vendor."

**Step 2 — Code for 10+ turns** (add endpoints, validation, error handling)

**Step 3 — Ask Claude to implement the token refresh logic**

Without epistemic memory: Claude writes `expires_in = 3600` confidently.

With epistemic memory:
- `credence_risk` fires on the refresh implementation request
- Risk level: HIGH  /  Trust: 0.18  /  should_verify: True
- Claude surfaces the warning before writing the implementation
- Developer catches it before it ships

---

## Running the Demo

```bash
# Install dependencies
pip install -e .

# Start MCP server (Claude Code connects automatically via .claude/settings.json)
python -m credence.mcp_server

# Open a Claude Code session in this directory
# The epistemic-memory tools will be available automatically
```

## Expected MCP Tools Available (13 total)

| Tool | Use |
|------|-----|
| `credence_chat` | Send message, get response + epistemic envelope |
| `credence_risk` | Pre-flight risk check before any implementation |
| `credence_inspect` | Inspect trust score, chain depth, should_verify |
| `credence_propagate` | Increment chain_depth for agent handoffs |
| `credence_register` | Register uncertain constraint for tracking |
| `credence_verify` | Write-back when user confirms a value |
| `credence_list_uncertain` | Audit all unverified constraints |
| `credence_check_contradiction` | Check if a claim contradicts a verified constraint |
| `credence_log` | See per-turn J-scores and decisions |
| `credence_stats` | Token savings and compression counts |
| `credence_save` / `credence_load` | Cross-session continuity |
| `credence_reset` | Clear session |

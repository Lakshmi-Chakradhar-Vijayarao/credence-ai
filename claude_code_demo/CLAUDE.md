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
em_propagation_risk(content="<the uncertain statement>", chain_depth=0)
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
cams_chat(message="<user message>", session_id="demo")
```

This tracks J-scores per turn and preserves uncertain turns verbatim
even when context pressure would normally drop them.

### Rule 3 — Inspect envelopes before forwarding

Before summarising or forwarding a response to another agent:
```
cams_inspect_envelope(envelope=<envelope dict>)
```

If `safe_to_compress = False` — do not summarise. Preserve verbatim.

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
- `em_propagation_risk` fires on the refresh implementation request
- Risk level: HIGH  /  Trust: 0.18  /  should_verify: True
- Claude surfaces the warning before writing the implementation
- Developer catches it before it ships

---

## Running the Demo

```bash
# Install dependencies
pip install -e .

# Start MCP server (Claude Code connects automatically via .claude/settings.json)
python -m cams.mcp_server

# Open a Claude Code session in this directory
# The epistemic-memory tools will be available automatically
```

## Expected MCP Tools Available

| Tool | Use |
|------|-----|
| `cams_chat` | Send message, get response + epistemic envelope |
| `em_propagation_risk` | Pre-flight risk check before any implementation |
| `cams_inspect_envelope` | Inspect trust score, chain depth, should_verify |
| `cams_get_decision_log` | See per-turn J-scores and decisions |
| `cams_get_stats` | Token savings and compression counts |

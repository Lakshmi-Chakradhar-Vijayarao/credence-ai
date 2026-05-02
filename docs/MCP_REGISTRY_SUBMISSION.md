# MCP Registry Submission

Submit to: https://registry.modelcontextprotocol.io/
Process: Web UI → GitHub OAuth → submit server.json

The registry uses the `mcp-publisher` CLI / web interface, not a direct GitHub PR.

---

## server.json (paste into publisher wizard)

```json
{
  "$schema": "https://registry.modelcontextprotocol.io/schema/v0/server.json",
  "name": "credence-guard",
  "description": "Epistemic enforcement layer that prevents uncertainty qualifiers from being silently stripped during LLM context compression. Measured: 60% qualifier strip rate under naive Haiku compression → 0% with faithfulness probe (0.017ms, zero API calls). 22 tools covering registration, enforcement, cross-session memory, and Rust PreToolUse gate (3.4ms). Fully local — no data leaves the machine.",
  "version": "1.0.0",
  "repository": {
    "url": "https://github.com/Lakshmi-Chakradhar-Vijayarao/credence-ai",
    "source": "github"
  },
  "packages": [
    {
      "registryType": "pypi",
      "identifier": "credence-guard",
      "version": "1.0.0",
      "transport": {
        "type": "stdio"
      },
      "runtimeHint": "credence-server",
      "environmentVariables": []
    }
  ]
}
```

---

## Submission Steps (10 minutes, free)

1. Go to https://registry.modelcontextprotocol.io/
2. Click **"📤 Publish my MCP server"**
3. Sign in with GitHub (OAuth — uses your `Lakshmi-Chakradhar-Vijayarao` account)
4. Namespace: `io.github.Lakshmi-Chakradhar-Vijayarao/credence-ai`
5. Paste the `server.json` above or fill in the wizard fields
6. Submit — review typically completes within 2-5 days

---

## Anthropic Connectors Directory (second listing, also free)

This is separate from the MCP community registry and requires a brief partner application.

- URL: https://anthropic.com/partners/mcp
- What to say: "Epistemic enforcement layer for Claude Code. Prevents qualifier loss during context compression. Measured result. Fully local, zero API calls."
- Category: Safety / Context Management
- Contact: partnerships@anthropic.com or use the form on the page

---

## Quick-install copy for announcements

```bash
pip install "credence-guard[mcp]"
```

`.mcp.json`:
```json
{
  "mcpServers": {
    "credence-guard": {
      "type": "stdio",
      "command": "credence-server"
    }
  }
}
```

---

## Announcement channels (ordered by leverage)

| Channel | Format | ETA |
|---|---|---|
| Hacker News Show HN | "Show HN: Credence – prevents Claude from forgetting what it didn't know" | Day 1 |
| r/ClaudeAI | Share the FCR table + demo GIF | Day 1 |
| r/LocalLLaMA | Technical framing — probe mechanism + Rust gate numbers | Day 1 |
| r/MachineLearning | arXiv preprint link (once submitted) | Day 3+ |
| LinkedIn | 3-paragraph post with 0%→46% headline | Day 1 |
| X / Twitter | Thread: problem → measurement → fix → install | Day 1 |

---

## arXiv Submission (free, establishes permanent record)

The paper is 90% written in `docs/TECHNICAL_REPORT.md`.

Steps:
1. Convert to LaTeX (`docs/TECHNICAL_REPORT.md` → `paper.tex`) — ~2 hours
2. Submit to cs.AI + cs.CL: https://arxiv.org/submit
3. arXiv IDs are assigned within 1-2 business days

Headline result for abstract: "We measure a 60% epistemic qualifier strip rate under naive LLM context compression and show a deterministic faithfulness probe reduces this to 0% at 0.017ms with zero API calls."

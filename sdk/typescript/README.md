# credence-guard TypeScript SDK

TypeScript SDK for the Credence epistemic enforcement layer.

## Current Scope

| Feature | Status | Notes |
|---|---|---|
| `runProbe()` | ✅ Production-ready | Zero deps, 198-marker faithfulness probe, works in Node 18+ and browser |
| `CredenceEnvelope` | ✅ Production-ready | ETP envelope: trust_score, chain_depth, should_verify, safe_to_compress |
| `CredenceRegistryClient` | ⚠️ Preview | HTTP client — requires credence-server running in HTTP mode (see below) |

## Installation

```bash
npm install credence-guard
# or
bun add credence-guard
```

## Usage

### Faithfulness Probe (zero dependencies, works everywhere)

```typescript
import { runProbe } from 'credence-guard';

const result = runProbe("The rate limit is probably around 50 — I haven't confirmed it");
console.log(result.blocked);       // true — uncertainty detected
console.log(result.markers_found); // ["probably", "haven't confirmed"]
console.log(result.latency_ms);    // ~0.02ms
```

### ETP Envelope (agent handoffs)

```typescript
import { CredenceEnvelope } from 'credence-guard';

const envelope = new CredenceEnvelope({
  content: "Auth token expiry is 3600s — unverified",
  j_score: 0.28,
  zone: "LOW",
  source: "user",
  verified: false,
});

console.log(envelope.trust_score);   // 0.18 (low — uncertainty_preserved + untrusted source)
console.log(envelope.should_verify); // true
console.log(envelope.safe_to_compress); // false

// Propagate to next agent (increments chain_depth, resets verified)
const propagated = envelope.propagate("agent-2");
```

### Registry Client (requires HTTP server)

`CredenceRegistryClient` communicates with `credence-server` over HTTP. The default MCP server runs on **stdio** (for Claude Code integration). To use the HTTP client, start the server in HTTP mode:

```bash
# HTTP mode (FastMCP SSE transport) — in development, target v1.1
credence-server --transport sse --port 3001
```

Once the HTTP server is running:

```typescript
import { CredenceRegistryClient } from 'credence-guard';

const client = new CredenceRegistryClient("http://localhost:3001", "my-session");

await client.register("rate limit is probably 50 req/min", 0.28, "LOW");
const { constraints } = await client.constraints();
await client.verify(constraints[0].constraint_id, "confirmed: 100 req/min");
```

> **Note:** HTTP transport is planned for v1.1. The stdio MCP server (for Claude Code) is the production integration path in v1.0. For non-Claude environments, use `runProbe()` directly.

## API Reference

### `runProbe(text: string): ProbeResult`
- `blocked: boolean` — true if uncertainty markers detected
- `markers_found: string[]` — which markers triggered
- `marker_count: number`
- `latency_ms: number`

### `CredenceEnvelope`
- `trust_score: number` — j_score − (chain_depth × 0.05) − source_penalty
- `should_verify: boolean` — trust_score < 0.40 and not verified
- `safe_to_compress: boolean` — trust ≥ 0.40, zone=HIGH, not uncertainty_preserved
- `.propagate(newSource): CredenceEnvelope` — chain_depth+1, verified=false
- `.verify(): CredenceEnvelope` — marks verified=true
- `.toDict() / CredenceEnvelope.fromDict()` — MCP-serializable

## Running Tests

```bash
bun install
bun test
```

/**
 * credence-guard TypeScript SDK
 *
 * Provides:
 *   - runProbe()             — faithfulness probe (zero-dependency, ~0.017ms)
 *   - CredenceEnvelope       — epistemic provenance wrapper for multi-agent pipelines
 *   - CredenceRegistryClient — HTTP client for the credence-server MCP server
 *
 * Zero runtime dependencies. Works in Node.js 18+ and browser (probe + envelope only).
 */

export { runProbe, UNCERTAINTY_MARKERS, type ProbeResult } from "./probe";
export {
  CredenceEnvelope,
  type EpistemicZone,
  type ContentType,
  type EnvelopeFields,
} from "./envelope";
export {
  CredenceRegistryClient,
  type Constraint,
  type RegisterResult,
  type SessionBrief,
  type ProjectStatus,
  type DiffResult,
} from "./registry-client";

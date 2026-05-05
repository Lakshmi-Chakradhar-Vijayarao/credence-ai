/**
 * CredenceRegistryClient — lightweight client for the Credence MCP server.
 *
 * Wraps HTTP calls to the credence-server JSON API.
 * For use in TypeScript/Node environments (Copilot, Cursor extensions, etc.)
 * without needing Python.
 *
 * All methods are async and return typed results.
 */

export interface Constraint {
  constraint_id:   string;
  content:         string;
  session_id:      string;
  j_score:         number;
  zone:            "LOW" | "MEDIUM" | "HIGH";
  verified:        boolean;
  verified_value:  string | null;
  created_at:      string;
  updated_at:      string;
  source:          string;
  constraint_type: string;
}

export interface RegisterResult {
  constraint_id: string;
  source_type:   string;
  j_score:       number;
  zone:          string;
  message:       string;
}

export interface SessionSummary {
  brief:              string;
  unverified_count:   number;
  high_risk_count:    number;
  action_required:    boolean;
  constraint_summaries: Array<{
    constraint_id: string;
    content:       string;
    tier:          string;
    j_score:       number;
  }>;
}

export interface ProjectStatus {
  project_id:        string;
  total_constraints: number;
  verified_count:    number;
  unverified_count:  number;
  epistemic_debt:    number;
  verified_rate:     number;
  high_risk_count:   number;
  disputed_count:    number;
  health:            string;
  health_message:    string;
  top_unresolved:    Constraint[];
  session_breakdown: Record<string, { total: number; verified: number; unverified: number }>;
}

export interface DiffResult {
  matched_claims:      unknown[];
  contradictions:      unknown[];
  registry_conflicts:  unknown[];
  divergence_score:    number;
  contradiction_count: number;
  recommendation:      string;
}

/**
 * Thin HTTP client for the Credence MCP server.
 *
 * @param baseUrl  URL of the credence-server (default: http://localhost:3001)
 * @param sessionId  Default session to use for all calls
 */
export class CredenceRegistryClient {
  private readonly baseUrl: string;
  private readonly sessionId: string;

  constructor(baseUrl = "http://localhost:3001", sessionId = "default") {
    this.baseUrl   = baseUrl.replace(/\/$/, "");
    this.sessionId = sessionId;
  }

  private async call<T>(tool: string, args: Record<string, unknown>): Promise<T> {
    const resp = await fetch(`${this.baseUrl}/tools/${tool}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(args),
    });
    if (!resp.ok) {
      throw new Error(`credence-server error ${resp.status}: ${await resp.text()}`);
    }
    return resp.json() as Promise<T>;
  }

  /** Register an uncertain constraint. */
  async register(
    content:     string,
    jScore      = 0.30,
    zone        = "LOW",
    sourceType  = "observation",
  ): Promise<RegisterResult> {
    return this.call<RegisterResult>("credence_register", {
      content, session_id: this.sessionId, j_score: jScore, zone, source_type: sourceType,
    });
  }

  /** Mark a constraint as verified. */
  async verify(constraintId: string, verifiedValue: string): Promise<Constraint> {
    return this.call<Constraint>("credence_verify", {
      constraint_id: constraintId, verified_value: verifiedValue,
    });
  }

  /** List unverified constraints for the current session. */
  async constraints(): Promise<{ constraints: Constraint[]; count: number }> {
    return this.call("credence_constraints", { session_id: this.sessionId });
  }

  /** Get a plain-English session summary of inherited uncertainties. */
  async sessionSummary(projectId = ""): Promise<SessionSummary> {
    return this.call<SessionSummary>("credence_session_summary", {
      session_id: this.sessionId, project_id: projectId,
    });
  }

  /** Compare two texts for epistemic divergence. */
  async diff(textA: string, textB: string): Promise<DiffResult> {
    return this.call<DiffResult>("credence_diff", {
      text_a: textA, text_b: textB, session_id: this.sessionId,
    });
  }

  /** Project-wide epistemic health. */
  async projectStatus(projectId: string): Promise<ProjectStatus> {
    return this.call<ProjectStatus>("credence_project_status", { project_id: projectId });
  }

  /** Check text before compression. */
  async preCompress(text: string): Promise<{ decision: string; blocked: boolean; message: string }> {
    return this.call("credence_pre_compress", { text, session_id: this.sessionId });
  }

  /** Gate before irreversible tool execution. */
  async gate(
    toolName:          string,
    argumentsSummary:  string,
  ): Promise<{ proceed: boolean; recommendation: string; blocked_by: unknown[] }> {
    return this.call("credence_gate", {
      tool_name: toolName, arguments_summary: argumentsSummary, session_id: this.sessionId,
    });
  }
}

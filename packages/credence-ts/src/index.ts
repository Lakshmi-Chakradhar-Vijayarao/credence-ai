/**
 * credence-ai — TypeScript SDK
 *
 * Epistemic memory layer for AI pipelines. Preserves uncertainty qualifiers
 * through compression, agent handoffs, and session boundaries.
 * Reference implementation of ETP v1.
 *
 * Usage:
 *   import { CredenceClient } from 'credence-ai'
 *
 *   const client = new CredenceClient({ apiKey: 'cr-your-key' })
 *
 *   const result = await client.chat({
 *     message: "I think the rate limit is 100 req/min — unconfirmed",
 *     sessionId: "payment-v2",
 *   })
 *   console.log(result.response)
 *   console.log(`J=${result.jScore}  zone=${result.zone}  decision=${result.decision}`)
 *
 *   if (result.alignmentWarnings.length > 0) {
 *     console.log("Governor flagged:", result.alignmentWarnings[0].suggestedCaveat)
 *   }
 */

// ---------------------------------------------------------------------------
// ETP Types
// ---------------------------------------------------------------------------

export type EpistemicZone = 'LOW' | 'MEDIUM' | 'HIGH'
export type EventType = 'register' | 'scout' | 'chat_update' | 'verify' | 'contradict'
export type RiskLevel = 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'

export interface EpistemicConstraint {
  constraintId:  string
  sessionId:     string
  content:       string
  jScore:        number
  zone:          EpistemicZone
  verified:      boolean
  verifiedValue: string | null
  createdAt:     string
  updatedAt:     string
}

export interface EpistemicEvent {
  eventId:      number
  constraintId: string
  timestamp:    string
  eventType:    EventType
  jScore:       number | null
  zone:         EpistemicZone | null
  notes:        string | null
}

export interface EpistemicEnvelope {
  content:              string
  jScore:               number
  zone:                 EpistemicZone
  source:               string
  verified:             boolean
  chainDepth:           number
  trustScore:           number
  shouldVerify:         boolean
  safeToCompress:       boolean
  uncertaintyPreserved: boolean
  contentType:          string
  sessionId:            string | null
}

export interface AlignmentWarning {
  constraintId:      string
  constraintContent: string
  ledgerZone:        EpistemicZone
  responseZone:      EpistemicZone
  overlapWords:      string[]
  suggestedCaveat:   string
}

export interface EpistemicLedger {
  sessionId:        string
  totalConstraints: number
  unverifiedCount:  number
  verifiedCount:    number
  constraints:      EpistemicConstraint[]
  etpVersion:       string
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

export interface ChatResult {
  response:             string
  jScore:               number
  zone:                 EpistemicZone
  decision:             'COMPRESS' | 'TRIM' | 'PRESERVE'
  tokensSaved:          number
  driftState:           boolean
  uncertaintyPreserved: boolean
  truthBufferCount:     number
  scoutExtractions:     number
  alignmentWarnings:    AlignmentWarning[]
  caveatInjected:       boolean
  autoRegistered:       boolean
  adaptiveThetaHigh:    number
  adaptiveThetaLow:     number
  envelope:             EpistemicEnvelope
}

export interface RiskResult {
  riskLevel:               RiskLevel
  jScore:                  number
  zone:                    EpistemicZone
  effectiveTrust:          number
  chainDepth:              number
  hasUncertainty:          boolean
  uncertaintyMarkersFound: string[]
  safeToCompress:          boolean
  shouldVerify:            boolean
  action:                  string
  reasoning:               string
}

export interface AlignResult {
  warningCount:      number
  alignmentWarnings: AlignmentWarning[]
  caveatNeeded:      boolean
  suggestedCaveats:  string[]
  governorActive:    boolean
}

export interface GateResult {
  proceed:          boolean
  blockedBy:        EpistemicConstraint[]
  unverifiedCount:  number
  recommendation:   string
}

export interface ContradictionResult {
  hasContradiction: boolean
  matchCount:       number
  matches:          EpistemicConstraint[]
  recommendation:   string
}

export interface SessionStats {
  totalTokensIn:    number
  totalTokensOut:   number
  totalTokensSaved: number
  totalCostUsd:     number
  compressionRatio: number
  turnsCompressed:  number
  turnsTrimmed:     number
  turnsPreserved:   number
  turnCount:        number
  driftState:       boolean
  regimeActive:     boolean
}

// ---------------------------------------------------------------------------
// Client config
// ---------------------------------------------------------------------------

export interface CredenceClientConfig {
  apiKey:   string
  baseUrl?: string
  timeout?: number
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

class CredenceAPIError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`Credence API error ${status}: ${detail}`)
    this.name = 'CredenceAPIError'
  }
}

async function httpRequest<T>(
  method:  'GET' | 'POST' | 'DELETE',
  url:     string,
  body:    unknown,
  apiKey:  string,
  timeout: number,
): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const resp = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key':    apiKey,
        'User-Agent':   'credence-ts-sdk/1.0.0',
      },
      body:   body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })

    if (!resp.ok) {
      let detail = resp.statusText
      try { detail = (await resp.json() as { detail?: string }).detail ?? detail } catch {}
      throw new CredenceAPIError(resp.status, detail)
    }

    return await resp.json() as T
  } finally {
    clearTimeout(timer)
  }
}

// ---------------------------------------------------------------------------
// CredenceClient
// ---------------------------------------------------------------------------

/** Convert snake_case API response keys to camelCase. */
function toCamel(data: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(data)) {
    const camel = k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase())
    out[camel] = v
  }
  return out
}

function parseChatResult(raw: Record<string, unknown>): ChatResult {
  const d = toCamel(raw)
  return {
    response:             d['response'] as string,
    jScore:               d['jScore'] as number,
    zone:                 d['zone'] as EpistemicZone,
    decision:             d['decision'] as ChatResult['decision'],
    tokensSaved:          d['tokensSaved'] as number,
    driftState:           d['driftState'] as boolean,
    uncertaintyPreserved: d['uncertaintyPreserved'] as boolean,
    truthBufferCount:     d['truthBufferCount'] as number,
    scoutExtractions:     d['scoutExtractions'] as number,
    alignmentWarnings:    (d['alignmentWarnings'] as unknown[] ?? []).map(w =>
      toCamel(w as Record<string, unknown>) as unknown as AlignmentWarning
    ),
    caveatInjected:       d['caveatInjected'] as boolean ?? false,
    autoRegistered:       d['autoRegistered'] as boolean ?? false,
    adaptiveThetaHigh:    d['adaptiveThetaHigh'] as number ?? 0.70,
    adaptiveThetaLow:     d['adaptiveThetaLow'] as number ?? 0.45,
    envelope:             toCamel(d['envelope'] as Record<string, unknown> ?? {}) as unknown as EpistemicEnvelope,
  }
}

export class CredenceClient {
  private readonly apiKey:  string
  private readonly baseUrl: string
  private readonly timeout: number

  constructor(config: CredenceClientConfig) {
    this.apiKey  = config.apiKey
    this.baseUrl = (config.baseUrl ?? process.env['CREDENCE_API_URL'] ?? 'https://api.credence-ai.io').replace(/\/$/, '')
    this.timeout = config.timeout ?? 30_000
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return httpRequest<T>('POST', `${this.baseUrl}${path}`, body, this.apiKey, this.timeout)
  }

  private get<T>(path: string): Promise<T> {
    return httpRequest<T>('GET', `${this.baseUrl}${path}`, undefined, this.apiKey, this.timeout)
  }

  private delete<T>(path: string): Promise<T> {
    return httpRequest<T>('DELETE', `${this.baseUrl}${path}`, undefined, this.apiKey, this.timeout)
  }

  // ------------------------------------------------------------------
  // Core: chat
  // ------------------------------------------------------------------

  /** Send a message and receive a response with epistemic envelope. */
  async chat(params: { message: string; sessionId?: string }): Promise<ChatResult> {
    const raw = await this.post<Record<string, unknown>>('/v1/chat', {
      session_id: params.sessionId ?? 'default',
      message:    params.message,
    })
    return parseChatResult(raw)
  }

  // ------------------------------------------------------------------
  // Epistemic risk
  // ------------------------------------------------------------------

  /** Pre-flight risk assessment before compressing or forwarding content. */
  async risk(params: { content: string; chainDepth?: number }): Promise<RiskResult> {
    const raw = await this.post<Record<string, unknown>>('/v1/risk', {
      content:     params.content,
      chain_depth: params.chainDepth ?? 0,
    })
    return toCamel(raw) as unknown as RiskResult
  }

  // ------------------------------------------------------------------
  // Output Alignment (Governor)
  // ------------------------------------------------------------------

  /** Check if a response is more confident than the ledger warrants. */
  async align(params: { responseText: string; sessionId?: string }): Promise<AlignResult> {
    const raw = await this.post<Record<string, unknown>>('/v1/align', {
      session_id:    params.sessionId ?? 'default',
      response_text: params.responseText,
    })
    const d = toCamel(raw)
    return {
      warningCount:      d['warningCount'] as number,
      alignmentWarnings: (d['alignmentWarnings'] as unknown[] ?? []).map(w =>
        toCamel(w as Record<string, unknown>) as unknown as AlignmentWarning
      ),
      caveatNeeded:      d['caveatNeeded'] as boolean,
      suggestedCaveats:  d['suggestedCaveats'] as string[],
      governorActive:    d['governorActive'] as boolean,
    }
  }

  // ------------------------------------------------------------------
  // Epistemic ledger
  // ------------------------------------------------------------------

  /** Register an uncertain constraint. Returns constraint_id. */
  async register(params: {
    content:   string
    sessionId?: string
    jScore?:   number
    zone?:     EpistemicZone
  }): Promise<string> {
    const result = await this.post<{ constraint_id: string }>('/v1/register', {
      content:    params.content,
      session_id: params.sessionId ?? 'default',
      j_score:    params.jScore ?? 0.30,
      zone:       params.zone  ?? 'LOW',
    })
    return result.constraint_id
  }

  /** Mark a constraint as verified with its confirmed value. */
  async verify(params: {
    constraintId:  string
    verifiedValue: string
    sessionId?:    string
  }): Promise<Record<string, unknown>> {
    return this.post('/v1/verify', {
      constraint_id:  params.constraintId,
      verified_value: params.verifiedValue,
      session_id:     params.sessionId ?? 'default',
    })
  }

  /** Return the full epistemic ledger for a session. */
  async ledger(sessionId = 'default'): Promise<EpistemicLedger> {
    const raw = await this.get<Record<string, unknown>>(`/v1/ledger/${sessionId}`)
    return toCamel(raw) as unknown as EpistemicLedger
  }

  /** Return only unverified constraints for a session. */
  async uncertain(sessionId = 'default'): Promise<EpistemicConstraint[]> {
    const result = await this.get<{ constraints: EpistemicConstraint[] }>(`/v1/ledger/${sessionId}/uncertain`)
    return result.constraints
  }

  /** Return a single constraint with full certainty trajectory. */
  async constraint(constraintId: string): Promise<{ constraint: EpistemicConstraint; trajectory: EpistemicEvent[] }> {
    return this.get(`/v1/constraint/${constraintId}`)
  }

  /** Check if a claim contradicts verified constraints. */
  async contradiction(params: { claim: string; sessionId?: string }): Promise<ContradictionResult> {
    const raw = await this.post<Record<string, unknown>>('/v1/contradiction', {
      claim:      params.claim,
      session_id: params.sessionId ?? 'default',
    })
    return toCamel(raw) as unknown as ContradictionResult
  }

  // ------------------------------------------------------------------
  // Agentic gate
  // ------------------------------------------------------------------

  /** Block tool calls when unverified constraints may affect the action. */
  async gate(params: {
    toolName:          string
    argumentsSummary:  string
    sessionId?:        string
  }): Promise<GateResult> {
    const raw = await this.post<Record<string, unknown>>('/v1/gate', {
      tool_name:          params.toolName,
      arguments_summary:  params.argumentsSummary,
      session_id:         params.sessionId ?? 'default',
    })
    return toCamel(raw) as unknown as GateResult
  }

  // ------------------------------------------------------------------
  // Session management
  // ------------------------------------------------------------------

  /** Return session statistics. */
  async stats(sessionId = 'default'): Promise<SessionStats> {
    const raw = await this.get<Record<string, unknown>>(`/v1/stats/${sessionId}`)
    return toCamel(raw) as unknown as SessionStats
  }

  /** Return per-turn decision log. */
  async log(sessionId = 'default'): Promise<Record<string, unknown>[]> {
    return this.get(`/v1/log/${sessionId}`)
  }

  /** Reset a session, clearing all history and stats. */
  async reset(sessionId = 'default'): Promise<{ status: string }> {
    return this.delete(`/v1/session/${sessionId}`)
  }

  // ------------------------------------------------------------------
  // Health
  // ------------------------------------------------------------------

  async health(): Promise<{ status: string; version: string; uptime_s: number }> {
    return this.get('/health')
  }
}

// Re-export error class
export { CredenceAPIError }

/**
 * CredenceEnvelope — epistemic provenance wrapper.
 * Port of credence/envelope.py.
 *
 * Attach to any AI-generated response before passing it to the next agent.
 * Trust degrades by 0.05 per agent hop. Unknown sources incur +0.10 penalty.
 */

export type EpistemicZone = "LOW" | "MEDIUM" | "HIGH";
export type ContentType   = "text" | "code" | "error" | "math";

const TRUSTED_SOURCES      = new Set(["credence", "user", "system"]);
const CHAIN_DEPTH_PENALTY  = 0.05;
const VERIFY_THRESHOLD     = 0.40;
const UNKNOWN_SRC_PENALTY  = 0.10;

export interface EnvelopeFields {
  content:               string;
  jScore:                number;
  zone:                  EpistemicZone;
  source:                string;
  verified:              boolean;
  chainDepth:            number;
  uncertaintyPreserved:  boolean;
  contentType:           ContentType;
  sessionId?:            string | null;
}

/**
 * Immutable epistemic provenance wrapper.
 *
 * Use .propagate() to create the next-hop envelope (increments chainDepth).
 * Use .verify()   to create a verified copy.
 * Use .toDict()   to get a JSON-serializable plain object (MCP-transportable).
 */
export class CredenceEnvelope {
  readonly content:               string;
  readonly jScore:                number;
  readonly zone:                  EpistemicZone;
  readonly source:                string;
  readonly verified:              boolean;
  readonly chainDepth:            number;
  readonly uncertaintyPreserved:  boolean;
  readonly contentType:           ContentType;
  readonly sessionId:             string | null;

  constructor(fields: EnvelopeFields) {
    this.content              = fields.content;
    this.jScore               = fields.jScore;
    this.zone                 = fields.zone;
    this.source               = fields.source;
    this.verified             = fields.verified;
    this.chainDepth           = fields.chainDepth;
    this.uncertaintyPreserved = fields.uncertaintyPreserved;
    this.contentType          = fields.contentType;
    this.sessionId            = fields.sessionId ?? null;
    Object.freeze(this);
  }

  get sourceTrustPenalty(): number {
    return TRUSTED_SOURCES.has(this.source) ? 0 : UNKNOWN_SRC_PENALTY;
  }

  get trustScore(): number {
    const raw = this.jScore
      - this.chainDepth * CHAIN_DEPTH_PENALTY
      - this.sourceTrustPenalty;
    return Math.round(Math.max(0, raw) * 10000) / 10000;
  }

  get shouldVerify(): boolean {
    return this.trustScore < VERIFY_THRESHOLD && !this.verified;
  }

  get safeToCompress(): boolean {
    return (
      this.trustScore >= VERIFY_THRESHOLD &&
      !this.uncertaintyPreserved &&
      !this.shouldVerify &&
      this.zone === "HIGH"
    );
  }

  /** Create the next-hop envelope (chain_depth+1, verified reset to false). */
  propagate(newSource?: string): CredenceEnvelope {
    return new CredenceEnvelope({
      content:              this.content,
      jScore:               this.jScore,
      zone:                 this.zone,
      source:               newSource ?? this.source,
      verified:             false,
      chainDepth:           this.chainDepth + 1,
      uncertaintyPreserved: this.uncertaintyPreserved,
      contentType:          this.contentType,
      sessionId:            this.sessionId,
    });
  }

  /** Return a verified copy. */
  verify(): CredenceEnvelope {
    return new CredenceEnvelope({ ...this.toFields(), verified: true });
  }

  private toFields(): EnvelopeFields {
    return {
      content:              this.content,
      jScore:               this.jScore,
      zone:                 this.zone,
      source:               this.source,
      verified:             this.verified,
      chainDepth:           this.chainDepth,
      uncertaintyPreserved: this.uncertaintyPreserved,
      contentType:          this.contentType,
      sessionId:            this.sessionId,
    };
  }

  /** Plain object — JSON-serializable, MCP-transportable. */
  toDict(): Record<string, unknown> {
    return {
      content:               this.content,
      j_score:               this.jScore,
      zone:                  this.zone,
      source:                this.source,
      verified:              this.verified,
      chain_depth:           this.chainDepth,
      uncertainty_preserved: this.uncertaintyPreserved,
      content_type:          this.contentType,
      session_id:            this.sessionId,
      trust_score:           this.trustScore,
      should_verify:         this.shouldVerify,
      safe_to_compress:      this.safeToCompress,
    };
  }

  /** Reconstruct from a plain dict (e.g. received over MCP). */
  static fromDict(d: Record<string, unknown>): CredenceEnvelope {
    return new CredenceEnvelope({
      content:              d["content"] as string,
      jScore:               d["j_score"] as number,
      zone:                 d["zone"] as EpistemicZone,
      source:               d["source"] as string,
      verified:             d["verified"] as boolean,
      chainDepth:           d["chain_depth"] as number,
      uncertaintyPreserved: d["uncertainty_preserved"] as boolean,
      contentType:          d["content_type"] as ContentType,
      sessionId:            (d["session_id"] as string | null | undefined) ?? null,
    });
  }
}

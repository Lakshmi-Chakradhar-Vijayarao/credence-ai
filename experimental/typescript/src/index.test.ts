/**
 * credence-guard TypeScript SDK tests
 *
 * Pure unit tests — no network, no server required.
 * Tests probe accuracy and envelope trust decay math.
 */

import { runProbe }          from "./probe";
import { CredenceEnvelope }  from "./envelope";

// ---------------------------------------------------------------------------
// Probe tests
// ---------------------------------------------------------------------------

describe("runProbe", () => {
  test("blocks on 'i think'", () => {
    const r = runProbe("I think the rate limit is 50 req/min");
    expect(r.shouldBlock).toBe(true);
    expect(r.triggeredMarkers).toContain("i think");
  });

  test("blocks on 'approximately'", () => {
    const r = runProbe("The timeout is approximately 30 seconds");
    expect(r.shouldBlock).toBe(true);
    expect(r.triggeredMarkers).toContain("approximately");
  });

  test("blocks on 'unverified'", () => {
    const r = runProbe("This value is unverified and needs checking");
    expect(r.shouldBlock).toBe(true);
    expect(r.triggeredMarkers).toContain("unverified");
  });

  test("does not block on confident factual text", () => {
    const r = runProbe("The HTTP 200 status code means success.");
    expect(r.shouldBlock).toBe(false);
    expect(r.triggeredMarkers).toHaveLength(0);
  });

  test("latency is measured", () => {
    const r = runProbe("some text");
    expect(typeof r.latencyMs).toBe("number");
    expect(r.latencyMs).toBeGreaterThanOrEqual(0);
  });

  test("case-insensitive matching", () => {
    const r = runProbe("PROBABLY around 100");
    expect(r.shouldBlock).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Envelope tests
// ---------------------------------------------------------------------------

describe("CredenceEnvelope", () => {
  const makeEnv = (overrides: Partial<{
    jScore: number; source: string; chainDepth: number; verified: boolean;
    uncertaintyPreserved: boolean; zone: "LOW" | "MEDIUM" | "HIGH";
  }> = {}) =>
    new CredenceEnvelope({
      content:              "test content",
      jScore:               overrides.jScore    ?? 0.80,
      zone:                 overrides.zone       ?? "HIGH",
      source:               overrides.source     ?? "credence",
      verified:             overrides.verified   ?? false,
      chainDepth:           overrides.chainDepth ?? 0,
      uncertaintyPreserved: overrides.uncertaintyPreserved ?? false,
      contentType:          "text",
      sessionId:            "test-session",
    });

  test("trustScore at depth=0 trusted source equals jScore", () => {
    const env = makeEnv({ jScore: 0.80 });
    expect(env.trustScore).toBeCloseTo(0.80, 4);
  });

  test("trustScore degrades by 0.05 per hop", () => {
    const env = makeEnv({ jScore: 0.80, chainDepth: 2 });
    expect(env.trustScore).toBeCloseTo(0.70, 4);
  });

  test("unknown source gets 0.10 penalty", () => {
    const env = makeEnv({ jScore: 0.80, source: "external-agent" });
    expect(env.trustScore).toBeCloseTo(0.70, 4);
  });

  test("trustScore floors at 0", () => {
    const env = makeEnv({ jScore: 0.30, chainDepth: 10 });
    expect(env.trustScore).toBe(0);
  });

  test("shouldVerify when trust below 0.40", () => {
    const env = makeEnv({ jScore: 0.30, chainDepth: 0 });
    expect(env.shouldVerify).toBe(true);
  });

  test("shouldVerify=false when verified=true even if trust low", () => {
    const env = makeEnv({ jScore: 0.20, verified: true });
    expect(env.shouldVerify).toBe(false);
  });

  test("safeToCompress only when HIGH zone + trust sufficient", () => {
    const env = makeEnv({ jScore: 0.80, zone: "HIGH" });
    expect(env.safeToCompress).toBe(true);
  });

  test("safeToCompress=false when uncertaintyPreserved", () => {
    const env = makeEnv({ jScore: 0.80, zone: "HIGH", uncertaintyPreserved: true });
    expect(env.safeToCompress).toBe(false);
  });

  test("propagate increments chainDepth", () => {
    const env  = makeEnv({ chainDepth: 0 });
    const next = env.propagate();
    expect(next.chainDepth).toBe(1);
  });

  test("propagate resets verified to false", () => {
    const env  = makeEnv({ verified: true });
    const next = env.propagate();
    expect(next.verified).toBe(false);
  });

  test("propagate can update source", () => {
    const env  = makeEnv({ source: "credence" });
    const next = env.propagate("agent-b");
    expect(next.source).toBe("agent-b");
  });

  test("verify returns verified copy", () => {
    const env      = makeEnv({ verified: false });
    const verified = env.verify();
    expect(verified.verified).toBe(true);
    expect(env.verified).toBe(false); // original immutable
  });

  test("toDict is JSON-serializable with all computed fields", () => {
    const env  = makeEnv();
    const dict = env.toDict();
    expect(dict).toHaveProperty("trust_score");
    expect(dict).toHaveProperty("should_verify");
    expect(dict).toHaveProperty("safe_to_compress");
    expect(() => JSON.stringify(dict)).not.toThrow();
  });

  test("fromDict round-trips via toDict", () => {
    const env  = makeEnv({ jScore: 0.65, chainDepth: 1, source: "agent-x" });
    const copy = CredenceEnvelope.fromDict(env.toDict());
    expect(copy.jScore).toBe(env.jScore);
    expect(copy.chainDepth).toBe(env.chainDepth);
    expect(copy.source).toBe(env.source);
    expect(copy.trustScore).toBeCloseTo(env.trustScore, 4);
  });

  test("envelope is frozen (immutable)", () => {
    const env = makeEnv();
    expect(() => {
      // @ts-expect-error — testing runtime immutability
      env.jScore = 0.99;
    }).toThrow();
  });
});

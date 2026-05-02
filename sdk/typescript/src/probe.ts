/**
 * Faithfulness probe — deterministic, zero-dependency.
 * Port of credence/context_manager.py _UNCERTAINTY_MARKERS + _has_uncertainty().
 *
 * Checks whether text contains canonical uncertainty markers before compression.
 * If it does, compression should be BLOCKED to preserve epistemic qualifiers.
 */

export const UNCERTAINTY_MARKERS: ReadonlySet<string> = new Set([
  // Hedging phrases
  "i think", "i believe", "i'm not sure", "i am not sure", "not certain",
  "not sure", "i'm uncertain", "i am uncertain", "unclear", "unsure",
  "i guess", "i suppose", "i assume", "roughly", "approximately",
  "around", "about", "maybe", "perhaps", "possibly", "probably",
  "might be", "could be", "seems like", "seems to be", "appears to",
  "likely", "unlikely", "presumably", "supposedly",
  // Explicit uncertainty flags
  "unverified", "unconfirmed", "not verified", "not confirmed",
  "needs verification", "needs to be verified", "needs checking",
  "not checked", "haven't confirmed", "haven't verified",
  "not yet confirmed", "not yet verified", "pending",
  // Open questions
  "open question", "still open", "to be determined", "to be confirmed",
  "not yet decided", "under discussion", "awaiting", "tbd", "tbc",
  // Estimation language
  "estimate", "estimated", "estimation", "rough estimate",
  "ballpark", "order of magnitude", "in the range", "somewhere between",
  // Conditional uncertainty
  "depends on", "depending on", "subject to", "contingent on",
  "if this is correct", "assuming this is right",
  // Source attribution (implicit uncertainty)
  "according to", "reportedly", "allegedly", "claimed", "purportedly",
  "i was told", "someone mentioned", "i heard", "i read",
  // Numerical hedging
  "or so", "give or take", "more or less", "at least", "at most",
  "up to", "as many as", "no more than", "no less than",
  // Code comment hedging
  "fixme", "hack", "workaround", "temporary", "tentative",
  "not production", "do not use in prod",
  // Domain hedging
  "hypotheses", "hypothesis", "conjecture", "speculation",
  "preliminary", "provisional", "ambiguous",
]);

/**
 * Result of running the faithfulness probe on a text.
 */
export interface ProbeResult {
  /** Whether compression should be blocked. */
  shouldBlock: boolean;
  /** List of markers that triggered the probe. */
  triggeredMarkers: string[];
  /** Latency in milliseconds. */
  latencyMs: number;
}

/**
 * Run the faithfulness probe on text.
 *
 * Returns shouldBlock=true if any uncertainty marker is found.
 * This is the core guard that prevents Haiku (or any compressor) from silently
 * stripping epistemic qualifiers.
 *
 * Complexity: O(n_markers × n_words) — ~0.017ms P50 on typical responses.
 */
export function runProbe(text: string): ProbeResult {
  const t0 = performance.now();
  const lower = text.toLowerCase();
  const triggered: string[] = [];

  for (const marker of UNCERTAINTY_MARKERS) {
    if (lower.includes(marker)) {
      triggered.push(marker);
    }
  }

  return {
    shouldBlock:      triggered.length > 0,
    triggeredMarkers: triggered,
    latencyMs:        Math.round((performance.now() - t0) * 1000) / 1000,
  };
}

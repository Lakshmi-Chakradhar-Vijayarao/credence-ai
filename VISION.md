# Credence: Vision and Research Arc

## The Thesis

Context compression is routinely treated as a token-density problem: given a budget, maximize information retained. Every major compressor — LLMLingua-2, SnapKV, StreamingLLM — optimizes this objective. None of them ask whether the information that was retained is epistemically consistent with what was discarded. When Haiku compresses ten turns of conversation, it makes implicit decisions about which qualifier 54.0% of uncertainty markers survive naive Haiku compression (n=50). The downstream effect is a 34.0% False Certainty Rate — the model answers questions about uncertain constraints as if they were confirmed facts, one in three times, with no indication to the user that anything changed. This is not a failure of intelligence. It is a failure of infrastructure.

The pivot is recognizing that **compression is an epistemic event.** The decision to summarize a sentence is simultaneously a decision about how certain that sentence's content is. Uncertainty Quantification research (Kuhn et al., Semantic Entropy, ICLR 2023; UProp, ACL 2025) has built rigorous tools for detecting and measuring uncertainty — but none of these tools intervene at the compression layer, and none of them are wired to enforcement. Memory systems (MemGPT/Letta, Mem0, Zep) persist facts across sessions, but strip epistemic qualifiers at write time by design. Credence is the intersection that was missing: a deterministic enforcement layer that intercepts the compression event, preserves epistemic qualifiers through it, and carries that preservation forward through agent handoffs and cross-session memory.

---

## Architecture Value

Five checkpoints, each encoding a distinct architectural principle:

- **CP1 — Faithfulness Probe (0.07ms, zero API calls):** The probe is deterministic, not probabilistic. A 167-marker frozenset scan on user turns only. FCR 34.0% → 0% (n=50, 95% CI [0%, 7.1%]). *Principle: enforcement that requires a model call is not enforcement — it is a suggestion.*

- **CP2 — Truth Buffer + Consistency Enforcer:** Injects all unverified constraints into the system prompt every turn, with imperative prohibition when the user's query keyword-overlaps a registered constraint (32 domain synonym clusters, 0% FP rate). *Principle: the model should never have to remember an uncertain constraint; inject it explicitly at generation time.*

- **CP3 — Generation-Time Scanner:** Annotates numeric and string literals in generated code and prose with confidence tiers (HIGH RISK / UNVERIFIED / CHECK) derived from the live registry. Catches `RATE_LIMIT = 50`, `ALGORITHM = "RS256"`, `BASE_URL = "/api/v2"`. *Principle: enforcement must extend to the artifact, not just the conversation.*

- **CP4 — Rust Gate (3.4ms, 98× faster than Python, 0% FP rate):** Native PreToolUse hook. Blocks Write/Edit/Bash when tool arguments overlap unverified constraints. *Principle: irreversible actions are where epistemic errors become real costs — gate the action, not the text.*

- **CP5 — Epistemic Memory:** Cross-session constraint registry with certainty trajectories and confidence decay. CS-FCR 50% (no memory) → 0% (Credence Memory), n=16 callbacks. *Principle: epistemic state is session-persistent by nature, not by accident of context window size.*

---

## The Research Arc

**Now (deployed):** Deterministic enforcement at five checkpoints. 22-tool MCP server. 178 passing tests (S1–S26, 11 skipped offline-only). Ghost Gauntlet: BothRate 0.200 → 1.000 (n=10 sessions). E6: 19.6% → 100% correction recall (n=23). Precision eval: 0% FP on CE, GTS, and probe. ETP schema defined. The system prevents the measured failure with no false positives on the precision eval set.

**6 months — calibrated epistemic compression:** Replace binary block/proceed with a continuous epistemic importance weight per sentence. Analogous to LLMLingua-2's token importance scoring, but the weight is derived from the constraint registry: sentences containing registered uncertain values receive 10× importance in compression scheduling. This gives a principled hybrid — compress everything else aggressively; treat uncertain sentences as near-incompressible. Technical path: replace the binary Haiku gate with a weighted retention policy that annotates compression-safe vs. compression-risky sentences before the compressor runs.

**2 years — ETP as open standard:** Make the Epistemic Transport Protocol a community standard adopted by AutoGPT, LangChain, CrewAI, and native model provider APIs. The model: HTTP headers carry request metadata; ETP headers carry epistemic metadata. Every agent handoff in every pipeline passes `{j_score, zone, verified, chain_depth}` alongside the content. The Ghost Detector and SE probe become standard middleware callable by any framework. Credence becomes the reference implementation of a protocol.

---

## Limitations as Research Directions

**FCR measures hedging absence, not factual incorrectness.** A response stating "the rate limit is 50 req/min" fails FCR if the source said "I think around 50." That is the right harm to measure. But FCR conflates two harms — stripped qualifier and wrong fact — that have different remedies. Disentangling them requires ground-truth verification, which Credence does not currently perform. *Research direction: integrate the registry with external verification sources (API docs, database schemas) to enable factual FCR alongside hedging FCR.*

**Truth Buffer and CE are probabilistic.** CP1 and CP4 are deterministic. CP2 depends on the model following injected instructions. Prompt-only instruction achieves 90.0% qualifier survival (not 100%), versus 100% with the probe. The gap is the model's compliance probability. *Research direction: apply conformal prediction to give PAC-style bounds — characterize per-constraint-type compliance rates and surface the gap to users.*

**GTS over-annotation on common literals.** A constraint registering value `50` annotates every `= 50` assignment in generated code, including unrelated loop indices. Over-annotation is safer than under-annotation; but it is noise. *Research direction: sentence-level embedding similarity between the constraint text and the code context around the matched literal, with a configurable annotation threshold.*

**Confident-wrong ceiling.** J-score measures linguistic assertiveness (ρ = −0.034 with factual correctness). A confidently wrong statement scores HIGH-J, bypasses the faithfulness probe, and is neither flagged nor blocked. Ghost Detector catches many of these via Opus reasoning. *Research direction: distill a lightweight confident-wrong classifier from Ghost Detector training data, runnable at probe speed without an API call.*

**Short sessions.** Compression fires at turn 16 (COMPRESS) or 20 (TRIM). Sessions shorter than this do not trigger CP1. *Research direction: lower compression threshold adaptively for sessions with early high-stakes constraint registration.*

---

## Why This Becomes Mandatory

Type checking became a production requirement not because developers wanted more tooling, but because silent type errors at scale were more expensive than the overhead of enforcement. The same cost structure applies here. At 10 agent hops with a 10% per-hop FCR, the probability that at least one agent in the chain encounters a false certainty is 65%. Credence's deterministic probe brings that to near zero. The compounding math is unambiguous: the per-hop cost of enforcement is constant; the per-chain cost of not enforcing grows exponentially with chain length.

The Rust gate is the key infrastructure signal. The PreToolUse hook pattern — running before every irreversible action, not just occasionally — is the same model as security scanners before every git commit. Developers accept that 3.4ms overhead because the cost of a security failure dominates the cost of the scan. Epistemic failures in deployed AI systems have the same asymmetric cost structure: a wrong rate limit baked into a deployed service, a wrong auth algorithm committed to a codebase, a wrong expiry value shipped to a client. Once a team has experienced zero FCR in production, the gate becomes load-bearing infrastructure that nobody removes.

---

## The Standard

`etp-v1.json` is a model-agnostic JSON Schema for epistemic metadata transport. It defines four primitives: `EpistemicConstraint` (a tracked uncertain claim), `EpistemicEnvelope` (a provenance wrapper for AI-generated content with trust decay per hop), `EpistemicLedger` (full session state), and `AlignmentWarning` (fired when a response is more confident than the ledger warrants).

The design principle: *every AI system today passes information between agents by value. Nobody passes it by epistemic weight.* ETP proposes to fix this by making epistemic metadata first-class in agent protocols — the same way HTTP headers made request metadata first-class in web protocols.

Credence is the reference implementation. The standard is the destination.

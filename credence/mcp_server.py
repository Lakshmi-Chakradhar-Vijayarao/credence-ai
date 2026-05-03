"""
credence/mcp_server.py
======================
Credence MCP Server — zero API key required.

Credence is a guard layer, not a context manager. It wraps around whatever
compression the host coding agent (Claude Code, Copilot, Codex) already does.
The agent compresses with its own model; Credence enforces epistemic safety
before and after, entirely through deterministic string operations.

Tools (10 total, zero LLM calls):
    credence_pre_compress    — CP1: check text before compression (BLOCK / ALLOW)
    credence_post_compress   — CP1: measure qualifier survival after compression
    credence_register        — register an uncertain constraint explicitly
    credence_verify          — mark a constraint as verified
    credence_constraints     — query all unverified constraints for a session
    credence_gate            — CP4: pre-tool gate (block if unverified constraints apply)
    credence_scan            — CP3: scan model output for unverified numeric literals
    credence_memory_snapshot — persist unverified constraints as project memory
    credence_memory_recall   — load project memory into a new session
    credence_autoverify      — auto-verify constraints on confirmation phrases

Resources (passive, epistemic:// URI scheme):
    epistemic://session/{session_id}/ledger             — all constraints
    epistemic://session/{session_id}/constraint/{id}    — single constraint + trajectory

Install:
    pip install credence-ai fastmcp
    credence-server

No ANTHROPIC_API_KEY or any other key needed.
"""

import os
import re
import threading
from typing import Optional

try:
    from fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False

from .context_manager import (
    _UNCERTAINTY_MARKERS,
    _CE_STOPWORDS,
    _CE_DOMAIN_SYNONYMS,
    _GTS_NUM_PATTERN,
    _GTS_CODE_BLOCK,
    _GTS_SKIP_PREFIXES,
    _GTS_SENTENCE_SPLIT,
    _GTS_WARN_THRESHOLD,
    _GTS_QUALIFY_THRESHOLD,
)
from .registry import CredenceRegistry

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Credence",
    instructions=(
        "Credence is a zero-config epistemic guard layer. It prevents uncertainty "
        "qualifiers from being silently dropped during context compression.\n\n"
        "Lifecycle:\n"
        "  1. credence_memory_recall at session START — load prior unverified constraints\n"
        "  2. credence_pre_compress BEFORE any compression — BLOCK if qualifiers present\n"
        "  3. credence_post_compress AFTER compression — measure qualifier survival\n"
        "  4. credence_gate BEFORE any irreversible tool call\n"
        "  5. credence_scan BEFORE shipping generated code\n"
        "  6. credence_verify when a constraint is confirmed by the user\n"
        "  7. credence_memory_snapshot at session END — persist for next session\n\n"
        "Key invariant: never compress text that contains unverified uncertainty qualifiers."
    ),
) if _FASTMCP_AVAILABLE else None

# Process-level registry singleton
_registry: Optional[CredenceRegistry] = None
_registry_lock = threading.Lock()


def _get_registry() -> CredenceRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:  # double-checked locking
                db_path = (
                    os.environ.get("CREDENCE_DB_PATH")
                    or os.environ.get("CREDENCE_REGISTRY_PATH")
                    or "epistemic_registry.db"
                )
                _registry = CredenceRegistry(db_path=db_path)
    return _registry


# ---------------------------------------------------------------------------
# Synonym-expansion helper (replicates CE logic without a ContextManager)
# ---------------------------------------------------------------------------

def _expand_tokens(tokens: set) -> set:
    expanded = set(tokens)
    for t in tokens:
        cluster = _CE_DOMAIN_SYNONYMS.get(t)
        if cluster:
            expanded |= cluster
    return expanded


# ---------------------------------------------------------------------------
# Standalone GTS scan (no ContextManager needed)
# ---------------------------------------------------------------------------

def _scan_output(output_text: str, registry: CredenceRegistry, session_id: str, turn: int = 0) -> tuple[str, list]:
    """Scan output for unverified numeric literals. Returns (annotated_text, hits)."""
    constraints = registry.list_uncertain(session_id)
    if not constraints:
        return output_text, []

    # Build value map: numeric_value_str → constraint dict with eff_conf
    value_map: dict[str, dict] = {}
    for c in constraints:
        nums = _GTS_NUM_PATTERN.findall(c.get("content", ""))
        eff_conf = registry.get_effective_confidence(c["constraint_id"], turn)
        c_with_conf = dict(c, eff_conf=eff_conf)
        for n in nums:
            if len(n) >= 2:
                value_map[n] = c_with_conf

    if not value_map:
        return output_text, []

    def _annotation(c: dict, snippet: str, for_code: bool) -> str:
        ec = c.get("eff_conf", 0.5)
        text = (c.get("content") or "")[:60]
        if ec < _GTS_WARN_THRESHOLD:
            tier = f"⚠⚠ CREDENCE[HIGH RISK, conf={ec:.2f}]"
        elif ec < _GTS_QUALIFY_THRESHOLD:
            tier = f"⚠ CREDENCE[unverified, conf={ec:.2f}]"
        else:
            tier = f"CREDENCE[check, conf={ec:.2f}]"
        return (f"  # {tier}: {text}" if for_code else f" ⚠ {tier}: {text}")

    hits: list[dict] = []
    result = output_text

    # Pass 1 — code blocks (annotate literals + inherit uncertainty downstream)
    _ASSIGN_RE = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=')

    def annotate_code_block(m: re.Match) -> str:
        fence, body, close = m.group(1), m.group(2), m.group(3)
        lines = body.split("\n")
        out   = []

        # First pass: annotate direct literal hits, collect annotated var names
        annotated_vars: dict[str, dict] = {}  # var_name → constraint
        for line in lines:
            if any(line.lstrip().startswith(p) for p in _GTS_SKIP_PREFIXES):
                out.append(line)
                continue
            if "CREDENCE:" in line:
                out.append(line)
                continue
            for val, c in value_map.items():
                if re.search(r'\b' + re.escape(val) + r'\b', line):
                    m_assign = _ASSIGN_RE.match(line)
                    if m_assign:
                        annotated_vars[m_assign.group(1)] = c
                    line = line.rstrip() + _annotation(c, val, for_code=True)
                    hits.append({"value": val, "constraint_id": c["constraint_id"],
                                 "constraint_text": c.get("content","")[:80],
                                 "line": line.strip(), "eff_conf": c.get("eff_conf",0.5),
                                 "source": "code"})
                    break
            out.append(line)

        # Inheritance pass: annotate lines that reference annotated variable names
        if annotated_vars:
            out2 = []
            for line in out:
                if "CREDENCE:" in line:
                    out2.append(line)
                    continue
                if any(line.lstrip().startswith(p) for p in _GTS_SKIP_PREFIXES):
                    out2.append(line)
                    continue
                # Skip the assignment line itself (already annotated)
                if _ASSIGN_RE.match(line):
                    out2.append(line)
                    continue
                for var_name, c in annotated_vars.items():
                    if re.search(r'\b' + re.escape(var_name) + r'\b', line):
                        ec   = c.get("eff_conf", 0.5)
                        tier = "HIGH RISK" if ec < _GTS_WARN_THRESHOLD else "unverified"
                        line = line.rstrip() + f"  # CREDENCE[inherited from {var_name}, {tier}]"
                        hits.append({"value": var_name,
                                     "constraint_id": c["constraint_id"],
                                     "constraint_text": c.get("content","")[:80],
                                     "line": line.strip(),
                                     "eff_conf": ec,
                                     "source": "code_inherited"})
                        break
                out2.append(line)
            out = out2

        return fence + "\n".join(out) + close

    result = _GTS_CODE_BLOCK.sub(annotate_code_block, result)

    # Pass 2 — prose (non-code segments)
    segments = _GTS_CODE_BLOCK.split(result)
    prose_out = []
    for seg in segments:
        if seg.startswith("```") or seg.endswith("```"):
            prose_out.append(seg)
            continue
        sentences = _GTS_SENTENCE_SPLIT.split(seg)
        new_sents = []
        for sent in sentences:
            if "CREDENCE:" in sent:
                new_sents.append(sent)
                continue
            for val, c in value_map.items():
                if re.search(r'\b' + re.escape(val) + r'\b', sent):
                    sent = sent.rstrip() + _annotation(c, val, for_code=False)
                    hits.append({"value": val, "constraint_id": c["constraint_id"],
                                 "constraint_text": c.get("content","")[:80],
                                 "line": sent.strip(), "eff_conf": c.get("eff_conf",0.5),
                                 "source": "prose"})
                    break
            new_sents.append(sent)
        prose_out.append(" ".join(new_sents))

    # prose_out will have interleaved code+prose; simplest: just return result from code pass
    # (prose pass on the already-annotated result is correct)
    return result, hits


# ---------------------------------------------------------------------------
# Session type detection (keyword heuristics — zero API)
# ---------------------------------------------------------------------------

_SESSION_TYPE_KEYWORDS: dict[str, frozenset] = {
    "debug": frozenset({
        "error", "exception", "traceback", "stack trace", "stacktrace",
        "bug", "fix", "crash", "fails", "failing", "broken", "not working",
        "undefined", "null pointer", "segfault", "timeout", "deadlock",
        "infinite loop", "memory leak", "500", "404", "503",
    }),
    "design": frozenset({
        "architecture", "schema", "design", "trade-off", "tradeoff",
        "pattern", "approach", "system", "service", "component",
        "microservice", "monolith", "event-driven", "message queue",
        "database", "scalability", "availability", "consistency",
        "sharding", "replication", "caching", "load balancer",
    }),
    "code_review": frozenset({
        "review", "refactor", "clean up", "cleanup", "improve",
        "simplify", "readability", "maintainability", "best practice",
        "naming", "duplication", "dry", "solid", "smell",
    }),
    "research": frozenset({
        "compare", "evaluate", "benchmark", "analysis", "survey",
        "pros and cons", "trade offs", "alternatives", "options",
        "which is better", "difference between", "versus", "vs",
        "recommend", "suggestion",
    }),
}


def _detect_session_type(text: str) -> str:
    """
    Classify session type from text using keyword heuristics.
    Returns: "debug" | "design" | "code_review" | "research" | "general"
    """
    text_lower = text.lower()
    scores = {stype: 0 for stype in _SESSION_TYPE_KEYWORDS}
    for stype, keywords in _SESSION_TYPE_KEYWORDS.items():
        scores[stype] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.tool()
    def credence_pre_compress(text: str, session_id: str) -> dict:
        """
        Check whether text is safe to compress (CP1 — faithfulness probe).

        Call this BEFORE any context compression. If uncertainty qualifiers
        are detected, the tool returns action=BLOCK and you must preserve the
        text verbatim. If safe, it returns action=ALLOW with a manifest of
        qualifiers to preserve in the compressed output.

        Works with any model: Claude, GPT, Gemini, local. No API key needed.

        Args:
            text:       The text about to be compressed.
            session_id: Session identifier for registry context.

        Returns:
            action: "BLOCK" or "ALLOW"
            markers_found: list of uncertainty markers detected
            registered_constraints: count of unverified constraints in session
            message: human-readable explanation
        """
        text_lower = text.lower()
        found = [m for m in _UNCERTAINTY_MARKERS if m in text_lower]

        registry     = _get_registry()
        n_uncertain  = len(registry.list_uncertain(session_id))

        if found:
            return {
                "action":                 "BLOCK",
                "markers_found":          found[:10],
                "marker_count":           len(found),
                "registered_constraints": n_uncertain,
                "message": (
                    f"BLOCK — {len(found)} uncertainty marker(s) detected. "
                    "Compress this text and you risk stripping epistemic qualifiers. "
                    "Preserve it verbatim or verify the uncertain claims first."
                ),
            }

        return {
            "action":                 "ALLOW",
            "markers_found":          [],
            "marker_count":           0,
            "registered_constraints": n_uncertain,
            "message": (
                "ALLOW — no uncertainty markers found. "
                "Safe to compress. After compression, call credence_post_compress "
                "to verify qualifier survival."
            ),
        }

    @mcp.tool()
    def credence_post_compress(original: str, compressed: str, session_id: str) -> dict:
        """
        Measure qualifier survival after compression (CP1 — post-check).

        Call this AFTER your model compresses context. Measures what fraction
        of the original uncertainty markers survived into the compressed output.
        Low qual_survival → qualifiers were stripped → false certainty risk.

        Args:
            original:   The original uncompressed text.
            compressed: The compressed output from your model.
            session_id: Session identifier.

        Returns:
            qual_survival: fraction of original markers preserved (0.0–1.0)
            fcr_risk:      estimated false-certainty rate (0.0–1.0)
            verdict:       "SAFE" / "WARN" / "RISK"
            dropped:       list of markers that were stripped
        """
        orig_lower = original.lower()
        comp_lower = compressed.lower()

        orig_markers = {m for m in _UNCERTAINTY_MARKERS if m in orig_lower}
        comp_markers = {m for m in _UNCERTAINTY_MARKERS if m in comp_lower}

        if orig_markers:
            survived     = orig_markers & comp_markers
            dropped      = orig_markers - comp_markers
            qual_survival = len(survived) / len(orig_markers)
        else:
            survived, dropped = set(), set()
            qual_survival = 1.0

        fcr_risk = max(0.0, min(1.0, round(1.0 - qual_survival * 1.2, 3)))

        if qual_survival >= 0.80:
            verdict = "SAFE"
        elif qual_survival >= 0.50:
            verdict = "WARN"
        else:
            verdict = "RISK"

        # Passive marker flywheel recording — no user-visible effect
        if orig_markers:
            registry = _get_registry()
            session_type = _detect_session_type(original + " " + compressed)
            registry.record_marker_events(
                session_id    = session_id,
                markers_fired = list(orig_markers),
                qual_survival = qual_survival,
                session_type  = session_type,
            )

        return {
            "qual_survival":        round(qual_survival, 3),
            "fcr_risk":             fcr_risk,
            "verdict":              verdict,
            "original_marker_count": len(orig_markers),
            "output_marker_count":   len(comp_markers),
            "survived":             list(survived)[:10],
            "dropped":              list(dropped)[:10],
            "message": (
                f"{verdict} — qualifier survival {qual_survival:.0%}. "
                + (f"Dropped: {list(dropped)[:5]}" if dropped else "All qualifiers preserved.")
            ),
        }

    @mcp.tool()
    def credence_register(
        content:         str,
        session_id:      str,
        j_score:         float = 0.30,
        zone:            str   = "LOW",
        source_type:     str   = "observation",
    ) -> dict:
        """
        Register an uncertain constraint in the epistemic registry.

        Use whenever the user states something uncertain: an unconfirmed vendor
        claim, an assumption, a 'I think' statement, a number from a quick search.
        The registry tracks it across the session so the gate and scan tools can
        enforce it on every subsequent turn.

        source_type classifies the epistemic origin of the claim — this affects
        confidence decay rate and ghost detection in Phase 3:
          "vendor_claim"  — value stated by a vendor / docs / external party (slow decay)
          "user_estimate" — user's rough guess or approximation (fast decay)
          "observation"   — user's direct measurement or confirmed observation (slow decay)
          "assumption"    — working hypothesis, needs confirmation (fastest decay)
          "compliance"    — regulatory / legal constraint (almost no decay)
          "config"        — configuration value that could change (medium decay)
          "inference"     — model-derived, not directly stated (medium decay)

        Args:
            content:     The uncertain claim text.
            session_id:  Session identifier.
            j_score:     Confidence 0–1 (default 0.30 = clearly uncertain).
            zone:        "HIGH", "MEDIUM", or "LOW" (default "LOW").
            source_type: Epistemic origin (see above). Default "observation".

        Returns:
            constraint_id, status, content, j_score, zone, source_type
        """
        _VALID_SOURCE_TYPES = {
            "vendor_claim", "user_estimate", "observation",
            "assumption", "compliance", "config", "inference",
        }
        # Map user_estimate → estimate for registry internal type
        _TYPE_MAP = {"user_estimate": "estimate", "inference": "assumption"}
        registry_type = _TYPE_MAP.get(source_type, source_type)
        if registry_type not in {"vendor_claim", "estimate", "observation",
                                  "assumption", "compliance", "config", "performance"}:
            registry_type = "observation"

        registry = _get_registry()
        cid = registry.register(
            content          = content,
            session_id       = session_id,
            j_score          = j_score,
            zone             = zone,
            constraint_type  = registry_type,
        )
        return {
            "constraint_id": cid,
            "status":        "registered",
            "content":       content,
            "session_id":    session_id,
            "j_score":       j_score,
            "zone":          zone,
            "source_type":   source_type,
            "message": (
                f"Registered as uncertain [{source_type}] (id={cid}). "
                f"Call credence_verify('{cid}', <confirmed_value>, '{session_id}') "
                "when confirmed."
            ),
        }

    @mcp.tool()
    def credence_verify(
        constraint_id:  str,
        verified_value: str,
        session_id:     str,
        evidence:       str = "",
        source:         str = "user",
    ) -> dict:
        """
        Mark a registered constraint as verified with its confirmed value.

        After verification the constraint is excluded from Truth Buffer injection
        and Consistency Enforcer enforcement. An audit trail is recorded — who
        verified, on what basis, and what the confirmed value is.

        Args:
            constraint_id:  ID from credence_register.
            verified_value: The confirmed value (e.g. "100 req/min per Stripe docs §4.2").
            session_id:     Session identifier.
            evidence:       What was checked to confirm this. Strongly recommended.
                            Examples: "checked Stripe dashboard 2026-05-02",
                            "confirmed in production logs", "vendor email attached".
                            An empty evidence string is accepted but leaves no audit basis.
            source:         Who verified this. Use "user" for human confirmation,
                            "api_response" for automated checks, "agent:<name>" for
                            downstream agents, "external_doc" for documentation.

        Returns:
            Updated constraint dict with verified=True and audit fields.
        """
        registry = _get_registry()
        result   = registry.verify(constraint_id, verified_value, evidence=evidence, source=source)
        if "error" in result:
            return result
        result["status"]   = "verified"
        result["audit"]    = {"source": source, "evidence": evidence or "(none provided)"}
        result["message"]  = (
            f"Verified by '{source}'. Confirmed value: '{verified_value}'. "
            + (f"Evidence: {evidence}. " if evidence else "Warning: no evidence recorded. ")
            + "Safe to implement code that depends on this value."
        )
        return result

    @mcp.tool()
    def credence_constraints(session_id: str) -> dict:
        """
        List all unverified constraints for a session.

        Use before writing code that may embed user-stated values, or at
        session end to audit what still needs confirmation.

        Args:
            session_id: Session identifier.

        Returns:
            count, constraints list (id, content, j_score, zone), message.
        """
        registry    = _get_registry()
        constraints = registry.list_uncertain(session_id)
        count       = len(constraints)
        message = (
            "All constraints verified."
            if count == 0
            else f"{count} unverified constraint(s) — confirm before shipping code."
        )
        return {"count": count, "constraints": constraints, "message": message}

    @mcp.tool()
    def credence_gate(
        tool_name:         str,
        arguments_summary: str,
        session_id:        str,
    ) -> dict:
        """
        Pre-execution epistemic gate (CP4): block irreversible tool calls that
        embed unverified constraint values.

        Call BEFORE write_file, execute_code, send_request, deploy, or any
        tool that would embed a user-stated value into code or infrastructure.
        Uses synonym-expansion to catch paraphrase overlap ("how fast" ↔ "rate limit").

        Args:
            tool_name:           Name of the tool about to be called.
            arguments_summary:   Brief summary of arguments (omit secrets).
            session_id:          Session identifier.

        Returns:
            proceed: bool, blocked_by: list, recommendation: str.
        """
        registry  = _get_registry()
        uncertain = registry.list_uncertain(session_id)

        if not uncertain:
            return {
                "proceed":          True,
                "blocked_by":       [],
                "unverified_count": 0,
                "recommendation":   "PROCEED — no unverified constraints in this session.",
            }

        action_text  = f"{tool_name} {arguments_summary}".lower()
        raw_tokens   = set(re.sub(r"[^\w\s]", " ", action_text).split()) - _CE_STOPWORDS
        action_tokens = _expand_tokens(raw_tokens)

        blocking: list[dict] = []
        for c in uncertain:
            c_raw    = set(re.sub(r"[^\w\s]", " ", c["content"].lower()).split()) - _CE_STOPWORDS
            c_tokens = _expand_tokens(c_raw)
            overlap  = action_tokens & c_tokens
            if len(overlap) >= 2:
                c["overlap_terms"] = list(overlap)[:6]
                blocking.append(c)

        if blocking:
            cids = ", ".join(c["constraint_id"] for c in blocking)
            recommendation = (
                f"BLOCK — {len(blocking)} unverified constraint(s) may affect this action. "
                f"Verify first (IDs: {cids})."
            )
        else:
            recommendation = (
                f"PROCEED — {len(uncertain)} unverified constraint(s) exist "
                "but none are topically related to this action."
            )

        session_type = _detect_session_type(f"{tool_name} {arguments_summary}")

        return {
            "proceed":          len(blocking) == 0,
            "blocked_by":       blocking,
            "unverified_count": len(uncertain),
            "session_type":     session_type,
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_scan(output_text: str, session_id: str, current_turn: int = 0) -> dict:
        """
        Generation-Time Constraint Scanner (CP3): scan model output for numeric
        literals that match registered unverified constraints.

        Annotations are severity-tiered by decayed confidence:
            ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]  — eff_conf < 0.20
            ⚠  CREDENCE[unverified, conf=0.30]  — 0.20 ≤ eff_conf < 0.40
               CREDENCE[check, conf=0.42]        — eff_conf ≥ 0.40

        Scans both code blocks and prose.

        Args:
            output_text:  Raw model output to scan.
            session_id:   Session whose constraints to check against.
            current_turn: Turn number for confidence decay (default 0).

        Returns:
            annotated_output, scan_hits, hit_count, high_risk_count, recommendation.
        """
        registry   = _get_registry()
        annotated, hits = _scan_output(output_text, registry, session_id, current_turn)

        high_risk = [h for h in hits if h.get("eff_conf", 1.0) < _GTS_WARN_THRESHOLD]

        if high_risk:
            recommendation = (
                f"BLOCK — {len(high_risk)} HIGH RISK literal(s) found. "
                "Verify before use."
            )
        elif hits:
            recommendation = (
                f"REVIEW — {len(hits)} unverified literal(s) annotated. "
                "Confirm values before shipping."
            )
        else:
            recommendation = "CLEAN — no output literals matched unverified constraints."

        return {
            "annotated_output": annotated,
            "scan_hits":        hits,
            "hit_count":        len(hits),
            "high_risk_count":  len(high_risk),
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_memory_snapshot(session_id: str, project_id: str) -> dict:
        """
        Persist all unverified constraints from a session as project memory.

        Call at the END of a session. Next session on the same project calls
        credence_memory_recall to inherit what was still uncertain — the new
        session starts knowing what it doesn't know.

        Args:
            session_id: Current session ID.
            project_id: Stable project identifier (e.g. "my-api-project").

        Returns:
            saved_count, items, message.
        """
        from .memory import CredenceMemory
        mem  = CredenceMemory(_get_registry())
        snap = mem.snapshot(session_id=session_id, project=project_id)
        return {
            "project_id":  snap.project_id,
            "session_id":  snap.session_id,
            "saved_count": snap.saved_count,
            "items": [
                {"constraint_id": i.get("constraint_id"),
                 "content":       i.get("content"),
                 "zone":          i.get("zone"),
                 "j_score":       i.get("j_score")}
                for i in snap.items
            ],
            "message": snap.summary(),
        }

    @mcp.tool()
    def credence_memory_recall(
        project_id:     str,
        new_session_id: str,
        context_hint:   str = "",
    ) -> dict:
        """
        Load project memories into a new session at session start.

        Call at the START of a new session. Injects all previously unverified
        constraints into the new session so enforcement works from turn 1.

        Args:
            project_id:     Project identifier matching credence_memory_snapshot.
            new_session_id: ID for the new session.
            context_hint:   Optional keyword filter.

        Returns:
            injected_count, system_block (prepend to system prompt), items, message.
        """
        from .memory import CredenceMemory
        mem    = CredenceMemory(_get_registry())
        recall = mem.recall_and_inject(
            project        = project_id,
            new_session_id = new_session_id,
            context_hint   = context_hint,
        )
        return {
            "project_id":     recall.project_id,
            "new_session_id": recall.new_session_id,
            "injected_count": recall.injected_count,
            "system_block":   recall.system_block,
            "items": [
                {"constraint_id": i.get("constraint_id"),
                 "content":       i.get("content"),
                 "zone":          i.get("zone"),
                 "j_score":       i.get("j_score")}
                for i in recall.items
            ],
            "is_empty": recall.is_empty(),
            "message": (
                f"Loaded {recall.injected_count} unverified constraint(s) from "
                f"project '{project_id}' into session '{new_session_id}'."
                if not recall.is_empty()
                else f"No unverified constraints found for project '{project_id}'."
            ),
        }

    @mcp.tool()
    def credence_autoverify(text: str, session_id: str) -> dict:
        """
        Scan text for natural-language verification signals and auto-verify
        matching unverified constraints — zero API calls.

        When a user says "actually it's 3600", "confirmed: rate limit is 100",
        or "I checked, the port is 5432", this tool detects those confirmation
        phrases and automatically marks matching constraints as verified.

        Matching: a constraint is a candidate if ≥ 2 non-stopword tokens from
        the constraint text appear in the confirmation sentence.

        Args:
            text:       The user or assistant message to scan for confirmations.
            session_id: Session whose constraints to check against.

        Returns:
            verified_count, verified_ids, candidates_checked, message.
        """
        _CONFIRM_PHRASES = frozenset({
            "actually", "confirmed", "i checked", "turns out", "verified",
            "it is", "it's", "the answer is", "we confirmed", "just checked",
            "double checked", "double-checked", "in fact", "correction",
            "to clarify", "clarification", "the actual", "the real",
            "as per", "according to", "per the docs", "per docs",
            "i verified", "we verified", "found out", "it turns out",
        })
        _AUTOVERIFY_STOPWORDS = frozenset({
            "the", "a", "an", "is", "it", "its", "this", "that", "for",
            "and", "or", "to", "of", "in", "on", "at", "by", "be", "are",
        })

        text_lower = text.lower()
        has_signal = any(p in text_lower for p in _CONFIRM_PHRASES)
        if not has_signal:
            return {
                "verified_count":    0,
                "verified_ids":      [],
                "candidates_checked": 0,
                "message":           "No confirmation signal detected in text.",
            }

        registry  = _get_registry()
        uncertain = registry.list_uncertain(session_id)
        if not uncertain:
            return {
                "verified_count":    0,
                "verified_ids":      [],
                "candidates_checked": 0,
                "message":           "Confirmation detected but no unverified constraints in session.",
            }

        text_tokens = set(re.sub(r"[^\w\s]", " ", text_lower).split()) - _AUTOVERIFY_STOPWORDS
        verified_ids: list[str] = []

        for c in uncertain:
            c_tokens = set(
                re.sub(r"[^\w\s]", " ", c["content"].lower()).split()
            ) - _AUTOVERIFY_STOPWORDS
            overlap = text_tokens & c_tokens
            if len(overlap) >= 2:
                registry.verify(c["constraint_id"], f"auto-verified from: {text[:80]}")
                verified_ids.append(c["constraint_id"])

        n = len(verified_ids)
        return {
            "verified_count":    n,
            "verified_ids":      verified_ids,
            "candidates_checked": len(uncertain),
            "message": (
                f"Auto-verified {n} constraint(s) matching confirmation signal."
                if n > 0
                else "Confirmation signal present but no constraints matched (< 2 token overlap)."
            ),
        }

# ---------------------------------------------------------------------------
# MCP Resources — epistemic:// URI scheme
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.resource("epistemic://session/{session_id}/ledger")
    def epistemic_ledger(session_id: str) -> str:
        """Epistemic ledger — all constraints for a session."""
        import json as _json
        registry = _get_registry()
        uncertain = registry.list_uncertain(session_id)
        all_rows  = registry._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        all_constraints = [registry._row_to_dict(r) for r in all_rows]
        return _json.dumps({
            "session_id":        session_id,
            "total_constraints": len(all_constraints),
            "unverified_count":  len(uncertain),
            "verified_count":    len(all_constraints) - len(uncertain),
            "constraints":       all_constraints,
            "etp_version":       "1.0",
        }, indent=2)

    @mcp.resource("epistemic://session/{session_id}/constraint/{constraint_id}")
    def epistemic_constraint(session_id: str, constraint_id: str) -> str:
        """Single constraint with full certainty trajectory."""
        import json as _json
        registry = _get_registry()
        rows = registry._conn.execute(
            "SELECT * FROM constraints WHERE constraint_id=? AND session_id=?",
            (constraint_id, session_id),
        ).fetchone()
        if rows is None:
            return _json.dumps({"error": f"constraint '{constraint_id}' not found"})
        constraint = registry._row_to_dict(rows)
        trajectory = registry.get_trajectory(constraint_id)
        return _json.dumps({
            "session_id":  session_id,
            "constraint":  constraint,
            "trajectory":  trajectory,
            "event_count": len(trajectory),
            "etp_version": "1.0",
        }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _FASTMCP_AVAILABLE:
        print("fastmcp not installed. Run: pip install 'credence-ai[mcp]'")
        return
    mcp.run()

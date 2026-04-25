"""
credence/mcp_server.py
==================
Credence MCP Server — context management conditioned on epistemic state.

Exposes ContextManager as 8 MCP tools callable from Claude Desktop
or any MCP-compatible agent framework.

Core principle: memory allocation decisions should be conditioned on epistemic
state. Only HIGH-J (epistemically resolved) content is safe to compress. Uncertain
content is preserved verbatim through every compression and trim operation.

Every credence_chat response includes a CredenceEnvelope dict so downstream agents
can inspect epistemic provenance before compressing or acting on information.

Run:
    python -m credence.mcp_server

Or directly:
    python credence/mcp_server.py

Requires:
    pip install fastmcp anthropic

Trust boundary:
    Envelopes from unknown sources (not in _TRUSTED_SOURCES) receive a
    trust penalty, making should_verify=True more likely. This prevents
    the system from blindly trusting envelopes injected from untrusted agents.

Model-agnostic:
    The epistemic signal reads output text. It works regardless of which
    model produced the response — Claude, GPT-4o, Llama, or any other LLM.
"""

import os
import re
from typing import Optional

try:
    from fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False

from .context_manager import ContextManager, _UNCERTAINTY_MARKERS
from .confidence_proxy import CredenceProxy
from .envelope import CredenceEnvelope, _TRUSTED_SOURCES
from .registry import CredenceRegistry

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Credence",
    instructions=(
        "Credence is a context management layer that conditions memory "
        "allocation on epistemic state. Only epistemically resolved (HIGH-J) content "
        "is compressed. Uncertain content is preserved verbatim.\n\n"
        "Every credence_chat response includes an 'envelope' with a J-score and trust "
        "metadata. ALWAYS check envelope['should_verify'] before treating uncertain "
        "information as fact. ALWAYS check envelope['safe_to_compress'] before "
        "summarizing or passing content to another agent.\n\n"
        "Use credence_risk before compressing or forwarding any response. "
        "Use credence_save / credence_load for cross-session continuity."
    ),
) if _FASTMCP_AVAILABLE else None

# Session registry: session_id → ContextManager
_sessions: dict[str, ContextManager] = {}

# Epistemic registry singleton — one per process
_registry: Optional[CredenceRegistry] = None


def _get_registry() -> CredenceRegistry:
    """Return the process-level CredenceRegistry, creating it on first call."""
    global _registry
    if _registry is None:
        db_path    = os.environ.get("CREDENCE_REGISTRY_PATH", "epistemic_registry.db")
        _registry  = CredenceRegistry(db_path=db_path)
    return _registry


def _get_session(session_id: str) -> ContextManager:
    """Return or create a ContextManager for the given session ID.

    Passes the shared registry and session_id so the Truth Buffer and Scout
    Classifier have access to the constraint store. Models are configurable
    via CREDENCE_MAIN_MODEL and CREDENCE_COMPRESSION_MODEL env vars.
    """
    if session_id not in _sessions:
        use_scout = os.environ.get("CREDENCE_USE_SCOUT", "").lower() in ("1", "true", "yes")
        _sessions[session_id] = ContextManager(
            main_model        = os.environ.get("CREDENCE_MAIN_MODEL")        or None,
            compression_model = os.environ.get("CREDENCE_COMPRESSION_MODEL") or None,
            registry          = _get_registry(),
            session_id        = session_id,
            use_scout         = use_scout,
        )
    return _sessions[session_id]


def _validate_source(source: str) -> str:
    """Normalize and validate envelope source field."""
    return source.strip() or "unknown"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.tool()
    def credence_chat(session_id: str, message: str) -> dict:
        """
        Send a message through Credence and receive a response with epistemic envelope.

        The 'envelope' in the response contains:
          - j_score: confidence of this response (0–1)
          - zone: HIGH / MEDIUM / LOW
          - trust_score: j_score degraded by chain depth + source trust
          - should_verify: True when trust too low to act without verification
          - safe_to_compress: True only when HIGH-J, not uncertainty-preserved, trusted
          - uncertainty_preserved: True when faithfulness probe kept this turn verbatim

        If the user message contains uncertainty markers, it is automatically
        registered in the epistemic registry for tracking and verification.

        Args:
            session_id: Unique session identifier (create any string for new sessions)
            message: User message to send

        Returns:
            response, envelope, decision, j_score, zone, tokens_saved, drift_state,
            uncertainty_preserved, auto_registered (bool)
        """
        # Auto-register user message if it contains uncertainty markers
        msg_lower       = message.lower()
        auto_registered = False
        if any(m in msg_lower for m in _UNCERTAINTY_MARKERS):
            proxy    = CredenceProxy(theta_high=0.70, theta_low=0.45)
            cr_msg   = proxy.compute(message)
            _get_registry().register(
                content    = message[:1000],   # cap length for DB storage
                session_id = session_id,
                j_score    = cr_msg.j_score,
                zone       = cr_msg.zone,
            )
            auto_registered = True

        mgr    = _get_session(session_id)
        result = mgr.chat(message)
        return {
            "response":              result.response,
            "envelope":              result.envelope,
            "decision":              result.decision,
            "j_score":               result.j_score,
            "zone":                  result.zone,
            "tokens_saved":          result.tokens_saved,
            "drift_state":           result.drift_state,
            "adaptive_theta_high":   result.adaptive_theta_high,
            "adaptive_theta_low":    result.adaptive_theta_low,
            "uncertainty_preserved": result.uncertainty_preserved,
            "auto_registered":       auto_registered,
            "truth_buffer_count":    result.truth_buffer_count,
            "scout_extractions":     result.scout_extractions,
        }

    @mcp.tool()
    def credence_inspect(envelope_dict: dict) -> dict:
        """
        Inspect a CredenceEnvelope received from another agent.

        Use this tool when you receive an envelope from an upstream agent and
        need to decide whether to trust, verify, or compress its content.

        Returns trust analysis with actionable recommendations.

        Args:
            envelope_dict: The envelope dict from a previous credence_chat or agent handoff

        Returns:
            trust_score, should_verify, safe_to_compress, recommendation
        """
        try:
            env = CredenceEnvelope.from_dict(envelope_dict)
        except (KeyError, TypeError) as e:
            return {"error": f"Invalid envelope: {e}"}

        if env.trust_score < 0.30:
            recommendation = "BLOCK — trust too low. Verify original source before using."
        elif env.should_verify:
            recommendation = "VERIFY — trust marginal. Cross-check before compressing or acting."
        elif not env.safe_to_compress:
            recommendation = "PRESERVE — do not summarize or compress this content."
        else:
            recommendation = "PROCEED — HIGH-J, trusted, no uncertainty flags."

        return {
            "trust_score":         env.trust_score,
            "should_verify":       env.should_verify,
            "safe_to_compress":    env.safe_to_compress,
            "uncertainty_preserved": env.uncertainty_preserved,
            "chain_depth":         env.chain_depth,
            "source":              env.source,
            "source_trusted":      env.source in _TRUSTED_SOURCES,
            "recommendation":      recommendation,
        }

    @mcp.tool()
    def credence_propagate(envelope_dict: dict, new_source: str) -> dict:
        """
        Propagate a CredenceEnvelope to the next agent hop.

        Call this when passing information from one agent to another.
        Increments chain_depth (degrading trust) and updates source.
        The verified flag is reset — the new agent hasn't confirmed this.

        Args:
            envelope_dict: The envelope to propagate
            new_source: ID of the receiving agent

        Returns:
            Updated envelope dict with incremented chain_depth
        """
        try:
            env = CredenceEnvelope.from_dict(envelope_dict)
        except (KeyError, TypeError) as e:
            return {"error": f"Invalid envelope: {e}"}

        propagated = env.propagate(new_source=_validate_source(new_source))
        return propagated.to_dict()

    @mcp.tool()
    def credence_stats(session_id: str) -> dict:
        """
        Return session statistics: tokens used, saved, cost, compression counts.

        Args:
            session_id: Session identifier

        Returns:
            Full session stats including compression_ratio and per-decision counts
        """
        mgr = _get_session(session_id)
        return {
            "total_tokens_in":    mgr.stats.total_tokens_in,
            "total_tokens_out":   mgr.stats.total_tokens_out,
            "total_tokens_saved": mgr.stats.total_tokens_saved,
            "total_cost_usd":     round(mgr.stats.total_cost_usd, 4),
            "compression_ratio":  round(mgr.stats.compression_ratio, 3),
            "turns_compressed":   mgr.stats.turns_compressed,
            "turns_trimmed":      mgr.stats.turns_trimmed,
            "turns_preserved":    mgr.stats.turns_preserved,
            "turn_count":         mgr._turn_idx,
            "drift_state":        mgr._drift_state,
            "regime_active":      mgr._should_enable_credence(),
        }

    @mcp.tool()
    def credence_log(session_id: str) -> list:
        """
        Return per-turn decision log with J-scores, zones, decisions, and reasoning.

        Useful for debugging why a specific turn was compressed or preserved.

        Args:
            session_id: Session identifier

        Returns:
            List of per-turn dicts with turn, j_score, zone, decision, reasoning
        """
        return _get_session(session_id).decision_log

    @mcp.tool()
    def credence_save(session_id: str, path: str) -> dict:
        """
        Persist full session state to disk for cross-session continuity.

        Saves history, J-scores, vocabulary, adaptive thresholds, and stats.
        Load with credence_load in a future session to resume with full context.

        Args:
            session_id: Session to save
            path: File path for the JSON state file

        Returns:
            Status confirmation with session metadata
        """
        mgr = _get_session(session_id)
        mgr.save(path)
        return {
            "status":     "saved",
            "path":       path,
            "turn_count": mgr._turn_idx,
            "version":    "1.1",
        }

    @mcp.tool()
    def credence_load(session_id: str, path: str) -> dict:
        """
        Load a previously saved session state.

        Restores history, J-scores, vocabulary, and stats from a prior session.
        Migrates older session formats automatically.

        Args:
            session_id: Session ID to load state into (creates new if not exists)
            path: File path of the saved JSON state

        Returns:
            Status confirmation with loaded session metadata
        """
        mgr = _get_session(session_id)
        mgr.load(path)
        return {
            "status":     "loaded",
            "turn_count": mgr._turn_idx,
            "history_len": len(mgr._history),
        }

    @mcp.tool()
    def credence_reset(session_id: str) -> dict:
        """
        Reset a Credence session, clearing all history and stats.

        Args:
            session_id: Session to reset

        Returns:
            Status confirmation
        """
        if session_id in _sessions:
            _sessions[session_id].reset()
        return {"status": "reset", "session_id": session_id}

    @mcp.tool()
    def credence_risk(content: str, chain_depth: int = 0) -> dict:
        """
        Pre-flight epistemic risk assessment before compressing or forwarding content.

        Call this before:
          - Summarising a conversation segment
          - Passing content to another agent
          - Storing content in long-term memory
          - Including content in a system prompt

        Returns a risk level and recommended action based on the epistemic
        state of the content.

        Args:
            content:     The text to assess (response or conversation segment)
            chain_depth: How many agent hops this content has already traversed (default 0)

        Returns:
            risk_level, j_score, zone, action, reasoning, safe_to_compress, should_verify
        """
        proxy = CredenceProxy(theta_high=0.70, theta_low=0.45)
        cr = proxy.compute(content)

        # Check for uncertainty markers
        lower = content.lower()
        uncertainty_hits = [m for m in _UNCERTAINTY_MARKERS if m in lower]
        has_uncertainty = len(uncertainty_hits) > 0

        # Trust degradation with chain depth
        chain_penalty = chain_depth * 0.05
        effective_trust = max(0.0, cr.j_score - chain_penalty)

        # Risk assessment
        if has_uncertainty and cr.zone in ("LOW", "MEDIUM"):
            risk_level = "HIGH"
            action = "PRESERVE — content contains explicit uncertainty markers. Do not compress or summarise."
        elif has_uncertainty and cr.zone == "HIGH":
            risk_level = "MEDIUM"
            action = "VERIFY — linguistic J-score is high but uncertainty markers detected. Check for confident-sounding uncertain claims."
        elif cr.zone == "LOW" or effective_trust < 0.40:
            risk_level = "MEDIUM"
            action = "PRESERVE — low confidence zone or trust degraded by chain depth. Keep verbatim."
        elif cr.zone == "MEDIUM":
            risk_level = "LOW"
            action = "TRIM — medium confidence. Safe to trim but not compress. Preserve exact wording of any hedged claims."
        else:
            risk_level = "NONE"
            action = "COMPRESS — high confidence, no uncertainty markers, trust sufficient. Safe to summarise."

        return {
            "risk_level":       risk_level,
            "j_score":          cr.j_score,
            "zone":             cr.zone,
            "effective_trust":  round(effective_trust, 3),
            "chain_depth":      chain_depth,
            "has_uncertainty":  has_uncertainty,
            "uncertainty_markers_found": uncertainty_hits[:5],
            "safe_to_compress": risk_level == "NONE",
            "should_verify":    risk_level in ("HIGH", "MEDIUM"),
            "action":           action,
            "reasoning":        cr.reasoning,
        }

    @mcp.tool()
    def credence_register(
        content:    str,
        session_id: str,
        j_score:    float = 0.30,
        zone:       str   = "LOW",
    ) -> dict:
        """
        Explicitly register an uncertain constraint in the epistemic registry.

        Use when you want to manually track an uncertain claim for later
        verification. credence_chat auto-registers messages with uncertainty markers;
        use this tool for explicit registration or custom j_score/zone values.

        Args:
            content:    The uncertain constraint text
            session_id: Session identifier for grouping constraints
            j_score:    Confidence score 0–1 (default 0.30 = clearly uncertain)
            zone:       "HIGH", "MEDIUM", or "LOW" (default "LOW")

        Returns:
            constraint_id, status, content, session_id, j_score, zone
        """
        registry = _get_registry()
        cid = registry.register(
            content    = content,
            session_id = session_id,
            j_score    = j_score,
            zone       = zone,
        )
        return {
            "constraint_id": cid,
            "status":        "registered",
            "content":       content,
            "session_id":    session_id,
            "j_score":       j_score,
            "zone":          zone,
            "message":       (
                f"Registered as uncertain (id={cid}). "
                f"Call credence_verify('{cid}', <confirmed_value>, '{session_id}') "
                "when the value is confirmed."
            ),
        }

    @mcp.tool()
    def credence_verify(
        constraint_id:  str,
        verified_value: str,
        session_id:     str,
    ) -> dict:
        """
        Mark a registered uncertain constraint as verified with its confirmed value.

        This is the write-back operation — it closes the loop on an uncertain claim.
        After verification:
          - The constraint is excluded from credence_list_uncertain
          - credence_check_contradiction will flag new claims that conflict with this value
          - The envelope's should_verify flag becomes False for downstream agents

        Args:
            constraint_id:  The ID returned by credence_register or credence_chat auto-register
            verified_value: The confirmed factual value (e.g. "86400 seconds per vendor docs")
            session_id:     Session identifier (for audit context)

        Returns:
            Updated constraint dict with verified=True and verified_value set
        """
        registry = _get_registry()
        result   = registry.verify(constraint_id, verified_value)
        if "error" in result:
            return result
        result["status"]  = "verified"
        result["message"] = (
            f"Constraint verified. Confirmed value: '{verified_value}'. "
            "Safe to implement code that depends on this value."
        )
        return result

    @mcp.tool()
    def credence_list_uncertain(session_id: str) -> dict:
        """
        List all unverified uncertain constraints registered for this session.

        Use this before implementing code that may depend on unconfirmed values,
        or at the end of a session to audit what still needs verification.

        Args:
            session_id: Session identifier

        Returns:
            count, constraints list (each with constraint_id, content, j_score, zone),
            and a human-readable message
        """
        registry    = _get_registry()
        constraints = registry.list_uncertain(session_id)
        count       = len(constraints)
        if count == 0:
            message = "All constraints verified. No unresolved uncertainties for this session."
        elif count == 1:
            message = "1 unverified constraint — confirm before implementing code that depends on it."
        else:
            message = (
                f"{count} unverified constraints — confirm each before implementing "
                "code that depends on them."
            )
        return {
            "count":       count,
            "constraints": constraints,
            "message":     message,
        }

    @mcp.tool()
    def credence_gate(
        tool_name:           str,
        arguments_summary:   str,
        session_id:          str,
    ) -> dict:
        """
        Agentic pre-execution gate: check for unverified constraints before running a tool.

        Call this BEFORE any tool execution that may depend on user-stated constraints
        (e.g. write_file, execute_code, send_request, deploy). If unverified constraints
        are topically related to the planned action, block and surface a warning.

        This converts Credence from a passive monitor into an active safety gate that
        intercepts tool calls before they bake uncertain assumptions into irreversible
        actions.

        Args:
            tool_name:           Name of the tool about to be called (e.g. "write_file")
            arguments_summary:   Brief summary of the arguments (sensitive values omitted)
            session_id:          Session identifier

        Returns:
            proceed: bool — True if safe to proceed, False if verification needed
            blocked_by: list of constraint dicts that triggered the block
            recommendation: human-readable action
        """
        registry    = _get_registry()
        uncertain   = registry.list_uncertain(session_id)

        if not uncertain:
            return {
                "proceed":         True,
                "blocked_by":      [],
                "unverified_count": 0,
                "recommendation":  "PROCEED — no unverified constraints in this session.",
            }

        # Keyword overlap between planned action and unverified constraints
        action_text    = f"{tool_name} {arguments_summary}".lower()
        action_words   = set(re.sub(r"[^\w\s]", " ", action_text).split())
        blocking: list[dict] = []

        for c in uncertain:
            constraint_words = set(re.sub(r"[^\w\s]", " ", c["content"].lower()).split())
            overlap = action_words & constraint_words - {
                "the", "and", "for", "that", "this", "with", "from",
                "are", "was", "not", "but", "all", "any"
            }
            if len(overlap) >= 2:
                c["overlap_terms"] = list(overlap)[:6]
                blocking.append(c)

        if blocking:
            cids  = ", ".join(c["constraint_id"] for c in blocking)
            recommendation = (
                f"BLOCK — {len(blocking)} unverified constraint(s) may affect this action. "
                f"Verify them first with credence_verify (IDs: {cids}), "
                "or call credence_list_uncertain for the full list."
            )
        else:
            recommendation = (
                f"PROCEED — {len(uncertain)} unverified constraint(s) exist in session "
                "but none are topically related to this action."
            )

        return {
            "proceed":          len(blocking) == 0,
            "blocked_by":       blocking,
            "unverified_count": len(uncertain),
            "recommendation":   recommendation,
        }

    @mcp.tool()
    def credence_align(session_id: str, response_text: str) -> dict:
        """
        Output Alignment Layer — post-generation epistemic Governor check.

        Call this after any AI response to detect when the response asserts confidence
        about constraints the ledger marks as unverified. This is the Governor:
        the system that closes the loop between what was generated and what the
        epistemic record says is safe to assert.

        When warnings are found, the suggested caveats are automatically queued for
        injection into the NEXT turn's system prompt (async auditor pattern — zero
        latency impact on the current response, corrects the next one).

        Use in Claude Code CLAUDE.md:
          After every AI response in a session with registered constraints,
          call credence_align to check for epistemic misalignment.
          If warning_count > 0, include the caveats in your next prompt.

        Args:
            session_id:     Session identifier
            response_text:  The AI response to check against the epistemic ledger

        Returns:
            warning_count, alignment_warnings list, caveat_needed bool,
            suggested_caveats list, governor_active bool
        """
        mgr      = _get_session(session_id)
        warnings = mgr._align_output(response_text)

        if warnings:
            # Queue caveat for next turn (already done inside _align_output via
            # chat() wiring, but when called as standalone MCP tool we set it directly)
            mgr._pending_alignment_caveat = "\n".join(w.suggested_caveat for w in warnings)

        warning_dicts = [w.to_dict() for w in warnings]
        return {
            "warning_count":      len(warnings),
            "alignment_warnings": warning_dicts,
            "caveat_needed":      len(warnings) > 0,
            "suggested_caveats":  [w["suggested_caveat"] for w in warning_dicts],
            "governor_active":    mgr._registry is not None,
            "message": (
                f"{len(warnings)} alignment warning(s) detected. "
                "Caveats queued for next turn injection."
                if warnings else
                "No alignment issues detected — response confidence matches ledger state."
            ),
        }

    @mcp.tool()
    def credence_trajectory(constraint_id: str) -> dict:
        """
        Return the certainty trajectory for a constraint — its full event history.

        A trajectory shows how a constraint's confidence evolved from first observation
        through Scout extraction, chat updates, and finally verification. Useful for
        auditing how long an assumption was left unverified.

        Args:
            constraint_id: The ID returned by credence_register or auto-registration

        Returns:
            constraint (current state), events (ordered event log), event_count
        """
        registry = _get_registry()
        events   = registry.get_trajectory(constraint_id)
        # Also return current constraint state
        rows = registry._conn.execute(
            "SELECT * FROM constraints WHERE constraint_id=?", (constraint_id,)
        ).fetchone()
        if rows is None:
            return {"error": f"constraint_id '{constraint_id}' not found"}
        return {
            "constraint":   registry._row_to_dict(rows),
            "events":       events,
            "event_count":  len(events),
        }

    @mcp.tool()
    def credence_claims(session_id: str) -> dict:
        """
        List all claim nodes for a session — both manually registered and auto-extracted.

        Returns the full epistemic ledger including auto-extracted claims from
        use_claim_extraction turns, organized by status (unverified/verified).
        """
        reg       = _get_registry()
        uncertain = reg.list_uncertain(session_id)
        verified  = reg.list_verified(session_id)
        return {
            "session_id":       session_id,
            "total_claims":     len(uncertain) + len(verified),
            "unverified_count": len(uncertain),
            "verified_count":   len(verified),
            "unverified":       uncertain,
            "verified":         verified,
            "message": (
                f"{len(uncertain)} unverified, {len(verified)} verified claims "
                f"for session '{session_id}'."
            ),
        }

    @mcp.tool()
    def credence_check_contradiction(claim: str, session_id: str) -> dict:
        """
        Check whether a new claim contradicts verified constraints in this session.

        Computes Jaccard word-overlap similarity between the claim and all verified
        constraints. High-similarity verified constraints may represent factual conflicts.

        Use before asserting a new value that was previously uncertain, to detect
        cases where the newly stated value conflicts with what was already confirmed.

        Args:
            claim:      The new claim or fact to check
            session_id: Session to check against

        Returns:
            has_contradiction, matches with similarity scores, recommendation
        """
        registry = _get_registry()
        matches  = registry.check_contradiction(claim, session_id)
        has_contradiction = len(matches) > 0
        if has_contradiction:
            recommendation = (
                f"REVIEW — {len(matches)} verified constraint(s) are topically similar "
                "to this claim. Check for factual conflict before proceeding."
            )
        else:
            recommendation = "PROCEED — no similar verified constraints found. No contradiction detected."
        return {
            "has_contradiction": has_contradiction,
            "match_count":       len(matches),
            "matches":           matches,
            "recommendation":    recommendation,
        }

    @mcp.tool()
    def credence_scan_output(output_text: str, session_id: str) -> dict:
        """
        Generation-Time Constraint Scanner (GTS) with Confidence Policy Layer.

        Scans model output (code blocks AND prose) for numeric literals that match
        registered unverified constraints. Annotations are severity-tiered by
        effective confidence (decayed j_score):

            HIGH RISK  (eff_conf < 0.20):
                RATE_LIMIT = 50  # ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: ...
            UNVERIFIED (0.20 ≤ eff_conf < 0.40):
                RATE_LIMIT = 50  # ⚠ CREDENCE[unverified, conf=0.30]: ...
            CHECK      (eff_conf ≥ 0.40, still unverified):
                RATE_LIMIT = 50  # CREDENCE[check, conf=0.42]: ...
            VERIFIED:  no annotation (clean output).

        This closes the generation gap. Full lifecycle coverage:
            Storage   → registry (credence_register)
            Injection → Truth Buffer + Consistency Enforcer (credence_chat)
            Generation → GTS (this tool)

        Args:
            output_text: The raw model output to scan (code blocks and prose)
            session_id:  Session whose unverified constraints to scan against

        Returns:
            annotated_output  — output with inline CREDENCE annotations inserted
            scan_hits         — list of {value, constraint_id, constraint_text, eff_conf, source, line}
            hit_count         — number of literals annotated
            high_risk_count   — hits where eff_conf < 0.20 (should block before shipping)
            recommendation    — action summary
        """
        mgr = _get_session(session_id)
        annotated, hits = mgr._scan_output_for_constraints(output_text)

        high_risk = [h for h in hits if h.get("eff_conf", 1.0) < 0.20]

        if high_risk:
            recommendation = (
                f"BLOCK — {len(high_risk)} HIGH RISK literal(s) found. "
                "These values have very low confidence and must be verified before use."
            )
        elif hits:
            recommendation = (
                f"REVIEW — {len(hits)} unverified literal(s) annotated. "
                "Confirm values with source before shipping."
            )
        else:
            recommendation = "CLEAN — no output literals matched registered unverified constraints."

        return {
            "annotated_output": annotated,
            "scan_hits":        hits,
            "hit_count":        len(hits),
            "high_risk_count":  len(high_risk),
            "recommendation":   recommendation,
        }


# ---------------------------------------------------------------------------
# Cross-session memory tools
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:
    @mcp.tool()
    def credence_memory_snapshot(session_id: str, project_id: str) -> dict:
        """
        Save all unverified constraints from session_id as persistent project memory.

        Call this at the END of a Claude Code session to preserve epistemic state
        across session boundaries. Any constraint that was stated but never verified
        during the session will be remembered for future sessions on the same project.

        Unlike regular memory systems (Mem0, Zep), Credence memory carries epistemic
        provenance: it remembers not just WHAT you told Claude, but WHETHER it was verified.

        Args:
            session_id: The current session ID (used to identify which constraints to save).
            project_id: A stable project identifier (e.g. "my-api-project", "payment-service").
                        All sessions for the same codebase should use the same project_id.

        Returns:
            saved_count: Number of unverified constraints saved to project memory.
            items: List of saved constraints with their content and confidence levels.
            message: Human-readable summary.

        Example:
            At end of session where you discussed an unconfirmed rate limit:
            → credence_memory_snapshot("session-abc", "payment-service")
            → "Saved 2 unverified constraints to project 'payment-service'"
        """
        from .memory import CredenceMemory
        mem = CredenceMemory(_get_registry())
        snap = mem.snapshot(session_id=session_id, project=project_id)
        return {
            "project_id":  snap.project_id,
            "session_id":  snap.session_id,
            "saved_count": snap.saved_count,
            "items": [
                {
                    "constraint_id": item.get("constraint_id"),
                    "content":       item.get("content"),
                    "zone":          item.get("zone"),
                    "j_score":       item.get("j_score"),
                }
                for item in snap.items
            ],
            "message": snap.summary(),
        }

    @mcp.tool()
    def credence_memory_recall(
        project_id: str,
        new_session_id: str,
        context_hint: str = "",
    ) -> dict:
        """
        Load project memories into a new session at session start.

        Call this at the START of a new Claude Code session before any other
        credence_chat calls. It injects all previously unverified constraints
        from the project into the new session's registry so the Truth Buffer
        and Consistency Enforcer work from turn 1.

        This is the key capability that no other memory system provides:
        the new session starts KNOWING what it doesn't know.

        Args:
            project_id:     Project identifier matching what was used in credence_memory_snapshot.
            new_session_id: ID for the new session (e.g. a fresh UUID).
            context_hint:   Optional keyword string to filter relevant memories.
                            E.g. "rate limit authentication" to load only auth-related memories.
                            Leave empty to load all project memories.

        Returns:
            injected_count: Number of constraints injected into the new session.
            system_block:   Ready-to-use system prompt prefix — prepend to your session's
                            system prompt so Claude starts aware of pending uncertainties.
            items:          List of injected constraints.
            message:        Human-readable summary.

        Example:
            At start of new session on the same project:
            → credence_memory_recall("payment-service", "session-xyz")
            → system_block includes: "⚠ [LOW] rate limit ~50 req/min — UNVERIFIED"
            → Claude now knows to hedge when discussing rate limits from turn 1
        """
        from .memory import CredenceMemory
        mem = CredenceMemory(_get_registry())
        recall = mem.recall_and_inject(
            project=project_id,
            new_session_id=new_session_id,
            context_hint=context_hint,
        )
        return {
            "project_id":      recall.project_id,
            "new_session_id":  recall.new_session_id,
            "injected_count":  recall.injected_count,
            "system_block":    recall.system_block,
            "items": [
                {
                    "constraint_id": item.get("constraint_id"),
                    "content":       item.get("content"),
                    "zone":          item.get("zone"),
                    "j_score":       item.get("j_score"),
                    "session_id":    item.get("session_id"),
                }
                for item in recall.items
            ],
            "is_empty": recall.is_empty(),
            "message":  (
                f"Loaded {recall.injected_count} unverified constraint(s) from project '{project_id}' "
                f"into session '{new_session_id}'."
                if not recall.is_empty()
                else f"No unverified constraints found for project '{project_id}'."
            ),
        }

    @mcp.tool()
    def credence_memory_status(project_id: str) -> dict:
        """
        Show epistemic debt for a project — all unverified constraints across all sessions.

        Use this to audit what assumptions are still pending verification for a project.
        'Epistemic debt' = number of unverified constraints still outstanding.
        High epistemic debt means future sessions will carry more uncertainty overhead.

        Args:
            project_id: Project identifier to query.

        Returns:
            epistemic_debt: Total unverified + disputed constraints.
            verified_count: Constraints that have been confirmed.
            unverified: List of pending uncertain constraints.
        """
        from .memory import CredenceMemory
        mem = CredenceMemory(_get_registry())
        status = mem.project_status(project_id)
        return status

    @mcp.tool()
    def credence_pipeline_intercept(
        agent_output: str,
        from_session:  str,
        to_session:    str,
        use_ghost_detector: bool = False,
    ) -> dict:
        """
        Epistemic middleware for multi-agent pipelines.

        Call this BETWEEN agents — after Agent A produces output and BEFORE
        Agent B is invoked. The monitor scans Agent A's output for uncertain
        claims, registers them in the shared registry, and returns a
        system_block to prepend to Agent B's system prompt.

        Without this tool: Agent A's uncertain estimates arrive at Agent B as
        apparent facts. Agent B embeds them in code without qualification.
        With this tool: Agent B's Truth Buffer sees the constraints and enforces
        qualifier propagation automatically.

        Args:
            agent_output:       The text output from the upstream agent (Agent A).
            from_session:       Session ID of the upstream agent.
            to_session:         Session ID of the downstream agent.
            use_ghost_detector: If True, use Ghost Detector for implicit claims
                                (requires ANTHROPIC_API_KEY set server-side).
                                If False, use probe-only (free, deterministic).

        Returns:
            system_block:   Prepend this to Agent B's system prompt.
            n_extracted:    Uncertain claims found in Agent A's output.
            n_injected:     Claims registered and injected.
            strategy:       "probe" | "ghost_detector" | "none"
            has_uncertain:  True if any uncertain claims were found.
            claims:         List of extracted claims with confidence scores.
        """
        from .pipeline_monitor import PipelineMonitor
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        monitor = PipelineMonitor(
            registry=_get_registry(),
            api_key=api_key if use_ghost_detector else None,
            use_ghost_detector=use_ghost_detector,
        )
        handoff = monitor.intercept(
            agent_output=agent_output,
            from_session=from_session,
            to_session=to_session,
        )
        return {
            "system_block":  handoff.system_block,
            "n_extracted":   handoff.n_extracted,
            "n_injected":    handoff.n_injected,
            "strategy":      handoff.strategy,
            "has_uncertain": handoff.has_uncertain,
            "claims": [
                {
                    "content":    c.content,
                    "confidence": c.confidence,
                    "source":     c.source,
                    "cid":        c.cid,
                }
                for c in handoff.claims
            ],
        }


# ---------------------------------------------------------------------------
# MCP Resources — epistemic:// URI scheme
#
# Resources expose the epistemic ledger as passive context any MCP-compatible
# agent can read without calling a tool. This is the architectural shift from
# "a set of tools epistemically-aware code calls" to "a memory substrate any
# agent in the pipeline inherits automatically."
#
# URI scheme:
#   epistemic://session/{session_id}/ledger              — all constraints
#   epistemic://session/{session_id}/constraint/{id}     — single constraint + trajectory
#   epistemic://session/{session_id}/alignment           — pending governor caveats
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.resource("epistemic://session/{session_id}/ledger")
    def epistemic_ledger(session_id: str) -> str:
        """
        Epistemic ledger for a session — all registered uncertain constraints.

        Returns a JSON document listing every constraint registered for this
        session, including verified and unverified states. Downstream agents
        read this resource to know what is uncertain before acting.

        URI: epistemic://session/{session_id}/ledger
        """
        import json as _json
        registry    = _get_registry()
        uncertain   = registry.list_uncertain(session_id)
        all_rows    = registry._conn.execute(
            "SELECT * FROM constraints WHERE session_id=? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        all_constraints = [registry._row_to_dict(r) for r in all_rows]
        return _json.dumps({
            "session_id":          session_id,
            "total_constraints":   len(all_constraints),
            "unverified_count":    len(uncertain),
            "verified_count":      len(all_constraints) - len(uncertain),
            "constraints":         all_constraints,
            "etp_version":         "1.0",
        }, indent=2)

    @mcp.resource("epistemic://session/{session_id}/constraint/{constraint_id}")
    def epistemic_constraint(session_id: str, constraint_id: str) -> str:
        """
        Single constraint with full certainty trajectory.

        Returns the constraint's current state plus its complete event log —
        every register, scout, verify, and contradict event with timestamps
        and J-scores. Use to audit how long an assumption was left unverified.

        URI: epistemic://session/{session_id}/constraint/{constraint_id}
        """
        import json as _json
        registry = _get_registry()
        rows     = registry._conn.execute(
            "SELECT * FROM constraints WHERE constraint_id=? AND session_id=?",
            (constraint_id, session_id),
        ).fetchone()
        if rows is None:
            return _json.dumps({"error": f"constraint '{constraint_id}' not found in session '{session_id}'"})
        constraint = registry._row_to_dict(rows)
        trajectory = registry.get_trajectory(constraint_id)
        return _json.dumps({
            "session_id":  session_id,
            "constraint":  constraint,
            "trajectory":  trajectory,
            "event_count": len(trajectory),
            "etp_version": "1.0",
        }, indent=2)

    @mcp.resource("epistemic://session/{session_id}/alignment")
    def epistemic_alignment(session_id: str) -> str:
        """
        Pending Governor alignment caveats for this session.

        Returns any queued alignment caveats from the Output Alignment Layer.
        These are warnings that the previous AI response was more confident
        than the ledger warrants — they will be injected into the next turn.

        URI: epistemic://session/{session_id}/alignment
        """
        import json as _json
        mgr    = _sessions.get(session_id)
        caveat = mgr._pending_alignment_caveat if mgr else None
        return _json.dumps({
            "session_id":              session_id,
            "pending_caveat":          caveat,
            "governor_active":         mgr is not None and mgr._registry is not None,
            "caveat_pending":          caveat is not None,
            "etp_version":             "1.0",
        }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _FASTMCP_AVAILABLE:
        print("fastmcp not installed. Run: pip install fastmcp")
        print("Then: python -m credence.mcp_server")
        return
    mcp.run()


if __name__ == "__main__":
    main()

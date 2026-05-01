"""
credence/mcp_server.py
==================
Credence MCP Server — 10 validated tools for epistemic context management.

Core principle: memory allocation decisions should be conditioned on epistemic
state. Only HIGH-J (epistemically resolved) content is safe to compress. Uncertain
content is preserved verbatim through every compression and trim operation.

Tools (10 validated):
    credence_chat           — Chat with full enforcement stack (probe + truth buffer + CE)
    credence_register       — Register an uncertain constraint
    credence_verify         — Mark a constraint as verified
    credence_list_uncertain — Query unverified constraints for a session
    credence_gate           — Agentic pre-tool gate: block if uncertain constraints apply
    credence_scan_output    — GTS: scan model output for unverified numeric literals
    credence_memory_snapshot — Save unverified constraints as project memory
    credence_memory_recall  — Load project memory into a new session
    credence_stats          — Session statistics (tokens, cost, compressions)
    credence_reset          — Reset a session

Resources (passive, epistemic:// URI scheme):
    epistemic://session/{session_id}/ledger             — all constraints
    epistemic://session/{session_id}/constraint/{id}    — single constraint + trajectory

Run:
    python -m credence.mcp_server

Requires:
    pip install fastmcp anthropic
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
from .registry import CredenceRegistry

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Credence",
    instructions=(
        "Credence is an epistemic memory layer for AI systems. It preserves "
        "uncertainty state — which constraints are unverified — through context "
        "compression and across multi-agent pipelines.\n\n"
        "Lifecycle:\n"
        "  1. credence_memory_recall at session START — load prior uncertainties\n"
        "  2. credence_chat for each turn — enforcement fires automatically\n"
        "  3. credence_gate BEFORE any irreversible tool call\n"
        "  4. credence_verify when a constraint is confirmed\n"
        "  5. credence_scan_output BEFORE shipping generated code\n"
        "  6. credence_memory_snapshot at session END — persist for next session\n\n"
        "The key invariant: never state an uncertain constraint as a confirmed fact."
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
        db_path   = os.environ.get("CREDENCE_REGISTRY_PATH", "epistemic_registry.db")
        _registry = CredenceRegistry(db_path=db_path)
    return _registry


def _session_state_path(session_id: str) -> str:
    session_dir = os.environ.get("CREDENCE_SESSION_DIR", ".credence/sessions")
    os.makedirs(session_dir, exist_ok=True)
    safe_id = re.sub(r"[^\w\-.]", "_", session_id)[:128]
    return os.path.join(session_dir, f"{safe_id}.json")


def _get_session(session_id: str) -> ContextManager:
    """Return or create a ContextManager for the given session ID."""
    if session_id not in _sessions:
        mgr = ContextManager(
            main_model        = os.environ.get("CREDENCE_MAIN_MODEL") or None,
            compression_model = os.environ.get("CREDENCE_COMPRESSION_MODEL") or None,
            registry          = _get_registry(),
            session_id        = session_id,
        )
        state_path = _session_state_path(session_id)
        if os.path.exists(state_path):
            try:
                mgr.load(state_path)
            except Exception:
                pass
        _sessions[session_id] = mgr
    return _sessions[session_id]


def _persist_session(session_id: str) -> None:
    """Write the current session state to disk. Called after every chat turn."""
    mgr = _sessions.get(session_id)
    if mgr is None:
        return
    auto_persist = os.environ.get("CREDENCE_AUTO_PERSIST", "1").lower() not in ("0", "false", "no")
    if not auto_persist:
        return
    try:
        mgr.save(_session_state_path(session_id))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tools (10 validated)
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.tool()
    def credence_chat(session_id: str, message: str) -> dict:
        """
        Send a message through Credence and receive a response with epistemic metadata.

        Every turn runs the full enforcement stack:
          1. Truth Buffer — injects all unverified constraints into system prompt
          2. Consistency Enforcer — fires imperative enforcement on direct queries
          3. Faithfulness probe — blocks compression if uncertainty markers present
          4. J-score routing — HIGH→compress, MEDIUM→trim, LOW→preserve
          5. Auto-registration — uncertain user messages registered automatically

        Args:
            session_id: Session identifier (use the same ID across a conversation)
            message:    User message

        Returns:
            response, j_score, zone, decision, enforcement_active,
            truth_buffer_count, tokens_in, tokens_out, cost_usd
        """
        mgr = _get_session(session_id)
        try:
            result = mgr.chat(message)
        except Exception as e:
            return {"error": str(e)}

        _persist_session(session_id)

        return {
            "response":           result.response,
            "j_score":            round(result.j_score, 4),
            "zone":               result.zone,
            "decision":           result.decision,
            "enforcement_active": getattr(result, "enforcement_active", False),
            "truth_buffer_count": getattr(result, "truth_buffer_count", 0),
            "scout_extractions":  getattr(result, "scout_extractions", 0),
            "tokens_in":          result.tokens_in,
            "tokens_out":         result.tokens_out,
            "tokens_saved":       result.tokens_saved,
            "cost_usd":           round(result.cost_usd, 6),
        }

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
    def credence_register(
        content:    str,
        session_id: str,
        j_score:    float = 0.30,
        zone:       str   = "LOW",
    ) -> dict:
        """
        Explicitly register an uncertain constraint in the epistemic registry.

        credence_chat auto-registers messages with uncertainty markers.
        Use this tool when you want to manually track an uncertain claim
        or supply custom j_score/zone values.

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
            "message": (
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

        This closes the loop on an uncertain claim. After verification:
          - The constraint is excluded from credence_list_uncertain
          - The Truth Buffer stops injecting it into system prompts
          - The Consistency Enforcer no longer fires for it

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

        Use before implementing code that may depend on unconfirmed values,
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

        Args:
            tool_name:           Name of the tool about to be called (e.g. "write_file")
            arguments_summary:   Brief summary of the arguments (sensitive values omitted)
            session_id:          Session identifier

        Returns:
            proceed: bool — True if safe to proceed, False if verification needed
            blocked_by: list of constraint dicts that triggered the block
            recommendation: human-readable action
        """
        from .context_manager import _CE_STOPWORDS, _CE_DOMAIN_SYNONYMS

        registry  = _get_registry()
        uncertain = registry.list_uncertain(session_id)

        if not uncertain:
            return {
                "proceed":          True,
                "blocked_by":       [],
                "unverified_count": 0,
                "recommendation":   "PROCEED — no unverified constraints in this session.",
            }

        # Reuse the CE synonym-expansion for more accurate overlap
        cm = ContextManager.__new__(ContextManager)
        action_text = f"{tool_name} {arguments_summary}".lower()
        raw_tokens  = set(re.sub(r"[^\w\s]", " ", action_text).split()) - _CE_STOPWORDS
        if hasattr(cm, "_expand_tokens"):
            action_tokens = cm._expand_tokens(raw_tokens)
        else:
            action_tokens = raw_tokens

        blocking: list[dict] = []
        for c in uncertain:
            c_raw  = set(re.sub(r"[^\w\s]", " ", c["content"].lower()).split()) - _CE_STOPWORDS
            c_exp  = cm._expand_tokens(c_raw) if hasattr(cm, "_expand_tokens") else c_raw
            overlap = action_tokens & c_exp
            if len(overlap) >= 2:
                c["overlap_terms"] = list(overlap)[:6]
                blocking.append(c)

        if blocking:
            cids = ", ".join(c["constraint_id"] for c in blocking)
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
    def credence_scan_output(output_text: str, session_id: str) -> dict:
        """
        Generation-Time Constraint Scanner (GTS) with Confidence Policy Layer.

        Scans model output (code blocks AND prose) for numeric literals that match
        registered unverified constraints. Annotations are severity-tiered by
        effective confidence (decayed j_score):

            HIGH RISK  (eff_conf < 0.20):  ⚠⚠ CREDENCE[HIGH RISK, conf=0.15]: ...
            UNVERIFIED (0.20 ≤ eff_conf < 0.40): ⚠ CREDENCE[unverified, conf=0.30]: ...
            CHECK      (eff_conf ≥ 0.40): CREDENCE[check, conf=0.42]: ...
            VERIFIED:  no annotation (clean output).

        Args:
            output_text: The raw model output to scan (code blocks and prose)
            session_id:  Session whose unverified constraints to scan against

        Returns:
            annotated_output, scan_hits, hit_count, high_risk_count, recommendation
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
            session_id: The current session ID.
            project_id: A stable project identifier (e.g. "my-api-project").

        Returns:
            saved_count, items, message
        """
        from .memory import CredenceMemory
        mem  = CredenceMemory(_get_registry())
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
        project_id:     str,
        new_session_id: str,
        context_hint:   str = "",
    ) -> dict:
        """
        Load project memories into a new session at session start.

        Call this at the START of a new Claude Code session before any other
        credence_chat calls. It injects all previously unverified constraints
        from the project into the new session's registry so the Truth Buffer
        and Consistency Enforcer work from turn 1.

        The new session starts KNOWING what it doesn't know.

        Args:
            project_id:     Project identifier matching credence_memory_snapshot.
            new_session_id: ID for the new session.
            context_hint:   Optional keyword filter (e.g. "rate limit authentication").

        Returns:
            injected_count, system_block (prepend to system prompt), items, message
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
                {
                    "constraint_id": item.get("constraint_id"),
                    "content":       item.get("content"),
                    "zone":          item.get("zone"),
                    "j_score":       item.get("j_score"),
                }
                for item in recall.items
            ],
            "is_empty": recall.is_empty(),
            "message": (
                f"Loaded {recall.injected_count} unverified constraint(s) from project "
                f"'{project_id}' into session '{new_session_id}'."
                if not recall.is_empty()
                else f"No unverified constraints found for project '{project_id}'."
            ),
        }


# ---------------------------------------------------------------------------
# MCP Resources — epistemic:// URI scheme
#
# Resources expose the epistemic ledger as passive context any MCP-compatible
# agent can read without calling a tool.
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.resource("epistemic://session/{session_id}/ledger")
    def epistemic_ledger(session_id: str) -> str:
        """
        Epistemic ledger for a session — all registered uncertain constraints.

        URI: epistemic://session/{session_id}/ledger
        """
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
        """
        Single constraint with full certainty trajectory.

        URI: epistemic://session/{session_id}/constraint/{constraint_id}
        """
        import json as _json
        registry   = _get_registry()
        rows       = registry._conn.execute(
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
        print("fastmcp not installed. Run: pip install fastmcp")
        print("Then: python -m credence.mcp_server")
        return
    mcp.run()


if __name__ == "__main__":
    main()

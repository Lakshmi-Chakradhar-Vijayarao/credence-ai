"""
cams/mcp_server.py
==================
CAMS MCP Server — Epistemic Integrity Layer as a Model Context Protocol service.

Exposes CAMSContextManager as 7 MCP tools callable from Claude Desktop
or any MCP-compatible agent framework.

Every cams_chat response includes a CAMSEnvelope dict so downstream agents
can inspect epistemic provenance before compressing or acting on information.

Run:
    python -m cams.mcp_server

Or directly:
    python cams/mcp_server.py

Requires:
    pip install fastmcp anthropic

Trust boundary:
    Envelopes from unknown sources (not in _TRUSTED_SOURCES) receive a
    trust penalty, making should_verify=True more likely. This prevents
    CAMS from blindly trusting envelopes injected from untrusted agents.
"""

import os
from typing import Optional

try:
    from fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False

from .context_manager import CAMSContextManager
from .envelope import CAMSEnvelope, _TRUSTED_SOURCES

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "CAMS",
    instructions=(
        "CAMS is an epistemic integrity layer. Every cams_chat response includes "
        "an 'envelope' field with a J-score and trust metadata. Check "
        "envelope['should_verify'] before treating uncertain information as fact. "
        "Check envelope['safe_to_compress'] before summarizing a response. "
        "Use cams_save / cams_load for cross-session continuity."
    ),
) if _FASTMCP_AVAILABLE else None

# Session registry: session_id → CAMSContextManager
_sessions: dict[str, CAMSContextManager] = {}


def _get_session(session_id: str) -> CAMSContextManager:
    if session_id not in _sessions:
        _sessions[session_id] = CAMSContextManager()
    return _sessions[session_id]


def _validate_source(source: str) -> str:
    """Normalize and validate envelope source field."""
    return source.strip() or "unknown"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

if _FASTMCP_AVAILABLE:

    @mcp.tool()
    def cams_chat(session_id: str, message: str) -> dict:
        """
        Send a message through CAMS and receive a response with epistemic envelope.

        The 'envelope' in the response contains:
          - j_score: confidence of this response (0–1)
          - zone: HIGH / MEDIUM / LOW
          - trust_score: j_score degraded by chain depth + source trust
          - should_verify: True when trust too low to act without verification
          - safe_to_compress: True only when HIGH-J, not uncertainty-preserved, trusted
          - uncertainty_preserved: True when faithfulness probe kept this turn verbatim

        Args:
            session_id: Unique session identifier (create any string for new sessions)
            message: User message to send

        Returns:
            response, envelope, decision, j_score, zone, tokens_saved, drift_state
        """
        mgr    = _get_session(session_id)
        result = mgr.chat(message)
        return {
            "response":     result.response,
            "envelope":     result.envelope,
            "decision":     result.decision,
            "j_score":      result.j_score,
            "zone":         result.zone,
            "tokens_saved": result.tokens_saved,
            "drift_state":  result.drift_state,
            "adaptive_theta_high": result.adaptive_theta_high,
            "adaptive_theta_low":  result.adaptive_theta_low,
        }

    @mcp.tool()
    def cams_inspect_envelope(envelope_dict: dict) -> dict:
        """
        Inspect a CAMSEnvelope received from another agent.

        Use this tool when you receive an envelope from an upstream agent and
        need to decide whether to trust, verify, or compress its content.

        Returns trust analysis with actionable recommendations.

        Args:
            envelope_dict: The envelope dict from a previous cams_chat or agent handoff

        Returns:
            trust_score, should_verify, safe_to_compress, recommendation
        """
        try:
            env = CAMSEnvelope.from_dict(envelope_dict)
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
    def cams_propagate_envelope(envelope_dict: dict, new_source: str) -> dict:
        """
        Propagate a CAMSEnvelope to the next agent hop.

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
            env = CAMSEnvelope.from_dict(envelope_dict)
        except (KeyError, TypeError) as e:
            return {"error": f"Invalid envelope: {e}"}

        propagated = env.propagate(new_source=_validate_source(new_source))
        return propagated.to_dict()

    @mcp.tool()
    def cams_get_stats(session_id: str) -> dict:
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
            "regime_active":      mgr._should_enable_cams(),
        }

    @mcp.tool()
    def cams_get_decision_log(session_id: str) -> list:
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
    def cams_save(session_id: str, path: str) -> dict:
        """
        Persist full session state to disk for cross-session continuity.

        Saves history, J-scores, vocabulary, adaptive thresholds, and stats.
        Load with cams_load in a future session to resume with full context.

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
    def cams_load(session_id: str, path: str) -> dict:
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
    def cams_reset(session_id: str) -> dict:
        """
        Reset a CAMS session, clearing all history and stats.

        Args:
            session_id: Session to reset

        Returns:
            Status confirmation
        """
        if session_id in _sessions:
            _sessions[session_id].reset()
        return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _FASTMCP_AVAILABLE:
        print("fastmcp not installed. Run: pip install fastmcp")
        print("Then: python -m cams.mcp_server")
        return
    mcp.run()


if __name__ == "__main__":
    main()

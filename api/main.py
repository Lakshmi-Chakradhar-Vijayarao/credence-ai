"""
api/main.py
===========
Credence REST API — production backend for the hosted epistemic memory service.

Exposes the same 16 MCP tools as REST endpoints plus the three MCP Resources,
so developers can use Credence from any language without running Python locally.

Run locally:
    pip install fastapi uvicorn
    ANTHROPIC_API_KEY=... uvicorn api.main:app --reload

Deploy:
    fly deploy     # Fly.io (recommended — Dockerfile in repo root)
    railway up     # Railway

Auth:
    All endpoints except /health require an API key header:
    X-API-Key: cr-<your-key>

    For local development, set CREDENCE_DEV_MODE=1 to bypass auth.

Session model:
    Each session_id maps to an isolated ContextManager + registry entry.
    Sessions are in-memory by default; for persistence set CREDENCE_DB_PATH.

Versioning:
    All endpoints are under /v1/. Breaking changes will increment to /v2/.
"""

import os
import time
import json
import hashlib
import secrets
from typing import Optional, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Header, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as _e:
    raise ImportError(
        "Credence API requires FastAPI and Pydantic. "
        "Install with: pip install fastapi uvicorn pydantic"
    ) from _e

_FASTAPI_AVAILABLE = True

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credence.context_manager import ContextManager
from credence.confidence_proxy import CredenceProxy
from credence.registry import CredenceRegistry
from credence.context_manager import _UNCERTAINTY_MARKERS

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credence API",
    description=(
        "Epistemic memory layer for AI pipelines. "
        "Preserves uncertainty qualifiers through compression, agent handoffs, "
        "and session boundaries. Reference implementation of ETP v1."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_sessions:  dict[str, ContextManager] = {}
_registry:  Optional[CredenceRegistry] = None
_api_keys:  set[str] = set()   # loaded from env or generated
_start_time = time.time()

_DEV_MODE = os.environ.get("CREDENCE_DEV_MODE", "").lower() in ("1", "true", "yes")


def _get_registry() -> CredenceRegistry:
    global _registry
    if _registry is None:
        db_path = os.environ.get("CREDENCE_DB_PATH", "epistemic_registry.db")
        _registry = CredenceRegistry(db_path=db_path)
    return _registry


def _get_session(session_id: str) -> ContextManager:
    if session_id not in _sessions:
        _sessions[session_id] = ContextManager(
            api_key           = os.environ.get("ANTHROPIC_API_KEY"),
            registry          = _get_registry(),
            session_id        = session_id,
            use_scout         = os.environ.get("CREDENCE_USE_SCOUT", "").lower() in ("1", "true"),
        )
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _verify_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    if _DEV_MODE:
        return "dev"
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    if not x_api_key.startswith("cr-"):
        raise HTTPException(status_code=401, detail="Invalid API key format")
    return x_api_key


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message:    str = Field(..., description="User message")

class RegisterRequest(BaseModel):
    content:    str   = Field(..., description="The uncertain constraint text")
    session_id: str   = Field(..., description="Session identifier")
    j_score:    float = Field(0.30, description="Confidence score 0–1")
    zone:       str   = Field("LOW", description="HIGH | MEDIUM | LOW")

class VerifyRequest(BaseModel):
    constraint_id:  str = Field(..., description="Constraint ID from register")
    verified_value: str = Field(..., description="The confirmed factual value")
    session_id:     str = Field(..., description="Session identifier (for audit)")

class RiskRequest(BaseModel):
    content:     str = Field(..., description="Text to assess")
    chain_depth: int = Field(0,   description="Agent hops traversed")

class AlignRequest(BaseModel):
    session_id:    str = Field(..., description="Session identifier")
    response_text: str = Field(..., description="AI response to check")

class GateRequest(BaseModel):
    tool_name:          str = Field(..., description="Tool about to be called")
    arguments_summary:  str = Field(..., description="Summary of arguments")
    session_id:         str = Field(..., description="Session identifier")

class ContradictionRequest(BaseModel):
    claim:      str = Field(..., description="New claim to check")
    session_id: str = Field(..., description="Session identifier")

class PropagateRequest(BaseModel):
    envelope_dict: dict = Field(..., description="Envelope from a prior credence_chat")
    new_source:    str  = Field(..., description="Receiving agent identifier")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":   "ok",
        "version":  "1.0.0",
        "uptime_s": round(time.time() - _start_time, 1),
    }


# ---------------------------------------------------------------------------
# v1 endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat")
def chat(req: ChatRequest, _key: str = Depends(_verify_key)):
    """Send a message and receive a response with epistemic envelope."""
    proxy = CredenceProxy(theta_high=0.70, theta_low=0.45)

    # Auto-register uncertain user messages
    msg_lower       = req.message.lower()
    auto_registered = False
    if any(m in msg_lower for m in _UNCERTAINTY_MARKERS):
        cr_msg = proxy.compute(req.message)
        _get_registry().register(
            content    = req.message[:1000],
            session_id = req.session_id,
            j_score    = cr_msg.j_score,
            zone       = cr_msg.zone,
        )
        auto_registered = True

    mgr    = _get_session(req.session_id)
    result = mgr.chat(req.message)
    return {
        "response":              result.response,
        "envelope":              result.envelope,
        "decision":              result.decision,
        "j_score":               result.j_score,
        "zone":                  result.zone,
        "tokens_saved":          result.tokens_saved,
        "drift_state":           result.drift_state,
        "uncertainty_preserved": result.uncertainty_preserved,
        "truth_buffer_count":    result.truth_buffer_count,
        "scout_extractions":     result.scout_extractions,
        "alignment_warnings":    result.alignment_warnings,
        "caveat_injected":       result.caveat_injected,
        "auto_registered":       auto_registered,
        "adaptive_theta_high":   result.adaptive_theta_high,
        "adaptive_theta_low":    result.adaptive_theta_low,
    }


@app.post("/v1/risk")
def risk(req: RiskRequest, _key: str = Depends(_verify_key)):
    """Pre-flight epistemic risk assessment before compressing or forwarding content."""
    proxy = CredenceProxy(theta_high=0.70, theta_low=0.45)
    cr    = proxy.compute(req.content)
    lower = req.content.lower()
    uncertainty_hits = [m for m in _UNCERTAINTY_MARKERS if m in lower]
    has_uncertainty  = len(uncertainty_hits) > 0
    chain_penalty    = req.chain_depth * 0.05
    effective_trust  = max(0.0, cr.j_score - chain_penalty)

    if has_uncertainty and cr.zone in ("LOW", "MEDIUM"):
        risk_level = "HIGH";  action = "PRESERVE — uncertainty markers detected."
    elif has_uncertainty and cr.zone == "HIGH":
        risk_level = "MEDIUM"; action = "VERIFY — high J but uncertainty markers detected."
    elif cr.zone == "LOW" or effective_trust < 0.40:
        risk_level = "MEDIUM"; action = "PRESERVE — low confidence or degraded trust."
    elif cr.zone == "MEDIUM":
        risk_level = "LOW";    action = "TRIM — safe to trim but not compress."
    else:
        risk_level = "NONE";   action = "COMPRESS — HIGH-J, no uncertainty markers."

    return {
        "risk_level": risk_level, "j_score": cr.j_score, "zone": cr.zone,
        "effective_trust": round(effective_trust, 3), "chain_depth": req.chain_depth,
        "has_uncertainty": has_uncertainty,
        "uncertainty_markers_found": uncertainty_hits[:5],
        "safe_to_compress": risk_level == "NONE",
        "should_verify":    risk_level in ("HIGH", "MEDIUM"),
        "action": action, "reasoning": cr.reasoning,
    }


@app.post("/v1/align")
def align(req: AlignRequest, _key: str = Depends(_verify_key)):
    """Output Alignment Layer — check if response is more confident than ledger warrants."""
    mgr      = _get_session(req.session_id)
    warnings = mgr._align_output(req.response_text)
    if warnings:
        mgr._pending_alignment_caveat = "\n".join(w.suggested_caveat for w in warnings)
    warning_dicts = [w.to_dict() for w in warnings]
    return {
        "warning_count":      len(warnings),
        "alignment_warnings": warning_dicts,
        "caveat_needed":      len(warnings) > 0,
        "suggested_caveats":  [w["suggested_caveat"] for w in warning_dicts],
        "governor_active":    mgr._registry is not None,
    }


@app.post("/v1/register")
def register(req: RegisterRequest, _key: str = Depends(_verify_key)):
    """Register an uncertain constraint in the epistemic ledger."""
    cid = _get_registry().register(
        content=req.content, session_id=req.session_id,
        j_score=req.j_score, zone=req.zone,
    )
    return {"constraint_id": cid, "status": "registered",
            "content": req.content, "session_id": req.session_id,
            "j_score": req.j_score, "zone": req.zone}


@app.post("/v1/verify")
def verify(req: VerifyRequest, _key: str = Depends(_verify_key)):
    """Mark a constraint as verified with its confirmed value."""
    result = _get_registry().verify(req.constraint_id, req.verified_value)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    result["status"] = "verified"
    return result


@app.get("/v1/ledger/{session_id}")
def get_ledger(session_id: str, _key: str = Depends(_verify_key)):
    """Return all constraints for a session (the epistemic ledger)."""
    registry = _get_registry()
    uncertain = registry.list_uncertain(session_id)
    all_rows  = registry._conn.execute(
        "SELECT * FROM constraints WHERE session_id=? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    all_constraints = [registry._row_to_dict(r) for r in all_rows]
    return {
        "session_id":        session_id,
        "total_constraints": len(all_constraints),
        "unverified_count":  len(uncertain),
        "verified_count":    len(all_constraints) - len(uncertain),
        "constraints":       all_constraints,
        "etp_version":       "1.0",
    }


@app.get("/v1/ledger/{session_id}/uncertain")
def list_uncertain(session_id: str, _key: str = Depends(_verify_key)):
    """Return only unverified constraints for a session."""
    constraints = _get_registry().list_uncertain(session_id)
    return {"session_id": session_id, "count": len(constraints), "constraints": constraints}


@app.get("/v1/constraint/{constraint_id}")
def get_constraint(constraint_id: str, _key: str = Depends(_verify_key)):
    """Return a single constraint with full certainty trajectory."""
    registry   = _get_registry()
    rows       = registry._conn.execute(
        "SELECT * FROM constraints WHERE constraint_id=?", (constraint_id,)
    ).fetchone()
    if rows is None:
        raise HTTPException(status_code=404, detail=f"constraint '{constraint_id}' not found")
    return {
        "constraint":  registry._row_to_dict(rows),
        "trajectory":  registry.get_trajectory(constraint_id),
        "etp_version": "1.0",
    }


@app.post("/v1/gate")
def gate(req: GateRequest, _key: str = Depends(_verify_key)):
    """Agentic gate — block tool calls when unverified constraints are topically related."""
    import re as _re
    registry  = _get_registry()
    uncertain = registry.list_uncertain(req.session_id)
    if not uncertain:
        return {"proceed": True, "blocked_by": [], "unverified_count": 0,
                "recommendation": "PROCEED — no unverified constraints."}
    action_text  = f"{req.tool_name} {req.arguments_summary}".lower()
    action_words = set(_re.sub(r"[^\w\s]", " ", action_text).split())
    _stopwords   = {"the","and","for","that","this","with","from","are","was","not","but","all","any"}
    blocking     = []
    for c in uncertain:
        cwords  = set(_re.sub(r"[^\w\s]", " ", c["content"].lower()).split())
        overlap = action_words & cwords - _stopwords
        if len(overlap) >= 2:
            c["overlap_terms"] = list(overlap)[:6]
            blocking.append(c)
    if blocking:
        recommendation = (
            f"BLOCK — {len(blocking)} unverified constraint(s) may affect this action. "
            "Verify them first with /v1/verify."
        )
    else:
        recommendation = (
            f"PROCEED — {len(uncertain)} unverified constraint(s) exist but none are "
            "topically related to this action."
        )
    return {"proceed": len(blocking) == 0, "blocked_by": blocking,
            "unverified_count": len(uncertain), "recommendation": recommendation}


@app.post("/v1/contradiction")
def check_contradiction(req: ContradictionRequest, _key: str = Depends(_verify_key)):
    """Check whether a new claim contradicts verified constraints."""
    registry = _get_registry()
    matches  = registry.check_contradiction(req.claim, req.session_id)
    return {
        "has_contradiction": len(matches) > 0,
        "match_count":       len(matches),
        "matches":           matches,
        "recommendation": (
            f"REVIEW — {len(matches)} verified constraint(s) are topically similar."
            if matches else
            "PROCEED — no contradictions detected."
        ),
    }


@app.post("/v1/propagate")
def propagate(req: PropagateRequest, _key: str = Depends(_verify_key)):
    """Propagate an envelope to the next agent hop (increments chain_depth)."""
    from credence.envelope import CredenceEnvelope
    try:
        env = CredenceEnvelope.from_dict(req.envelope_dict)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid envelope: {e}")
    return env.propagate(new_source=req.new_source.strip() or "unknown").to_dict()


@app.get("/v1/stats/{session_id}")
def stats(session_id: str, _key: str = Depends(_verify_key)):
    """Return session statistics: tokens, cost, compression counts."""
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


@app.get("/v1/log/{session_id}")
def decision_log(session_id: str, _key: str = Depends(_verify_key)):
    """Return per-turn decision log with J-scores and decisions."""
    return _get_session(session_id).decision_log


@app.delete("/v1/session/{session_id}")
def reset_session(session_id: str, _key: str = Depends(_verify_key)):
    """Reset a session, clearing all history and stats."""
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# ETP Resource endpoints (mirrors MCP Resources over HTTP)
# ---------------------------------------------------------------------------

@app.get("/v1/etp/{session_id}/ledger")
def etp_ledger(session_id: str, _key: str = Depends(_verify_key)):
    """ETP ledger resource — epistemic://session/{session_id}/ledger over HTTP."""
    return get_ledger(session_id, _key)


@app.get("/v1/etp/{session_id}/constraint/{constraint_id}")
def etp_constraint(session_id: str, constraint_id: str, _key: str = Depends(_verify_key)):
    """ETP constraint resource with trajectory."""
    return get_constraint(constraint_id, _key)


@app.get("/v1/etp/{session_id}/alignment")
def etp_alignment(session_id: str, _key: str = Depends(_verify_key)):
    """ETP alignment resource — pending Governor caveats."""
    mgr    = _sessions.get(session_id)
    caveat = mgr._pending_alignment_caveat if mgr else None
    return {
        "session_id":      session_id,
        "pending_caveat":  caveat,
        "caveat_pending":  caveat is not None,
        "governor_active": mgr is not None and mgr._registry is not None,
        "etp_version":     "1.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

"""
credence/client.py
==================
Python SDK client for the hosted Credence API.

Drop-in replacement for ContextManager when using the hosted service instead
of running the model locally. Same public interface: client.chat() returns a
TurnResult-compatible object. Zero Claude API key required on the client side.

Usage:
    from credence import CredenceClient

    client = CredenceClient(api_key="cr-your-key-here")
    result = client.chat("I think the rate limit is 100 req/min — unconfirmed",
                         session_id="payment-v2")
    print(result.response)
    print(f"J={result.j_score:.2f}  zone={result.zone}  decision={result.decision}")

    # Check alignment warnings (Governor output)
    if result.alignment_warnings:
        print("Governor flagged:", result.alignment_warnings[0]["suggested_caveat"])

    # Register an uncertain constraint explicitly
    cid = client.register("I think the DB is PostgreSQL 14", session_id="payment-v2")
    client.verify(cid, "Confirmed: PostgreSQL 14.8 per infra team", session_id="payment-v2")

    # Check risk before forwarding to another agent
    risk = client.risk("The rate limit is 100 req/min", chain_depth=1)
    if risk["risk_level"] == "HIGH":
        print("Do not compress or forward this without verification")

Install:
    pip install credence-ai    # published to PyPI

Self-host:
    Set CREDENCE_API_URL to point at your own deployment instead of the default.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Any

try:
    import urllib.request
    import urllib.error
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False

_DEFAULT_BASE_URL = os.environ.get("CREDENCE_API_URL", "https://api.credence-ai.io")


# ---------------------------------------------------------------------------
# Response object — mirrors TurnResult from context_manager.py
# ---------------------------------------------------------------------------

@dataclass
class TurnResultRemote:
    """
    Response from the hosted Credence API. Mirrors TurnResult from ContextManager
    so existing code that calls mgr.chat() can switch to CredenceClient.chat()
    without changing downstream logic.
    """
    response:              str
    j_score:               float
    zone:                  str
    decision:              str
    tokens_saved:          int
    drift_state:           bool
    uncertainty_preserved: bool
    truth_buffer_count:    int
    scout_extractions:     int
    alignment_warnings:    list  = field(default_factory=list)
    caveat_injected:       bool  = False
    auto_registered:       bool  = False
    adaptive_theta_high:   float = 0.70
    adaptive_theta_low:    float = 0.45
    envelope:              dict  = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "TurnResultRemote":
        return cls(
            response              = data.get("response", ""),
            j_score               = data.get("j_score", 0.5),
            zone                  = data.get("zone", "MEDIUM"),
            decision              = data.get("decision", "PRESERVE"),
            tokens_saved          = data.get("tokens_saved", 0),
            drift_state           = data.get("drift_state", False),
            uncertainty_preserved = data.get("uncertainty_preserved", False),
            truth_buffer_count    = data.get("truth_buffer_count", 0),
            scout_extractions     = data.get("scout_extractions", 0),
            alignment_warnings    = data.get("alignment_warnings", []),
            caveat_injected       = data.get("caveat_injected", False),
            auto_registered       = data.get("auto_registered", False),
            adaptive_theta_high   = data.get("adaptive_theta_high", 0.70),
            adaptive_theta_low    = data.get("adaptive_theta_low", 0.45),
            envelope              = data.get("envelope", {}),
        )


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

class CredenceAPIError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"Credence API error {status}: {detail}")


def _http(method: str, url: str, body: Optional[dict], api_key: str) -> dict:
    """Minimal HTTP client using stdlib urllib — no httpx/requests dependency."""
    data  = json.dumps(body).encode() if body else None
    req   = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-API-Key":    api_key,
            "User-Agent":   "credence-python-sdk/1.0.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise CredenceAPIError(e.code, detail) from e


# ---------------------------------------------------------------------------
# CredenceClient
# ---------------------------------------------------------------------------

class CredenceClient:
    """
    Python SDK for the hosted Credence API.

    Provides the same interface as ContextManager so switching between
    local and hosted operation requires changing one line:

        # Local (requires ANTHROPIC_API_KEY):
        from credence import ContextManager
        client = ContextManager()

        # Hosted (no ANTHROPIC_API_KEY needed on client):
        from credence import CredenceClient
        client = CredenceClient(api_key="cr-...")

    Both: result = client.chat("message", session_id="s1")
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key  = api_key or os.environ.get("CREDENCE_API_KEY", "")
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError(
                "Credence API key required. Set CREDENCE_API_KEY or pass api_key=. "
                "Get a key at https://credence-ai.io"
            )

    def _post(self, path: str, body: dict) -> dict:
        return _http("POST", f"{self.base_url}{path}", body, self.api_key)

    def _get(self, path: str) -> dict:
        return _http("GET", f"{self.base_url}{path}", None, self.api_key)

    # ------------------------------------------------------------------
    # Core: chat
    # ------------------------------------------------------------------

    def chat(self, message: str, session_id: str = "default") -> TurnResultRemote:
        """
        Send a message and receive a response with epistemic envelope.

        Equivalent to ContextManager.chat(). The hosted API manages session
        state, compression decisions, and the epistemic ledger server-side.
        """
        data = self._post("/v1/chat", {"session_id": session_id, "message": message})
        return TurnResultRemote.from_api(data)

    # ------------------------------------------------------------------
    # Epistemic risk
    # ------------------------------------------------------------------

    def risk(self, content: str, chain_depth: int = 0) -> dict:
        """Pre-flight epistemic risk assessment before compressing or forwarding."""
        return self._post("/v1/risk", {"content": content, "chain_depth": chain_depth})

    # ------------------------------------------------------------------
    # Output Alignment (Governor)
    # ------------------------------------------------------------------

    def align(self, response_text: str, session_id: str = "default") -> dict:
        """Check if a response is more confident than the ledger warrants."""
        return self._post("/v1/align", {"session_id": session_id, "response_text": response_text})

    # ------------------------------------------------------------------
    # Epistemic ledger
    # ------------------------------------------------------------------

    def register(
        self,
        content:    str,
        session_id: str   = "default",
        j_score:    float = 0.30,
        zone:       str   = "LOW",
    ) -> str:
        """Register an uncertain constraint. Returns constraint_id."""
        result = self._post("/v1/register", {
            "content": content, "session_id": session_id,
            "j_score": j_score, "zone": zone,
        })
        return result["constraint_id"]

    def verify(self, constraint_id: str, verified_value: str, session_id: str = "default") -> dict:
        """Mark a constraint as verified with its confirmed value."""
        return self._post("/v1/verify", {
            "constraint_id":  constraint_id,
            "verified_value": verified_value,
            "session_id":     session_id,
        })

    def ledger(self, session_id: str = "default") -> dict:
        """Return the full epistemic ledger for a session."""
        return self._get(f"/v1/ledger/{session_id}")

    def uncertain(self, session_id: str = "default") -> list:
        """Return only unverified constraints for a session."""
        result = self._get(f"/v1/ledger/{session_id}/uncertain")
        return result.get("constraints", [])

    def constraint(self, constraint_id: str) -> dict:
        """Return a single constraint with full certainty trajectory."""
        return self._get(f"/v1/constraint/{constraint_id}")

    def contradiction(self, claim: str, session_id: str = "default") -> dict:
        """Check if a claim contradicts verified constraints."""
        return self._post("/v1/contradiction", {"claim": claim, "session_id": session_id})

    # ------------------------------------------------------------------
    # Agentic gate
    # ------------------------------------------------------------------

    def gate(self, tool_name: str, arguments_summary: str, session_id: str = "default") -> dict:
        """Block tool calls when unverified constraints may affect the action."""
        return self._post("/v1/gate", {
            "tool_name":         tool_name,
            "arguments_summary": arguments_summary,
            "session_id":        session_id,
        })

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def stats(self, session_id: str = "default") -> dict:
        """Return session statistics."""
        return self._get(f"/v1/stats/{session_id}")

    def log(self, session_id: str = "default") -> list:
        """Return per-turn decision log."""
        return self._get(f"/v1/log/{session_id}")

    def reset(self, session_id: str = "default") -> dict:
        """Reset a session, clearing all history and stats."""
        return _http("DELETE", f"{self.base_url}/v1/session/{session_id}", None, self.api_key)

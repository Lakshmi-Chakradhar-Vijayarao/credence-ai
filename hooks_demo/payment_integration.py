"""
Payment Integration — Demo Project for Epistemic Memory

This file is intentionally incomplete.
The demo shows what happens when Claude Code implements the token refresh
logic WITHOUT knowing the auth token expiry is uncertain.

The uncertain constraint (established in the conversation, not here):
  "I think the auth token expires in 3600 seconds, but it might be 86400 —
   I haven't confirmed with the vendor yet."

With epistemic memory: Claude Code catches this before hardcoding the value.
Without epistemic memory: Claude Code writes expires_in = 3600 confidently.
"""

import os
import time
import requests
from typing import Optional


class PaymentAPIClient:
    """
    Client for the payment API integration.

    Uncertain constraints from the planning session:
    - Auth token expiry: 3600s or 86400s? (UNVERIFIED — needs confirmation)
    - Rate limit: 100 req/min or 50 req/min for sandbox? (UNVERIFIED)
    """

    BASE_URL = "https://api.payments.example.com/v2"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._token:       Optional[str] = None
        self._token_expiry: Optional[float] = None

    def _get_token(self) -> str:
        """Obtain a Bearer token via client credentials grant."""
        resp = requests.post(
            f"{self.BASE_URL}/oauth/token",
            json={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]

        # ── EPISTEMIC MEMORY FLAG POINT ────────────────────────────────────
        # The expiry value below should trigger credence_risk BEFORE
        # being hardcoded.  The uncertain constraint from the planning session:
        # "token expires in 3600s or 86400s — unconfirmed."
        # Without epistemic memory: Claude writes expires_in = 3600 here.
        # With epistemic memory: Claude surfaces a ⚠ warning instead.
        # ──────────────────────────────────────────────────────────────────
        expires_in = data.get("expires_in", None)  # intentionally left as None
        if expires_in:
            self._token_expiry = time.time() + expires_in
        return self._token

    def _is_token_valid(self) -> bool:
        if self._token is None or self._token_expiry is None:
            return False
        return time.time() < self._token_expiry - 60  # 60s buffer

    def _ensure_token(self) -> str:
        if not self._is_token_valid():
            self._get_token()
        return self._token

    def charge(self, amount_cents: int, currency: str,
               idempotency_key: str) -> dict:
        """Create a charge. Amount in smallest currency unit (cents for USD)."""
        headers = {
            "Authorization":    f"Bearer {self._ensure_token()}",
            "X-Idempotency-Key": idempotency_key,
            "Content-Type":     "application/json",
        }
        resp = requests.post(
            f"{self.BASE_URL}/charges",
            headers=headers,
            json={
                "amount":   amount_cents,
                "currency": currency,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

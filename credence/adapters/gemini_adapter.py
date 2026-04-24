"""
credence/adapters/gemini_adapter.py
=====================================
Gemini adapter for Credence — Gemini 1.5 Pro, Flash.

Brings full Credence epistemic memory to Google Gemini models:
  - J-score routing (compress/trim/preserve)
  - Faithfulness probe before compression
  - Output Alignment Layer (Governor)
  - Truth Buffer (epistemic ledger injection)

Usage:
    from credence.adapters import GeminiAdapter

    mgr = GeminiAdapter(api_key="AI...", model="gemini-1.5-pro")
    result = mgr.chat("I think the rate limit is 100 req/min — unconfirmed")
    print(result.response, result.j_score, result.decision)

Requires: pip install google-generativeai
"""

import os
from typing import Optional

from .base import BaseAdapter, AdapterTurnResult


class GeminiAdapter(BaseAdapter):
    """
    Credence epistemic memory for Google Gemini models.
    Uses gemini-1.5-pro for generation and gemini-1.5-flash for compression (cheaper).
    """

    def __init__(
        self,
        api_key:           Optional[str] = None,
        model:             str  = "gemini-1.5-pro",
        compression_model: str  = "gemini-1.5-flash",
        theta_high:        float = 0.70,
        theta_low:         float = 0.45,
        system_prompt:     Optional[str] = None,
        max_tokens:        int   = 1024,
        registry=None,
        session_id:        Optional[str] = None,
    ):
        super().__init__(
            theta_high    = theta_high,
            theta_low     = theta_low,
            system_prompt = system_prompt,
            max_tokens    = max_tokens,
            registry      = registry,
            session_id    = session_id,
        )
        self._model             = model
        self._compression_model = compression_model
        self._api_key           = api_key or os.environ.get("GOOGLE_API_KEY", "")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._genai         = genai
            self._gen_client    = genai.GenerativeModel(self._model)
            self._compress_client = genai.GenerativeModel(self._compression_model)
        except ImportError:
            raise ImportError("Gemini adapter requires: pip install google-generativeai")

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _generate(self, messages: list, system: str) -> tuple[str, int, int]:
        # Gemini uses a different conversation format — rebuild with system injected
        gemini_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        # Inject system prompt into first user message if history not empty
        if gemini_history:
            first = gemini_history[0]
            if first["role"] == "user":
                first["parts"] = [f"{system}\n\n{first['parts'][0]}"]

        # Split last user message from history for send_message
        if gemini_history and gemini_history[-1]["role"] == "user":
            last_user_content = gemini_history[-1]["parts"][0]
            prior_history     = gemini_history[:-1]
        else:
            last_user_content = ""
            prior_history     = gemini_history

        chat = self._gen_client.start_chat(history=prior_history)
        resp = chat.send_message(
            last_user_content,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
            ),
        )

        text       = resp.text or ""
        # Gemini usage metadata
        tokens_in  = getattr(getattr(resp, "usage_metadata", None), "prompt_token_count",     0) or 0
        tokens_out = getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", 0) or 0
        return text, tokens_in, tokens_out

    def _compress_with_model(self, text: str) -> str:
        prompt = (
            "Summarize this conversation in 2-3 concise sentences. "
            "Preserve all key facts. "
            "CRITICAL: preserve all uncertainty qualifiers verbatim "
            "(e.g. 'I think', 'might be', 'not confirmed', 'approximately').\n\n"
            + text
        )
        resp = self._compress_client.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(max_output_tokens=200),
        )
        return resp.text or ""

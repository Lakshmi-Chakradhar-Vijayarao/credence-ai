"""
credence/adapters/openai_adapter.py
=====================================
OpenAI adapter for Credence — GPT-4o, o1, o3, GPT-3.5-turbo.

Brings full Credence epistemic memory to OpenAI models:
  - J-score routing (compress/trim/preserve)
  - Faithfulness probe before compression
  - Output Alignment Layer (Governor)
  - Truth Buffer (epistemic ledger injection)

Usage:
    from credence.adapters import OpenAIAdapter

    mgr = OpenAIAdapter(api_key="sk-...", model="gpt-4o")
    result = mgr.chat("I think the rate limit is 100 req/min — unconfirmed")
    print(result.response, result.j_score, result.decision)

Requires: pip install openai
"""

import os
from typing import Optional

from .base import BaseAdapter, AdapterTurnResult


class OpenAIAdapter(BaseAdapter):
    """
    Credence epistemic memory for OpenAI models.
    Uses GPT-4o for generation and GPT-3.5-turbo for compression (cheaper).
    """

    def __init__(
        self,
        api_key:           Optional[str] = None,
        model:             str  = "gpt-4o",
        compression_model: str  = "gpt-3.5-turbo",
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
        self._api_key           = api_key or os.environ.get("OPENAI_API_KEY", "")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("OpenAI adapter requires: pip install openai")

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def _generate(self, messages: list, system: str) -> tuple[str, int, int]:
        openai_messages = [{"role": "system", "content": system}] + messages
        resp = self._client.chat.completions.create(
            model      = self._model,
            messages   = openai_messages,
            max_tokens = self.max_tokens,
        )
        text       = resp.choices[0].message.content or ""
        tokens_in  = resp.usage.prompt_tokens     if resp.usage else 0
        tokens_out = resp.usage.completion_tokens if resp.usage else 0
        return text, tokens_in, tokens_out

    def _compress_with_model(self, text: str) -> str:
        resp = self._client.chat.completions.create(
            model      = self._compression_model,
            messages   = [{
                "role":    "user",
                "content": (
                    "Summarize this conversation in 2-3 concise sentences. "
                    "Preserve all key facts. "
                    "CRITICAL: preserve all uncertainty qualifiers verbatim "
                    "(e.g. 'I think', 'might be', 'not confirmed', 'approximately').\n\n"
                    + text
                ),
            }],
            max_tokens = 200,
        )
        return resp.choices[0].message.content or ""

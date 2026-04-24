"""
credence/adapters/ollama_adapter.py
=====================================
Ollama adapter for Credence — any local model (llama3.2, mistral, qwen, etc.)

Brings full Credence epistemic memory to locally-run Ollama models:
  - J-score routing (compress/trim/preserve)
  - Faithfulness probe before compression
  - Output Alignment Layer (Governor)
  - Truth Buffer (epistemic ledger injection)

Usage:
    from credence.adapters import OllamaAdapter

    mgr = OllamaAdapter(model="llama3.2")
    result = mgr.chat("I think the rate limit is 100 req/min — unconfirmed")
    print(result.response, result.j_score, result.decision)

Requires: Ollama running locally (https://ollama.com) + pip install ollama
"""

import os
from typing import Optional

from .base import BaseAdapter, AdapterTurnResult


class OllamaAdapter(BaseAdapter):
    """
    Credence epistemic memory for locally-run Ollama models.
    Uses the same model for both generation and compression (local = zero API cost).
    A smaller compression_model can be specified for faster summarisation.
    """

    def __init__(
        self,
        model:             str  = "llama3.2",
        compression_model: Optional[str] = None,
        host:              Optional[str] = None,
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
        self._compression_model = compression_model or model
        self._host              = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        try:
            import ollama
            self._ollama = ollama
            # Use custom host if specified
            if host or os.environ.get("OLLAMA_HOST"):
                self._client = ollama.Client(host=self._host)
            else:
                self._client = ollama.Client()
        except ImportError:
            raise ImportError("Ollama adapter requires: pip install ollama")

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _generate(self, messages: list, system: str) -> tuple[str, int, int]:
        resp = self._client.chat(
            model   = self._model,
            messages= [{"role": "system", "content": system}] + messages,
            options = {"num_predict": self.max_tokens},
        )
        text       = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        # Ollama usage stats
        if isinstance(resp, dict):
            tokens_in  = resp.get("prompt_eval_count",  0) or 0
            tokens_out = resp.get("eval_count",          0) or 0
        else:
            tokens_in  = getattr(resp, "prompt_eval_count",  0) or 0
            tokens_out = getattr(resp, "eval_count",          0) or 0
        return text or "", tokens_in, tokens_out

    def _compress_with_model(self, text: str) -> str:
        resp = self._client.chat(
            model   = self._compression_model,
            messages= [{
                "role":    "user",
                "content": (
                    "Summarize this conversation in 2-3 concise sentences. "
                    "Preserve all key facts. "
                    "CRITICAL: preserve all uncertainty qualifiers verbatim "
                    "(e.g. 'I think', 'might be', 'not confirmed', 'approximately').\n\n"
                    + text
                ),
            }],
            options = {"num_predict": 200},
        )
        if isinstance(resp, dict):
            return resp.get("message", {}).get("content", "") or ""
        return resp.message.content or ""

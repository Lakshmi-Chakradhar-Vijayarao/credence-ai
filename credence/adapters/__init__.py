"""
credence/adapters/
==================
Model-agnostic adapters for the Credence epistemic memory system.

Each adapter wraps a different LLM provider's API in a common interface
so J-score routing, faithfulness probes, and the epistemic ledger work
with any model — not just Claude.

Available adapters:
    from credence.adapters import OpenAIAdapter, GeminiAdapter, OllamaAdapter

    # OpenAI (GPT-4o, o1, o3)
    mgr = OpenAIAdapter(api_key="sk-...", session_id="s1")

    # Gemini
    mgr = GeminiAdapter(api_key="AI...", session_id="s1")

    # Local models via Ollama
    mgr = OllamaAdapter(model="llama3.2", session_id="s1")

    # All three: same interface
    result = mgr.chat("I think the rate limit is 100 req/min")
    print(result.response, result.j_score, result.decision)
"""

from .openai_adapter import OpenAIAdapter
from .gemini_adapter  import GeminiAdapter
from .ollama_adapter  import OllamaAdapter
from .base            import BaseAdapter, AdapterTurnResult

__all__ = ["BaseAdapter", "AdapterTurnResult", "OpenAIAdapter", "GeminiAdapter", "OllamaAdapter"]

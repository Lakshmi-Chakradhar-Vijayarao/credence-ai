"""
Credence — Epistemic Enforcement Layer.

Prevents uncertainty qualifiers from being silently dropped during LLM
context compression. Works with any model, any coding agent, zero API key.

Four deterministic checkpoints:
  CP1  Pre-compression  → faithfulness probe blocks compression when qualifiers present
  CP2  Pre-generation   → Truth Buffer + Consistency Enforcer enforce uncertainty in output
  CP3  Post-generation  → Generation-Time Scanner annotates unverified literals in code
  CP4  Pre-tool-use     → credence-gate (Rust) / hooks.py (Python) blocks irreversible writes

Primary interface — MCP server (10 tools, zero API key):
    credence-server              — MCP server entry point

Python API (no API key required):
    wrap(compress_fn, context)   — wrap any compress function with CP1 guard
    CredenceRegistry             — SQLite constraint store
    CredenceMemory               — cross-session epistemic memory

Advanced (requires Anthropic API key):
    ContextManager               — full enforcement engine with compression loop
"""

from .registry import CredenceRegistry
from .memory import CredenceMemory
from .wrap import wrap, WrapResult, measure_fcr
from .enforce import enforce, CredenceViolation
from .context_manager import ContextManager, TurnResult, SessionStats
from .confidence_proxy import CredenceProxy, CredenceResult
from .epistemic_manifest import EpistemicManifest

__version__ = "1.0.0"
__all__ = [
    # Python API — zero API key
    "wrap", "WrapResult", "measure_fcr",
    "CredenceRegistry",
    "CredenceMemory",
    "enforce", "CredenceViolation",
    # Advanced — requires API key
    "ContextManager", "TurnResult", "SessionStats",
    "CredenceProxy", "CredenceResult",
    "EpistemicManifest",
]

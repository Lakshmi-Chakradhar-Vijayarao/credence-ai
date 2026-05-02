"""
Credence — Epistemic Enforcement Layer.

Prevents uncertainty qualifiers from being silently dropped during LLM
context compression. Works with any model, any coding agent, zero API key.

Four deterministic checkpoints:
  CP1  Pre-compression  → faithfulness probe blocks compression when qualifiers present
  CP2  Pre-generation   → Truth Buffer + Consistency Enforcer enforce uncertainty in output
  CP3  Post-generation  → Generation-Time Scanner annotates unverified literals in code
  CP4  Pre-tool-use     → credence-gate (Rust) / hooks.py (Python) blocks irreversible writes

Model-agnostic API (no API key required):
    wrap(compress_fn, context)   — wrap any compress function with CP1 guard
    credence-server              — MCP server with 12 tools, zero LLM calls

Cross-session: CredenceMemory persists j_score + zone + verified=False so new
sessions inherit which facts were unverified, not just what the values were.

Multi-agent: PipelineMonitor intercepts Agent A → Agent B handoffs, extracts
uncertain claims from Agent A's output, registers them, and injects an
epistemic handoff block into Agent B's system prompt.
"""

from .context_manager import ContextManager, TurnResult, SessionStats
from .registry import CredenceRegistry
from .memory import CredenceMemory
from .pipeline_monitor import PipelineMonitor, EpistemicHandoff
from .enforce import enforce, CredenceViolation
from .wrap import wrap, WrapResult, measure_fcr
from .confidence_proxy import CredenceProxy, CredenceResult
from .epistemic_manifest import EpistemicManifest

__version__ = "1.0.0"
__all__ = [
    # Model-agnostic wrapper — primary open-source API
    "wrap", "WrapResult", "measure_fcr",
    # Constraint registry (SQLite, zero deps)
    "CredenceRegistry",
    # Cross-session epistemic memory
    "CredenceMemory",
    # Multi-agent middleware
    "PipelineMonitor", "EpistemicHandoff",
    # Decorator-based enforcement
    "enforce", "CredenceViolation",
    # Full enforcement engine (requires API key — optional power-user feature)
    "ContextManager", "TurnResult", "SessionStats",
    # J-score proxy and confidence result
    "CredenceProxy", "CredenceResult",
    # Structured epistemic manifest (XML injection)
    "EpistemicManifest",
]

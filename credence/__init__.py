"""
Credence — Epistemic Enforcement Layer for Claude.

Four checkpoints ensure uncertain constraints keep their epistemic status
through every downstream operation:

  CP1  Compression      → faithfulness probe blocks Haiku before it strips qualifiers
  CP2  Generation       → Truth Buffer + Consistency Enforcer inject and enforce
  CP3  Code output      → Generation-Time Scanner annotates uncertain literals
  CP4  Tool execution   → credence-gate (Rust) / hooks.py (Python) blocks writes

Cross-session: CredenceMemory persists j_score + zone + verified=False so new
sessions inherit which facts were unverified, not just what the values were.

Multi-agent: PipelineMonitor intercepts Agent A → Agent B handoffs, extracts
uncertain claims from Agent A's output, registers them, and injects an
epistemic handoff block into Agent B's system prompt.

Public API:
    ContextManager   — main enforcement engine
    CredenceRegistry — constraint store (SQLite)
    CredenceProxy    — J-score computation (offline)
    PipelineMonitor  — multi-agent middleware
    CredenceMemory   — cross-session epistemic persistence
"""

from .confidence_proxy import CredenceProxy, CredenceResult
from .context_manager import ContextManager, TurnResult, SessionStats
from .registry import CredenceRegistry
from .memory import CredenceMemory
from .pipeline_monitor import PipelineMonitor, EpistemicHandoff
from .agent import CredenceAgent
from .envelope import CredenceEnvelope
from .semantic_entropy import SemanticEntropyProbe, SemanticEntropyResult

__version__ = "2.0.0"
__all__ = [
    # Core enforcement
    "ContextManager", "TurnResult", "SessionStats",
    "CredenceRegistry",
    "CredenceProxy", "CredenceResult",
    # Cross-session
    "CredenceMemory",
    # Multi-agent
    "PipelineMonitor", "EpistemicHandoff",
    # Extras
    "CredenceAgent",
    "CredenceEnvelope",
    "SemanticEntropyProbe", "SemanticEntropyResult",
]

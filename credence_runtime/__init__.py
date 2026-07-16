"""
credence_runtime — Cognitive Infrastructure for AI inference.

Every AI system already computes an internal epistemic state.
Credence exposes that state and uses it to adapt memory, retrieval,
reasoning, and compute — before generation occurs.

The only AI infrastructure cost that decreases as your models improve:
the smarter the model, the more it knows, the less retrieval it needs.

Install:
    pip install credence-runtime

Quick start:
    from credence_runtime import Credence

    cred = Credence.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        calibration="cal.json",
    )

    result = cred.complete("Who wrote Hamlet?")
    print(result.text)        # "William Shakespeare"
    print(result.routing)     # ANSWER
    print(result.j_know)      # 18.3

    if result.needs_retrieval:
        context = my_retriever(result.query)
        result  = cred.complete_with_context(result.query, context)

    cred.print_cost_summary()
    # Credence Session Summary
    #   RAG skip rate:        38.5%
    #   Est. cost reduction:  28.4%

Serve over HTTP:
    credence serve --model Qwen/Qwen2.5-3B-Instruct --calibration cal.json

Five-way epistemic routing:
    ANSWER   — model knows from parametric memory, committed
    VERIFY   — model thinks it knows but shows confabulation fingerprint
    RETRIEVE — needs context; caller fetches and retries
    DEFER    — low certainty; generate with reduced budget
    ESCALATE — context-dependent + low confidence; escalate to higher tier

Three structurally independent signals at gen-step-1:
    j_know     = 18.3   Fisher LDA projection (positive = PARAM)
    j_velocity = 2.1    Commitment slope (J_deep - J_shallow)
    entropy    = 0.28   Output certainty (low = confident)

    corr(j_know, entropy) = 0.0039   [n=800, structurally independent]
    Output entropy anti-predictive: AUROC 0.36 (Qwen3-32B, Groq study)
    Fisher AUROC: 0.866–0.994 (GQA-family models)

Company:  Credence AI
Product:  Credence Runtime
Category: Cognitive Infrastructure
"""

from esm.adaptive_runtime import (  # noqa: F401
    Credence,
    AdaptiveRuntime,       # backwards compat alias
    CompletionResult,
    CostReport,
    SessionCostSummary,
    epistemic_wrap,
)
from esm.runtime import (  # noqa: F401
    EpistemicRuntime,
    EpistemicTag,
    EpistemicTrace,
    CalibrationState,
    ANSWER,
    VERIFY,
    RETRIEVE,
    DEFER,
    ESCALATE,
)

__version__ = "0.2.0"
__author__  = "Credence AI"

__all__ = [
    # Primary API
    "Credence",
    "CompletionResult",
    "CostReport",
    "SessionCostSummary",
    "epistemic_wrap",
    # Low-level engine
    "EpistemicRuntime",
    "EpistemicTag",
    "EpistemicTrace",
    "CalibrationState",
    # Routing constants
    "ANSWER",
    "VERIFY",
    "RETRIEVE",
    "DEFER",
    "ESCALATE",
    # Backwards compat
    "AdaptiveRuntime",
    "__version__",
    "__author__",
]

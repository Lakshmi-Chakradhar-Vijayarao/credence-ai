"""
credence/enforce.py
===================
@credence.enforce — decorator-based integration for Python functions.

Checks the Credence registry for unverified constraints before a decorated
function executes. Blocks (raises CredenceViolation) or warns when the
function arguments overlap a registered uncertain constraint.

Usage:
    from credence import enforce, CredenceRegistry

    registry = CredenceRegistry()
    registry.register("Rate limit is 50 req/min — unconfirmed", "session-1", j_score=0.30)

    @enforce(registry=registry, session_id="session-1", policy="strict")
    def call_api(endpoint: str, rate_limit: int):
        ...

    # Raises CredenceViolation: "rate" + "limit" match registered constraint
    call_api("/v1/messages", rate_limit=50)

Policies:
    strict  — raise CredenceViolation when any unverified constraint overlaps
    warn    — print a warning and continue
    log     — record the violation silently, continue

The decorator inspects both positional and keyword argument VALUES (as strings)
and the function's argument NAMES for keyword overlap with registered constraints.
This catches both direct value embedding and parameter name similarity.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import Callable, Optional

from .registry import CredenceRegistry


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CredenceViolation(RuntimeError):
    """Raised by @enforce(policy='strict') when an unverified constraint is violated."""

    def __init__(self, message: str, constraint_id: str, constraint_text: str,
                 overlap_terms: list[str], effective_confidence: float):
        super().__init__(message)
        self.constraint_id        = constraint_id
        self.constraint_text      = constraint_text
        self.overlap_terms        = overlap_terms
        self.effective_confidence = effective_confidence


# ---------------------------------------------------------------------------
# Stopwords (same set as Consistency Enforcer)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it",
    "its", "this", "that", "these", "those", "and", "or", "but", "not",
    "so", "if", "we", "our", "my", "your", "they", "their", "you",
    "i", "he", "she", "us", "me", "him", "her", "what", "how", "why",
    "when", "where", "which", "who", "just", "also", "tell", "know",
    "use", "get", "set", "run", "call", "send", "make", "need", "want",
    "let", "now", "then", "here", "there", "up", "down", "any", "all",
    "each", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "same", "than", "too", "very",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(str(text)) if t.lower() not in _STOPWORDS and len(t) > 2}


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def _check_constraints(
    registry:   CredenceRegistry,
    session_id: str,
    call_tokens: set[str],
    current_turn: int = 0,
    min_overlap:  int = 2,
) -> list[dict]:
    """
    Return all unverified constraints whose tokens overlap call_tokens by ≥ min_overlap.
    """
    constraints = registry.list_uncertain(session_id)
    violations  = []

    for c in constraints:
        c_tokens = _tokenize(c.get("content", ""))
        overlap  = call_tokens & c_tokens
        if len(overlap) >= min_overlap:
            eff_conf = registry.get_effective_confidence(c["constraint_id"], current_turn)
            violations.append({
                "constraint_id":        c["constraint_id"],
                "constraint_text":      c.get("content", ""),
                "overlap_terms":        sorted(overlap),
                "effective_confidence": round(eff_conf, 3),
                "zone":                 c.get("zone", "MEDIUM"),
            })

    return sorted(violations, key=lambda v: v["effective_confidence"])


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def enforce(
    registry:    Optional[CredenceRegistry] = None,
    session_id:  str = "default",
    policy:      str = "strict",
    min_overlap: int = 2,
    current_turn: int = 0,
):
    """
    Decorator factory. Wraps a function to check the Credence registry before execution.

    Args:
        registry:     CredenceRegistry instance. If None, enforcement is a no-op.
        session_id:   Session key for registry lookup.
        policy:       "strict" (raise), "warn" (print warning), "log" (silent).
        min_overlap:  Minimum non-stopword token overlap to trigger. Default 2.
        current_turn: Turn index for confidence decay calculation.

    Returns:
        Decorator that wraps the target function.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if registry is None:
                return fn(*args, **kwargs)

            # Build token set from function name + arg names + arg values
            call_text_parts = [fn.__name__]
            call_text_parts.extend(kwargs.keys())
            for v in list(args) + list(kwargs.values()):
                call_text_parts.append(str(v))
            call_tokens = _tokenize(" ".join(call_text_parts))

            violations = _check_constraints(
                registry, session_id, call_tokens, current_turn, min_overlap
            )

            if violations:
                v = violations[0]  # highest-severity (lowest confidence) first
                msg = (
                    f"\n╔══════════════════════════════════════════════════════╗\n"
                    f"║  CREDENCE ENFORCE — {'BLOCKED' if policy == 'strict' else 'WARNING'}                        ║\n"
                    f"╚══════════════════════════════════════════════════════╝\n"
                    f"  Function:   {fn.__name__}\n"
                    f"  Constraint: {v['constraint_text'][:80]}\n"
                    f"  Confidence: {v['effective_confidence']:.3f} ({v['zone']})\n"
                    f"  Overlap:    {', '.join(v['overlap_terms'])}\n"
                    f"\n"
                    f"  Call credence_verify('{v['constraint_id']}', confirmed_value)\n"
                    f"  to resolve and unblock this function.\n"
                )

                if policy == "strict":
                    raise CredenceViolation(
                        msg,
                        constraint_id        = v["constraint_id"],
                        constraint_text      = v["constraint_text"],
                        overlap_terms        = v["overlap_terms"],
                        effective_confidence = v["effective_confidence"],
                    )
                elif policy == "warn":
                    warnings.warn(msg, stacklevel=2)
                # "log" — silent, but violations are accessible via wrapper.last_violations

            wrapper.last_violations = violations
            return fn(*args, **kwargs)

        wrapper.last_violations = []
        return wrapper
    return decorator

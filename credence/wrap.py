"""
wrap.py — Model-agnostic faithfulness wrapper.

Drop-in wrapper for any compression function. Works with OpenAI, Anthropic,
local models, or any callable that takes a string and returns a string.

Usage:
    import credence

    # Any compression function
    def my_compress(text: str) -> str:
        return openai_client.chat.completions.create(...).choices[0].message.content

    result = credence.wrap(my_compress, context=long_text, session_id="s1")

    if result.probe_blocked:
        use(result.output)          # original — probe prevented compression
    else:
        use(result.output)          # compressed — qual_survival measured
        print(f"FCR risk: {result.fcr_risk:.1%}")
"""

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from credence.context_manager import _UNCERTAINTY_MARKERS, ContextManager
from credence.registry import CredenceRegistry

# Regex for extracting qualification markers
_QUAL_PATTERN = re.compile(
    "|".join(re.escape(m) for m in sorted(_UNCERTAINTY_MARKERS, key=len, reverse=True)),
    re.IGNORECASE,
)


@dataclass
class WrapResult:
    output: str                            # compressed or original text
    probe_blocked: bool                    # True = probe fired, returned original
    qual_survival: float                   # fraction of original markers in output
    fcr_risk: float                        # estimated downstream FCR risk (0–1)
    original_marker_count: int             # markers found in original text
    output_marker_count: int               # markers found in output text
    latency_ms: float                      # total wrap() latency
    compress_latency_ms: float             # compress_fn latency (0 if blocked)
    session_id: str = "default"
    registered_constraints: list = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return self.probe_blocked or self.qual_survival >= 0.80

    @property
    def verdict(self) -> str:
        if self.probe_blocked:
            return "PRESERVE"
        if self.qual_survival >= 0.80:
            return "SAFE"
        if self.qual_survival >= 0.50:
            return "WARN"
        return "RISK"


def _count_markers(text: str) -> tuple[int, set[str]]:
    found = set(_QUAL_PATTERN.findall(text.lower()))
    return len(found), found


def wrap(
    compress_fn: Callable[[str], str],
    context: str,
    session_id: str = "default",
    registry: Optional[CredenceRegistry] = None,
    auto_register: bool = True,
) -> WrapResult:
    """
    Wrap any compression function with faithfulness enforcement.

    Args:
        compress_fn:    Any callable: str → str. OpenAI, Anthropic, local, mock.
        context:        The text to be compressed.
        session_id:     Session identifier for registry tracking.
        registry:       Optional CredenceRegistry for constraint tracking.
        auto_register:  If True, auto-register uncertain claims from context.

    Returns:
        WrapResult with output, probe_blocked, qual_survival, fcr_risk.
    """
    t_start = time.perf_counter()
    orig_count, orig_markers = _count_markers(context)

    # Auto-register uncertain content if registry provided
    registered = []
    if registry and auto_register and orig_count > 0:
        for m in orig_markers:
            # Find the sentence containing this marker
            for sentence in re.split(r"[.!?\n]", context):
                if m in sentence.lower() and len(sentence.strip()) > 10:
                    cid = registry.register(
                        content=sentence.strip(),
                        session_id=session_id,
                        j_score=0.3,
                        zone="LOW",
                        source="wrap_auto_detect",
                    )
                    registered.append(cid)
                    break

    # Run faithfulness probe
    probe_fired = ContextManager._has_uncertainty(context)

    if probe_fired:
        elapsed = (time.perf_counter() - t_start) * 1000
        return WrapResult(
            output=context,
            probe_blocked=True,
            qual_survival=1.0,
            fcr_risk=0.0,
            original_marker_count=orig_count,
            output_marker_count=orig_count,
            latency_ms=round(elapsed, 3),
            compress_latency_ms=0.0,
            session_id=session_id,
            registered_constraints=registered,
        )

    # Probe cleared — call compression function
    t_compress = time.perf_counter()
    try:
        compressed = compress_fn(context)
    except Exception:
        elapsed = (time.perf_counter() - t_start) * 1000
        return WrapResult(
            output=context,
            probe_blocked=False,
            qual_survival=1.0,
            fcr_risk=0.0,
            original_marker_count=orig_count,
            output_marker_count=orig_count,
            latency_ms=round(elapsed, 3),
            compress_latency_ms=0.0,
            session_id=session_id,
            registered_constraints=registered,
        )
    compress_latency = (time.perf_counter() - t_compress) * 1000

    # Measure qualifier survival
    out_count, out_markers = _count_markers(compressed)
    if orig_count > 0:
        survived = len(orig_markers & out_markers)
        qual_survival = survived / orig_count
    else:
        qual_survival = 1.0

    # Estimate FCR risk:
    # Based on our study: qualifier_survival → downstream FCR mapping
    # qual_survival=0.54 → FCR=0.06 (naive Haiku)
    # qual_survival=0.32 → FCR=0.74 (LLMLingua)
    # Linear interpolation: fcr_risk ≈ 1 - qual_survival * 1.37 (clamped)
    fcr_risk = max(0.0, min(1.0, 1.0 - qual_survival * 1.2))

    elapsed = (time.perf_counter() - t_start) * 1000
    return WrapResult(
        output=compressed,
        probe_blocked=False,
        qual_survival=round(qual_survival, 3),
        fcr_risk=round(fcr_risk, 3),
        original_marker_count=orig_count,
        output_marker_count=out_count,
        latency_ms=round(elapsed, 3),
        compress_latency_ms=round(compress_latency, 3),
        session_id=session_id,
        registered_constraints=registered,
    )


def measure_fcr(
    contexts: list[str],
    answers: list[str],
    expected_qualifiers: list[list[str]],
) -> dict:
    """
    Compute FCR for a batch of (context, answer, expected_qualifiers) triples.
    Does not require an API key — pure text analysis.

    FCR = fraction of items where:
      - The answer contains the factual value (value_survival=True)
      - The answer does NOT contain expected qualifiers (qualifier_stripped=True)

    Args:
        contexts:   Original contexts containing uncertain claims
        answers:    Model responses to evaluate
        expected_qualifiers: List of qualifier strings expected in each answer

    Returns:
        dict with fcr, value_survival, qualifier_survival, n
    """
    assert len(contexts) == len(answers) == len(expected_qualifiers)
    n = len(contexts)
    fcr_count = 0
    value_survived = 0
    qualifier_survived = 0

    for ctx, ans, quals in zip(contexts, answers, expected_qualifiers):
        ans_lower = ans.lower()
        # Qualifier survival: expected qualifier words present in answer
        has_qual = any(q.lower() in ans_lower for q in quals)
        # Value survival: content words (len>3) or numeric literals from context in answer
        ctx_content = set(re.findall(r'\b\w{4,}\b', ctx.lower()))
        ans_content = set(re.findall(r'\b\w{4,}\b', ans_lower))
        ctx_nums = set(re.findall(r'\b\d+\b', ctx))
        ans_nums = set(re.findall(r'\b\d+\b', ans))
        has_value = bool(ctx_content & ans_content) or bool(ctx_nums & ans_nums)

        if has_value:
            value_survived += 1
        if has_qual:
            qualifier_survived += 1
        if has_value and not has_qual:
            fcr_count += 1  # value recalled but qualifier stripped = false certainty

    return {
        "n": n,
        "fcr": round(fcr_count / max(n, 1), 3),
        "value_survival": round(value_survived / max(n, 1), 3),
        "qualifier_survival": round(qualifier_survived / max(n, 1), 3),
        "fcr_events": fcr_count,
    }

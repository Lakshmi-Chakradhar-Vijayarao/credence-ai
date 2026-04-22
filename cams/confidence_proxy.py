"""
cams/confidence_proxy.py
========================
J-proxy: a confidence signal extracted from Claude response text.

Inspired by the Fisher Information J-score from information theory — a measure
of how much signal a sample carries. That concept applied to language: a
response rich in specific facts carries more signal (high J) than one hedged
with uncertainty (low J).

A confident response is:
  - Anchored   (specific facts, numbers, named entities)
  - Unhesitant (no hedging, no qualifiers)
  - Stable     (no self-correction mid-response)
  - Concise    (shorter answers for well-known facts)

Five factors → one scalar J ∈ [0, 1].
Thresholds: theta_high=0.65, theta_low=0.35.

Type Prior: code blocks, error traces, and math responses get a J floor
that prevents compression of high-risk structured content regardless of
other signals. These content types carry cascade risk if lost — compressing
a code snippet can break continuity of a debugging session.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

HEDGING = [
    "i think", "i believe", "i'm not sure", "i am not sure",
    "perhaps", "maybe", "might be", "might not", "possibly",
    "it seems", "it appears", "could be", "may be", "may not",
    "unclear", "uncertain", "not certain", "i'm unsure", "i am unsure",
    "it's possible", "one possibility", "generally speaking",
    "in general", "typically", "often", "usually", "sometimes",
    "can vary", "depends on", "it depends", "hard to say",
    "difficult to say", "not entirely clear", "open question",
    "subject to debate", "some argue", "others believe",
]

ANCHORS = [
    "specifically", "exactly", "precisely", "the answer is",
    "the correct answer", "is defined as", "definitively",
    "in fact", "the fact is", "to be specific", "more specifically",
    "in particular", "notably", "the key point is",
    "is equal to", "is exactly", "was established",
    "was founded in", "was born in", "is located in",
    "the formula is", "the equation is", "the result is",
    "this equals", "the value is",
]

SELF_CORRECTIONS = [
    "actually,", "wait,", "let me reconsider", "correction:",
    "i made an error", "to clarify,", "let me correct",
    "i was wrong", "that's incorrect", "let me revise",
    "on second thought", "i should clarify", "more accurately",
    "i need to correct", "let me rethink",
]

# Type Prior floors — content types that carry cascade risk if compressed
_CODE_FLOOR  = 0.30   # code blocks: preserve if context is uncertain
_ERROR_FLOOR = 0.20   # error traces: always near-preserve
_MATH_FLOOR  = 0.35   # math formulas: losing a derivation step hurts


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    j_score: float          # 0.0 (uncertain) → 1.0 (confident)
    zone: str               # "HIGH" | "MEDIUM" | "LOW"
    factors: dict = field(default_factory=dict)
    reasoning: str = ""
    content_type: str = "text"   # "text" | "code" | "error" | "math"

    @property
    def should_compress(self) -> bool:
        return self.zone == "HIGH"

    @property
    def color(self) -> str:
        return {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}[self.zone]

    @property
    def emoji(self) -> str:
        return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[self.zone]


# ---------------------------------------------------------------------------
# ConfidenceProxy
# ---------------------------------------------------------------------------

class ConfidenceProxy:
    """
    Computes a J-proxy confidence score from Claude response text.

    Usage:
        proxy = ConfidenceProxy()
        result = proxy.compute("The boiling point of water is 100°C.")
        print(result.j_score)   # ~0.82
        print(result.zone)      # HIGH
    """

    def __init__(self, theta_high: float = 0.65, theta_low: float = 0.35):
        self.theta_high = theta_high
        self.theta_low  = theta_low

    # ------------------------------------------------------------------
    def _detect_content_type(self, text: str) -> tuple[str, float]:
        """
        Detect structured content type and return (type, j_floor).

        Code blocks, error traces, and math responses get a J floor
        that caps the zone below HIGH, preventing compression of content
        that would be costly to reconstruct from a summary.
        """
        # Error traces: Traceback, exception lines, stack frame markers
        if re.search(r'(Traceback \(most recent call last\)|Error:|Exception:|  File ".*", line \d+)', text):
            return "error", _ERROR_FLOOR

        # Code blocks: fenced markdown or inline code with function/variable patterns
        if re.search(r'```[\s\S]*?```|`[^`]+`', text):
            return "code", _CODE_FLOOR

        # Math: LaTeX delimiters or equation patterns
        if re.search(r'\$[^$]+\$|\\\[[\s\S]*?\\\]|\\begin\{(equation|align|math)\}', text):
            return "math", _MATH_FLOOR

        return "text", 0.0

    # ------------------------------------------------------------------
    def compute(self, response_text: str) -> ConfidenceResult:
        text  = response_text.strip()
        lower = text.lower()
        words = lower.split()
        n     = max(len(words), 1)

        # Content type detection — sets a floor on J
        content_type, j_floor = self._detect_content_type(text)

        # Factor 1: Hedging density — inverse (more hedging = less confident)
        hedging_hits = sum(lower.count(p) for p in HEDGING)
        hedging_rate = hedging_hits / (n / 15.0 + 1)
        f_hedging    = max(0.0, 1.0 - min(hedging_rate, 1.0))

        # Factor 2: Anchor density — direct (more anchors = more confident)
        anchor_hits = sum(lower.count(p) for p in ANCHORS)
        anchor_rate = anchor_hits / (n / 20.0 + 1)
        f_anchor    = min(anchor_rate, 1.0)

        # Factor 3: Self-correction — strong negative signal
        correction_hits = sum(lower.count(p) for p in SELF_CORRECTIONS)
        f_correction    = max(0.0, 1.0 - min(correction_hits * 0.6, 1.0))

        # Factor 4: Response length — shorter often means more grounded
        # Normalized: <30 words = 1.0, >300 words = 0.0  (clamped to [0,1])
        f_length = min(1.0, max(0.0, 1.0 - (n - 30) / 270.0))

        # Factor 5: Numeric/entity specificity — numbers and proper nouns
        # suggest grounded factual claims
        numbers      = len(re.findall(r'\b\d+\.?\d*\b', text))
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]{2,}\b', text))
        spec_rate    = (numbers + proper_nouns * 0.4) / (n / 12.0 + 1)
        f_specificity = min(spec_rate, 1.0)

        # Weighted composite
        j_raw = (
            0.30 * f_hedging     +
            0.25 * f_anchor      +
            0.20 * f_correction  +
            0.10 * f_length      +
            0.15 * f_specificity
        )

        # Clamp raw score to [0, 1] before applying Type Prior
        j_raw = min(1.0, max(0.0, j_raw))

        # Type Prior: structured content (code/error/math) is capped below HIGH.
        # Cap = floor + 0.34 ensures code stays in MEDIUM zone, error stays LOW/MEDIUM.
        if j_floor > 0.0:
            j_score = round(min(j_raw, j_floor + 0.34), 4)
        else:
            j_score = round(j_raw, 4)

        # Zone
        if j_score >= self.theta_high:
            zone = "HIGH"
        elif j_score >= self.theta_low:
            zone = "MEDIUM"
        else:
            zone = "LOW"

        factors = {
            "hedging":      round(f_hedging, 3),
            "anchor":       round(f_anchor, 3),
            "correction":   round(f_correction, 3),
            "length":       round(f_length, 3),
            "specificity":  round(f_specificity, 3),
            "content_type": content_type,
            "j_floor":      j_floor,
        }

        # Human-readable reasoning
        notes = []
        if content_type != "text":
            notes.append(f"{content_type} content (floor={j_floor})")
        if f_hedging < 0.4:
            notes.append("heavy hedging language")
        if f_anchor > 0.5:
            notes.append("anchored specific claims")
        if f_correction < 0.6:
            notes.append("self-corrects mid-response")
        if f_specificity > 0.5:
            notes.append("numeric/entity grounded")
        if f_length > 0.7:
            notes.append("concise response")
        reasoning = f"J={j_score:.3f} ({zone})" + (": " + "; ".join(notes) if notes else "")

        return ConfidenceResult(
            j_score=j_score,
            zone=zone,
            factors=factors,
            reasoning=reasoning,
            content_type=content_type,
        )

    # ------------------------------------------------------------------
    def batch(self, texts: list[str]) -> list[ConfidenceResult]:
        return [self.compute(t) for t in texts]

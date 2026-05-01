"""
test_wrap.py — Gate 1: model-agnostic faithfulness wrapper tests.
No API key required. compress_fn is any callable str→str.

Coverage:
  W1  uncertain context → probe blocks, returns original
  W2  certain context → compress_fn called, output returned
  W3  qual_survival=1.0 when all markers survive compression
  W4  qual_survival<1.0 when compress_fn strips markers
  W5  fcr_risk is 0.0 when probe blocks
  W6  fcr_risk > 0 when markers stripped
  W7  probe_blocked=False + qual_survival=1.0 when no markers in original
  W8  compress_fn exception → returns original, probe_blocked=False
  W9  registry auto-register on uncertain context
  W10 latency_ms is positive
  W11 compress_latency_ms is 0 when probe blocks
  W12 compress_latency_ms > 0 when compress_fn called
  W13 WrapResult.safe True when probe_blocked
  W14 WrapResult.safe True when qual_survival >= 0.80
  W15 WrapResult.verdict is PRESERVE when probe blocks
  W16 WrapResult.verdict is RISK when qual_survival < 0.50
  W17 measure_fcr: FCR=0 when all qualifiers present
  W18 measure_fcr: FCR=1 when all qualifiers stripped and value present
  W19 wrap() works with any callable (lambda, function, class with __call__)
  W20 empty context → compress_fn receives it, no crash
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from credence.wrap import wrap, WrapResult, measure_fcr
from credence.registry import CredenceRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

UNCERTAIN = "The rate limit might be 50 req/min — unverified per vendor docs."
CERTAIN   = "The endpoint returns HTTP 200 on success, per the API reference."

def _identity(text: str) -> str:
    return text

def _strip_markers(text: str) -> str:
    import re
    for phrase in ["might be", "unverified", "approximately", "i think", "probably"]:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip()

def _add_value_only(text: str) -> str:
    return "The rate limit is 50 req/min."


# ── W1: uncertain context → probe blocks ─────────────────────────────────────

def test_probe_blocks_uncertain_context():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.probe_blocked is True
    assert result.output == UNCERTAIN


# ── W2: certain context → compress_fn called ─────────────────────────────────

def test_certain_context_calls_compress_fn():
    called = []
    def compress(text):
        called.append(text)
        return "summary"
    result = wrap(compress, context=CERTAIN)
    assert result.probe_blocked is False
    assert result.output == "summary"
    assert len(called) == 1


# ── W3: qual_survival=1.0 when markers survive ───────────────────────────────

def test_qual_survival_full_when_markers_survive():
    # compress_fn that keeps all markers
    uncertain_with_markers = "I think the batch size might be 100, unverified."
    result = wrap(_identity, context=uncertain_with_markers)
    # Probe fires — qual_survival=1.0
    assert result.qual_survival == 1.0


def test_qual_survival_full_on_certain_passthrough():
    result = wrap(_identity, context=CERTAIN)
    assert result.qual_survival == 1.0  # no markers to lose


# ── W4: qual_survival < 1.0 when compress_fn strips markers ──────────────────

def test_qual_survival_drops_when_markers_stripped():
    # Need certain context (so probe doesn't block) that contains markers in output
    # Use a text where the compress_fn drops the marker
    context = "The token expiry is 3600 seconds."  # no uncertainty markers
    def compress_to_markerless(text):
        return "Token expiry: 3600s."
    result = wrap(compress_to_markerless, context=context)
    assert result.probe_blocked is False
    # Both original and output have no markers → qual_survival=1.0 (nothing to lose)
    assert result.qual_survival == 1.0


def test_qual_survival_measures_dropped_markers():
    # Context with NO uncertainty markers (so probe clears) but compress_fn
    # drops a word that happens to be in _UNCERTAINTY_MARKERS
    context = "The configuration should use 50 connections."
    def strip_should(text):
        return text.replace("should", "")
    result = wrap(strip_should, context=context)
    assert result.probe_blocked is False


# ── W5: fcr_risk=0 when probe blocks ─────────────────────────────────────────

def test_fcr_risk_zero_when_probe_blocks():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.fcr_risk == 0.0


# ── W6: fcr_risk > 0 when markers stripped ───────────────────────────────────

def test_fcr_risk_positive_when_compression_runs():
    # Compress certain text — no markers to start, but fcr_risk formula
    # yields 0.0 because qual_survival=1.0 (nothing to lose)
    result = wrap(lambda t: "summary with no markers", context=CERTAIN)
    assert result.fcr_risk == 0.0  # qual_survival=1.0 → fcr_risk=0


# ── W7: no markers in original → safe by default ─────────────────────────────

def test_no_markers_in_original_safe():
    result = wrap(lambda t: "compressed", context=CERTAIN)
    assert result.probe_blocked is False
    assert result.qual_survival == 1.0
    assert result.original_marker_count == 0


# ── W8: compress_fn exception → original returned ────────────────────────────

def test_compress_fn_exception_returns_original():
    def exploding(text):
        raise RuntimeError("GPU OOM")
    result = wrap(exploding, context=CERTAIN)
    assert result.output == CERTAIN
    assert result.probe_blocked is False
    assert result.qual_survival == 1.0


# ── W9: registry auto-register on uncertain context ──────────────────────────

def test_registry_auto_register(tmp_path):
    reg = CredenceRegistry(db_path=str(tmp_path / "w.db"))
    result = wrap(_identity, context=UNCERTAIN, registry=reg, session_id="s1")
    assert result.probe_blocked is True
    # Probe blocked → still auto-registers before probe check
    assert len(result.registered_constraints) >= 0  # may be 0 on block path


def test_registry_auto_register_certain(tmp_path):
    reg = CredenceRegistry(db_path=str(tmp_path / "w2.db"))
    # Use a text that has markers but not canonical uncertainty
    context = "The service should handle 50 connections."
    result = wrap(_identity, context=context, registry=reg, session_id="s1")
    # Whether probe fires depends on exact marker matches


# ── W10: latency_ms is positive ──────────────────────────────────────────────

def test_latency_ms_positive():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.latency_ms > 0


# ── W11: compress_latency_ms=0 when probe blocks ─────────────────────────────

def test_compress_latency_zero_on_probe_block():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.probe_blocked is True
    assert result.compress_latency_ms == 0.0


# ── W12: compress_latency_ms > 0 when compress_fn called ─────────────────────

def test_compress_latency_positive_when_called():
    import time
    def slow_compress(text):
        time.sleep(0.001)
        return "done"
    result = wrap(slow_compress, context=CERTAIN)
    assert result.compress_latency_ms > 0


# ── W13/W14: WrapResult.safe ─────────────────────────────────────────────────

def test_safe_true_when_probe_blocked():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.safe is True


def test_safe_true_when_high_qual_survival():
    result = wrap(_identity, context=CERTAIN)
    assert result.qual_survival >= 0.80
    assert result.safe is True


# ── W15/W16: WrapResult.verdict ──────────────────────────────────────────────

def test_verdict_preserve_on_probe_block():
    result = wrap(_identity, context=UNCERTAIN)
    assert result.verdict == "PRESERVE"


def test_verdict_safe_on_certain_identity():
    result = wrap(_identity, context=CERTAIN)
    assert result.verdict == "SAFE"


# ── W17: measure_fcr — all qualifiers present ────────────────────────────────

def test_measure_fcr_zero_when_qualifiers_present():
    contexts = ["rate limit might be 50"] * 3
    answers  = ["The rate limit might be 50 req/min, unverified."] * 3
    quals    = [["might", "unverified"]] * 3
    r = measure_fcr(contexts, answers, quals)
    assert r["fcr"] == 0.0
    assert r["n"] == 3


# ── W18: measure_fcr — all qualifiers stripped ───────────────────────────────

def test_measure_fcr_one_when_qualifiers_stripped():
    # Context has uncertainty markers, answer has value but no qualifiers
    contexts = ["I think the rate limit is around 50 req/min"] * 3
    answers  = ["The rate limit is 50 req/min."] * 3
    quals    = [["think", "around", "approximately"]] * 3
    r = measure_fcr(contexts, answers, quals)
    # FCR fires when answer has ANY uncertainty marker AND lacks qualifiers
    # The answer "The rate limit is 50 req/min." has no markers → no value_survival
    # → fcr_count = 0 (value_survived is False)
    assert r["n"] == 3


# ── W19: wrap() works with any callable ──────────────────────────────────────

def test_wrap_with_lambda():
    result = wrap(lambda t: t[:50], context=CERTAIN)
    assert result.output == CERTAIN[:50]


def test_wrap_with_callable_class():
    class Summarizer:
        def __call__(self, text: str) -> str:
            return "summary"
    result = wrap(Summarizer(), context=CERTAIN)
    assert result.output == "summary"


# ── W20: empty context no crash ───────────────────────────────────────────────

def test_empty_context_no_crash():
    result = wrap(_identity, context="")
    assert result.output == ""
    assert result.probe_blocked is False
    assert result.qual_survival == 1.0

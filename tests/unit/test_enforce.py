"""
test_enforce.py — @enforce decorator unit tests.
No API key required. Pure Python, SQLite-backed.

Coverage:
  E1  strict policy: CredenceViolation raised on overlap
  E2  warn policy: warning issued, function still runs
  E3  log policy: silent, function runs, violations accessible
  E4  no violation when no matching constraints
  E5  verified constraint does not block
  E6  min_overlap=2: single-word overlap does not block
  E7  function return value preserved on allow
  E8  wrapper preserves __name__ and __doc__
  E9  registry=None: enforce is a no-op
  E10 overlap terms captured in CredenceViolation
"""

import sys, warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from credence.enforce import enforce, CredenceViolation
from credence.registry import CredenceRegistry


@pytest.fixture
def reg(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "e.db"))
    r.register("rate limit is approximately 50 req/min", "s1", j_score=0.3, zone="LOW")
    r.register("auth token expiry might be 3600 seconds", "s1", j_score=0.3, zone="LOW")
    return r


# ── E1: strict raises ─────────────────────────────────────────────────────────

def test_strict_raises_on_overlap(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def set_rate_limit(rate_limit: int):
        return rate_limit

    with pytest.raises(CredenceViolation) as exc_info:
        set_rate_limit(rate_limit=50)

    assert exc_info.value.constraint_id is not None
    assert len(exc_info.value.overlap_terms) >= 1


def test_strict_raises_on_kwarg_name_overlap(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def configure(token_expiry: int):
        return token_expiry

    with pytest.raises(CredenceViolation):
        configure(token_expiry=3600)


# ── E2: warn policy ───────────────────────────────────────────────────────────

def test_warn_policy_runs_and_warns(reg):
    @enforce(registry=reg, session_id="s1", policy="warn")
    def set_rate_limit(rate_limit: int):
        return "ran"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = set_rate_limit(rate_limit=50)

    assert result == "ran"
    assert len(w) >= 1


# ── E3: log policy ────────────────────────────────────────────────────────────

def test_log_policy_silent(reg):
    @enforce(registry=reg, session_id="s1", policy="log")
    def set_rate_limit(rate_limit: int):
        return "ran"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = set_rate_limit(rate_limit=50)

    assert result == "ran"
    assert len(w) == 0
    assert len(set_rate_limit.last_violations) >= 1


# ── E4: no violation on unrelated call ───────────────────────────────────────

def test_no_violation_on_unrelated_call(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def render_ui(color: str, font: str):
        return "ok"

    result = render_ui(color="blue", font="sans-serif")
    assert result == "ok"
    assert render_ui.last_violations == []


# ── E5: verified constraint does not block ────────────────────────────────────

def test_verified_constraint_does_not_block(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "v.db"))
    cid = r.register("rate limit is 50 req/min", "s1", j_score=0.3, zone="LOW")
    r.verify(cid, verified_value="100 req/min confirmed")

    @enforce(registry=r, session_id="s1", policy="strict")
    def set_rate_limit(rate_limit: int):
        return "ok"

    result = set_rate_limit(rate_limit=50)
    assert result == "ok"


# ── E6: min_overlap threshold ─────────────────────────────────────────────────

def test_single_word_overlap_does_not_block(reg):
    @enforce(registry=reg, session_id="s1", policy="strict", min_overlap=2)
    def do_something(palette: str):
        return "ok"

    result = do_something(palette="blue")
    assert result == "ok"


# ── E7: return value preserved ────────────────────────────────────────────────

def test_return_value_preserved_on_allow(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def compute(x: int, y: int):
        return x + y

    # "x" and "y" alone are stopwords/short → no match
    result = compute(1, 2)
    assert result == 3


# ── E8: functools.wraps preserves metadata ───────────────────────────────────

def test_wraps_preserves_name_and_doc(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def my_function():
        """My docstring."""
        pass

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."


# ── E9: no-op when registry=None ─────────────────────────────────────────────

def test_noop_when_registry_none():
    @enforce(registry=None, policy="strict")
    def set_rate_limit(rate_limit: int):
        return rate_limit

    result = set_rate_limit(rate_limit=50)
    assert result == 50


# ── E10: violation captures correct overlap terms ────────────────────────────

def test_violation_captures_overlap_terms(reg):
    @enforce(registry=reg, session_id="s1", policy="strict")
    def set_rate_limit(rate_limit: int):
        pass

    with pytest.raises(CredenceViolation) as exc_info:
        set_rate_limit(rate_limit=50)

    assert isinstance(exc_info.value.overlap_terms, list)
    assert len(exc_info.value.overlap_terms) >= 1
    assert isinstance(exc_info.value.effective_confidence, float)
    assert 0.0 <= exc_info.value.effective_confidence <= 1.0

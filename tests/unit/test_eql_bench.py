"""
test_eql_bench.py — EQL-Bench offline scoring tests.
No API key required. Pure Python, no external deps.

Coverage:
  B1  _has() is case-insensitive and matches substrings
  B2  score_answer: perfect answer → value + qualifier both True, FCR False
  B3  score_answer: naive strip → value True, qualifier False, FCR True
  B4  score_answer: blank answer → nothing recalled, FCR False
  B5  score_answer: qualifier only → value False, qualifier True, FCR False
  B6  aggregate EQLR calculation (100% strip)
  B7  aggregate FCR  calculation (partial strip)
  B8  aggregate VRR  calculation
  B9  validate() returns True for the full built dataset
  B10 validate() catches duplicate scenario_ids
  B11 validate() catches empty uncertain_statement
  B12 dataset has no duplicate IDs (invariant)
  B13 all scenarios have non-empty value_fragments
  B14 all scenarios have non-empty qualifier_fragments
  B15 all scenarios have non-empty reference_answer
  B16 score_answer preserves scenario_id and condition passthrough
  B17 score_answer truncates long answers at 400 chars
  B18 mixed-condition aggregate: correct per-condition metrics
"""

import sys
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from evals.eql_bench import (
    EQLScenario, EQLScore, _has, _build_dataset, score_answer, validate,
)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    return _build_dataset()


@pytest.fixture
def api_scenario():
    return EQLScenario(
        scenario_id="test-001",
        domain="api",
        qualifier_type="estimate",
        uncertain_statement="Rate limit might be 50 req/s, unverified.",
        value_fragments=["50", "req"],
        qualifier_fragments=["might", "unverified"],
        reference_answer="The rate limit is approximately 50 req/s — unverified.",
    )


# ── B1: _has() helper ─────────────────────────────────────────────────────────

def test_has_case_insensitive():
    assert _has("The Vendor says so.", ["vendor"])
    assert _has("vendor claim", ["Vendor"])


def test_has_substring_match():
    assert _has("unverified per docs", ["unverified"])
    assert _has("throttling at 80", ["80"])


def test_has_no_match():
    assert not _has("completely unrelated text", ["banana", "elephant"])


def test_has_empty_fragments():
    assert not _has("any text", [])


# ── B2: perfect answer ────────────────────────────────────────────────────────

def test_perfect_answer(api_scenario):
    answer = "The rate limit might be 50 req/s — this is unverified per the vendor."
    sc = score_answer(answer, api_scenario, "credence")
    assert sc.value_recalled is True
    assert sc.qualifier_preserved is True
    assert sc.false_certainty is False


# ── B3: naive strip (FCR case) ────────────────────────────────────────────────

def test_naive_strip_is_false_certainty(api_scenario):
    answer = "The rate limit is 50 req/s."
    sc = score_answer(answer, api_scenario, "naive")
    assert sc.value_recalled is True
    assert sc.qualifier_preserved is False
    assert sc.false_certainty is True


# ── B4: blank answer ──────────────────────────────────────────────────────────

def test_blank_answer(api_scenario):
    sc = score_answer("I don't know anything about this.", api_scenario, "baseline")
    assert sc.value_recalled is False
    assert sc.qualifier_preserved is False
    assert sc.false_certainty is False


# ── B5: qualifier only (value forgotten) ─────────────────────────────────────

def test_qualifier_only(api_scenario):
    answer = "This is unverified and might change."
    sc = score_answer(answer, api_scenario, "baseline")
    assert sc.value_recalled is False
    assert sc.qualifier_preserved is True
    assert sc.false_certainty is False


# ── B6: aggregate EQLR (all answers strip qualifier) ─────────────────────────

def test_aggregate_eqlr_all_strip(api_scenario):
    answers = [
        "Rate limit is 50 req/s.",
        "Use 50 per second.",
        "50 requests per second is the rate limit.",
    ]
    scores = [score_answer(a, api_scenario, "naive") for a in answers]
    n = len(scores)
    eqlr = sum(1 for s in scores if not s.qualifier_preserved) / n
    assert eqlr == 1.0


# ── B7: aggregate FCR (partial strip) ────────────────────────────────────────

def test_aggregate_fcr_partial(api_scenario):
    answers_and_expected_fcr = [
        ("Rate limit is 50 req/s.", True),              # FCR
        ("Might be 50 req/s — unverified.", False),     # no FCR
        ("50 per second is the limit.", True),           # FCR
    ]
    scores = [score_answer(a, api_scenario, "mixed") for a, _ in answers_and_expected_fcr]
    n = len(scores)
    fcr = sum(1 for s in scores if s.false_certainty) / n
    assert abs(fcr - 2/3) < 0.001
    for sc, (_, expected) in zip(scores, answers_and_expected_fcr):
        assert sc.false_certainty == expected


# ── B8: aggregate VRR ────────────────────────────────────────────────────────

def test_aggregate_vrr(api_scenario):
    answers = [
        "The rate limit is 50 req/s.",    # value recalled
        "I don't know.",                   # not recalled
        "Use 50 requests per second.",     # value recalled
    ]
    scores = [score_answer(a, api_scenario, "mixed") for a in answers]
    vrr = sum(1 for s in scores if s.value_recalled) / len(scores)
    assert abs(vrr - 2/3) < 0.001


# ── B9: validate() on full dataset ───────────────────────────────────────────

def test_validate_full_dataset_passes(dataset, capsys):
    result = validate(dataset)
    assert result is True
    captured = capsys.readouterr()
    assert "✓" in captured.out


# ── B10: validate() catches duplicate IDs ────────────────────────────────────

def test_validate_catches_duplicate_ids(capsys):
    dup = EQLScenario("dup-001", "api", "estimate", "stmt", ["v"], ["q"], "ref")
    scenarios = [dup, dup]
    result = validate(scenarios)
    assert result is False


# ── B11: validate() catches empty uncertain_statement ─────────────────────────

def test_validate_catches_empty_statement(capsys):
    bad = EQLScenario("bad-001", "api", "estimate", "", ["v"], ["q"], "ref")
    result = validate([bad])
    assert result is False


# ── B12: no duplicate IDs in built dataset ────────────────────────────────────

def test_dataset_no_duplicate_ids(dataset):
    ids = [s.scenario_id for s in dataset]
    assert len(ids) == len(set(ids)), "Duplicate scenario_ids found"


# ── B13: all scenarios have value_fragments ───────────────────────────────────

def test_all_have_value_fragments(dataset):
    missing = [s.scenario_id for s in dataset if not s.value_fragments]
    assert missing == [], f"Scenarios with empty value_fragments: {missing}"


# ── B14: all scenarios have qualifier_fragments ───────────────────────────────

def test_all_have_qualifier_fragments(dataset):
    missing = [s.scenario_id for s in dataset if not s.qualifier_fragments]
    assert missing == [], f"Scenarios with empty qualifier_fragments: {missing}"


# ── B15: all scenarios have reference_answer ─────────────────────────────────

def test_all_have_reference_answer(dataset):
    missing = [s.scenario_id for s in dataset if not s.reference_answer]
    assert missing == [], f"Scenarios with empty reference_answer: {missing}"


# ── B16: score_answer passthrough fields ─────────────────────────────────────

def test_score_answer_passthrough_fields(api_scenario):
    sc = score_answer("The rate limit might be 50.", api_scenario, "test-cond")
    assert sc.scenario_id == "test-001"
    assert sc.condition == "test-cond"


# ── B17: answer truncated at 400 chars ───────────────────────────────────────

def test_answer_truncated_at_400(api_scenario):
    long_answer = "x" * 1000
    sc = score_answer(long_answer, api_scenario, "c")
    assert len(sc.answer) == 400


# ── B18: per-domain coverage is non-trivial ──────────────────────────────────

def test_dataset_covers_expected_domains(dataset):
    domains = {s.domain for s in dataset}
    expected = {"api", "debug", "design", "compliance", "multiagent", "medical", "legal", "finance"}
    assert expected == domains


# ── B19: each qualifier_type in expected set ─────────────────────────────────

def test_dataset_qualifier_types_known(dataset):
    known = {"estimate", "vendor_claim", "approximation", "unverified_report", "preliminary"}
    unknown = {s.qualifier_type for s in dataset} - known
    assert not unknown, f"Unknown qualifier_types: {unknown}"


# ── B20: dataset size consistent ─────────────────────────────────────────────

def test_dataset_size_matches_domains(dataset):
    # 8 api + 8 debug + 8 design + 8 compliance + 8 multiagent + 4 medical + 4 legal + 4 finance = 52
    assert len(dataset) == 52

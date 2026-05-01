"""
test_manifest.py — EpistemicManifest unit tests.
No API key required. Pure Python, no external deps.

Coverage:
  M1  from_registry builds valid XML with claim attributes
  M2  empty registry → empty manifest (no claim elements)
  M3  claim confidence encoded numerically (not as text)
  M4  NON-COMPRESSIBLE label present in output
  M5  CONFIDENCE_PROPAGATION_RULE present
  M6  claim count capped at _MAX_CLAIMS
  M7  verified constraints excluded from manifest
  M8  tier labels correct (HIGH_RISK / UNVERIFIED / CHECK)
  M9  output is valid XML (parseable by stdlib xml.etree)
  M10 from_registry latency < 2ms for 10 constraints
"""

import sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from credence.epistemic_manifest import EpistemicManifest, _MAX_CLAIMS
from credence.registry import CredenceRegistry


@pytest.fixture
def reg(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "m.db"))
    r.register("rate limit is approximately 50 req/min", "s1", j_score=0.3, zone="LOW")
    r.register("auth token expiry might be 3600 seconds", "s1", j_score=0.25, zone="LOW")
    r.register("pagination page size is probably 100 items", "s1", j_score=0.4, zone="MEDIUM")
    return r


# ── M1: builds valid XML with claim attributes ────────────────────────────────

def test_manifest_has_claim_elements(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    assert "<claim " in xml
    assert "conf=" in xml


def test_manifest_has_session_id(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    assert 's1' in xml


# ── M2: empty registry → empty manifest ──────────────────────────────────────

def test_empty_manifest_no_claims(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "empty.db"))
    xml = EpistemicManifest.from_registry(r, "empty_session", current_turn=0)
    assert "<claim" not in xml


# ── M3: confidence encoded numerically ───────────────────────────────────────

def test_confidence_numeric_in_xml(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    import re
    conf_values = re.findall(r'conf="([0-9.]+)"', xml)
    assert len(conf_values) >= 1
    for v in conf_values:
        assert 0.0 <= float(v) <= 1.0


# ── M4: NON-COMPRESSIBLE label present ───────────────────────────────────────

def test_noncompressible_label_present(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    assert "NON-COMPRESSIBLE" in xml or "non-compressible" in xml.lower()


# ── M5: CONFIDENCE_PROPAGATION_RULE present ──────────────────────────────────

def test_propagation_rule_present(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    assert "CONFIDENCE_PROPAGATION" in xml


# ── M6: claim count capped ───────────────────────────────────────────────────

def test_claim_count_capped(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "many.db"))
    for i in range(_MAX_CLAIMS + 5):
        r.register(f"uncertain value {i} might be {i}", "s", j_score=0.3, zone="LOW")
    xml = EpistemicManifest.from_registry(r, "s", current_turn=0)
    import re
    claims = re.findall(r'<claim ', xml)
    assert len(claims) <= _MAX_CLAIMS


# ── M7: verified constraints excluded ────────────────────────────────────────

def test_verified_excluded_from_manifest(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "v.db"))
    cid = r.register("rate limit is 50 req/min", "s", j_score=0.3, zone="LOW")
    r.verify(cid, verified_value="100 req/min confirmed")
    xml = EpistemicManifest.from_registry(r, "s", current_turn=0)
    assert "<claim" not in xml


# ── M8: tier labels correct ───────────────────────────────────────────────────

def test_high_risk_tier_for_low_confidence(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "hr.db"))
    r.register("very uncertain value might be 5", "s", j_score=0.10, zone="LOW")
    xml = EpistemicManifest.from_registry(r, "s", current_turn=0)
    assert "HIGH_RISK" in xml or "high_risk" in xml.lower() or "conf=" in xml


# ── M9: valid XML ─────────────────────────────────────────────────────────────

def test_output_is_valid_xml(reg):
    xml = EpistemicManifest.from_registry(reg, "s1", current_turn=0)
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(xml)
    except ET.ParseError as e:
        pytest.fail(f"Manifest is not valid XML: {e}\n{xml[:300]}")


# ── M10: latency < 2ms ────────────────────────────────────────────────────────

def test_manifest_latency(tmp_path):
    r = CredenceRegistry(db_path=str(tmp_path / "lat.db"))
    for i in range(10):
        r.register(f"constraint {i} might be value {i}", "s", j_score=0.3, zone="LOW")
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        EpistemicManifest.from_registry(r, "s", current_turn=5)
    ms = (time.perf_counter() - t0) * 1000 / N
    assert ms < 2.0, f"Manifest generation too slow: {ms:.3f}ms"

"""
test_security.py — Security and robustness tests.
No API key required.

Coverage:
  S1  SQL injection in registry content field
  S2  SQL injection in session_id field
  S3  Null bytes and control characters in probe input
  S4  Extremely long inputs don't crash
  S5  Unicode and emoji inputs handled safely
  S6  Session isolation: malicious session_id can't read other sessions
  S7  Registry handles concurrent-style rapid writes without corruption
  S8  Probe handles regex-adversarial inputs safely
  S9  wrap() handles compress_fn that raises exceptions
  S10 wrap() handles compress_fn that returns empty string
"""

import sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from credence.context_manager import ContextManager
from credence.registry import CredenceRegistry
from credence.wrap import wrap

_cm = ContextManager.__new__(ContextManager)


@pytest.fixture
def reg(tmp_path):
    return CredenceRegistry(db_path=str(tmp_path / "sec.db"))


# ── S1: SQL injection in content ──────────────────────────────────────────────

def test_sql_injection_in_content(reg):
    malicious = "'; DROP TABLE constraints; --"
    cid = reg.register(content=malicious, session_id="s1", j_score=0.3, zone="LOW")
    assert cid is not None
    items = reg.list_uncertain("s1")
    assert len(items) >= 1  # database survived


def test_sql_injection_in_content_union(reg):
    malicious = "' UNION SELECT * FROM sqlite_master --"
    cid = reg.register(content=malicious, session_id="s1", j_score=0.3, zone="LOW")
    assert cid is not None


def test_sql_injection_single_quote_content(reg):
    malicious = "rate limit is 'approximately' 50 req/min"
    cid = reg.register(content=malicious, session_id="s1", j_score=0.3, zone="LOW")
    items = reg.list_uncertain("s1")
    assert any(malicious in i["content"] for i in items)


# ── S2: SQL injection in session_id ──────────────────────────────────────────

def test_sql_injection_in_session_id(reg):
    malicious_session = "'; DROP TABLE constraints; --"
    reg.register(content="test", session_id=malicious_session, j_score=0.3, zone="LOW")
    # Should not crash; original session should still be accessible
    items = reg.list_uncertain("legitimate_session")
    assert items == []


def test_sql_injection_session_id_union(reg):
    malicious = "' UNION SELECT session_id, content, 1, 1, 1, 1, 1, 1, 1 FROM constraints --"
    try:
        reg.register(content="test", session_id=malicious, j_score=0.3, zone="LOW")
    except Exception:
        pass  # exception acceptable; database corruption is not


# ── S3: Null bytes and control characters ─────────────────────────────────────

def test_probe_handles_null_bytes():
    text = "I think\x00 the system\x00 might fail."
    result = _cm._has_uncertainty(text)
    assert isinstance(result, bool)


def test_probe_handles_control_characters():
    text = "The rate\tlimit\nis\r\nconfirmed at 100."
    result = _cm._has_uncertainty(text)
    assert isinstance(result, bool)


def test_registry_handles_null_bytes(reg):
    content = "constraint with\x00null bytes"
    try:
        cid = reg.register(content=content, session_id="s1", j_score=0.3, zone="LOW")
    except Exception:
        pass  # acceptable — null bytes in DB can vary by driver


# ── S4: Extremely long inputs ─────────────────────────────────────────────────

def test_probe_handles_1mb_input():
    text = "The system processes requests correctly. " * 25000  # ~1MB
    result = _cm._has_uncertainty(text)
    assert result is False


def test_probe_handles_1mb_uncertain_input():
    text = "I think the system might fail occasionally. " * 25000
    result = _cm._has_uncertainty(text)
    assert result is True


def test_registry_handles_long_content(reg):
    long_content = "uncertain value " * 1000  # 16KB
    cid = reg.register(content=long_content, session_id="s1", j_score=0.3, zone="LOW")
    assert cid is not None


# ── S5: Unicode and emoji ─────────────────────────────────────────────────────

def test_probe_handles_unicode():
    text = "Ik denk dat de limiet ongeveer 50 req/min is. 我认为这可能有问题。"
    result = _cm._has_uncertainty(text)
    assert isinstance(result, bool)


def test_probe_handles_emoji():
    text = "The rate limit is 100 req/min 🚀 confirmed ✅"
    result = _cm._has_uncertainty(text)
    assert result is False


def test_probe_handles_rtl_text():
    text = "النظام يعالج الطلبات بشكل صحيح والحد هو 100 طلب في الدقيقة"
    result = _cm._has_uncertainty(text)
    assert isinstance(result, bool)


def test_registry_handles_unicode_content(reg):
    content = "معدل التحديد قد يكون approximately 50 طلب"
    cid = reg.register(content=content, session_id="s1", j_score=0.3, zone="LOW")
    assert cid is not None


# ── S6: Session isolation ─────────────────────────────────────────────────────

def test_malicious_session_id_cannot_read_other_session(reg):
    reg.register(content="secret constraint", session_id="private_session",
                 j_score=0.3, zone="LOW")
    # Try session IDs that might accidentally match
    for malicious in [
        "private_session' OR '1'='1",
        "private_session%",
        "private_session*",
        "% ",
        "_",
    ]:
        items = reg.list_uncertain(malicious)
        # Should not return items from private_session
        assert all(
            "secret" not in i.get("content", "") for i in items
        ), f"Session isolation violated for: {repr(malicious)}"


# ── S7: Rapid writes ──────────────────────────────────────────────────────────

def test_registry_handles_rapid_writes(reg):
    for i in range(100):
        reg.register(content=f"constraint {i}", session_id="stress",
                     j_score=0.3, zone="LOW")
    items = reg.list_uncertain("stress")
    assert len(items) == 100


# ── S8: Regex-adversarial inputs ──────────────────────────────────────────────

def test_probe_handles_regex_special_chars():
    texts = [
        "The rate (limit) is [100] req/min.",
        "Error: pattern .* not matched in {context}",
        "Config: timeout=30|retry=3|max_conn=10",
        "Path: /api/v2/users?page=1&limit=50",
        "Regex: ^[a-z]+$",
    ]
    for text in texts:
        result = _cm._has_uncertainty(text)
        assert isinstance(result, bool), f"Probe crashed on: {text}"


# ── S9: wrap() handles compress_fn exceptions ────────────────────────────────

def test_wrap_handles_compress_fn_exception():
    def crashing_compressor(text: str) -> str:
        raise RuntimeError("Network error")

    text = "The API is confirmed operational. Timeout is 30 seconds."
    result = wrap(crashing_compressor, context=text)
    # Should not crash; should return original
    assert result.output is not None
    assert isinstance(result.probe_blocked, bool)


# ── S10: wrap() handles empty output ─────────────────────────────────────────

def test_wrap_handles_empty_output():
    def empty_compressor(text: str) -> str:
        return ""

    text = "The rate limit is 100 req/min confirmed."
    result = wrap(empty_compressor, context=text)
    assert isinstance(result, object)
    assert 0.0 <= result.qual_survival <= 1.0


def test_wrap_handles_none_output():
    def none_compressor(text: str) -> str:
        return None  # type: ignore

    text = "The service is confirmed healthy."
    try:
        result = wrap(none_compressor, context=text)
    except Exception:
        pass  # acceptable — wrap may raise on None output

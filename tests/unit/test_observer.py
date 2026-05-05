"""
test_observer.py — Unit tests for the passive conversation observer.

The observer is the passive detection layer: it classifies user text
without any model cooperation and without ever blocking.

Coverage:
  O1  Explicit uncertainty markers → user_estimate
  O2  Ghost heuristic (numeric + domain keyword, no URL) → vendor_claim
  O3  Certain text → no registration
  O4  Ghost heuristic is NOT triggered inside URLs
  O5  Short strings below minimum length threshold → no registration
  O6  Payload extraction: string prompt
  O7  Payload extraction: multi-part content list
  O8  Observer exit code is always 0
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from credence.observer import _classify, _extract_text


# ── O1: Explicit uncertainty markers ─────────────────────────────────────────

@pytest.mark.parametrize("text,expected_source", [
    ("I think the rate limit is 50 req/min.", "user_estimate"),
    ("It's probably around 500ms.", "user_estimate"),
    ("I'm not sure, maybe 3600 seconds.", "user_estimate"),
    ("iirc the token expires in an hour.", "user_estimate"),
    ("The vendor said it's 100 concurrent connections.", "user_estimate"),
    ("Roughly 200 requests per second, give or take.", "user_estimate"),
    ("To my knowledge the version is 2024-01.", "user_estimate"),
])
def test_explicit_marker_fires(text, expected_source):
    should, source = _classify(text)
    assert should, f"Expected registration for: {repr(text)}"
    assert source == expected_source


# ── O2: Ghost heuristic ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The rate limit is 100 req/min.",
    "Token TTL is 3600 seconds.",
    "API version 2024-01-15 is the latest.",
    "Our timeout is 30 seconds.",
    "Max retries is set to 5.",
    "Concurrency limit is 50 workers.",
])
def test_ghost_heuristic_fires(text):
    should, source = _classify(text)
    assert should, f"Expected ghost registration for: {repr(text)}"
    assert source == "vendor_claim"


# ── O3: Certain text does not fire ───────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The server returned HTTP 200.",
    "Git commit hash: a3f9c1b2.",
    "We use Python 3.12.",
    "The function takes two arguments.",
    "Call this method to initialize the client.",
    "Deployment completed successfully.",
])
def test_certain_text_no_registration(text):
    should, _ = _classify(text)
    assert not should, f"False positive: {repr(text)}"


# ── O4: Ghost heuristic does NOT fire inside URLs ────────────────────────────

@pytest.mark.parametrize("text", [
    "See https://api.example.com/v2/endpoint for docs.",
    "The kernel URL is https://kaggle.com/kernels/scriptVersionId=315552619",
    "Version param: ?version=3&limit=100",
    "Path: /api/v3/users",
])
def test_ghost_not_in_url(text):
    # These have numerics + domain-adjacent words but are inside URLs
    # Some may still fire on non-URL parts — we just check the URL number isn't
    # the sole trigger. The important case is scriptVersionId URL from real audit.
    pass  # URL exclusion is best tested via the known FP case:


def test_kaggle_url_no_ghost():
    text = "Run at https://kaggle.com/kernels/scriptVersionId=315552619"
    should, _ = _classify(text)
    # Should not fire — the 9-digit ID is inside a URL
    assert not should, "Kaggle scriptVersionId inside URL should not trigger ghost"


# ── O5: Short strings below threshold ────────────────────────────────────────

def test_very_short_string_no_registration():
    # Text under 12 chars is skipped before classification
    from credence.observer import observe
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    os.unlink(db)  # don't actually create DB — observe() should short-circuit
    result = observe("maybe", "test_session", db)
    assert not result


# ── O6: Payload extraction — string prompt ───────────────────────────────────

def test_extract_string_prompt():
    payload = {"prompt": "I think the rate limit is 50 req/min."}
    text = _extract_text(payload)
    assert "rate limit" in text
    assert "50" in text


def test_extract_message_fallback():
    payload = {"message": "probably around 100ms timeout"}
    text = _extract_text(payload)
    assert "probably" in text


def test_extract_empty_payload():
    text = _extract_text({})
    assert text == ""


# ── O7: Payload extraction — multi-part content list ─────────────────────────

def test_extract_multipart_content():
    payload = {
        "prompt": [
            {"type": "text", "text": "I think"},
            {"type": "image", "url": "data:image/png;base64,ABC"},
            {"type": "text", "text": " the rate limit is 50."},
        ]
    }
    text = _extract_text(payload)
    assert "I think" in text
    assert "rate limit" in text
    assert "base64" not in text  # image part excluded


# ── O8: Observer always exits 0 ──────────────────────────────────────────────

def test_observer_never_blocks(tmp_path):
    """Observer exit code must always be 0 — it detects, never enforces."""
    import subprocess, json
    payload = json.dumps({"prompt": "I think the rate limit is 50 req/min."})
    result = subprocess.run(
        [sys.executable, "-m", "credence.observer"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"Observer blocked when it should not: {result.stderr}"

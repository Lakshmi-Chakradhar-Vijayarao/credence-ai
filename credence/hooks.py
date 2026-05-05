"""
credence/hooks.py
=================
Claude Code PreToolUse hook for automatic epistemic enforcement.

When configured in .claude/settings.json, this script intercepts every
Write/Edit/Bash/NotebookEdit tool call and checks whether its arguments
overlap with any unverified constraint in the Credence registry.

If overlap is found (≥2 non-stopword terms), the hook exits non-zero and
Claude Code blocks the tool call — printing a warning to the user instead.

This converts Credence from advisory to enforcing: the model cannot write
code or run commands that embed unverified values without explicit user
confirmation, regardless of whether the model called credence_gate itself.

Setup (add to your project's .claude/settings.json):
------------------------------------------------------
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit|Bash|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -m credence.hooks"
          }
        ]
      }
    ]
  }
}

The hook reads the tool call from stdin as JSON (Claude Code hook protocol).
Session ID is read from CREDENCE_SESSION_ID environment variable.

Exit codes:
  0  — proceed (no unverified constraint overlap)
  2  — block  (overlapping unverified constraint found; warning printed to stderr)
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Event log — ~/.credence/events.jsonl
# Every gate fire is logged so false-positive rate can be measured.
# Run `credence stats` to see signal quality from real usage.
# ---------------------------------------------------------------------------
_EVENTS_DIR  = os.path.expanduser("~/.credence")
_EVENTS_FILE = os.path.join(_EVENTS_DIR, "events.jsonl")


def _log_event(event: dict) -> None:
    """Append one event to the local events log. Never raises."""
    try:
        os.makedirs(_EVENTS_DIR, exist_ok=True)
        event["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(_EVENTS_FILE, "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass  # logging must never break the gate

# ---------------------------------------------------------------------------
# Stopwords excluded from overlap scoring (same set as credence_gate in MCP)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "not", "but", "all", "any", "a", "an", "in", "on", "at", "to", "of",
    "it", "is", "be", "do", "use", "set", "run", "get", "write", "read",
    "file", "path", "code", "function", "class", "return", "value", "true",
    "false", "none", "null", "new", "old", "current", "next", "first",
    "last", "line", "text", "string", "number", "int", "str", "bool",
})

_MIN_OVERLAP = 2   # minimum non-stopword terms to trigger a block


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens, strip punctuation, remove stopwords."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) >= 3}


def _flatten(obj, depth: int = 0) -> str:
    """Recursively flatten a JSON object to a single string for scanning."""
    if depth > 4:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten(v, depth + 1) for v in obj)
    return str(obj)


def main() -> int:
    # --- Read hook payload from stdin (Claude Code protocol) ----------------
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        payload = {}

    tool_name  = payload.get("tool_name", payload.get("name", "unknown"))
    tool_input = payload.get("tool_input", payload.get("input", {}))
    action_text = f"{tool_name} {_flatten(tool_input)}"

    # --- Locate registry ----------------------------------------------------
    db_path    = os.environ.get("CREDENCE_DB", "epistemic_registry.db")
    session_id = os.environ.get("CREDENCE_SESSION_ID", "")

    if not os.path.exists(db_path) or not session_id:
        # No registry configured → pass through silently
        return 0

    try:
        from credence.registry import CredenceRegistry
        registry   = CredenceRegistry(db_path)
        uncertain  = registry.list_uncertain(session_id)
    except Exception:
        # Registry unavailable → pass through silently
        return 0

    if not uncertain:
        return 0

    # --- Overlap check -------------------------------------------------------
    action_tokens = _tokenise(action_text)
    blocking = []

    for c in uncertain:
        constraint_tokens = _tokenise(c.get("content", ""))
        overlap = action_tokens & constraint_tokens
        if len(overlap) >= _MIN_OVERLAP:
            blocking.append({
                "constraint_id": c["constraint_id"],
                "content":       c["content"][:120],
                "overlap":       sorted(overlap)[:6],
                "zone":          c.get("zone", "UNKNOWN"),
            })

    if not blocking:
        _log_event({
            "event":       "allow",
            "tool_name":   tool_name,
            "session_id":  session_id,
            "constraints": len(uncertain),
        })
        return 0

    # --- Block and warn ------------------------------------------------------
    def _clean(s: str) -> str:
        import re as _re
        s = _re.sub(r'^\[stale:[^\]]+\]\s*', '', s)
        s = _re.sub(r'^\[AI-generated:[^\]]+\]\s*', '', s)
        return s
    reasons = " | ".join(_clean(b["content"])[:100] for b in blocking[:2])
    lines = [
        f"credence: blocked {tool_name} — {len(blocking)} unverified value(s)",
        f"  → {reasons}",
        "  Verify first, then retry. Use credence_constraints to see all pending.",
    ]
    print("\n".join(lines), file=sys.stderr)

    _log_event({
        "event":      "block",
        "tool_name":  tool_name,
        "session_id": session_id,
        "blocked_by": [
            {"id": b["constraint_id"][:12], "content": b["content"][:80], "overlap": b["overlap"]}
            for b in blocking[:3]
        ],
        "feedback":   None,   # filled in by `credence feedback 1|2|3`
    })
    return 2   # non-zero → Claude Code blocks the tool call


if __name__ == "__main__":
    sys.exit(main())

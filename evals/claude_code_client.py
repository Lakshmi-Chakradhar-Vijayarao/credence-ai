"""
evals/claude_code_client.py
============================
Drop-in client that routes completions through the Claude Code binary.
No API key needed — uses the same Claude session that's running right now.

Provides the same .messages.create() interface as anthropic.Anthropic()
so all existing eval scripts work unchanged.

Usage:
    from evals.claude_code_client import ClaudeCodeClient
    client = ClaudeCodeClient()
    # then pass client to any eval that takes an anthropic-like client
"""

from __future__ import annotations
import subprocess, json, os, re, shutil, textwrap
from dataclasses import dataclass

_CLAUDE_BIN_CANDIDATES = [
    "/Users/chakrivijayarao/.vscode/extensions/anthropic.claude-code-2.1.126-darwin-arm64/resources/native-binary/claude",
    "/Users/chakrivijayarao/.vscode/extensions/anthropic.claude-code-2.1.123-darwin-arm64/resources/native-binary/claude",
    "/Users/chakrivijayarao/.bun/install/cache/@anthropic-ai/claude-agent-sdk-darwin-arm64@0.2.117@@@1/claude",
]


def _find_binary() -> str:
    # Check PATH first
    path_claude = shutil.which("claude")
    if path_claude:
        return path_claude
    for candidate in _CLAUDE_BIN_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Claude Code binary not found. Checked PATH and known install locations.\n"
        "Set CLAUDE_BIN env var to override."
    )


_CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "") or _find_binary()


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Message:
    content: list[_TextBlock]
    stop_reason: str = "end_turn"
    model: str = "claude-code"


class _Messages:
    def create(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> _Message:
        # Build the prompt from messages
        parts = []
        if system:
            parts.append(f"[System]\n{system}\n")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            parts.append(f"[{role.upper()}]\n{content}")
        prompt = "\n\n".join(parts)

        # Call the Claude Code binary
        result = subprocess.run(
            [_CLAUDE_BIN, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:200]
            raise RuntimeError(f"Claude Code binary failed (exit {result.returncode}): {stderr}")

        text = result.stdout.strip()
        return _Message(content=[_TextBlock(text=text)])


class ClaudeCodeClient:
    """
    Drop-in replacement for anthropic.Anthropic() that routes calls through
    the Claude Code binary already running in this session.

    The model parameter passed to .messages.create() is ignored —
    all calls go to Claude Code (Sonnet 4.6 in this session).
    """

    def __init__(self):
        # Verify binary works
        try:
            r = subprocess.run(
                [_CLAUDE_BIN, "--version"], capture_output=True, text=True, timeout=10
            )
            self._version = r.stdout.strip()
        except Exception as e:
            raise RuntimeError(f"Claude Code binary test failed: {e}")

        self.messages = _Messages()

    def __repr__(self):
        return f"ClaudeCodeClient(binary={_CLAUDE_BIN!r}, version={self._version!r})"


if __name__ == "__main__":
    client = ClaudeCodeClient()
    print(f"Client: {client}")
    resp = client.messages.create(
        model="any",
        max_tokens=100,
        messages=[{"role": "user", "content": "Reply with exactly: CREDENCE_CLIENT_OK"}],
    )
    print(f"Test response: {resp.content[0].text!r}")

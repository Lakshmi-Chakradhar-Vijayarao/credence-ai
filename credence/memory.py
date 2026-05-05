"""
credence/memory.py
==================
Cross-session epistemic memory for Credence.

Claude Code forgets everything when you close a session. Including what it was UNSURE about.
This module persists epistemic state — specifically unverified uncertain constraints — across
session boundaries so the next session starts epistemically aware.

Usage:
    from credence.memory import CredenceMemory
    from credence.registry import CredenceRegistry

    reg = CredenceRegistry()
    mem = CredenceMemory(reg)

    # At the END of a session — snapshot what's still uncertain
    snapshot = mem.snapshot("session-abc", project="my-api-project")

    # At the START of a new session — inject prior uncertainties
    ctx = mem.recall_and_inject("my-api-project", new_session_id="session-xyz")
    # ctx.system_block is ready to prepend to your system prompt
    # ctx.injected_count is how many memories were loaded
"""

from __future__ import annotations

from dataclasses import dataclass, field

from credence.registry import CredenceRegistry


@dataclass
class MemorySnapshot:
    """Result of snapshot() — what was saved from a session."""
    project_id: str
    session_id: str
    saved_count: int
    items: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.saved_count == 0:
            return f"[Memory] No unverified constraints in session '{self.session_id}' to snapshot."
        lines = [f"[Memory] Snapshotted {self.saved_count} unverified constraint(s) → project '{self.project_id}'"]
        for item in self.items[:5]:
            zone = item.get("zone", "?")
            content = item.get("content", "")[:80]
            lines.append(f"  ⚠ [{zone}] {content}")
        if self.saved_count > 5:
            lines.append(f"  … and {self.saved_count - 5} more")
        return "\n".join(lines)


@dataclass
class MemoryRecall:
    """Result of recall_and_inject() — what was loaded into the new session."""
    project_id: str
    new_session_id: str
    injected_count: int
    items: list[dict] = field(default_factory=list)
    system_block: str = ""

    def is_empty(self) -> bool:
        return self.injected_count == 0


class CredenceMemory:
    """
    Cross-session epistemic memory layer.

    Wraps CredenceRegistry to provide project-scoped persistent uncertain constraints
    that survive session rotation. This is what makes Credence unique vs all other
    memory tools: it tracks not just WHAT you told Claude, but WHETHER it was verified.

    A Zep or Mem0 memory node says "rate limit = 50 req/min".
    A Credence memory node says "rate limit = 50 req/min — UNVERIFIED, j=0.28, session s1".
    The epistemic provenance travels with the fact.
    """

    # Maximum memories to inject (avoid bloating system prompt)
    _MAX_INJECT = 8

    def __init__(self, registry: CredenceRegistry):
        self._reg = registry

    def snapshot(self, session_id: str, project: str) -> MemorySnapshot:
        """
        Save all unverified constraints from session_id to the project memory store.

        Call this at the END of a session (PostToolUse hook, explicit MCP call, or on exit).
        Idempotent: safe to call multiple times for the same session.
        """
        saved = self._reg.snapshot_to_project(session_id=session_id, project_id=project)
        return MemorySnapshot(
            project_id=project,
            session_id=session_id,
            saved_count=len(saved),
            items=saved,
        )

    def recall_and_inject(
        self,
        project: str,
        new_session_id: str,
        context_hint: str = "",
    ) -> MemoryRecall:
        """
        Inject project memories into new_session_id and return a formatted system block.

        Call this at the START of a new session. The returned system_block should be
        prepended to the ContextManager's system_prompt so the Truth Buffer picks up
        the injected constraints on turn 1.

        context_hint: optional free text to filter relevant memories (keyword match).
                      Leave empty to inject all project memories.
        """
        # Inject memories into the new session's registry
        injected_ids = self._reg.inject_memories_into_session(
            project_id=project,
            new_session_id=new_session_id,
        )

        # Fetch the injected items (they're now in the registry under new_session_id)
        all_memories = self._reg.recall_project_memories(project)

        # Filter by context_hint if provided
        if context_hint:
            hint_words = set(context_hint.lower().split())
            filtered = []
            for m in all_memories:
                content_words = set(m["content"].lower().split())
                if hint_words & content_words:
                    filtered.append(m)
            memories_to_show = filtered if filtered else all_memories
        else:
            memories_to_show = all_memories

        memories_to_show = memories_to_show[:self._MAX_INJECT]

        system_block = self._format_system_block(memories_to_show, project)

        return MemoryRecall(
            project_id=project,
            new_session_id=new_session_id,
            injected_count=len(injected_ids),
            items=memories_to_show,
            system_block=system_block,
        )

    def project_status(self, project: str) -> dict:
        """
        Summary of all epistemic memory for a project.
        Returns counts of verified vs unverified constraints across all sessions.
        Excludes cross_session_memory copies — counts originals only.
        """
        all_constraints = [
            c for c in self._reg.get_all_project_constraints(project)
            if c.get("source") != "cross_session_memory"
        ]
        verified = [c for c in all_constraints if c.get("verified") == 1]
        unverified = [c for c in all_constraints if c.get("verified") == 0
                      and c.get("validation_status") != "disputed"]
        disputed = [c for c in all_constraints if c.get("validation_status") == "disputed"]

        return {
            "project_id": project,
            "total_memories": len(all_constraints),
            "verified_count": len(verified),
            "unverified_count": len(unverified),
            "disputed_count": len(disputed),
            "epistemic_debt": len(unverified) + len(disputed),
            "unverified": [
                {
                    "constraint_id": c["constraint_id"],
                    "content": c["content"],
                    "zone": c["zone"],
                    "j_score": c["j_score"],
                    "session_id": c["session_id"],
                    "created_at": c["created_at"],
                }
                for c in unverified[:10]
            ],
        }

    def _format_system_block(self, memories: list[dict], project: str) -> str:
        """
        Format memory items as a system prompt injection block.
        The block is designed to work alongside the Truth Buffer.
        """
        if not memories:
            return ""

        lines = [
            f"EPISTEMIC MEMORY — PROJECT '{project}' (cross-session unverified constraints):",
            "These were stated in previous sessions and have NOT been verified.",
            "Treat them as working assumptions, not confirmed facts. Flag uncertainty when recalled.",
            "",
        ]
        for m in memories:
            zone = m.get("zone", "LOW")
            content = m.get("content", "")
            if len(content) > 80:
                content = content[:77] + "…"
            j = m.get("j_score", 0.0)
            session = m.get("session_id", "unknown")[:12]
            lines.append(f"  ⚠ [{zone}, conf={j:.2f}, from={session}] {content}")

        lines.append("")
        lines.append("When referring to these values, always state they are unverified.")

        return "\n".join(lines)

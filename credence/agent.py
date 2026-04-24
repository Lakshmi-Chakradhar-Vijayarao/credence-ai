"""
credence/agent.py
=============
CredenceAgent: Claude Managed Agent with confidence-adaptive memory.

Designed for long-running, multi-step tasks that would otherwise exhaust
Claude's context window — document analysis, research Q&A, iterative
reasoning chains. Credence memory management keeps the agent running
efficiently across all turns without hitting context limits.

Pattern: Claude as both the reasoner AND the memory manager.
  - Opus 4.7 answers each sub-task
  - Opus 4.7 compresses its own history when confident
  - The agent survives contexts that would break a naive implementation

Compatible with Claude Managed Agents infrastructure for production
deployment of long-running tasks.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from .confidence_proxy import CredenceProxy
from .context_manager import ContextManager, _cost


# ---------------------------------------------------------------------------
# Task and result types
# ---------------------------------------------------------------------------

@dataclass
class SubTaskResult:
    question:     str
    answer:       str
    j_score:      float
    zone:         str
    decision:     str
    tokens_in:    int
    tokens_out:   int
    tokens_saved: int
    elapsed_ms:   float


@dataclass
class AgentResult:
    task:         str
    sub_results:  list[SubTaskResult] = field(default_factory=list)
    final_report: str = ""
    total_tokens_used:  int   = 0
    total_tokens_saved: int   = 0
    total_cost_usd:     float = 0.0
    total_savings_usd:  float = 0.0
    compression_ratio:  float = 0.0
    elapsed_sec:        float = 0.0

    @property
    def summary(self) -> str:
        pct = self.compression_ratio * 100
        return (
            f"Completed {len(self.sub_results)} sub-tasks | "
            f"Tokens used: {self.total_tokens_used:,} | "
            f"Tokens saved: {self.total_tokens_saved:,} ({pct:.0f}%) | "
            f"Cost: ${self.total_cost_usd:.4f} | "
            f"Saved: ${self.total_savings_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# CredenceAgent
# ---------------------------------------------------------------------------

class CredenceAgent:
    """
    Agentic task runner with Credence memory management.

    Two modes:
      document_qa  — split a long document into chunks, answer questions
                     across all chunks with shared Credence memory
      research     — iterative research loop: pose questions, synthesize
                     answers, maintain coherent context under Credence

    Usage:
        agent = CredenceAgent()
        result = agent.document_qa(
            document=long_text,
            questions=["What is X?", "How does Y work?", ...],
        )
        print(result.summary)
        print(result.final_report)
    """

    MODEL          = "claude-opus-4-7"
    CHUNK_TOKENS   = 1500   # approximate tokens per document chunk

    def __init__(
        self,
        api_key:    Optional[str] = None,
        theta_high: float = 0.65,
        theta_low:  float = 0.35,
        max_tokens: int   = 512,
        on_turn:    Optional[Callable] = None,   # progress callback
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("pip install anthropic")

        self.client  = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.proxy   = CredenceProxy(theta_high, theta_low)
        self.on_turn = on_turn   # called after each sub-task with SubTaskResult

        self._mgr = ContextManager(
            api_key    = api_key,
            theta_high = theta_high,
            theta_low  = theta_low,
            max_tokens = max_tokens,
            system_prompt = (
                "You are a precise research assistant working through a "
                "long document. Answer each question using only information "
                "from the document. Be concise and specific."
            ),
        )

    # ------------------------------------------------------------------
    # Document Q&A mode
    # ------------------------------------------------------------------

    def document_qa(
        self,
        document:  str,
        questions: list[str],
    ) -> AgentResult:
        """
        Answer a list of questions about a long document.
        Credence manages context across all turns — the agent doesn't hit
        context limits even with 20+ questions over a long document.
        """
        t0 = time.perf_counter()
        self._mgr.reset()
        result = AgentResult(task=f"Document Q&A ({len(questions)} questions)")

        # Ingest document in chunks
        chunks   = self._chunk(document)
        n_chunks = len(chunks)

        # Prime context with document overview
        intro = (
            f"I'm going to help you analyze a document. "
            f"The document has been split into {n_chunks} section(s). "
            f"I'll feed each section before answering questions about it."
        )
        self._mgr.chat(intro)

        # Feed document chunks
        for i, chunk in enumerate(chunks):
            self._mgr.chat(
                f"[Document section {i+1}/{n_chunks}]:\n{chunk}\n"
                f"Acknowledge that you've read this section."
            )

        # Answer each question with Credence managing memory
        for q in questions:
            t_q = time.perf_counter()
            turn = self._mgr.chat(q)
            elapsed = (time.perf_counter() - t_q) * 1000

            sub = SubTaskResult(
                question     = q,
                answer       = turn.response,
                j_score      = turn.j_score,
                zone         = turn.zone,
                decision     = turn.decision,
                tokens_in    = turn.tokens_in,
                tokens_out   = turn.tokens_out,
                tokens_saved = turn.tokens_saved,
                elapsed_ms   = round(elapsed, 1),
            )
            result.sub_results.append(sub)

            if self.on_turn:
                self.on_turn(sub)

        # Final synthesis
        qs_text = "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in result.sub_results)
        synthesis = self._mgr.chat(
            f"Based on all the questions and answers above, write a concise "
            f"executive summary (3-5 bullet points) of the key findings:\n\n{qs_text}"
        )
        result.final_report = synthesis.response

        # Aggregate stats
        s = self._mgr.stats
        result.total_tokens_used  = s.total_tokens_in + s.total_tokens_out
        result.total_tokens_saved = s.total_tokens_saved
        result.total_cost_usd     = round(s.total_cost_usd, 4)
        result.total_savings_usd  = round(s.total_savings_usd, 4)
        result.compression_ratio  = round(s.compression_ratio, 3)
        result.elapsed_sec        = round(time.perf_counter() - t0, 2)

        return result

    # ------------------------------------------------------------------
    # Research loop mode
    # ------------------------------------------------------------------

    def research(
        self,
        topic:      str,
        questions:  list[str],
        background: Optional[str] = None,
    ) -> AgentResult:
        """
        Iterative research loop. Ask a series of questions on a topic,
        building understanding turn-by-turn with Credence memory.
        """
        t0 = time.perf_counter()
        self._mgr.reset()
        self._mgr.system_prompt = (
            "You are a knowledgeable research assistant. Answer questions "
            "accurately and concisely. Be specific when you know the answer; "
            "express genuine uncertainty when you don't."
        )
        result = AgentResult(task=f"Research: {topic}")

        if background:
            self._mgr.chat(f"Research topic: {topic}\nBackground: {background}")

        for q in questions:
            t_q  = time.perf_counter()
            turn = self._mgr.chat(q)
            elapsed = (time.perf_counter() - t_q) * 1000

            sub = SubTaskResult(
                question     = q,
                answer       = turn.response,
                j_score      = turn.j_score,
                zone         = turn.zone,
                decision     = turn.decision,
                tokens_in    = turn.tokens_in,
                tokens_out   = turn.tokens_out,
                tokens_saved = turn.tokens_saved,
                elapsed_ms   = round(elapsed, 1),
            )
            result.sub_results.append(sub)

            if self.on_turn:
                self.on_turn(sub)

        # Synthesis
        synth = self._mgr.chat(
            "Synthesize a concise 3-5 bullet summary of everything we've "
            "established in this research session."
        )
        result.final_report = synth.response

        s = self._mgr.stats
        result.total_tokens_used  = s.total_tokens_in + s.total_tokens_out
        result.total_tokens_saved = s.total_tokens_saved
        result.total_cost_usd     = round(s.total_cost_usd, 4)
        result.total_savings_usd  = round(s.total_savings_usd, 4)
        result.compression_ratio  = round(s.compression_ratio, 3)
        result.elapsed_sec        = round(time.perf_counter() - t0, 2)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk(self, text: str, chars_per_chunk: int = 4000) -> list[str]:
        """Split text into chunks of approximately equal size."""
        paragraphs = text.split("\n\n")
        chunks, current, current_len = [], [], 0

        for para in paragraphs:
            if current_len + len(para) > chars_per_chunk and current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks or [text]

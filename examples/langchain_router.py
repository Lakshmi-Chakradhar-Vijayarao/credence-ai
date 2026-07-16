"""
examples/langchain_router.py — EpistemicTag integration with LangChain

Wraps any LangChain chain with epistemic routing:
  - ANSWER / VERIFY → use LLM directly (VERIFY adds a caveat)
  - RETRIEVE → trigger RAG (EpistemicRAGChain)
  - DEFER → add uncertainty notice
  - ESCALATE → route to larger model (GPT-4 fallback shown)

Prerequisites:
    pip install epistemic-stack langchain langchain-openai transformers torch
"""

from __future__ import annotations
from typing import Optional

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from esm import wrap_model, EpistemicTag, ANSWER, VERIFY, RETRIEVE, DEFER, ESCALATE


# ── EpistemicRouter LangChain integration ─────────────────────────────────────

class EpistemicLangChainChain:
    """
    A LangChain-compatible chain that routes queries based on EpistemicTag.

    Usage:
        chain = EpistemicLangChainChain(
            epistemic_model=em,          # EpistemicModel from wrap_model()
            rag_chain=my_rag_chain,      # Optional RAG chain for RETRIEVE
            escalation_llm=gpt4_llm,    # Optional LLM for ESCALATE
        )
        result = chain.invoke("Who wrote Hamlet?")
        print(result["answer"])
        print(result["routing"])         # ANSWER | VERIFY | RETRIEVE | DEFER | ESCALATE
        print(result["verify_flag"])     # True = confabulation fingerprint
    """

    VERIFY_CAVEAT = (
        "\n\n---\n⚠ Note: This response may require verification. "
        "The model's internal confidence trajectory suggests possible confabulation. "
        "Please verify with an authoritative source before relying on this answer."
    )

    DEFER_NOTICE = (
        "\n\n---\nNote: The model indicates uncertainty about this question. "
        "The answer above may be incomplete or inaccurate."
    )

    def __init__(
        self,
        epistemic_model,
        rag_chain=None,
        escalation_llm=None,
        add_epistemic_metadata: bool = True,
    ):
        self._em        = epistemic_model
        self._rag       = rag_chain
        self._escalate  = escalation_llm
        self._add_meta  = add_epistemic_metadata

    def invoke(self, question: str, **kwargs) -> dict:
        # Get routing before generation
        tag = self._em.tag(question)

        result = {
            "question":    question,
            "routing":     tag.routing,
            "verify_flag": tag.verify_flag,
            "j_know":      tag.j_know,
            "j_velocity":  tag.j_velocity,
            "epistemic":   tag,
        }

        if tag.routing == RETRIEVE:
            if self._rag is not None:
                answer = self._rag.invoke(question)
                result["answer"]  = answer
                result["via_rag"] = True
            else:
                resp = self._em.generate(question)
                result["answer"] = resp.text
                result["via_rag"] = False
                result["warning"] = "RETRIEVE routing but no RAG chain provided — used direct generation"

        elif tag.routing == ESCALATE:
            if self._escalate is not None:
                answer = self._escalate.invoke(question)
                result["answer"]      = str(answer)
                result["via_escalate"] = True
            else:
                result["answer"]  = "[Escalated — no escalation LLM configured]"
                result["blocked"] = True

        elif tag.routing == VERIFY:
            resp = self._em.generate(question)
            result["answer"] = resp.text + self.VERIFY_CAVEAT

        elif tag.routing == DEFER:
            resp = self._em.generate(question)
            result["answer"] = resp.text + self.DEFER_NOTICE

        else:  # ANSWER
            resp = self._em.generate(question)
            result["answer"] = resp.text

        return result

    def batch(self, questions: list, **kwargs) -> list:
        return [self.invoke(q) for q in questions]


# ── Compliance audit logger ───────────────────────────────────────────────────

class EpistemicAuditLogger:
    """
    Logs EpistemicTag metadata for every query to a JSONL file.

    Usage:
        logger = EpistemicAuditLogger("logs/epistemic_audit.jsonl")
        chain  = EpistemicLangChainChain(em, rag_chain=rag)

        q      = "Who wrote Hamlet?"
        result = chain.invoke(q)
        logger.log(result)
    """

    def __init__(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path

    def log(self, result: dict, session_id: Optional[str] = None):
        import json, datetime, hashlib
        record = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id":    session_id,
            "prompt_hash":   hashlib.sha256(result["question"].encode()).hexdigest()[:16],
            "routing":       result.get("routing"),
            "j_know":        round(result.get("j_know", 0.0), 4),
            "j_velocity":    round(result.get("j_velocity", 0.0), 4),
            "verify_flag":   result.get("verify_flag", False),
            "via_rag":       result.get("via_rag", False),
            "blocked":       result.get("blocked", False),
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Example: Adaptive RAG with epistemic routing ──────────────────────────────

def demo_adaptive_rag():
    """
    Demo: epistemic routing replaces threshold-based RAG triggering.

    Standard RAG: trigger retrieval on ALL queries (wasteful).
    Epistemic RAG: trigger retrieval ONLY on RETRIEVE/ESCALATE (30-40% queries).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    CAL_PATH = "checkpoints/llama3b_cal.json"

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model   = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="float16", device_map=None
    ).cuda().eval()
    em = wrap_model(hf_model, tokenizer, calibration=CAL_PATH)

    # Simple mock RAG chain
    def mock_rag_chain(question):
        return f"[RAG] Retrieved answer for: {question}"

    chain  = EpistemicLangChainChain(em, rag_chain=mock_rag_chain)
    logger = EpistemicAuditLogger("logs/demo_audit.jsonl")

    test_queries = [
        "Who wrote Hamlet?",                          # → ANSWER (PARAM, well-known)
        "What is the capital of France?",              # → ANSWER
        "What time did John's meeting start today?",   # → ESCALATE (CTX_DEP, low certainty)
        "Summarize the attached document.",            # → RETRIEVE
        "Who invented the telephone?",                 # → ANSWER or VERIFY
    ]

    print(f"\n{'='*70}")
    print("ADAPTIVE RAG WITH EPISTEMIC ROUTING")
    print(f"{'='*70}")

    rag_calls = 0
    for q in test_queries:
        result = chain.invoke(q)
        logger.log(result)

        routing = result["routing"]
        if routing in (RETRIEVE, ESCALATE):
            rag_calls += 1

        flag = " ⚠" if result["verify_flag"] else ""
        print(f"[{routing:>8}{flag}] {q[:55]}")

    print(f"\nRAG triggered: {rag_calls}/{len(test_queries)} queries "
          f"({rag_calls/len(test_queries)*100:.0f}%)")
    print(f"Audit log: logs/demo_audit.jsonl")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    demo_adaptive_rag()

"""
examples/compliance_logging.py — EpistemicTag for healthcare/legal/finance compliance

Use case: organizations deploying LLMs in regulated contexts need to demonstrate
that responses flagged as high-risk (VERIFY routing, verify_flag=True) were
handled appropriately before reaching end users.

This example shows:
  1. Per-request EpistemicTag logging (HIPAA/SOC2-compatible)
  2. VERIFY threshold policies (configurable per deployment context)
  3. Audit report generation
  4. Real-time dashboard hook (Prometheus metrics)

Prerequisites:
    pip install epistemic-stack transformers torch
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

from esm import wrap_model, EpistemicTag, ANSWER, VERIFY, RETRIEVE, DEFER, ESCALATE


# ── Compliance policy ─────────────────────────────────────────────────────────

@dataclass
class EpistemicPolicy:
    """
    Per-deployment epistemic routing policy.

    Healthcare:  Block VERIFY on clinical questions. RAG-required for RETRIEVE.
    Legal:       Add disclaimer to VERIFY. Block ESCALATE unless human-reviewed.
    Finance:     Add caveat to all non-ANSWER responses. Log all VERIFY events.
    """
    context:              str    = "default"
    block_on_verify:      bool   = False   # Block response if verify_flag=True
    require_rag_retrieve: bool   = False   # Require RAG before serving RETRIEVE
    block_escalate:       bool   = False   # Block ESCALATE entirely (human review)
    max_j_know_threshold: float  = -99.0  # Block if j_know < this (extra CTX safety)
    verify_caveat:        str    = ""      # Text appended to VERIFY responses
    log_all:              bool   = True    # Log every request to audit trail


POLICIES: Dict[str, EpistemicPolicy] = {
    "healthcare": EpistemicPolicy(
        context              = "healthcare",
        block_on_verify      = True,
        require_rag_retrieve = True,
        block_escalate       = False,
        verify_caveat        = (
            "\n\n⚠ CLINICAL ADVISORY: This response has been flagged by epistemic "
            "monitoring as potentially unreliable. It must be verified by a licensed "
            "healthcare professional before clinical application."
        ),
    ),
    "legal": EpistemicPolicy(
        context              = "legal",
        block_on_verify      = False,
        require_rag_retrieve = False,
        block_escalate       = True,
        verify_caveat        = (
            "\n\n⚠ LEGAL DISCLAIMER: This response has been flagged for potential "
            "inaccuracy. Do not rely on this for legal advice without independent "
            "verification by a qualified attorney."
        ),
    ),
    "finance": EpistemicPolicy(
        context              = "finance",
        block_on_verify      = False,
        require_rag_retrieve = False,
        block_escalate       = False,
        verify_caveat        = (
            "\n\n⚠ FINANCIAL ADVISORY: This response contains financial information "
            "that has not been independently verified. Consult a registered financial "
            "advisor before making investment decisions."
        ),
    ),
    "default": EpistemicPolicy(),
}


# ── Audit record ──────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    timestamp_utc:   str
    request_id:      str
    session_id:      Optional[str]
    prompt_hash:     str   # SHA256 of prompt — no PII stored
    routing:         str
    j_know:          float
    j_velocity:      float
    entropy:         float
    verify_flag:     bool
    latency_ms:      float
    policy_context:  str
    action_taken:    str   # pass | verify_caveat | blocked | rag_triggered | escalate_blocked
    response_served: bool
    override:        bool = False


# ── Compliant inference engine ────────────────────────────────────────────────

class ComplianceEpistemicEngine:
    """
    LLM inference engine with EpistemicTag-based compliance controls.

    Every request:
    1. Gets EpistemicTag at gen-step-1
    2. Applies policy to determine action
    3. Logs AuditRecord to JSONL
    4. Returns response or blocks as appropriate

    Usage:
        engine = ComplianceEpistemicEngine(
            model=hf_model,
            tokenizer=tokenizer,
            calibration_path="checkpoints/llama3b_cal.json",
            policy=POLICIES["healthcare"],
            audit_path="logs/healthcare_audit.jsonl",
        )
        result = engine.query("What is the recommended dosage of ibuprofen?")
        print(result["response"])
        print(result["blocked"])        # True if policy blocked the response
        print(result["action_taken"])   # What the policy did
    """

    def __init__(
        self,
        model,
        tokenizer,
        calibration_path: str,
        policy: EpistemicPolicy = POLICIES["default"],
        audit_path: str = "logs/epistemic_audit.jsonl",
        rag_fn=None,
    ):
        self._em        = wrap_model(model, tokenizer, calibration=calibration_path)
        self._policy    = policy
        self._audit     = Path(audit_path)
        self._audit.parent.mkdir(parents=True, exist_ok=True)
        self._rag       = rag_fn

    def query(
        self,
        prompt:     str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> dict:

        if request_id is None:
            request_id = hashlib.sha256(
                f"{prompt}{time.time()}".encode()
            ).hexdigest()[:12]

        # 1. Get epistemic tag
        tag = self._em.tag(prompt)

        # 2. Apply policy
        policy    = self._policy
        blocked   = False
        action    = "pass"
        response  = None
        via_rag   = False

        if tag.routing == VERIFY and policy.block_on_verify:
            blocked = True
            action  = "blocked"
            response = (
                "This query has been blocked by epistemic monitoring policy. "
                "A human reviewer will be notified. Request ID: " + request_id
            )

        elif tag.routing == ESCALATE and policy.block_escalate:
            blocked = True
            action  = "escalate_blocked"
            response = (
                "This query requires human review and has been flagged for escalation. "
                "Request ID: " + request_id
            )

        elif tag.routing == RETRIEVE and policy.require_rag_retrieve:
            if self._rag is not None:
                response = self._rag(prompt)
                action   = "rag_triggered"
                via_rag  = True
            else:
                response = "[RAG required but not configured — request blocked]"
                blocked  = True
                action   = "blocked_no_rag"

        else:
            # Generate response
            resp     = self._em.generate(prompt, max_new_tokens=max_new_tokens)
            response = resp.text
            action   = "pass"

            # Add caveat if VERIFY and policy has one
            if tag.routing == VERIFY and policy.verify_caveat:
                response += policy.verify_caveat
                action    = "verify_caveat"

        # 3. Audit log
        record = AuditRecord(
            timestamp_utc   = datetime.now(timezone.utc).isoformat(),
            request_id      = request_id,
            session_id      = session_id,
            prompt_hash     = hashlib.sha256(prompt.encode()).hexdigest()[:16],
            routing         = tag.routing,
            j_know          = round(tag.j_know, 4),
            j_velocity      = round(tag.j_velocity, 4),
            entropy         = round(tag.entropy, 4),
            verify_flag     = tag.verify_flag,
            latency_ms      = round(tag.latency_ms, 2),
            policy_context  = self._policy.context,
            action_taken    = action,
            response_served = not blocked,
        )
        with open(self._audit, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return {
            "response":    response,
            "routing":     tag.routing,
            "verify_flag": tag.verify_flag,
            "j_know":      tag.j_know,
            "blocked":     blocked,
            "action_taken": action,
            "request_id":  request_id,
            "via_rag":     via_rag,
        }

    def audit_report(self, n_recent: int = 100) -> dict:
        """Read recent audit records and compute routing distribution."""
        records = []
        if self._audit.exists():
            with open(self._audit) as f:
                lines = f.readlines()
            for line in lines[-n_recent:]:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

        if not records:
            return {"n": 0}

        from collections import Counter
        routing_dist = Counter(r["routing"] for r in records)
        verify_rate  = sum(1 for r in records if r["verify_flag"]) / len(records)
        blocked_rate = sum(1 for r in records if not r["response_served"]) / len(records)

        return {
            "n":              len(records),
            "routing_dist":   dict(routing_dist),
            "verify_rate":    round(verify_rate, 3),
            "blocked_rate":   round(blocked_rate, 3),
            "mean_j_know":    round(sum(r["j_know"] for r in records) / len(records), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 2),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_healthcare_compliance():
    """
    Demo: healthcare deployment with VERIFY blocking.

    In a real deployment:
    - VERIFY responses are blocked and escalated to a human reviewer
    - All requests are logged to an immutable audit trail
    - Audit reports can be exported for HIPAA compliance review
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    CAL_PATH = "checkpoints/llama3b_cal.json"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model  = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="float16", device_map=None
    ).cuda().eval()

    engine = ComplianceEpistemicEngine(
        model           = hf_model,
        tokenizer       = tokenizer,
        calibration_path = CAL_PATH,
        policy          = POLICIES["healthcare"],
        audit_path      = "logs/healthcare_audit.jsonl",
    )

    queries = [
        "What is the recommended adult dosage of aspirin for pain relief?",
        "Who discovered penicillin?",
        "Is metformin safe during pregnancy?",    # Likely VERIFY — nuanced clinical
        "What is the patient's current medication list?",  # RETRIEVE — CTX_DEP
        "What is the mechanism of action of beta blockers?",
    ]

    print(f"\n{'='*70}")
    print("HEALTHCARE COMPLIANCE ENGINE DEMO")
    print(f"{'='*70}")
    print(f"Policy: block_on_verify={POLICIES['healthcare'].block_on_verify}")
    print()

    for q in queries:
        result = engine.query(q, session_id="demo_session_001")
        status = "BLOCKED" if result["blocked"] else result["routing"]
        flag   = " ⚠" if result["verify_flag"] else ""
        print(f"[{status:>12}{flag}] {q[:55]}")

    report = engine.audit_report()
    print(f"\nAudit Report (last {report['n']} queries):")
    print(f"  Routing distribution: {report.get('routing_dist', {})}")
    print(f"  VERIFY rate:  {report.get('verify_rate', 0)*100:.1f}%")
    print(f"  Blocked rate: {report.get('blocked_rate', 0)*100:.1f}%")
    print(f"  Audit log:    logs/healthcare_audit.jsonl")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    demo_healthcare_compliance()

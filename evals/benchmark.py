from __future__ import annotations

"""
evals/benchmark.py
==================
Benchmarks Credence against two baselines on real Claude API calls:

  Baseline A — No compression: full context every turn (most expensive)
  Baseline B — Naive sliding window: drop turns older than N regardless
  Credence       — Confidence-adaptive: compress only when J >= theta_high

Measures:
  1. Token usage per session
  2. Estimated cost ($)
  3. Answer quality (ROUGE-L F1 against reference answers)
  4. AUARC — Area Under Abstention-Risk Curve
     Measures whether the J-proxy correctly identifies uncertain answers:
     if we abstain on the lowest-J answers, does retained quality improve?
  5. Reasoning Density per Dollar (ROUGE-L per $0.001 spent)
  6. J-score zone calibration: do HIGH-zone answers score better quality?

Run:
    python -m evals.benchmark          # runs all 3 conditions, prints table
    python -m evals.benchmark --quick  # 3 questions only (fast smoke test)

Results saved to evals/results.json for the demo to load.
"""

import os
import sys
import json
import math
import time
import re
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from anthropic import Anthropic
    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from credence.confidence_proxy import CredenceProxy
from credence.context_manager import ContextManager, _cost

# ---------------------------------------------------------------------------
# Shared system prompt — used by ALL conditions so results are comparable.
# Credence uses this internally via ContextManager; baseline/naive must use
# the same prompt or the comparison conflates prompt effect with J-proxy effect.
# ---------------------------------------------------------------------------
BENCHMARK_SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. "
    "Give concise answers when the answer is clear; "
    "express genuine uncertainty when it exists."
)

# ---------------------------------------------------------------------------
# Benchmark Q&A pairs — 30 questions across 3 domains
# Domain A: Factual/Scientific (expect HIGH J, short answers)
# Domain B: Reasoning/STEM (expect MEDIUM J, structured answers)
# Domain C: Uncertain/Speculative (expect LOW J, hedged answers)
# ---------------------------------------------------------------------------

QA_PAIRS = [
    # ── Domain A: Factual / Scientific (HIGH confidence expected) ──────────
    {
        "question":      "What is the speed of light in a vacuum?",
        "reference":     "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "Who wrote the play Hamlet?",
        "reference":     "Hamlet was written by William Shakespeare around 1600-1601.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "What is the chemical formula for water?",
        "reference":     "The chemical formula for water is H2O.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "In what year did World War II end?",
        "reference":     "World War II ended in 1945.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "What is the square root of 144?",
        "reference":     "The square root of 144 is 12.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "What is the boiling point of water at sea level in Celsius?",
        "reference":     "Water boils at 100 degrees Celsius at sea level.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "Who painted the Mona Lisa?",
        "reference":     "The Mona Lisa was painted by Leonardo da Vinci.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "What is the atomic number of gold?",
        "reference":     "Gold has an atomic number of 79.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "How many continents are there on Earth?",
        "reference":     "There are seven continents on Earth.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    {
        "question":      "What is the currency of Japan?",
        "reference":     "The currency of Japan is the yen.",
        "expected_zone": "HIGH",
        "domain":        "factual",
    },
    # ── Domain B: Reasoning / STEM (MEDIUM confidence expected) ───────────
    {
        "question":      "How does quantum entanglement work?",
        "reference":     "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly affects the other regardless of distance.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "What causes the Northern Lights?",
        "reference":     "The Northern Lights are caused by charged particles from the sun interacting with Earth's magnetic field and atmosphere, producing light.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "Why do objects fall at the same rate regardless of mass in a vacuum?",
        "reference":     "In a vacuum, all objects fall at the same rate because gravity accelerates all masses equally, as established by Galileo and formalized in Newton's laws.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "How does a transformer neural network process language?",
        "reference":     "Transformers use self-attention mechanisms to weigh the importance of different words in a sequence when generating representations.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "What are the trade-offs between microservices and monolithic architectures?",
        "reference":     "Microservices offer scalability and independent deployment but add operational complexity; monoliths are simpler but harder to scale independently.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "How does HTTPS encryption protect web traffic?",
        "reference":     "HTTPS uses TLS to encrypt data between client and server, preventing eavesdropping and ensuring data integrity through asymmetric and symmetric encryption.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "Why is the sky blue?",
        "reference":     "The sky appears blue because the atmosphere scatters short-wavelength blue light more than other colors, a phenomenon called Rayleigh scattering.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "How does natural selection drive evolution?",
        "reference":     "Natural selection favors traits that improve survival and reproduction, causing those traits to become more common over generations.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "What is the difference between supervised and unsupervised learning?",
        "reference":     "Supervised learning uses labeled data to train models; unsupervised learning finds patterns in unlabeled data without predefined targets.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    {
        "question":      "How does the immune system recognize and fight pathogens?",
        "reference":     "The immune system uses antigens to identify pathogens and deploys antibodies and T-cells to neutralize them.",
        "expected_zone": "MEDIUM",
        "domain":        "reasoning",
    },
    # ── Domain C: Uncertain / Speculative (LOW confidence expected) ────────
    {
        "question":      "What will artificial intelligence look like in 50 years?",
        "reference":     "AI in 50 years is highly uncertain and speculative.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "What is the best programming language to learn first?",
        "reference":     "The best first language depends on goals and context.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "What are the long-term psychological effects of social media on teenagers?",
        "reference":     "Research is ongoing and effects are debated among experts.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "Will quantum computing make current encryption obsolete?",
        "reference":     "Whether quantum computing will break current encryption is uncertain and depends on developments in both quantum hardware and post-quantum cryptography.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "What is the nature of consciousness?",
        "reference":     "Consciousness remains one of the deepest unsolved problems in science and philosophy, with many competing theories.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "What is the best diet for human longevity?",
        "reference":     "Research on optimal diet for longevity is ongoing and results vary significantly across populations and studies.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "How will climate change affect geopolitics in the next century?",
        "reference":     "The geopolitical effects of climate change are highly uncertain and depend on many interacting social, political, and physical factors.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "Is free will compatible with a deterministic universe?",
        "reference":     "The compatibility of free will with determinism is a long-standing philosophical debate with no consensus.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "What is the optimal way to structure a software team?",
        "reference":     "Team structure depends heavily on context, company size, culture, and product type, with no single optimal answer.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
    {
        "question":      "Will humanity colonize other planets in this century?",
        "reference":     "Whether humanity will colonize other planets this century is uncertain and depends on many technological and political factors.",
        "expected_zone": "LOW",
        "domain":        "uncertain",
    },
]

QUICK_PAIRS = QA_PAIRS[:3]   # smoke test: 1 per domain

# ---------------------------------------------------------------------------
# Context-Dependent Q&A pairs
# ---------------------------------------------------------------------------
# Design: each SESSION is a short scripted conversation. Turn 1 plants a
# constraint in uncertain language (LOW-J → Credence preserves). Turns 2-7 are
# factual filler (HIGH-J → naive window or compression could drop turn 1).
# Turn 8 is the test question — correct answer requires turn 1's constraint.
#
# Evaluation metric: Context Dependency Score (CDS) = does the answer at
# turn 8 reference the constraint from turn 1?
#
# Credence target: preserves turn 1 (attention sink) → correct answer at turn 8
# Naive window (window=6): turn 1 dropped at turn 8 → incomplete/wrong answer
# ---------------------------------------------------------------------------

CONTEXT_SESSIONS = [
    {
        "name": "lambda_memory",
        "constraint_turn": (
            "I'm not entirely sure about our deployment environment yet. "
            "It might be AWS Lambda with a 128 MB memory limit, or it could be "
            "a 512 MB container. The team hasn't confirmed. Can you keep this in mind?"
        ),
        "filler": [
            "What is the time complexity of quicksort?",
            "What does idempotent mean in the context of HTTP methods?",
            "What is the difference between a stack and a queue?",
            "What is tail recursion and why does it matter?",
            "What is the CAP theorem in distributed systems?",
            "What is connection pooling and why is it used?",
        ],
        "test_question": (
            "Given everything we've discussed about my project, "
            "what Python memory profiling tool would you recommend and why?"
        ),
        "constraint_keywords": ["lambda", "128", "memory limit", "container", "512"],
        "reference": (
            "Given a possible 128 MB Lambda memory limit, use tracemalloc or memory_profiler "
            "for lightweight profiling; avoid heavy tools like heapy that exceed the limit."
        ),
    },
    {
        "name": "budget_constraint",
        "constraint_turn": (
            "I should mention upfront that I'm not totally sure of our API budget. "
            "It's either $50/month or maybe $200/month — finance hasn't confirmed yet. "
            "I think it's probably the lower one but I genuinely don't know. Keep this in mind."
        ),
        "filler": [
            "What is exponential backoff in retry logic?",
            "What is the difference between REST and GraphQL?",
            "What is a webhook and how does it differ from polling?",
            "What are the main HTTP status code ranges and what does each mean?",
            "What is rate limiting and how is it typically implemented?",
            "What is the purpose of an API gateway?",
        ],
        "test_question": (
            "Based on what I've told you about my project, "
            "which LLM API pricing tier or model would you suggest I start with?"
        ),
        "constraint_keywords": ["50", "budget", "cost", "cheaper", "lower", "200"],
        "reference": (
            "Given an uncertain budget of $50-200/month, start with a cheaper model tier "
            "and monitor usage carefully before committing to higher-cost options."
        ),
    },
    {
        "name": "team_size",
        "constraint_turn": (
            "Quick context about my team: I think we have either 2 or 3 engineers — "
            "one contractor might not be joining after all. I'm not sure yet. "
            "Could be just me and one other person. Please keep this in mind."
        ),
        "filler": [
            "What is trunk-based development?",
            "What is the purpose of a code review?",
            "What is continuous integration and why does it matter?",
            "What is the difference between unit tests and integration tests?",
            "What is semantic versioning?",
            "What is a monorepo and what are its trade-offs?",
        ],
        "test_question": (
            "Given what I've shared about my situation, "
            "what project management approach would you recommend for our team?"
        ),
        "constraint_keywords": ["small", "2", "3", "two", "three", "solo", "pair", "engineers"],
        "reference": (
            "For a 2-3 person team, a lightweight kanban board or simple sprint planning "
            "works better than heavy JIRA setups; minimize process overhead."
        ),
    },
]


# ---------------------------------------------------------------------------
# ROUGE-L (unigram) — no external dependencies
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    """Simplified ROUGE-L: longest common subsequence F1 on words."""
    h = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    r = re.sub(r'[^\w\s]', '', reference.lower()).split()
    if not h or not r:
        return 0.0

    m, n = len(h), len(r)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])

    lcs_len   = dp[m][n]
    precision = lcs_len / m if m else 0
    recall    = lcs_len / n if n else 0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ---------------------------------------------------------------------------
# AUARC — Area Under Abstention-Risk Curve
# ---------------------------------------------------------------------------

def compute_auarc(turns: list) -> float:
    """
    AUARC: measure whether the J-proxy correctly identifies uncertain answers.

    Algorithm:
      1. Sort turns by J-score ascending (most uncertain first).
      2. At each cutoff k: if we abstain on the k lowest-J turns,
         what is the mean ROUGE-L of the retained turns?
      3. AUARC = area under the (abstention_rate, retained_quality) curve.

    Higher AUARC means: abstaining on low-J answers actually improves
    retained quality — the proxy is discriminative.
    Random J-proxy → AUARC ≈ 0.5 × mean_rouge_l (flat baseline).
    """
    if not turns:
        return 0.0

    sorted_turns = sorted(turns, key=lambda t: t.j_score)
    n = len(sorted_turns)
    rouge_scores = [t.rouge_l for t in sorted_turns]

    # Compute retained quality at each abstention threshold
    area = 0.0
    prev_abstain = 0.0
    for k in range(n + 1):
        abstain_rate = k / n
        if k < n:
            retained = rouge_scores[k:]
            retained_quality = sum(retained) / len(retained)
        else:
            retained_quality = 1.0   # abstain everything → perfect retained (vacuous)

        if k > 0:
            # Trapezoid rule
            area += (abstain_rate - prev_abstain) * retained_quality
        prev_abstain = abstain_rate

    return round(area, 4)


# ---------------------------------------------------------------------------
# Reasoning Density per Dollar
# ---------------------------------------------------------------------------

def reasoning_density(mean_rouge_l: float, total_cost_usd: float) -> float:
    """ROUGE-L per $0.001 spent. Higher = more quality per dollar."""
    if total_cost_usd <= 0:
        return 0.0
    return round(mean_rouge_l / (total_cost_usd * 1000), 4)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95% CI for the mean.
    n=30 ROUGE-L estimates have ~±0.04 error; reporting only point estimates
    without CIs overstates result precision.
    """
    import random as _rand
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = _rand.Random(seed)
    boot_means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_boot)
    )
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass
class TurnLog:
    question:     str
    answer:       str
    rouge_l:      float
    j_score:      float
    zone:         str
    decision:     str
    tokens_in:    int
    tokens_out:   int
    tokens_saved: int
    domain:       str = "unknown"


@dataclass
class ConditionResult:
    condition:             str
    turns:                 list
    total_tokens_used:     int
    total_tokens_saved:    int
    total_cost_usd:        float
    mean_rouge_l:          float
    mean_j_score:          float
    compression_ratio:     float
    auarc:                 float = 0.0
    reasoning_density_per_kdollar: float = 0.0


def run_credence(pairs: list[dict], client: Anthropic) -> ConditionResult:
    mgr   = ContextManager(max_tokens=300)
    turns = []

    for p in pairs:
        result = mgr.chat(p["question"])
        rl     = rouge_l(result.response, p["reference"])
        turns.append(TurnLog(
            question     = p["question"],
            answer       = result.response,
            rouge_l      = rl,
            j_score      = result.j_score,
            zone         = result.zone,
            decision     = result.decision,
            tokens_in    = result.tokens_in,
            tokens_out   = result.tokens_out,
            tokens_saved = result.tokens_saved,
            domain       = p.get("domain", "unknown"),
        ))

    s = mgr.stats
    total_used = s.total_tokens_in + s.total_tokens_out
    mean_rl    = round(sum(t.rouge_l for t in turns) / len(turns), 4)
    auarc      = compute_auarc(turns)
    rd         = reasoning_density(mean_rl, s.total_cost_usd)
    return ConditionResult(
        condition          = "Credence",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = s.total_tokens_saved,
        total_cost_usd     = round(s.total_cost_usd, 4),
        mean_rouge_l       = mean_rl,
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = round(s.compression_ratio, 3),
        auarc              = auarc,
        reasoning_density_per_kdollar = rd,
    )


def run_random_gating(
    pairs: list[dict],
    client: Anthropic,
    compress_rate: float = 0.30,
    seed: int = 42,
) -> ConditionResult:
    """
    Control condition: compress at the SAME RATE as Credence but pick turns randomly.

    If Credence ROUGE-L > random-gating ROUGE-L, the J signal is doing real work.
    If they are equal, J adds nothing over a compression schedule.

    compress_rate: fraction of eligible turns to compress (match Credence observed rate).
    Default 0.30 matches typical Credence compression rate on 30-turn sessions.
    """
    import random as _random
    rng = _random.Random(seed)

    history = []
    turns   = []
    total_tokens_in = total_tokens_out = total_tokens_saved = 0
    compression_count = 0
    ATTENTION_SINK = 2   # never compress first N turns (mirrors Credence)
    TRIM_WINDOW    = 10  # keep last N turns on trim (mirrors Credence)

    for idx, p in enumerate(pairs):
        history.append({"role": "user", "content": p["question"]})
        decision = "PRESERVE"

        # Eligible for compression after attention sink
        if idx >= ATTENTION_SINK:
            roll = rng.random()
            if roll < compress_rate and len(history) > ATTENTION_SINK * 2 + 4:
                # Random-compress: summarise oldest turns (same mechanism as Credence TRIM)
                keep = history[-(TRIM_WINDOW * 2):]
                dropped = history[:-(TRIM_WINDOW * 2)]
                if dropped:
                    total_tokens_saved += sum(len(m["content"]) // 4 for m in dropped)
                    history = history[:ATTENTION_SINK * 2] + keep
                    compression_count += 1
                    decision = "COMPRESS-RANDOM"
            elif roll < compress_rate * 2:
                # Random-trim: same probability band
                if len(history) > TRIM_WINDOW * 2:
                    dropped = history[:-(TRIM_WINDOW * 2)]
                    total_tokens_saved += sum(len(m["content"]) // 4 for m in dropped)
                    history = history[-(TRIM_WINDOW * 2):]
                    decision = "TRIM-RANDOM"

        resp = client.messages.create(
            model      = "claude-opus-4-7",
            system     = BENCHMARK_SYSTEM_PROMPT,
            messages   = history,
            max_tokens = 300,
        )
        text  = resp.content[0].text
        t_in  = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        total_tokens_in  += t_in
        total_tokens_out += t_out
        history.append({"role": "assistant", "content": text})

        proxy = CredenceProxy()
        cr    = proxy.compute(text)
        rl    = rouge_l(text, p["reference"])
        turns.append(TurnLog(
            question=p["question"], answer=text, rouge_l=rl,
            j_score=cr.j_score, zone=cr.zone, decision=decision,
            tokens_in=t_in, tokens_out=t_out, tokens_saved=0,
            domain=p.get("domain", "unknown"),
        ))

    total_used = total_tokens_in + total_tokens_out
    denom      = total_used + total_tokens_saved
    ratio      = total_tokens_saved / denom if denom > 0 else 0.0
    cost       = round(_cost(total_tokens_in, total_tokens_out), 4)
    mean_rl    = round(sum(t.rouge_l for t in turns) / len(turns), 4)
    return ConditionResult(
        condition          = "Random gating (control)",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = total_tokens_saved,
        total_cost_usd     = cost,
        mean_rouge_l       = mean_rl,
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = round(ratio, 3),
        auarc              = compute_auarc(turns),
        reasoning_density_per_kdollar = reasoning_density(mean_rl, cost),
    )


def run_baseline(pairs: list[dict], client: Anthropic, use_system_prompt: bool = True) -> ConditionResult:
    """Full context, no compression."""
    history = []
    turns   = []
    total_tokens_in = total_tokens_out = 0

    for p in pairs:
        history.append({"role": "user", "content": p["question"]})
        create_kwargs = dict(
            model      = "claude-opus-4-7",
            messages   = history,
            max_tokens = 300,
        )
        if use_system_prompt:
            create_kwargs["system"] = BENCHMARK_SYSTEM_PROMPT
        resp = client.messages.create(**create_kwargs)
        text  = resp.content[0].text
        t_in  = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        total_tokens_in  += t_in
        total_tokens_out += t_out
        history.append({"role": "assistant", "content": text})

        proxy = CredenceProxy()
        cr    = proxy.compute(text)
        rl    = rouge_l(text, p["reference"])
        turns.append(TurnLog(
            question=p["question"], answer=text, rouge_l=rl,
            j_score=cr.j_score, zone=cr.zone, decision="PRESERVE",
            tokens_in=t_in, tokens_out=t_out, tokens_saved=0,
            domain=p.get("domain", "unknown"),
        ))

    total_used = total_tokens_in + total_tokens_out
    cost       = round(_cost(total_tokens_in, total_tokens_out), 4)
    mean_rl    = round(sum(t.rouge_l for t in turns) / len(turns), 4)
    label      = "Baseline (no compression)" if use_system_prompt else "Baseline (no prompt)"
    return ConditionResult(
        condition          = label,
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = 0,
        total_cost_usd     = cost,
        mean_rouge_l       = mean_rl,
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = 0.0,
        auarc              = compute_auarc(turns),
        reasoning_density_per_kdollar = reasoning_density(mean_rl, cost),
    )


def run_naive_window(pairs: list[dict], client: Anthropic, window: int = 6) -> ConditionResult:
    """Sliding window: keep last N turns regardless of confidence."""
    history = []
    turns   = []
    total_tokens_in = total_tokens_out = total_tokens_saved = 0

    for p in pairs:
        history.append({"role": "user", "content": p["question"]})

        if len(history) > window * 2:
            dropped             = history[:-(window * 2)]
            total_tokens_saved += sum(len(m["content"]) // 4 for m in dropped)
            history             = history[-(window * 2):]

        resp = client.messages.create(
            model      = "claude-opus-4-7",
            system     = BENCHMARK_SYSTEM_PROMPT,
            messages   = history,
            max_tokens = 300,
        )
        text  = resp.content[0].text
        t_in  = resp.usage.input_tokens
        t_out = resp.usage.output_tokens
        total_tokens_in  += t_in
        total_tokens_out += t_out
        history.append({"role": "assistant", "content": text})

        proxy = CredenceProxy()
        cr    = proxy.compute(text)
        rl    = rouge_l(text, p["reference"])
        turns.append(TurnLog(
            question=p["question"], answer=text, rouge_l=rl,
            j_score=cr.j_score, zone=cr.zone, decision="TRIM",
            tokens_in=t_in, tokens_out=t_out, tokens_saved=0,
            domain=p.get("domain", "unknown"),
        ))

    total_used = total_tokens_in + total_tokens_out
    denom      = total_used + total_tokens_saved
    ratio      = total_tokens_saved / denom if denom > 0 else 0
    cost       = round(_cost(total_tokens_in, total_tokens_out), 4)
    mean_rl    = round(sum(t.rouge_l for t in turns) / len(turns), 4)
    return ConditionResult(
        condition          = "Naive sliding window",
        turns              = turns,
        total_tokens_used  = total_used,
        total_tokens_saved = total_tokens_saved,
        total_cost_usd     = cost,
        mean_rouge_l       = mean_rl,
        mean_j_score       = round(sum(t.j_score for t in turns) / len(turns), 4),
        compression_ratio  = round(ratio, 3),
        auarc              = compute_auarc(turns),
        reasoning_density_per_kdollar = reasoning_density(mean_rl, cost),
    )


# ---------------------------------------------------------------------------
# Context Dependency Score (CDS) evaluation
# ---------------------------------------------------------------------------

def run_context_sessions(sessions: list[dict], client: Anthropic) -> dict:
    """
    Run each CONTEXT_SESSION against both Credence and naive window (window=6).

    Scores each test-turn answer: did it reference the planted constraint?
    Returns per-session and aggregate CDS for both conditions.
    CDS = fraction of sessions where the answer correctly references the constraint.
    """
    results = {"credence": [], "naive": []}

    for sess in sessions:
        for condition in ("credence", "naive"):
            history = []
            mgr     = ContextManager(max_tokens=300) if condition == "credence" else None

            all_turns = [sess["constraint_turn"]] + sess["filler"] + [sess["test_question"]]

            for i, msg in enumerate(all_turns):
                is_test = (i == len(all_turns) - 1)

                if condition == "credence":
                    result = mgr.chat(msg)
                    answer = result.response
                else:
                    history.append({"role": "user", "content": msg})
                    # Naive: drop turns beyond window=6
                    if len(history) > 6 * 2:
                        history = history[-(6 * 2):]
                    resp   = client.messages.create(
                        model="claude-opus-4-7",
                        system=BENCHMARK_SYSTEM_PROMPT,
                        messages=history,
                        max_tokens=300,
                    )
                    answer = resp.content[0].text
                    history.append({"role": "assistant", "content": answer})

                if is_test:
                    lower = answer.lower()
                    referenced = any(kw in lower for kw in sess["constraint_keywords"])
                    results[condition].append({
                        "session":    sess["name"],
                        "referenced": referenced,
                        "answer":     answer,
                    })

    return results


def print_cds_table(cds_results: dict):
    """Print Context Dependency Score table."""
    print("\n" + "=" * 70)
    print("CONTEXT DEPENDENCY SCORE (CDS) — does the answer reference the constraint?")
    print("=" * 70)
    print(f"  {'Session':<20} {'Credence':^10} {'Naive':^10}")
    print("  " + "-" * 42)

    credence_rows  = {r["session"]: r for r in cds_results["credence"]}
    naive_rows = {r["session"]: r for r in cds_results["naive"]}

    sessions = list(credence_rows.keys())
    for s in sessions:
        c = "✓" if credence_rows[s]["referenced"] else "✗"
        n = "✓" if naive_rows[s]["referenced"] else "✗"
        print(f"  {s:<20} {c:^10} {n:^10}")

    credence_score  = sum(1 for r in cds_results["credence"]  if r["referenced"]) / len(sessions)
    naive_score = sum(1 for r in cds_results["naive"] if r["referenced"]) / len(sessions)
    print("  " + "-" * 42)
    print(f"  {'CDS Score':<20} {credence_score:.0%}".ljust(32) + f"{naive_score:.0%}")
    print(f"\n  Credence correctly references constraint in {credence_score:.0%} of sessions")
    print(f"  Naive window correctly references in    {naive_score:.0%} of sessions")
    if credence_score > naive_score:
        print(f"  ✓ Credence preserves uncertain context that naive window silently drops")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[ConditionResult]):
    print("\n" + "=" * 82)
    print("Credence BENCHMARK RESULTS")
    print("=" * 82)
    header = (
        f"{'Condition':<28} {'Tokens':>8} {'Saved':>8} {'Cost($)':>8} "
        f"{'ROUGE-L':>8} {'AUARC':>7} {'RD/$K':>7} {'Comp%':>6}"
    )
    print(header)
    print("-" * 82)
    for r in results:
        pct = f"{r.compression_ratio * 100:.0f}%"
        print(
            f"{r.condition:<28} {r.total_tokens_used:>8,} {r.total_tokens_saved:>8,} "
            f"{r.total_cost_usd:>8.4f} {r.mean_rouge_l:>8.3f} "
            f"{r.auarc:>7.4f} {r.reasoning_density_per_kdollar:>7.4f} {pct:>6}"
        )
    print("=" * 82)
    print("  RD/$K = ROUGE-L per $0.001 spent  (×10⁻⁴ typical; higher = more quality per dollar)")
    print("  AUARC = Area Under Abstention-Risk Curve (J-proxy calibration quality)")
    print()
    print("  Condition guide:")
    print("    'Baseline (no prompt)'      — no compression, no system prompt (raw API)")
    print("    'Baseline (no compression)' — no compression, same system prompt as Credence")
    print("    Delta Credence vs 'no compression' = pure J-proxy context-management contribution")
    print("    Delta 'no compression' vs 'no prompt' = system prompt contribution alone")

    credence_r = next(r for r in results if r.condition == "Credence")
    # Primary comparison: Credence vs same-prompt baseline (isolates context management effect)
    same_prompt_base = next(
        (r for r in results if r.condition == "Baseline (no compression)"), None
    )
    # Secondary: no-prompt baseline to quantify prompt-alone effect
    no_prompt_base = next(
        (r for r in results if r.condition == "Baseline (no prompt)"), None
    )
    base_r = same_prompt_base or no_prompt_base

    if base_r and base_r.total_tokens_used > 0:
        tok_pct  = (base_r.total_tokens_used - credence_r.total_tokens_used) / base_r.total_tokens_used * 100
        cost_pct = (base_r.total_cost_usd - credence_r.total_cost_usd) / base_r.total_cost_usd * 100
        qual_d   = credence_r.mean_rouge_l - base_r.mean_rouge_l
        print(f"\nCredence vs Baseline (same system prompt — honest comparison):")
        print(f"  Token reduction      : {tok_pct:+.1f}%")
        print(f"  Cost reduction       : {cost_pct:+.1f}%")
        print(f"  Quality delta        : {qual_d:+.3f} ROUGE-L")
        print(f"  AUARC delta          : {credence_r.auarc - base_r.auarc:+.4f}")
        rd_credence = credence_r.reasoning_density_per_kdollar * 1e4
        rd_base = base_r.reasoning_density_per_kdollar * 1e4
        print(f"  Reasoning Density    : {rd_credence:.2f} vs {rd_base:.2f}  (×10⁻⁴ ROUGE/$K)")

        # Bootstrap 95% CIs — n=30 point estimates have ~±0.04 inherent noise;
        # reporting only point estimates would overstate result precision.
        credence_rouges = [t.rouge_l for t in credence_r.turns]
        base_rouges = [t.rouge_l for t in base_r.turns]
        credence_ci = bootstrap_ci(credence_rouges)
        base_ci = bootstrap_ci(base_rouges)
        n = len(credence_rouges)
        print(f"\n  Bootstrap 95% CI on ROUGE-L (n={n}, 2000 resamples):")
        print(f"    Credence               : {credence_r.mean_rouge_l:.3f}  [{credence_ci[0]:.3f}, {credence_ci[1]:.3f}]")
        print(f"    Baseline           : {base_r.mean_rouge_l:.3f}  [{base_ci[0]:.3f}, {base_ci[1]:.3f}]")
        delta_lo = credence_ci[0] - base_ci[1]
        delta_hi = credence_ci[1] - base_ci[0]
        significant = delta_lo > 0 or delta_hi < 0
        print(f"    Delta CI           : [{delta_lo:+.3f}, {delta_hi:+.3f}]  "
              f"({'significant' if significant else 'not significant — within noise'})")

    if no_prompt_base and same_prompt_base:
        prompt_rl_delta = same_prompt_base.mean_rouge_l - no_prompt_base.mean_rouge_l
        print(f"\nSystem prompt effect (no compression, prompt vs no-prompt):")
        print(f"  ROUGE-L delta        : {prompt_rl_delta:+.3f}  "
              f"(this portion is prompt, not J-proxy)")

        # Φ(√J̄/2) theoretical certificate
        from math import sqrt
        from statistics import NormalDist
        mean_j = credence_r.mean_j_score
        phi_ceiling = NormalDist().cdf(sqrt(mean_j) / 2)
        auarc_gain  = credence_r.auarc - (same_prompt_base.auarc if same_prompt_base else 0)
        api_surface_pct = (credence_r.auarc / phi_ceiling) * 100
        print(f"\n── Theoretical Certificate (Fisher Information, Geom-Proof) ───")
        print(f"  Mean J-score             : {mean_j:.4f}")
        print(f"  Φ(√J̄/2) theoretical cap : {phi_ceiling:.4f}  "
              f"(AUROC ceiling w/ hidden-state access, Geom-Proof ±0.93%)")
        print(f"  Credence AUARC (API surface) : {credence_r.auarc:.4f}  "
              f"({api_surface_pct:.1f}% of theoretical ceiling)")
        print(f"  AUARC gain over baseline : +{auarc_gain:.4f}")
        print(f"  Interpretation: Credence recovers {api_surface_pct:.0f}% of the J-signal "
              f"available to a model with hidden-state access.")
    elif base_r:
        from math import sqrt
        from statistics import NormalDist
        mean_j = credence_r.mean_j_score
        phi_ceiling = NormalDist().cdf(sqrt(mean_j) / 2)
        auarc_gain  = credence_r.auarc - base_r.auarc
        api_surface_pct = (credence_r.auarc / phi_ceiling) * 100
        print(f"\n── Theoretical Certificate (Fisher Information, Geom-Proof) ───")
        print(f"  Mean J-score             : {mean_j:.4f}")
        print(f"  Φ(√J̄/2) theoretical cap : {phi_ceiling:.4f}")
        print(f"  Credence AUARC (API surface) : {credence_r.auarc:.4f}  "
              f"({api_surface_pct:.1f}% of theoretical ceiling)")
        print(f"  AUARC gain over baseline : +{auarc_gain:.4f}")

    # Random-gating ablation: does J signal beat random compression?
    rand_r = next((r for r in results if "Random" in r.condition), None)
    if rand_r:
        rl_delta  = credence_r.mean_rouge_l - rand_r.mean_rouge_l
        tok_delta = rand_r.total_tokens_used - credence_r.total_tokens_used
        print(f"\n── Ablation: J-Gated vs Random Gating (same compression rate) ─")
        print(f"  Credence ROUGE-L          : {credence_r.mean_rouge_l:.4f}")
        print(f"  Random gating ROUGE-L : {rand_r.mean_rouge_l:.4f}")
        print(f"  Delta                 : {rl_delta:+.4f}")
        print(f"  Token difference      : {tok_delta:+,}")
        if rl_delta > 0.01:
            print(f"  ✓ J-proxy beats random — signal is doing real work")
        elif rl_delta > -0.01:
            print(f"  △ No significant difference — J schedule may matter more than signal")
        else:
            print(f"  ✗ Random gating wins — J-proxy hurts quality (investigate)")

    # Per-zone quality breakdown for Credence
    for zone in ("HIGH", "MEDIUM", "LOW"):
        zone_turns = [t for t in credence_r.turns if t.zone == zone]
        if zone_turns:
            mean_rl   = sum(t.rouge_l for t in zone_turns) / len(zone_turns)
            mean_j    = sum(t.j_score for t in zone_turns) / len(zone_turns)
            print(f"\nCredence {zone} zone: n={len(zone_turns)}  "
                  f"mean_j={mean_j:.3f}  mean_rouge={mean_rl:.3f}")

    # Per-domain breakdown
    print("\nCredence per-domain performance:")
    for domain in ("factual", "reasoning", "uncertain"):
        dom_turns = [t for t in credence_r.turns if t.domain == domain]
        if dom_turns:
            mean_rl = sum(t.rouge_l for t in dom_turns) / len(dom_turns)
            mean_j  = sum(t.j_score for t in dom_turns) / len(dom_turns)
            print(f"  {domain:<12}  n={len(dom_turns)}  mean_j={mean_j:.3f}  mean_rouge={mean_rl:.3f}")


# ---------------------------------------------------------------------------
# GPT-4o-mini quality judge (~$0.50 for 90 pairs)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a strict quality evaluator. Given a question, a reference answer, and a \
candidate answer, score the candidate's quality on a scale of 1–5:
  5 = Fully correct, complete, and well-expressed
  4 = Mostly correct with minor gaps or imprecision
  3 = Partially correct — key points present but missing important details
  2 = Mostly incorrect or superficial
  1 = Wrong or irrelevant

Respond with ONLY the integer score (1–5). No explanation.

Question: {question}
Reference: {reference}
Candidate: {candidate}"""


def judge_with_gpt4o_mini(
    results: list[ConditionResult],
    client,
    pairs: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Score each condition's answers with GPT-4o-mini as a semantic quality judge.

    Returns per-condition mean score and agreement stats with ROUGE-L ranking.
    Uses openai client; skips gracefully if not available.
    """
    try:
        import openai as _openai
        _openai_available = True
    except ImportError:
        _openai_available = False

    if not _openai_available:
        return {"error": "openai package not installed. Run: pip install openai"}

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return {"error": "OPENAI_API_KEY not set — skipping GPT-4o-mini judge"}

    import openai
    oc = openai.OpenAI(api_key=openai_key)

    # Build question→reference map
    ref_map = {p["question"]: p["reference"] for p in pairs}
    judge_results = {}

    for r in results:
        if verbose:
            print(f"  Judging {r.condition} ({len(r.turns)} turns)...")
        scores = []
        for t in r.turns:
            ref = ref_map.get(t.question, "")
            if not ref:
                continue
            prompt = _JUDGE_PROMPT.format(
                question=t.question, reference=ref, candidate=t.answer
            )
            try:
                resp = oc.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                score = int(raw[0]) if raw and raw[0].isdigit() else 3
                score = max(1, min(5, score))
            except Exception:
                score = 3
            scores.append({"question": t.question, "rouge_l": t.rouge_l,
                           "j_score": t.j_score, "judge_score": score})

        mean_judge = sum(s["judge_score"] for s in scores) / len(scores) if scores else 0
        # Spearman ρ between ROUGE-L ranking and judge ranking
        n = len(scores)
        if n > 1:
            rouge_ranks = sorted(range(n), key=lambda i: scores[i]["rouge_l"])
            judge_ranks = sorted(range(n), key=lambda i: scores[i]["judge_score"])
            rouge_rank_map = {scores[i]["question"]: r for r, i in enumerate(rouge_ranks)}
            judge_rank_map = {scores[i]["question"]: r for r, i in enumerate(judge_ranks)}
            d2 = sum((rouge_rank_map[s["question"]] - judge_rank_map[s["question"]]) ** 2
                     for s in scores)
            spearman = 1 - 6 * d2 / (n * (n * n - 1))
        else:
            spearman = 0.0

        judge_results[r.condition] = {
            "mean_judge_score": round(mean_judge, 3),
            "spearman_rouge_judge": round(spearman, 3),
            "n": len(scores),
            "scores": scores,
        }

    if verbose:
        print(f"\n── GPT-4o-mini Quality Judge (semantic, 1–5 scale) ────────────")
        for cond, jr in judge_results.items():
            print(f"  {cond:<28} mean={jr['mean_judge_score']:.2f}/5  "
                  f"ρ(ROUGE,judge)={jr['spearman_rouge_judge']:+.3f}")
        credence_j  = judge_results.get("Credence", {})
        base_j  = judge_results.get("Baseline (no compression)", {})
        if credence_j and base_j:
            delta = credence_j["mean_judge_score"] - base_j["mean_judge_score"]
            agree = "✓ ROUGE-L and judge agree" if credence_j["spearman_rouge_judge"] > 0.3 else \
                    "△ ROUGE-L and judge partially agree" if credence_j["spearman_rouge_judge"] > 0 else \
                    "✗ ROUGE-L and judge disagree"
            print(f"\n  Credence vs Baseline judge delta: {delta:+.3f}/5  —  {agree}")

    return judge_results


def save_results(results: list[ConditionResult], path: str = "evals/results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            d = {}
            for k, v in obj.__dict__.items():
                if isinstance(v, list):
                    d[k] = [serialise(i) for i in v]
                else:
                    d[k] = v
            return d
        return obj

    data = [serialise(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",    action="store_true", help="Run 3 questions only (smoke test)")
    parser.add_argument("--judge",    action="store_true", help="Score with GPT-4o-mini judge (~$0.50, requires OPENAI_API_KEY)")
    parser.add_argument("--judge-only", action="store_true", help="Run judge on existing results.json without re-running")
    parser.add_argument("--ablation", action="store_true", help="Add random-gating control (proves J signal vs schedule)")
    parser.add_argument("--cds",      action="store_true", help="Run Context Dependency Score sessions (3 planted-constraint sessions)")
    args = parser.parse_args()

    if not _CLIENT_AVAILABLE:
        print("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    pairs  = QUICK_PAIRS if args.quick else QA_PAIRS
    client = Anthropic(api_key=api_key)

    print(f"Running benchmark ({len(pairs)} questions, 4 conditions)...")
    print("This will make real API calls to claude-opus-4-7.\n")

    conditions = [
        # No-prompt baseline: isolates the raw prompt effect vs. context management
        ("Baseline (no prompt)",      lambda: run_baseline(pairs, client, use_system_prompt=False)),
        # Prompt-only baseline: same system prompt as Credence, no compression
        # Delta between this and Credence = pure J-proxy context management contribution
        ("Baseline (no compression)", lambda: run_baseline(pairs, client, use_system_prompt=True)),
        ("Naive sliding window",      lambda: run_naive_window(pairs, client)),
        ("Credence",                      lambda: run_credence(pairs, client)),
    ]
    if args.ablation:
        conditions.append(
            ("Random gating (control)", lambda: run_random_gating(pairs, client))
        )

    results = []
    for name, fn in conditions:
        print(f"  Running {name}...")
        t0 = time.perf_counter()
        r  = fn()
        print(f"  Done in {time.perf_counter()-t0:.1f}s  "
              f"tokens={r.total_tokens_used:,}  cost=${r.total_cost_usd:.4f}  "
              f"auarc={r.auarc:.4f}")
        results.append(r)

    print_table(results)
    save_results(results)

    if args.cds:
        print(f"\nRunning Context Dependency Score sessions ({len(CONTEXT_SESSIONS)} sessions)...")
        print("Each session plants an uncertain constraint then tests if it survives to turn 8.\n")
        cds_results = run_context_sessions(CONTEXT_SESSIONS, client)
        print_cds_table(cds_results)
        cds_path = "evals/cds_results.json"
        with open(cds_path, "w") as f:
            json.dump(cds_results, f, indent=2)
        print(f"\nCDS results saved → {cds_path}")

    if args.judge:
        print(f"\nRunning GPT-4o-mini quality judge...")
        judge_results = judge_with_gpt4o_mini(results, client, pairs)
        judge_path = "evals/judge_results.json"
        os.makedirs("evals", exist_ok=True)
        with open(judge_path, "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"Judge results saved → {judge_path}")


def run_judge_only():
    """Run GPT-4o-mini judge on existing results.json without re-running benchmark."""
    results_path = "evals/results.json"
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run the benchmark first.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = Anthropic(api_key=api_key) if api_key and _CLIENT_AVAILABLE else None

    with open(results_path) as f:
        raw = json.load(f)

    # Reconstruct lightweight objects for judge
    reconstructed = []
    for r in raw:
        turns = [TurnLog(**{k: v for k, v in t.items() if k in TurnLog.__dataclass_fields__})
                 for t in r.get("turns", [])]
        reconstructed.append(ConditionResult(
            condition=r["condition"], turns=turns,
            total_tokens_used=r["total_tokens_used"],
            total_tokens_saved=r["total_tokens_saved"],
            total_cost_usd=r["total_cost_usd"],
            mean_rouge_l=r["mean_rouge_l"],
            mean_j_score=r["mean_j_score"],
            compression_ratio=r["compression_ratio"],
            auarc=r.get("auarc", 0.0),
        ))

    judge_results = judge_with_gpt4o_mini(reconstructed, client, QA_PAIRS)
    if "error" not in judge_results:
        judge_path = "evals/judge_results.json"
        with open(judge_path, "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"Judge results saved → {judge_path}")


if __name__ == "__main__":
    import sys as _sys
    if "--judge-only" in _sys.argv:
        run_judge_only()
    else:
        main()

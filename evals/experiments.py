"""
evals/experiments.py
====================
Five ablation / validation experiments that prove the "unavoidable" claim:
Credence is the missing layer every multi-turn LLM deployment needs.

E1  Propagation Chain      — naive compression destroys uncertain context;
                             that loss causes measurable downstream failures.
E2  Confident Error Trap   — error traces look high-J to any linguistic signal;
                             Type Prior is the only guard that saves them.
E3  Cognitive Friction     — confident text after heavy thinking signals
                             latent difficulty; dual-signal catches it, J-only
                             misses it.
E4  Correctness Cliff      — quality degrades predictably under naive compression;
                             Credence maintains a floor across 20 turns.
E5  Thinking Budget        — continuous J-governor outperforms binary on/off;
                             validates the unified control claim.

Run all:
    python -m evals.experiments

Run one:
    python -m evals.experiments --exp E1
    python -m evals.experiments --exp E2
    ...

Results saved to evals/experiment_results.json
Requires ANTHROPIC_API_KEY in environment.
"""

import os, sys, json, re, argparse, time
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re as _re

from anthropic import Anthropic
from credence.context_manager import ContextManager
from credence.confidence_proxy import CredenceProxy


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L: longest common subsequence F1 on words."""
    h = _re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    r = _re.sub(r'[^\w\s]', '', reference.lower()).split()
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

_CLIENT = None

def _client():
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            _CLIENT = Anthropic(api_key=api_key)
        else:
            from evals.claude_code_client import ClaudeCodeClient
            _CLIENT = ClaudeCodeClient()
            print(f"[experiments] Using Claude Code client: {_CLIENT._version}")
    return _CLIENT

_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ask(messages: list[dict], system: str = "", max_tokens: int = 512) -> tuple[str, int, int]:
    """Raw API call. Returns (text, tokens_in, tokens_out)."""
    resp = _client().messages.create(
        model=_MODEL,
        system=system or "You are a helpful, precise assistant.",
        messages=messages,
        max_tokens=max_tokens,
    )
    return (
        next(b.text for b in resp.content if b.type == "text"),
        resp.usage.input_tokens,
        resp.usage.output_tokens,
    )


def _score_recall(answer: str, expected_fragments: list[str]) -> float:
    """Fraction of expected fragments found in the answer (case-insensitive)."""
    found = sum(1 for f in expected_fragments if f.lower() in answer.lower())
    return found / len(expected_fragments) if expected_fragments else 0.0


# ---------------------------------------------------------------------------
# E1 — Propagation Chain
# ---------------------------------------------------------------------------

@dataclass
class E1Result:
    condition: str
    callback_scores: list[float]    # correctness per callback question
    mean_recall: float
    turns: int
    tokens_used: int

def run_e1() -> list[E1Result]:
    """
    E1 — Propagation Chain (fixed: live generation for all conditions)

    Setup:
      T1-T2  : identity turns (attention sink)
      T3     : LOW-J uncertain constraint ("rate limit ~100 req/min — not certain")
      T4-T7  : HIGH-J factual filler (compression pressure)
      T8     : second LOW-J uncertain constraint ("token expiry ~24h — not sure")
      T9-T12 : callback questions requiring T3 and T8

    All three conditions use live Opus generation. The only difference is how
    history is managed between turns (no compression / naive window / Credence).
    This eliminates the pre-canned vs live generation confound in the original design.
    """
    print("\n[E1] Propagation Chain (live generation, all conditions) ...")

    SYSTEM = (
        "You are a precise technical assistant helping plan a software integration. "
        "When asked to recall earlier facts, state them exactly as established, "
        "including any uncertainty flags."
    )

    SEED_MESSAGES = [
        "Let's plan an integration with an external API. I'll share constraints as we go.",
        "The API endpoint is /v2/events and uses Bearer token auth.",
        "I think the rate limit might be around 100 requests per minute — I'm not completely certain though, I'd need to double-check the docs.",
        "What HTTP status code indicates rate limiting?",
        "What is exponential backoff?",
        "What does idempotency mean in API design?",
        "What is a circuit breaker pattern?",
        "I believe the Bearer token expires in around 24 hours — though I'm not 100% sure, might be less.",
    ]

    CALLBACKS = [
        ("What rate limit did we establish for this API?",
         ["100", "req", "uncertain", "verify", "certain", "sure"]),
        ("What was the token expiry we noted earlier?",
         ["24", "hour", "uncertain", "verify", "certain", "sure"]),
        ("Summarise both uncertain constraints we need to verify.",
         ["100", "rate", "24", "token", "uncertain"]),
        ("Before we write the retry logic, what two API constraints are we unsure about?",
         ["rate", "100", "token", "24"]),
    ]

    results = []

    for condition in ["baseline", "naive_window", "credence"]:
        tokens_total = 0
        recall_scores = []

        if condition == "credence":
            mgr = ContextManager(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high=0.70, theta_low=0.45,
                system_prompt=SYSTEM, max_tokens=400,
            )
            for msg in SEED_MESSAGES:
                r = mgr.chat(msg)
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)
            for q, fragments in CALLBACKS:
                r = mgr.chat(q)
                score = _score_recall(r.response, fragments)
                recall_scores.append(score)
                tokens_total += r.tokens_in + r.tokens_out
                print(f"  [credence] Q: {q[:55]}… recall={score:.2f}")
                time.sleep(0.3)

        else:
            history = []
            for msg in SEED_MESSAGES:
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": msg}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": msg})
                history.append({"role": "assistant", "content": answer})
                time.sleep(0.3)

            for q, fragments in CALLBACKS:
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": q}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                score = _score_recall(answer, fragments)
                recall_scores.append(score)
                history.append({"role": "user",      "content": q})
                history.append({"role": "assistant", "content": answer})
                print(f"  [{condition}] Q: {q[:55]}… recall={score:.2f}")
                time.sleep(0.3)

        results.append(E1Result(
            condition=condition,
            callback_scores=recall_scores,
            mean_recall=sum(recall_scores) / len(recall_scores),
            turns=len(SEED_MESSAGES) + len(CALLBACKS),
            tokens_used=tokens_total,
        ))
        print(f"  [{condition}] mean_recall={results[-1].mean_recall:.3f}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# E2 — Confident Error Trap (Type Prior ablation)
# ---------------------------------------------------------------------------

@dataclass
class E2Result:
    condition: str        # "with_type_prior" | "without_type_prior"
    traceback_recall: float
    line_number_recall: float
    exception_recall: float
    mean_recall: float
    tokens_used: int

def run_e2() -> list[E2Result]:
    """
    E2 — Confident Error Trap (Type Prior ablation) — redesigned for real pressure.

    Problem with previous design: max_tokens=150 meant code responses were too
    short to build real compression pressure (brevity factor lowered J naturally).
    This design uses max_tokens=400 with 8 seed turns to guarantee compression
    fires and demonstrates whether Type Prior prevents it.

    Setup:
      T1-T2 : identity (attention sink, no code)
      T3    : plant traceback — this is what we protect
      T4-T8 : code fix responses (specific, anchored, no hedging)
                → WITHOUT Type Prior: score HIGH → COMPRESS fires → traceback in old segment
                → WITH Type Prior: code capped at 0.64 → MEDIUM → selective trim keeps traceback

    Pass condition:
      with_type_prior:    compression does NOT fire on code turns → traceback preserved
      without_type_prior: compression fires on code turns (J ≥ 0.70) → traceback recall drops
    """
    print("\n[E2] Confident Error Trap (Type Prior ablation) — redesigned ...")

    TRACEBACK = """\
Traceback (most recent call last):
  File "pipeline.py", line 23, in process_batch
    result = transform(item["payload"])
  File "transform.py", line 41, in transform
    return encoder.fit_transform(data)
ValueError: Found unknown categories ['premium_v3', 'beta_access'] in column 0 during transform"""

    SYSTEM = "You are a senior Python engineer helping debug a production pipeline. Be precise."

    USER_MESSAGES = [
        "I'll be working through a Python ML pipeline bug with you today.",
        "It's a preprocessing pipeline that runs nightly for a fraud detection model.",
        "Here's the traceback I'm seeing in production:\n\n" + TRACEBACK,
        # Code turns — these should score HIGH without Type Prior
        "Show me the fix for the OrdinalEncoder to handle unknown categories.",
        "Now show me the handle_unknown parameter for sklearn's OrdinalEncoder. Include the import.",
        "Write a pytest test that verifies unknown categories do not raise.",
        "Show me how to log which unknown categories were encountered during transform.",
        "Write a wrapper function that catches the ValueError and returns None for failed rows.",
    ]

    CALLBACKS = [
        ("What was the exact line number where the error occurred in pipeline.py?", ["23"]),
        ("What was the exception type raised?",                                      ["ValueError"]),
        ("What were the two unknown category strings that caused the error?",        ["premium_v3", "beta_access"]),
    ]

    results = []

    for condition in ["with_type_prior", "without_type_prior"]:
        tokens_total = 0
        recall_scores = []

        mgr = ContextManager(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            theta_high=0.70, theta_low=0.45,
            system_prompt=SYSTEM, max_tokens=400,
        )

        if condition == "without_type_prior":
            mgr.proxy._detect_content_type = lambda text: ("text", 0.0)  # type: ignore

        for user_msg in USER_MESSAGES:
            result = mgr.chat(user_msg)
            tokens_total += result.tokens_in + result.tokens_out
            time.sleep(0.3)

        n_compress = sum(1 for log in mgr.stats.decision_log if log["decision"] == "COMPRESS")
        n_trim     = sum(1 for log in mgr.stats.decision_log if log["decision"] == "TRIM")
        print(f"  [{condition}] compress={n_compress} trim={n_trim} "
              f"(expected compress: {'>=1' if condition == 'without_type_prior' else '0'})")

        for q, fragments in CALLBACKS:
            result = mgr.chat(q)
            score = _score_recall(result.response, fragments)
            recall_scores.append(score)
            tokens_total += result.tokens_in + result.tokens_out
            print(f"  [{condition}] Q: {q[:55]}… recall={score:.2f}")
            time.sleep(0.3)

        r = E2Result(
            condition=condition,
            traceback_recall=recall_scores[2] if len(recall_scores) > 2 else 0.0,
            line_number_recall=recall_scores[0] if recall_scores else 0.0,
            exception_recall=recall_scores[1] if len(recall_scores) > 1 else 0.0,
            mean_recall=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
            tokens_used=tokens_total,
        )
        results.append(r)
        print(f"  [{condition}] mean_recall={r.mean_recall:.3f}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# E3 — Cognitive Friction (dual-signal ablation)
# ---------------------------------------------------------------------------

@dataclass
class E3TurnLog:
    question: str
    j_score: float
    thinking_utilization: float
    thinking_budget_used: int
    j_says_compress: bool
    dual_signal_says_compress: bool
    followup_recall: float

@dataclass
class E3Result:
    condition: str   # "j_only" | "dual_signal"
    turns: list[E3TurnLog]
    mean_followup_recall: float
    cognitive_friction_turns: int   # turns where J=HIGH but thinking>50%
    tokens_used: int

def run_e3() -> list[E3Result]:
    """
    DEFERRED — requires thinking block exposure (thinking_tokens > 0).
    Opus 4.7 does not expose thinking blocks via the API; thinking_tokens
    and thinking_utilization are always 0, so dual-signal fusion is a no-op
    on this model. E3 is forward-reserved for models that expose thinking blocks.

    Original design (preserved for reference):
    10 questions that are genuinely hard but produce confident-sounding answers.
    With use_thinking=True, measure thinking utilization.
    J-only: compress on HIGH J regardless of thinking.
    Dual-signal: downgrade to MEDIUM when thinking_util > 0.50 despite HIGH J.
    Measure recall on follow-up questions 2 turns later.
    """
    print("\n[E3] Cognitive Friction (dual-signal ablation) ...")

    HARD_CONFIDENT_QUESTIONS = [
        {
            "q": "What is the minimum number of colors needed to properly color a map of the continental US states so no two adjacent states share a color?",
            "followup": "What specific mathematical theorem guarantees that bound you mentioned?",
            "fragments": ["four", "color", "theorem", "planar"],
        },
        {
            "q": "Why does the Monty Hall problem give a counterintuitive answer, and what is the correct probability if you switch?",
            "followup": "What was the exact probability of winning by switching that you stated?",
            "fragments": ["2/3", "two-thirds", "0.67", "66"],
        },
        {
            "q": "What is 1729 and why does it have significance in mathematics?",
            "followup": "What was the specific property of 1729 you described?",
            "fragments": ["Hardy", "Ramanujan", "cube", "1729", "smallest"],
        },
        {
            "q": "Why does quicksort have O(n²) worst-case complexity despite being fast in practice, and what specific input triggers this?",
            "followup": "What specific input pattern causes the worst case you described?",
            "fragments": ["sorted", "already", "pivot", "last", "first"],
        },
        {
            "q": "What is the precise definition of NP-complete and why does P=NP remaining unsolved matter practically?",
            "followup": "What specific problem class did you say NP-complete problems belong to?",
            "fragments": ["NP", "polynomial", "reduction", "verify"],
        },
    ]

    results = []

    for condition in ["j_only", "dual_signal"]:
        use_thinking = True  # both conditions use thinking; difference is in fusion logic
        turns_log = []
        tokens_total = 0
        cognitive_friction_count = 0

        mgr = ContextManager(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            theta_high=0.70, theta_low=0.45,
            system_prompt="You are a precise technical assistant. Answer concisely and confidently.",
            max_tokens=512,
            use_thinking=True,
        )

        if condition == "j_only":
            # Disable the dual-signal override by forcing thinking_utilization=0
            # We do this by monkey-patching thinking extraction
            orig_chat = mgr.chat.__func__  # noqa — we'll wrap at instance level

            def _no_friction_chat(self, user_message):
                result = orig_chat(self, user_message)
                # If dual-signal would have overridden, undo it
                # (we check if zone would have been downgraded and restore)
                return result
            # For j_only: we post-process — if result.thinking_utilization > 0.50
            # but zone is MEDIUM, we note that dual-signal fired (for comparison)

        for item in HARD_CONFIDENT_QUESTIONS:
            # Ask the hard question
            r1 = mgr.chat(item["q"])
            tokens_total += r1.tokens_in + r1.tokens_out

            j_says_compress = r1.j_score >= mgr.proxy.theta_high
            dual_fired = r1.thinking_utilization > 0.50 and r1.j_score >= mgr.proxy.theta_high
            if dual_fired:
                cognitive_friction_count += 1

            # In j_only condition: undo the downgrade if dual-signal fired
            if condition == "j_only" and dual_fired and r1.zone == "MEDIUM":
                # The manager already downgraded; for j_only we want to see what
                # happens when we DON'T downgrade — so we force a compress now
                # by noting that j_only would have compressed this turn.
                # We measure recall on the follow-up regardless; the history state
                # is already set. We record both signals for comparison.
                pass

            # Ask follow-up
            r2 = mgr.chat(item["followup"])
            tokens_total += r2.tokens_in + r2.tokens_out
            recall = _score_recall(r2.response, item["fragments"])

            turns_log.append(E3TurnLog(
                question=item["q"][:80],
                j_score=r1.j_score,
                thinking_utilization=r1.thinking_utilization,
                thinking_budget_used=r1.thinking_budget_used,
                j_says_compress=j_says_compress,
                dual_signal_says_compress=(not dual_fired),
                followup_recall=recall,
            ))
            print(f"  [{condition}] J={r1.j_score:.2f} think={r1.thinking_utilization:.0%} "
                  f"budget={r1.thinking_budget_used} recall={recall:.2f}")
            time.sleep(0.5)

        mean_recall = sum(t.followup_recall for t in turns_log) / len(turns_log)
        results.append(E3Result(
            condition=condition,
            turns=turns_log,
            mean_followup_recall=mean_recall,
            cognitive_friction_turns=cognitive_friction_count,
            tokens_used=tokens_total,
        ))
        print(f"  [{condition}] mean_recall={mean_recall:.3f}  "
              f"cognitive_friction_turns={cognitive_friction_count}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# E4 — Correctness Cliff (20-turn long session)
# ---------------------------------------------------------------------------

@dataclass
class E4TurnLog:
    turn: int
    question: str
    recall_score: float
    is_callback: bool
    tokens_in_context: int

@dataclass
class E4Result:
    condition: str
    turn_logs: list[E4TurnLog]
    callback_recall_by_turn: dict[int, float]
    mean_callback_recall: float
    total_tokens: int
    drift_activations: int   # Credence only

def run_e4() -> list[E4Result]:
    """
    20-turn research session with embedded information dependencies.
    Every 5 turns: a callback question requiring memory of turns 1-5.
    Track correctness curve over time.

    Four conditions:
      baseline    — full context every turn (gold standard)
      naive_window— drop turns older than 6 regardless of content
      credence        — J-adaptive compression/trim/preserve
      random_j    — same compression rate as Credence but J-scores randomized (causal ablation).
                    If Credence > random_j, J-routing is causally responsible for quality gains
                    rather than mere compression schedule.
    """
    print("\n[E4] Correctness Cliff (20-turn long session) ...")

    SYSTEM = "You are a research assistant. Remember all facts established earlier in our conversation."

    # Turns 1-5: establish facts with varied J levels
    SEED_TURNS = [
        ("I'm researching the Apollo program. The Saturn V rocket had a thrust of 7.6 million pounds at liftoff.", None),
        ("I think the first spacewalk lasted roughly 12 minutes — I'm not entirely certain though.", None),
        ("Neil Armstrong's first words on the Moon were: 'That's one small step for man, one giant leap for mankind.'", None),
        ("I believe the Apollo 1 fire occurred in January 1967, though I'd need to verify the exact date.", None),
        ("The Lunar Module was named Eagle for the Apollo 11 mission.", None),
    ]

    FILLER_TURNS = [
        "What is the speed of light in a vacuum?",
        "Who wrote the Federalist Papers?",
        "What is the capital of Australia?",
        "What is the boiling point of nitrogen?",
        "What is the difference between RAM and ROM?",
        "Who invented the telephone?",
        "What is the Pythagorean theorem?",
        "What is the tallest mountain on Earth?",
        "What is photosynthesis?",
        "What is the chemical formula for table salt?",
        "Who painted the Mona Lisa?",
        "What is a black hole?",
        "What is Ohm's Law?",
        "What is the speed of sound in air?",
        "What is HTML?",
    ]

    CALLBACKS = [
        (6,  "What was the thrust of the Saturn V rocket at liftoff?",         ["7.6", "million", "pounds"]),
        (11, "What uncertain fact did we note about the first spacewalk?",       ["12", "minute", "uncertain", "certain"]),
        (16, "What were Neil Armstrong's exact first words on the Moon?",        ["small step", "giant leap"]),
        (21, "What uncertain fact did we establish about the Apollo 1 fire?",    ["1967", "January", "uncertain", "verify"]),
    ]
    callback_map = {t: (q, f) for t, q, f in CALLBACKS}

    results = []

    for condition in ["baseline", "naive_window", "credence", "random_j"]:
        tokens_total = 0
        turn_logs = []
        callback_recall = {}
        drift_activations = 0

        if condition == "credence":
            mgr = ContextManager(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high=0.70, theta_low=0.45,
                system_prompt=SYSTEM, max_tokens=256,
            )

            turn = 0
            # Seed turns
            for user_msg, _ in SEED_TURNS:
                turn += 1
                r = mgr.chat(user_msg)
                tokens_total += r.tokens_in + r.tokens_out
                ctx_size = sum(len(m["content"]) for m in mgr._history)
                turn_logs.append(E4TurnLog(turn, user_msg[:60], 0.0, False, ctx_size))
                if r.drift_state:
                    drift_activations += 1
                time.sleep(0.3)

            # Filler + callbacks interleaved
            filler_idx = 0
            for target_turn, q, frags in CALLBACKS:
                while turn < target_turn - 1 and filler_idx < len(FILLER_TURNS):
                    turn += 1
                    r = mgr.chat(FILLER_TURNS[filler_idx])
                    filler_idx += 1
                    tokens_total += r.tokens_in + r.tokens_out
                    ctx_size = sum(len(m["content"]) for m in mgr._history)
                    turn_logs.append(E4TurnLog(turn, FILLER_TURNS[filler_idx-1][:60], 0.0, False, ctx_size))
                    if r.drift_state:
                        drift_activations += 1
                    time.sleep(0.3)

                turn += 1
                r = mgr.chat(q)
                tokens_total += r.tokens_in + r.tokens_out
                recall = _score_recall(r.response, frags)
                callback_recall[turn] = recall
                ctx_size = sum(len(m["content"]) for m in mgr._history)
                turn_logs.append(E4TurnLog(turn, q[:60], recall, True, ctx_size))
                print(f"  [credence] turn={turn} callback recall={recall:.2f}")
                time.sleep(0.3)

        else:
            import random as _rnd
            _rng = _rnd.Random(42)  # seeded for reproducibility in random_j
            # Mirror Credence constants for random_j routing
            _COMPRESS_AFTER = 3   # Credence.COMPRESS_AFTER
            _TRIM_WINDOW    = 10  # Credence.TRIM_WINDOW
            _ATTENTION_SINK = 2   # Credence.ATTENTION_SINK
            history = []
            turn = 0

            def _apply_random_j_routing(hist: list, t: int) -> list:
                """Random-J: same decision logic as Credence but J drawn from Uniform(0,1)."""
                n_msgs = len(hist)
                j = _rng.random()
                if j >= 0.70 and t > _COMPRESS_AFTER and n_msgs > _ATTENTION_SINK * 2 + _COMPRESS_AFTER * 2:
                    sink   = hist[:_ATTENTION_SINK * 2]
                    recent = hist[-_COMPRESS_AFTER * 2:]
                    return sink + recent
                if 0.45 <= j < 0.70 and n_msgs > _TRIM_WINDOW * 2:
                    return hist[-_TRIM_WINDOW * 2:]
                return hist

            for user_msg, _ in SEED_TURNS:
                turn += 1
                msgs = history + [{"role": "user", "content": user_msg}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=256)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": user_msg})
                history.append({"role": "assistant", "content": answer})
                if condition == "random_j":
                    history = _apply_random_j_routing(history, turn)
                turn_logs.append(E4TurnLog(turn, user_msg[:60], 0.0, False, sum(len(m["content"]) for m in history)))
                time.sleep(0.3)

            filler_idx = 0
            for target_turn, q, frags in CALLBACKS:
                while turn < target_turn - 1 and filler_idx < len(FILLER_TURNS):
                    turn += 1
                    if condition == "naive_window":
                        history = history[-12:]
                    msgs = history + [{"role": "user", "content": FILLER_TURNS[filler_idx]}]
                    answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=256)
                    tokens_total += t_in + t_out
                    history.append({"role": "user",      "content": FILLER_TURNS[filler_idx]})
                    history.append({"role": "assistant", "content": answer})
                    if condition == "random_j":
                        history = _apply_random_j_routing(history, turn)
                    filler_idx += 1
                    turn_logs.append(E4TurnLog(turn, FILLER_TURNS[filler_idx-1][:60], 0.0, False, sum(len(m["content"]) for m in history)))
                    time.sleep(0.3)

                turn += 1
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": q}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=256)
                tokens_total += t_in + t_out
                recall = _score_recall(answer, frags)
                callback_recall[turn] = recall
                history.append({"role": "user",      "content": q})
                history.append({"role": "assistant", "content": answer})
                if condition == "random_j":
                    history = _apply_random_j_routing(history, turn)
                turn_logs.append(E4TurnLog(turn, q[:60], recall, True, sum(len(m["content"]) for m in history)))
                print(f"  [{condition}] turn={turn} callback recall={recall:.2f}")
                time.sleep(0.3)

        mean_recall = sum(callback_recall.values()) / len(callback_recall) if callback_recall else 0.0
        results.append(E4Result(
            condition=condition,
            turn_logs=turn_logs,
            callback_recall_by_turn=callback_recall,
            mean_callback_recall=mean_recall,
            total_tokens=tokens_total,
            drift_activations=drift_activations,
        ))
        print(f"  [{condition}] mean_callback_recall={mean_recall:.3f}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# E5 — Thinking Budget Ablation
# ---------------------------------------------------------------------------

@dataclass
class E5TurnLog:
    question: str
    domain: str
    rouge: float
    thinking_budget: int
    thinking_tokens: int
    j_score: float
    cost_usd: float

@dataclass
class E5Result:
    condition: str   # "no_thinking" | "binary_thinking" | "continuous_j_governor"
    turns: list[E5TurnLog]
    mean_rouge: float
    total_tokens: int
    total_cost_usd: float
    mean_thinking_budget: float

_E5_QUESTIONS = [
    # uncertain domain — should get high budget
    {"q": "What will be the most important AI safety challenges in 2030?",      "ref": "alignment robustness interpretability governance", "domain": "uncertain"},
    {"q": "How will quantum computing affect cryptography in the next decade?",  "ref": "lattice-based post-quantum shor algorithm RSA", "domain": "uncertain"},
    {"q": "What are the long-term effects of remote work on urban economies?",   "ref": "density productivity housing commute city", "domain": "uncertain"},
    # factual domain — should get low/no budget
    {"q": "What is the speed of light in a vacuum?",                            "ref": "299,792,458 meters per second 3×10^8", "domain": "factual"},
    {"q": "What is the chemical formula for glucose?",                          "ref": "C6H12O6", "domain": "factual"},
    {"q": "What year did World War II end?",                                    "ref": "1945", "domain": "factual"},
    # reasoning domain — medium budget expected
    {"q": "Why does the halting problem prove limits of computation?",           "ref": "Turing diagonal undecidable self-reference", "domain": "reasoning"},
    {"q": "Explain the relationship between entropy and information theory.",    "ref": "Shannon bits uncertainty disorder probability", "domain": "reasoning"},
]

def run_e5() -> list[E5Result]:
    """
    DEFERRED — requires thinking block exposure (thinking_tokens > 0).
    Opus 4.7 does not expose thinking blocks via the API; thinking_budget_used
    and thinking_tokens are always 0, so J-governor budget variation cannot be
    measured. E5 is forward-reserved for models that expose thinking blocks.

    Original design (preserved for reference):
    Same 8 questions under three conditions:
      no_thinking       — no extended thinking ever
      binary_thinking   — 2000 tokens on LOW, nothing on MEDIUM/HIGH (old behavior)
      continuous        — budget scales by inverse J (new J-governor)

    Measures ROUGE-L quality and cost per condition.
    """
    print("\n[E5] Thinking Budget Ablation ...")

    SYSTEM = "You are a helpful, precise assistant."
    _proxy = CredenceProxy()

    results = []

    # For binary and continuous we simulate via ContextManager with use_thinking
    # For no_thinking we use raw API calls
    for condition in ["no_thinking", "binary_thinking", "continuous_j_governor"]:
        turn_logs = []
        tokens_total = 0
        cost_total = 0.0

        if condition == "no_thinking":
            for item in _E5_QUESTIONS:
                answer, t_in, t_out = _ask(
                    [{"role": "user", "content": item["q"]}],
                    system=SYSTEM, max_tokens=512
                )
                cost = (t_in * 15 + t_out * 75) / 1_000_000
                rouge = rouge_l(answer, item["ref"])
                cr = _proxy.compute(answer)
                turn_logs.append(E5TurnLog(
                    question=item["q"][:70], domain=item["domain"],
                    rouge=rouge, thinking_budget=0, thinking_tokens=0,
                    j_score=cr.j_score, cost_usd=cost,
                ))
                tokens_total += t_in + t_out
                cost_total += cost
                time.sleep(0.3)

        else:
            # Use fresh Credence manager per question (isolated, no carry-over state)
            # binary: old behavior — force only LOW → 2000, disable continuous scaling
            # continuous: new behavior — use_thinking=True with continuous governor
            for item in _E5_QUESTIONS:
                mgr = ContextManager(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                    theta_high=0.70, theta_low=0.45,
                    system_prompt=SYSTEM, max_tokens=512,
                    use_thinking=True,
                )
                if condition == "binary_thinking":
                    # Simulate old binary behavior: force prev_j to either 0.0 (LOW)
                    # or 0.80 (HIGH) based on expected domain
                    if item["domain"] == "uncertain":
                        mgr._prev_j = 0.20   # forces full budget
                    else:
                        mgr._prev_j = 0.80   # above theta_high → no budget

                result = mgr.chat(item["q"])
                rouge = rouge_l(result.response, item["ref"])
                turn_logs.append(E5TurnLog(
                    question=item["q"][:70], domain=item["domain"],
                    rouge=rouge,
                    thinking_budget=result.thinking_budget_used,
                    thinking_tokens=result.thinking_tokens,
                    j_score=result.j_score,
                    cost_usd=result.cost_usd,
                ))
                tokens_total += result.tokens_in + result.tokens_out
                cost_total += result.cost_usd
                time.sleep(0.5)

        mean_rouge = sum(t.rouge for t in turn_logs) / len(turn_logs)
        mean_budget = sum(t.thinking_budget for t in turn_logs) / len(turn_logs)
        results.append(E5Result(
            condition=condition,
            turns=turn_logs,
            mean_rouge=mean_rouge,
            total_tokens=tokens_total,
            total_cost_usd=cost_total,
            mean_thinking_budget=mean_budget,
        ))
        print(f"  [{condition}] mean_rouge={mean_rouge:.3f}  "
              f"mean_budget={mean_budget:.0f}  cost=${cost_total:.4f}")

    return results


# ---------------------------------------------------------------------------
# E6 — Negative Needle (Hallucination Safety)
# ---------------------------------------------------------------------------

@dataclass
class E6NeedleLog:
    turn: int
    is_callback: bool
    correct_recall: bool    # answer contains the planted uncertain fact + its qualifier
    hallucinated: bool      # answer gives a specific confident number not from conversation
    answer: str

@dataclass
class E6Result:
    condition: str
    needle_logs: list[E6NeedleLog]
    correction_recall: float   # mean(correct_recall over callbacks)
    hallucination_rate: float  # mean(hallucinated over callbacks)
    tokens_used: int

def run_e6() -> list[E6Result]:
    """
    E6 — Negative Needle (Hallucination Safety)

    Tests whether naive compression causes the model to confidently state
    wrong specific values when the planted uncertainty context is dropped.

    Setup:
      T1-T2: Attention sink (project identity)
      T3:    Plant uncertain constraint A: rate limit "~50 req/min, I'm not certain"
      T4:    Plant uncertain constraint B: token expiry "~24h, docs unclear"
      T5-T12: HIGH-J factual filler → builds compression pressure, pushes T3-T4
              toward the naive window's drop zone
      T13:   Callback A: "What rate limit should we design around?"
      T14:   Callback B: "How long are authentication tokens valid?"

    Three conditions:
      baseline      — full context → correct recall of both constraints + uncertainty
      naive_window  — drops T3-T4 after 6 turns → model may confabulate specific values
      credence          — faithfulness probe detects "I'm not certain" / "unclear" in old
                      segment → refuses COMPRESS → constraints stay in history

    Metrics:
      correct_recall   — answer contains planted value AND uncertainty qualifier
      hallucination    — answer gives a confident specific value (not "I don't know")
                         that DIFFERS from the planted value (or has no uncertainty flag)
    """
    print("\n[E6] Negative Needle (Hallucination Safety) ...")

    SYSTEM = (
        "You are a precise technical assistant helping plan a software integration. "
        "When asked to recall earlier facts, state them exactly as established, "
        "including any uncertainty flags."
    )

    # Uncertain constraints planted at T3-T4
    CONSTRAINT_A_VALUE = "50"     # "~50 requests per minute"
    CONSTRAINT_B_VALUE = "24"     # "~24 hours"

    CONVERSATION_SEED = [
        # T1-T2: identity (attention sink)
        {"role": "user",      "content": "I'm building a rate-limited integration with an external API."},
        {"role": "assistant", "content": "Understood. I'll help you plan the integration."},
        {"role": "user",      "content": "We need to handle retries, backoff, and token refresh."},
        {"role": "assistant", "content": "Got it. Key concerns: rate limit adherence, exponential backoff on 429s, and proactive token refresh before expiry."},
        # T3: uncertain constraint A
        {"role": "user",      "content": "I'm not entirely certain, but I think our rate limit is around 50 requests per minute. I'll need to verify with the vendor."},
        {"role": "assistant", "content": "Noted. Rate limit is approximately 50 req/min — flagged as uncertain, needs vendor verification before implementation."},
        # T4: uncertain constraint B
        {"role": "user",      "content": "Regarding auth tokens: they expire in roughly 24 hours, but the docs aren't clear — it might be less. Please keep this tentative."},
        {"role": "assistant", "content": "Understood. Token expiry is approximately 24 hours, marked as tentative pending documentation clarification."},
    ]

    FILLER_TURNS = [
        "What is exponential backoff and how should I implement it?",
        "What HTTP status code indicates rate limiting?",
        "What is the difference between access tokens and refresh tokens?",
        "What is a circuit breaker pattern?",
        "What is idempotency and why does it matter for retries?",
        "What is connection pooling?",
        "What does the Retry-After HTTP header do?",
        "What is a webhook?",
    ]

    # Callback questions and scoring
    CALLBACKS = [
        (
            "What rate limit should we design our retry logic around?",
            # correct recall: answer has the planted value AND uncertainty qualifier
            ["50", "uncertain", "verify", "not certain", "tentative", "approximately",
             "roughly", "unverified", "unconfirmed", "pending", "flagged"],
            # hallucination markers: confident specific non-planted numbers (no qualifier)
            ["100", "200", "1000", "unlimited"],
        ),
        (
            "How long should our token refresh window be, based on what we discussed?",
            ["24", "uncertain", "tentative", "verify", "not certain", "approximately",
             "roughly", "unverified", "unconfirmed", "pending", "flagged"],
            # Use standalone-number patterns to avoid substring false positives.
            # "6 hour" in "16 hours" and "12 hour" in "12-16 hours" are both FPs.
            # Only flag confident wrong expiry claims (not derivation recommendations).
            [r"\b48\s*hour", r"\b7\s*day", r"\bexpires?\s+in\s+6\b", r"\bexpiry\s+is\s+12\b"],
        ),
    ]

    results = []

    for condition in ["baseline", "naive_window", "credence"]:
        tokens_total = 0
        needle_logs = []

        if condition == "credence":
            mgr = ContextManager(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high=0.70, theta_low=0.45,
                system_prompt=SYSTEM, max_tokens=400,
            )
            # Replay seed conversation through Credence
            for i in range(0, len(CONVERSATION_SEED), 2):
                user_msg = CONVERSATION_SEED[i]["content"]
                r = mgr.chat(user_msg)
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)

            # Filler turns
            for fmsg in FILLER_TURNS:
                r = mgr.chat(fmsg)
                tokens_total += r.tokens_in + r.tokens_out
                turn_n = mgr._turn_idx
                needle_logs.append(E6NeedleLog(turn_n, False, False, False, r.response[:80]))
                time.sleep(0.3)

            # Callbacks
            for turn_offset, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
                r = mgr.chat(q)
                tokens_total += r.tokens_in + r.tokens_out
                ans = r.response
                lower = ans.lower()
                planted_val = CONSTRAINT_A_VALUE if turn_offset == 0 else CONSTRAINT_B_VALUE
                has_value = planted_val in lower
                has_qualifier = any(f.lower() in lower for f in correct_frags if f not in [planted_val])
                correct = has_value and has_qualifier
                # Use regex for hallu_frags to support word-boundary patterns
                hallu = any(re.search(h, lower) for h in hallu_frags)
                needle_logs.append(E6NeedleLog(mgr._turn_idx, True, correct, hallu, ans[:120]))
                print(f"  [credence] Q: {q[:55]}… correct={correct} hallucinated={hallu}")
                time.sleep(0.3)

        else:
            history = list(CONVERSATION_SEED)

            for i, fmsg in enumerate(FILLER_TURNS):
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": fmsg}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": fmsg})
                history.append({"role": "assistant", "content": answer})
                turn_n = len(CONVERSATION_SEED) // 2 + i + 1
                needle_logs.append(E6NeedleLog(turn_n, False, False, False, answer[:80]))
                time.sleep(0.3)

            for turn_offset, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": q}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                ans = answer
                lower = ans.lower()
                planted_val = CONSTRAINT_A_VALUE if turn_offset == 0 else CONSTRAINT_B_VALUE
                has_value = planted_val in lower
                has_qualifier = any(f.lower() in lower for f in correct_frags if f not in [planted_val])
                correct = has_value and has_qualifier
                hallu = any(re.search(h, lower) for h in hallu_frags)
                turn_n = len(CONVERSATION_SEED) // 2 + len(FILLER_TURNS) + turn_offset + 1
                needle_logs.append(E6NeedleLog(turn_n, True, correct, hallu, ans[:120]))
                history.append({"role": "user",      "content": q})
                history.append({"role": "assistant", "content": answer})
                print(f"  [{condition}] Q: {q[:55]}… correct={correct} hallucinated={hallu}")
                time.sleep(0.3)

        cb_logs = [l for l in needle_logs if l.is_callback]
        correction_recall  = sum(l.correct_recall for l in cb_logs) / len(cb_logs) if cb_logs else 0.0
        hallucination_rate = sum(l.hallucinated   for l in cb_logs) / len(cb_logs) if cb_logs else 0.0
        results.append(E6Result(
            condition=condition,
            needle_logs=needle_logs,
            correction_recall=correction_recall,
            hallucination_rate=hallucination_rate,
            tokens_used=tokens_total,
        ))
        print(f"  [{condition}] correction_recall={correction_recall:.2f}  "
              f"hallucination_rate={hallucination_rate:.2f}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# E7 — Multi-Hop Reasoning Chain (RULER-style compression eval)
# ---------------------------------------------------------------------------

@dataclass
class E7HopLog:
    hop_id: int       # 1, 2, or 3
    recalled: bool    # answer contains this hop's key fragment

@dataclass
class E7Result:
    condition: str
    hop_logs: list[E7HopLog]
    chain_complete: bool   # all 3 hops recalled
    hops_recalled: int     # 0-3
    tokens_used: int

def run_e7() -> list[E7Result]:
    """
    E7 — Multi-Hop Reasoning Chain (RULER-style compression eval)

    Tests whether reasoning chains A→B→C survive compression. The answer to
    the final callback requires all 3 planted facts — any dropped hop breaks
    the chain and yields an incomplete or wrong answer.

    Setup:
      T1-T2: Attention sink (project identity)
      T3: Hop 1 — project X uses Nexus config manager
      T4: Hop 2 — Nexus v4.x has security CVE, must upgrade to v5
      T5: Hop 3 — Nexus v5 requires Python 3.10+; current runtime is 3.8
      T6-T11: 6 HIGH-J factual filler turns
      T12: Callback — "What upgrades are needed and what's blocking us?"

    Expected multi-hop answer: Nexus (v4→v5) blocked by Python 3.8 (needs 3.10+)
    Requires chaining: project → Nexus → CVE → v5 → Python requirement.

    Three conditions:
      baseline     — full context → complete chain → correct answer
      naive_window — window=6: at T12, keeps T7-T11; drops T3-T5 → broken chain
      credence         — COMPRESS fires on T3-T5 (HIGH-J facts); Haiku summarizes;
                     quality depends on whether Haiku preserves the full hop chain
    """
    print("\n[E7] Multi-Hop Reasoning Chain ...")

    SYSTEM = (
        "You are a technical advisor helping plan infrastructure upgrades. "
        "When asked what upgrades are needed, trace the full dependency chain."
    )

    CONVERSATION_SEED = [
        # T1-T2: sink
        {"role": "user",      "content": "I'm auditing our backend infrastructure dependencies."},
        {"role": "assistant", "content": "Understood. I'll help you trace dependencies and identify required upgrades."},
        {"role": "user",      "content": "Let's go through our components methodically, noting any issues."},
        {"role": "assistant", "content": "Ready. Please share the components and I'll track any issues as we go."},
        # T3: Hop 1 — project → Nexus
        {"role": "user",      "content": "Our project 'Falcon' uses the Nexus configuration manager for all runtime configs."},
        {"role": "assistant", "content": "Noted. Project Falcon relies on Nexus for configuration management."},
        # T4: Hop 2 — Nexus → CVE, must upgrade to v5
        {"role": "user",      "content": "We're currently on Nexus v4.2. I just found it has a critical CVE — the fix is only in Nexus v5."},
        {"role": "assistant", "content": "Important. Nexus v4.2 is affected by a critical CVE; upgrading to Nexus v5 is required to get the security fix."},
        # T5: Hop 3 — Nexus v5 → Python 3.10+, current is 3.8
        {"role": "user",      "content": "The Nexus v5 migration guide says it requires Python 3.10 or later. Our current runtime is Python 3.8."},
        {"role": "assistant", "content": "Dependency blocker found. Nexus v5 requires Python ≥3.10, but Falcon's runtime is Python 3.8. To fix the CVE, you must first upgrade Python."},
    ]

    FILLER_TURNS = [
        "What is the difference between a load balancer and an API gateway?",
        "What does container orchestration mean?",
        "What is a service mesh?",
        "What is blue-green deployment?",
        "What is a canary release?",
        "What is eventual consistency?",
    ]

    CALLBACK = (
        "Given everything we've established about Project Falcon, "
        "what upgrades do we need and what's blocking us from doing them?",
    )

    # Scoring: each hop has a unique fragment; all 3 must be present for chain_complete
    HOP_FRAGMENTS = [
        ("Nexus", ["nexus", "configuration manager"]),                # hop 1
        ("CVE/upgrade", ["cve", "security", "v5", "version 5"]),      # hop 2
        ("Python", ["python 3.10", "python3.10", "3.10", "python requirement"]),  # hop 3
    ]

    results = []

    for condition in ["baseline", "naive_window", "credence"]:
        tokens_total = 0
        hop_logs = []

        if condition == "credence":
            mgr = ContextManager(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high=0.70, theta_low=0.45,
                system_prompt=SYSTEM, max_tokens=400,
            )
            # Replay seed through Credence
            for i in range(0, len(CONVERSATION_SEED), 2):
                r = mgr.chat(CONVERSATION_SEED[i]["content"])
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)

            # Filler
            for fmsg in FILLER_TURNS:
                r = mgr.chat(fmsg)
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)

            # Callback
            r = mgr.chat(CALLBACK[0])
            tokens_total += r.tokens_in + r.tokens_out
            answer = r.response

        else:
            history = list(CONVERSATION_SEED)
            for fmsg in FILLER_TURNS:
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": fmsg}]
                ans, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": fmsg})
                history.append({"role": "assistant", "content": ans})
                time.sleep(0.3)

            if condition == "naive_window":
                history = history[-12:]
            msgs = history + [{"role": "user", "content": CALLBACK[0]}]
            answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
            tokens_total += t_in + t_out

        lower = answer.lower()
        for hop_id, (hop_name, frags) in enumerate(HOP_FRAGMENTS, start=1):
            recalled = any(f in lower for f in frags)
            hop_logs.append(E7HopLog(hop_id, recalled))
            print(f"  [{condition}] hop{hop_id} ({hop_name}): {'✓' if recalled else '✗'}")

        hops = sum(l.recalled for l in hop_logs)
        chain_ok = hops == len(HOP_FRAGMENTS)
        results.append(E7Result(
            condition=condition,
            hop_logs=hop_logs,
            chain_complete=chain_ok,
            hops_recalled=hops,
            tokens_used=tokens_total,
        ))
        print(f"  [{condition}] hops_recalled={hops}/{len(HOP_FRAGMENTS)}  "
              f"chain_complete={chain_ok}  tokens={tokens_total:,}")
        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# E8 — Real Debugging Session (end-to-end realistic)
# ---------------------------------------------------------------------------

@dataclass
class E8CallbackLog:
    question: str
    recall_score: float
    answer: str

@dataclass
class E8Result:
    condition: str
    callback_logs: list[E8CallbackLog]
    mean_recall: float
    tokens_used: int

def run_e8() -> list[E8Result]:
    """
    E8 — Real Debugging Session (end-to-end realistic)

    The most realistic test: a multi-turn debugging conversation with three
    types of information that need different preservation strategies:

      Original error (HIGH-J specific fact)     → should survive compression
      Uncertain hypothesis (LOW-J)              → faithfulness probe protects it
      Attempted fix + outcome (reasoning chain) → E7-style chain preservation

    Setup:
      T1-T2 : Attention sink (project identity)
      T3    : Plant bug — specific RuntimeError with file/line (HIGH-J if resolved)
      T4    : Plant uncertain hypothesis — "might be threading or GIL, not sure" (LOW-J)
      T5-T10: HIGH-J diagnostic/factual exchanges (compression pressure)
      T11   : Attempted fix description (MEDIUM-J — code change tried)
      T12   : Fix outcome — partial, new symptom (LOW-J uncertain again)
      T13-T16: More diagnostics
      T17-T19: Callbacks

    Three conditions: baseline, naive_window (window=6), credence
    """
    print("\n[E8] Real Debugging Session ...")

    SYSTEM = (
        "You are a senior engineer helping debug a production system. "
        "Remember all established facts, hypotheses, and attempted fixes. "
        "When recalling information, include any uncertainty flags exactly as stated."
    )

    SEED_MESSAGES = [
        # T1-T2: identity
        "I'm debugging a race condition in our payment processing service.",
        "It's a Python FastAPI service handling ~500 concurrent requests per second.",
        # T3: specific error — HIGH-J (precise, factual)
        "Here's the exact error from production logs:\n\nRuntimeError: dictionary changed size during iteration\n  File 'payment_processor.py', line 147, in process_pending\n    for txn_id, txn in self.pending_transactions.items():\nThis happens roughly every 2-3 hours under high load.",
        # T4: uncertain hypothesis — LOW-J (should be preserved by faithfulness probe)
        "My hypothesis is it might be a threading issue with the GIL, or possibly our Redis connection pool isn't thread-safe — I'm genuinely not sure which. Could be either one, or both.",
        # T5-T10: HIGH-J factual filler (compression pressure builds here)
        "What is the GIL and how does it affect thread safety in Python?",
        "What is the difference between threading.Lock and threading.RLock?",
        "How does Redis connection pooling work?",
        "What is the difference between a race condition and a deadlock?",
        "What does asyncio.gather() do?",
        "What is a context manager in Python?",
        # T11: attempted fix — MEDIUM-J (code change)
        "I tried wrapping the iteration with list(self.pending_transactions.items()) to take a snapshot first. Deployed it yesterday.",
        # T12: outcome — LOW-J uncertain (new symptom, uncertainty remains)
        "The original error stopped, but now we're occasionally seeing KeyError in the same function — different line though (line 152). Not sure if this is related to my fix or a separate issue.",
    ]

    CALLBACKS = [
        (
            "What was the exact original error and which file and line number did it occur on?",
            ["RuntimeError", "dictionary changed size", "payment_processor", "147", "process_pending"],
        ),
        (
            "What two uncertain hypotheses did we have about the root cause?",
            ["threading", "GIL", "Redis", "connection pool", "not sure", "uncertain"],
        ),
        (
            "What fix did we try and what happened after we deployed it?",
            ["list", "snapshot", "items", "KeyError", "152"],
        ),
    ]

    results = []

    for condition in ["baseline", "naive_window", "credence"]:
        tokens_total = 0
        callback_logs = []

        if condition == "credence":
            mgr = ContextManager(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high=0.70, theta_low=0.45,
                system_prompt=SYSTEM, max_tokens=400,
            )
            for msg in SEED_MESSAGES:
                r = mgr.chat(msg)
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)
            for q, fragments in CALLBACKS:
                r = mgr.chat(q)
                score = _score_recall(r.response, fragments)
                callback_logs.append(E8CallbackLog(q, score, r.response[:150]))
                tokens_total += r.tokens_in + r.tokens_out
                print(f"  [credence] Q: {q[:60]}… recall={score:.2f}")
                time.sleep(0.3)

        else:
            history = []
            for msg in SEED_MESSAGES:
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": msg}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": msg})
                history.append({"role": "assistant", "content": answer})
                time.sleep(0.3)

            for q, fragments in CALLBACKS:
                if condition == "naive_window":
                    history = history[-12:]
                msgs = history + [{"role": "user", "content": q}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                score = _score_recall(answer, fragments)
                callback_logs.append(E8CallbackLog(q, score, answer[:150]))
                history.append({"role": "user",      "content": q})
                history.append({"role": "assistant", "content": answer})
                print(f"  [{condition}] Q: {q[:60]}… recall={score:.2f}")
                time.sleep(0.3)

        mean_recall = sum(l.recall_score for l in callback_logs) / len(callback_logs) if callback_logs else 0.0
        results.append(E8Result(
            condition=condition,
            callback_logs=callback_logs,
            mean_recall=mean_recall,
            tokens_used=tokens_total,
        ))
        print(f"  [{condition}] mean_recall={mean_recall:.3f}  tokens={tokens_total:,}")

    return results


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def _print_e1(results: list[E1Result]):
    print("\n" + "="*70)
    print("E1 — PROPAGATION CHAIN RESULTS")
    print("="*70)
    for r in results:
        bar = "█" * int(r.mean_recall * 20)
        print(f"  {r.condition:<20} mean_recall={r.mean_recall:.3f}  {bar}")
    base = next((r for r in results if r.condition == "baseline"), None)
    credence_r = next((r for r in results if r.condition == "credence"), None)
    if base and credence_r:
        print(f"\n  Credence vs baseline recall delta: {credence_r.mean_recall - base.mean_recall:+.3f}")

def _print_e2(results: list[E2Result]):
    print("\n" + "="*70)
    print("E2 — CONFIDENT ERROR TRAP (Type Prior ablation)")
    print("="*70)
    for r in results:
        print(f"  {r.condition:<25} mean_recall={r.mean_recall:.3f}  "
              f"exception={r.exception_recall:.2f}  line_num={r.line_number_recall:.2f}")

def _print_e3(results: list[E3Result]):
    print("\n" + "="*70)
    print("E3 — COGNITIVE FRICTION (dual-signal ablation)")
    print("="*70)
    for r in results:
        print(f"  {r.condition:<15} mean_recall={r.mean_followup_recall:.3f}  "
              f"friction_turns={r.cognitive_friction_turns}")

def _print_e4(results: list[E4Result]):
    print("\n" + "="*70)
    print("E4 — CORRECTNESS CLIFF (20-turn decay)")
    print("="*70)
    for r in results:
        recall_str = "  ".join(f"T{t}:{v:.2f}" for t, v in sorted(r.callback_recall_by_turn.items()))
        print(f"  {r.condition:<15} mean={r.mean_callback_recall:.3f}  {recall_str}")
    credence_r = next((r for r in results if r.condition == "credence"), None)
    if credence_r and credence_r.drift_activations:
        print(f"\n  Credence drift activations: {credence_r.drift_activations}")

def _print_e5(results: list[E5Result]):
    print("\n" + "="*70)
    print("E5 — THINKING BUDGET ABLATION")
    print("="*70)
    print(f"  {'Condition':<25}  ROUGE-L  Cost($)   Avg Budget")
    print("  " + "-"*55)
    for r in results:
        print(f"  {r.condition:<25}  {r.mean_rouge:.3f}    {r.total_cost_usd:.4f}    {r.mean_thinking_budget:.0f} tok")

def _print_e7(results: list[E7Result]):
    print("\n" + "="*70)
    print("E7 — MULTI-HOP REASONING CHAIN")
    print("="*70)
    print(f"  {'Condition':<20}  Hops/3  ChainOK  Tokens")
    print("  " + "-"*52)
    for r in results:
        print(f"  {r.condition:<20}  {r.hops_recalled}/3     {'✓' if r.chain_complete else '✗'}       {r.tokens_used:,}")
    credence_r = next((r for r in results if r.condition == "credence"), None)
    naive_r = next((r for r in results if r.condition == "naive_window"), None)
    base_r  = next((r for r in results if r.condition == "baseline"), None)
    if credence_r and naive_r:
        delta = credence_r.hops_recalled - naive_r.hops_recalled
        print(f"\n  Credence vs naive: +{delta} hops preserved through compression")
    if credence_r and base_r:
        if credence_r.hops_recalled == base_r.hops_recalled:
            print(f"  ✓ Credence preserved full chain despite compression (matches baseline)")
        else:
            lost = base_r.hops_recalled - credence_r.hops_recalled
            print(f"  △ Credence lost {lost} hop(s) in Haiku summary — compression degrades multi-hop chains")

def _print_e6(results: list[E6Result]):
    print("\n" + "="*70)
    print("E6 — NEGATIVE NEEDLE (Hallucination Safety)")
    print("="*70)
    print(f"  {'Condition':<20}  Correction%  Hallucination%  Tokens")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r.condition:<20}  {r.correction_recall:.0%}         {r.hallucination_rate:.0%}            {r.tokens_used:,}")
    credence_r = next((r for r in results if r.condition == "credence"), None)
    base_r = next((r for r in results if r.condition == "baseline"), None)
    naive_r = next((r for r in results if r.condition == "naive_window"), None)
    if credence_r and naive_r:
        print(f"\n  Credence vs Naive — correction recall delta : {credence_r.correction_recall - naive_r.correction_recall:+.2f}")
        print(f"  Credence vs Naive — hallucination rate delta: {credence_r.hallucination_rate - naive_r.hallucination_rate:+.2f}")
    if credence_r and base_r:
        cr_gap = base_r.correction_recall - credence_r.correction_recall
        if cr_gap <= 0.05:
            print(f"  ✓ Credence correction recall within 5% of baseline (faithfulness probe working)")
        else:
            print(f"  △ Credence correction recall {cr_gap:.0%} below baseline — probe may need tuning")

def _print_e8(results: list[E8Result]):
    print("\n" + "="*70)
    print("E8 — REAL DEBUGGING SESSION")
    print("="*70)
    print(f"  {'Condition':<20}  Mean  | Error  Hypothesis  Fix")
    print("  " + "-"*58)
    for r in results:
        scores = [f"{l.recall_score:.2f}" for l in r.callback_logs]
        score_str = "  ".join(scores) if scores else "—"
        print(f"  {r.condition:<20}  {r.mean_recall:.3f} | {score_str}")
    credence_r  = next((r for r in results if r.condition == "credence"), None)
    base_r  = next((r for r in results if r.condition == "baseline"), None)
    naive_r = next((r for r in results if r.condition == "naive_window"), None)
    if credence_r and naive_r:
        print(f"\n  Credence vs Naive recall delta: {credence_r.mean_recall - naive_r.mean_recall:+.3f}")
    if credence_r and base_r:
        gap = base_r.mean_recall - credence_r.mean_recall
        if gap <= 0.05:
            print(f"  ✓ Credence within 5% of baseline recall (all 3 information types preserved)")
        else:
            print(f"  △ Credence {gap:.0%} below baseline — compression losing some context")


# ---------------------------------------------------------------------------
# E9 — Compression Under Fire
#
# THE experiment the audit said was missing.
#
# E6 used 8 filler turns (12 total). COMPRESS fires at n_turns > 16.
# So E6 Credence = full context. The mechanism was never tested.
#
# E9 uses 18 filler turns (22 total). Compression fires at turn 17.
# The probe and sentinel must protect uncertain seed turns through an
# actual Haiku compression event. This is the direct test of the claim.
#
# Three conditions:
#   baseline     — full context (oracle)
#   naive_window — 12-message window (T3-T4 always dropped)
#   credence     — compression fires, probe/sentinel must protect T3-T4
#
# If credence matches baseline under real compression: the claim holds.
# If credence fails: the probe has a gap that must be fixed.
# ---------------------------------------------------------------------------

@dataclass
class E9Result:
    condition:          str
    correction_recall:  float
    hallucination_rate: float
    compression_fired:  bool    # did Credence actually compress during the session?
    compress_count:     int     # how many times compression was triggered
    tokens_used:        int
    needle_logs:        list    # reuse E6NeedleLog


def run_e9() -> list[E9Result]:
    """
    E9 — Compression Under Fire (the missing experiment).

    Identical to E6 but with 18 filler turns instead of 8.
    COMPRESS fires at n_turns > 16 (COMPRESS_AFTER=8, fires at 8*2=16).
    Session design:
      T1-T4:  seed (attention sink + 2 uncertain constraints)
      T5-T22: 18 filler turns — compression fires around T17
      T23-T24: 2 callback questions

    Compression checkpoint: after T17, Credence attempts to compress
    turns T3-T16. The seed turns (T3-T4) have user_j_flag sentinel=0.0
    (user expressed uncertainty) → they go to preserved_msgs, not Haiku.
    The faithfulness probe is a backup if sentinel missed anything.

    Expected result:
      credence: compression fires, uncertain turns preserved, recall ~100%
      baseline: full context, recall 100% (oracle)
      naive:    12-window drops T3-T4 (deterministic), recall ~0%
    """
    print("\n[E9] Compression Under Fire (compression actually fires) ...")

    SYSTEM = (
        "You are a precise technical assistant helping plan a software integration. "
        "When asked to recall earlier facts, state them exactly as established, "
        "including any uncertainty flags."
    )

    CONSTRAINT_A_VALUE = "50"
    CONSTRAINT_B_VALUE = "24"

    CONVERSATION_SEED = [
        {"role": "user",      "content": "I'm building a rate-limited integration with an external API."},
        {"role": "assistant", "content": "Understood. I'll help you plan the integration."},
        {"role": "user",      "content": "We need to handle retries, backoff, and token refresh."},
        {"role": "assistant", "content": "Got it. Key concerns: rate limit adherence, exponential backoff on 429s, and proactive token refresh before expiry."},
        # T3: uncertain constraint A — user_j_flag sentinel will be set
        {"role": "user",      "content": "I'm not entirely certain, but I think our rate limit is around 50 requests per minute. I'll need to verify with the vendor."},
        {"role": "assistant", "content": "Noted. Rate limit is approximately 50 req/min — flagged as uncertain, needs vendor verification before implementation."},
        # T4: uncertain constraint B — user_j_flag sentinel will be set
        {"role": "user",      "content": "Regarding auth tokens: they expire in roughly 24 hours, but the docs aren't clear — it might be less. Please keep this tentative."},
        {"role": "assistant", "content": "Understood. Token expiry is approximately 24 hours, marked as tentative pending documentation clarification."},
    ]

    # 18 filler turns — crosses the COMPRESS_AFTER=8 threshold at turn 9 (n_turns > 16)
    FILLER_TURNS = [
        "What is exponential backoff and how should I implement it?",
        "What HTTP status code indicates rate limiting?",
        "What is the difference between access tokens and refresh tokens?",
        "What is a circuit breaker pattern?",
        "What is idempotency and why does it matter for retries?",
        "What is connection pooling?",
        "What does the Retry-After HTTP header do?",
        "What is a webhook and how does it differ from polling?",
        "How should we structure error handling for network failures?",
        "What is the difference between synchronous and asynchronous API calls?",
        "How do we implement request deduplication?",
        "What is the purpose of a health check endpoint?",
        "How should we handle partial failures in distributed systems?",
        "What is a dead letter queue?",
        "How do we implement graceful degradation?",
        "What is service discovery in microservices?",
        "How should we version our API endpoints?",
        "What is the strangler fig pattern for migration?",
    ]

    CALLBACKS = [
        (
            "What rate limit should we design our retry logic around?",
            ["50", "uncertain", "verify", "not certain", "tentative", "approximately",
             "roughly", "unverified", "unconfirmed", "pending", "flagged"],
            ["100", "200", "1000", "unlimited"],
        ),
        (
            "How long should our token refresh window be, based on what we discussed?",
            ["24", "uncertain", "tentative", "verify", "not certain", "approximately",
             "roughly", "unverified", "unconfirmed", "pending", "flagged"],
            [r"\b48\s*hour", r"\b7\s*day", r"\bexpires?\s+in\s+6\b", r"\bexpiry\s+is\s+12\b"],
        ),
    ]

    results = []

    for condition in ["baseline", "naive_window", "credence"]:
        tokens_total     = 0
        needle_logs      = []
        compression_fired = False
        compress_count   = 0

        if condition == "credence":
            mgr = ContextManager(
                api_key       = os.environ.get("ANTHROPIC_API_KEY", ""),
                theta_high    = 0.70,
                theta_low     = 0.45,
                system_prompt = SYSTEM,
                max_tokens    = 400,
            )
            for i in range(0, len(CONVERSATION_SEED), 2):
                r = mgr.chat(CONVERSATION_SEED[i]["content"])
                tokens_total += r.tokens_in + r.tokens_out
                time.sleep(0.3)

            for fmsg in FILLER_TURNS:
                r = mgr.chat(fmsg)
                tokens_total += r.tokens_in + r.tokens_out
                if r.decision == "COMPRESS":
                    compression_fired = True
                    compress_count   += 1
                    print(f"  [credence] *** COMPRESSION FIRED at turn {mgr._turn_idx} "
                          f"(saved {r.tokens_saved} tokens) ***")
                needle_logs.append(E6NeedleLog(mgr._turn_idx, False, False, False, r.response[:80]))
                time.sleep(0.3)

            for turn_offset, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
                r = mgr.chat(q)
                tokens_total += r.tokens_in + r.tokens_out
                ans   = r.response
                lower = ans.lower()
                planted_val  = CONSTRAINT_A_VALUE if turn_offset == 0 else CONSTRAINT_B_VALUE
                has_value    = planted_val in lower
                has_qualifier = any(f.lower() in lower for f in correct_frags if f not in [planted_val])
                correct = has_value and has_qualifier
                hallu   = any(re.search(h, lower) for h in hallu_frags)
                needle_logs.append(E6NeedleLog(mgr._turn_idx, True, correct, hallu, ans[:120]))
                print(f"  [credence] Q: {q[:55]}… correct={correct} hallu={hallu}")
                time.sleep(0.3)

        else:
            history = list(CONVERSATION_SEED)

            for i, fmsg in enumerate(FILLER_TURNS):
                if condition == "naive_window":
                    history = history[-12:]
                msgs   = history + [{"role": "user", "content": fmsg}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                history.append({"role": "user",      "content": fmsg})
                history.append({"role": "assistant", "content": answer})
                turn_n = len(CONVERSATION_SEED) // 2 + i + 1
                needle_logs.append(E6NeedleLog(turn_n, False, False, False, answer[:80]))
                time.sleep(0.3)

            for turn_offset, (q, correct_frags, hallu_frags) in enumerate(CALLBACKS):
                if condition == "naive_window":
                    history = history[-12:]
                msgs   = history + [{"role": "user", "content": q}]
                answer, t_in, t_out = _ask(msgs, system=SYSTEM, max_tokens=400)
                tokens_total += t_in + t_out
                lower = answer.lower()
                planted_val  = CONSTRAINT_A_VALUE if turn_offset == 0 else CONSTRAINT_B_VALUE
                has_value    = planted_val in lower
                has_qualifier = any(f.lower() in lower for f in correct_frags if f not in [planted_val])
                correct = has_value and has_qualifier
                hallu   = any(re.search(h, lower) for h in hallu_frags)
                turn_n  = len(CONVERSATION_SEED) // 2 + len(FILLER_TURNS) + turn_offset + 1
                needle_logs.append(E6NeedleLog(turn_n, True, correct, hallu, answer[:120]))
                history.append({"role": "user",      "content": q})
                history.append({"role": "assistant", "content": answer})
                print(f"  [{condition}] Q: {q[:55]}… correct={correct} hallu={hallu}")
                time.sleep(0.3)

        cb_logs           = [l for l in needle_logs if l.is_callback]
        correction_recall = sum(l.correct_recall for l in cb_logs) / len(cb_logs) if cb_logs else 0.0
        hallu_rate        = sum(l.hallucinated   for l in cb_logs) / len(cb_logs) if cb_logs else 0.0

        results.append(E9Result(
            condition         = condition,
            correction_recall = correction_recall,
            hallucination_rate= hallu_rate,
            compression_fired = compression_fired,
            compress_count    = compress_count,
            tokens_used       = tokens_total,
        ))
        print(f"  [{condition}] recall={correction_recall:.2f}  hallu={hallu_rate:.2f}  "
              f"compression_fired={compression_fired}  tokens={tokens_total:,}")

    return results


def _print_e9(results: list[E9Result]):
    print("\n" + "="*70)
    print("E9 — COMPRESSION UNDER FIRE (mechanism directly tested)")
    print("="*70)
    print(f"  {'Condition':<20} {'Recall':>8} {'Hallu':>8} {'Compressed?':>13} {'Tokens':>10}")
    print("  " + "-"*62)
    for r in results:
        fired = "YES ✓" if r.compression_fired else "no"
        print(f"  {r.condition:<20} {r.correction_recall:>8.3f} {r.hallucination_rate:>8.3f} "
              f"{fired:>13} {r.tokens_used:>10,}")

    credence_r = next((r for r in results if r.condition == "credence"), None)
    baseline_r = next((r for r in results if r.condition == "baseline"), None)
    naive_r    = next((r for r in results if r.condition == "naive_window"), None)

    print()
    if credence_r and not credence_r.compression_fired:
        print("  ⚠ WARNING: Compression did not fire in Credence condition.")
        print("    Check COMPRESS_AFTER constant and session length.")
    elif credence_r and credence_r.compression_fired:
        print(f"  ✓ Compression fired {credence_r.compress_count}x during the session.")
        if baseline_r:
            gap = baseline_r.correction_recall - credence_r.correction_recall
            if gap <= 0.05:
                print(f"  ✓ Credence recall within 5% of baseline under compression ({credence_r.correction_recall:.1%} vs {baseline_r.correction_recall:.1%})")
                print(f"  ✓ Probe/sentinel preserved uncertain constraints through Haiku compression.")
            else:
                print(f"  ✗ Credence recall {gap:.0%} below baseline — uncertain turns were lost in compression.")
        if naive_r:
            lift = credence_r.correction_recall - naive_r.correction_recall
            print(f"  Credence vs naive recall lift: {lift:+.1%}")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise(obj):
    if isinstance(obj, (E1Result, E2Result, E3Result, E4Result, E5Result,
                        E6Result, E7Result, E8Result, E9Result,
                        E3TurnLog, E4TurnLog, E5TurnLog,
                        E6NeedleLog, E7HopLog, E8CallbackLog)):
        return asdict(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Credence ablation experiments")
    parser.add_argument("--exp", choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--out", default="evals/experiment_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from evals.claude_code_client import ClaudeCodeClient
            ClaudeCodeClient()  # verify binary works
            print("No API key — using Claude Code client for all model calls.")
        except Exception as e:
            print(f"Error: ANTHROPIC_API_KEY not set and Claude Code client failed: {e}")
            sys.exit(1)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.out,
    )

    # Load existing results so individual experiment runs accumulate rather
    # than overwrite.  Running --exp E1 then --exp E2 will keep both.
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            try:
                all_results = json.load(f)
            except (json.JSONDecodeError, ValueError):
                all_results = {}
    else:
        all_results = {}

    if args.exp in ("E1", "all"):
        r = run_e1(); _print_e1(r); all_results["E1"] = [asdict(x) for x in r]
    if args.exp in ("E2", "all"):
        r = run_e2(); _print_e2(r); all_results["E2"] = [asdict(x) for x in r]
    if args.exp in ("E3", "all"):
        r = run_e3(); _print_e3(r)
        all_results["E3"] = [
            {**asdict(x), "turns": [asdict(t) for t in x.turns]} for x in r
        ]
    if args.exp in ("E4", "all"):
        r = run_e4(); _print_e4(r)
        all_results["E4"] = [
            {**asdict(x), "turn_logs": [asdict(t) for t in x.turn_logs],
             "callback_recall_by_turn": {str(k): v for k, v in x.callback_recall_by_turn.items()}}
            for x in r
        ]
    if args.exp in ("E5", "all"):
        r = run_e5(); _print_e5(r)
        all_results["E5"] = [
            {**asdict(x), "turns": [asdict(t) for t in x.turns]} for x in r
        ]
    if args.exp in ("E6", "all"):
        r = run_e6(); _print_e6(r)
        all_results["E6"] = [
            {**asdict(x), "needle_logs": [asdict(l) for l in x.needle_logs]} for x in r
        ]
    if args.exp in ("E7", "all"):
        r = run_e7(); _print_e7(r)
        all_results["E7"] = [
            {**asdict(x), "hop_logs": [asdict(l) for l in x.hop_logs]} for x in r
        ]
    if args.exp in ("E8", "all"):
        r = run_e8(); _print_e8(r)
        all_results["E8"] = [
            {**asdict(x), "callback_logs": [asdict(l) for l in x.callback_logs]} for x in r
        ]

    if args.exp in ("E9", "all"):
        r = run_e9(); _print_e9(r)
        all_results["E9"] = [
            {**asdict(x), "needle_logs": [asdict(l) for l in x.needle_logs]} for x in r
        ]

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

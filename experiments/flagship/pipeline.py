"""
experiments/flagship/pipeline.py
=================================
EpistemicPipeline — three-condition experiment runner.

Conditions:
  baseline        — Full history, no compression. Oracle condition.
  naive_window    — Keep last NAIVE_WINDOW messages. Naive truncation.
  epistemic_memory — CAMS J-selective compression + faithfulness probe.

Each condition runs the same scenario:
  1. Inject seed turns (establishes uncertain constraints)
  2. Inject filler turns (HIGH-J content that triggers compression)
  3. Send each callback question and evaluate the response

Model-agnostic: the `model` parameter controls which Claude model answers.
The epistemic signal reads output text and is model-independent.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from .scenarios import Scenario, Callback
from .metrics import CallbackResult, ScenarioResult, evaluate_callback, aggregate_scenario

_MODEL_OPUS  = "claude-opus-4-7"
_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_NAIVE_WINDOW = 12   # messages (6 turn-pairs)


# ---------------------------------------------------------------------------
# Baseline condition — full history
# ---------------------------------------------------------------------------

def run_baseline(
    scenario: Scenario,
    client: "anthropic.Anthropic",
    model: str = _MODEL_OPUS,
    verbose: bool = False,
) -> ScenarioResult:
    history = []
    tokens_used = 0

    # Inject seed + filler turns verbatim (no API — they are planted)
    for turn in scenario.seed_turns + scenario.filler_turns:
        history.append({"role": turn.role, "content": turn.content})

    # Callback questions via live API
    callback_results = []
    for cb in scenario.callbacks:
        history.append({"role": "user", "content": cb.question})
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=history,
        )
        answer = resp.content[0].text.strip()
        tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
        history.append({"role": "assistant", "content": answer})

        cr = evaluate_callback(cb.question, answer, cb.required_fragments, cb.forbidden_fragments)
        callback_results.append(cr)
        if verbose:
            print(f"    [baseline] Q: {cb.question[:55]}…  recall={cr.fragment_recall:.2f}  prop_err={cr.propagation_error}")

    return aggregate_scenario(scenario.id, "baseline", callback_results, tokens_used)


# ---------------------------------------------------------------------------
# Naive window condition — last N messages only
# ---------------------------------------------------------------------------

def run_naive_window(
    scenario: Scenario,
    client: "anthropic.Anthropic",
    model: str = _MODEL_OPUS,
    window: int = _NAIVE_WINDOW,
    verbose: bool = False,
) -> ScenarioResult:
    full_history = []
    tokens_used = 0

    for turn in scenario.seed_turns + scenario.filler_turns:
        full_history.append({"role": turn.role, "content": turn.content})

    callback_results = []
    for cb in scenario.callbacks:
        full_history.append({"role": "user", "content": cb.question})

        # Truncate to last `window` messages
        windowed = full_history[-window:]
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=windowed,
        )
        answer = resp.content[0].text.strip()
        tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
        full_history.append({"role": "assistant", "content": answer})

        cr = evaluate_callback(cb.question, answer, cb.required_fragments, cb.forbidden_fragments)
        callback_results.append(cr)
        if verbose:
            print(f"    [naive]    Q: {cb.question[:55]}…  recall={cr.fragment_recall:.2f}  prop_err={cr.propagation_error}")

    return aggregate_scenario(scenario.id, "naive_window", callback_results, tokens_used)


# ---------------------------------------------------------------------------
# Epistemic Memory condition — CAMS J-selective compression
# ---------------------------------------------------------------------------

def run_epistemic_memory(
    scenario: Scenario,
    client: "anthropic.Anthropic",
    model: str = _MODEL_OPUS,
    verbose: bool = False,
) -> ScenarioResult:
    from cams.context_manager import CAMSContextManager

    mgr = CAMSContextManager(theta_high=0.70, theta_low=0.45)
    tokens_used = 0
    tokens_saved_total = 0

    # Prime the manager with seed + filler turns.
    # We call mgr.chat() for each user turn and inject assistant turns.
    # For planted turns (we control the assistant text), we bypass the API
    # and directly insert into history to keep the scenario deterministic.
    # Only callback turns use live API.
    for turn in scenario.seed_turns + scenario.filler_turns:
        if turn.role == "user":
            _pending_user = turn.content
        else:
            # Inject assistant turn directly — bypass API for planted content
            mgr._history.append({"role": "user", "content": _pending_user})
            mgr._history.append({"role": "assistant", "content": turn.content})
            mgr._turn_idx += 1

            # Compute J and update vocab for this planted turn
            from cams.confidence_proxy import ConfidenceProxy
            proxy = ConfidenceProxy(theta_high=mgr.proxy.theta_high, theta_low=mgr.proxy.theta_low)
            cr = proxy.compute(turn.content)
            mgr._history_j_scores.extend([0.0, cr.j_score])  # user=0, asst=j_score
            mgr._j_buffer.append(cr.j_score)
            mgr._update_content_vocab(turn.content)
            mgr._prev_zone = cr.zone

    # Callback turns use live API (via mgr.chat)
    callback_results = []
    for cb in scenario.callbacks:
        result = mgr.chat(cb.question)
        tokens_used += result.tokens_used
        tokens_saved_total += result.tokens_saved

        cr = evaluate_callback(cb.question, result.response, cb.required_fragments, cb.forbidden_fragments)
        callback_results.append(cr)
        if verbose:
            print(f"    [em]       Q: {cb.question[:55]}…  recall={cr.fragment_recall:.2f}  "
                  f"prop_err={cr.propagation_error}  J={result.j_score:.3f}  dec={result.decision}")

    return aggregate_scenario(scenario.id, "epistemic_memory", callback_results, tokens_used, tokens_saved_total)


# ---------------------------------------------------------------------------
# Run a single scenario across all three conditions
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    scenario_id: str
    trial: int
    baseline: ScenarioResult
    naive_window: ScenarioResult
    epistemic_memory: ScenarioResult


def run_scenario_trial(
    scenario: Scenario,
    client: "anthropic.Anthropic",
    trial: int = 1,
    model: str = _MODEL_OPUS,
    verbose: bool = False,
) -> TrialResult:
    if verbose:
        print(f"\n  [Scenario {scenario.id}: {scenario.name}] Trial {trial}")

    baseline = run_baseline(scenario, client, model=model, verbose=verbose)
    naive    = run_naive_window(scenario, client, model=model, verbose=verbose)
    em       = run_epistemic_memory(scenario, client, model=model, verbose=verbose)

    return TrialResult(
        scenario_id=scenario.id,
        trial=trial,
        baseline=baseline,
        naive_window=naive,
        epistemic_memory=em,
    )

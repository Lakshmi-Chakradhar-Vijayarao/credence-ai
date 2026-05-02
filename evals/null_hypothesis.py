"""
evals/null_hypothesis.py
========================
Control experiment: does explicit Haiku prompting preserve uncertainty qualifiers?

Research question
-----------------
The faithfulness probe prevents EQL by aborting compression entirely.
A simpler approach: add "CRITICAL: preserve all uncertainty qualifiers" to the
Haiku compression prompt. Does this instruction reliably achieve EQLR=0%?

Prediction (from instruction-following literature)
---------------------------------------------------
Tian et al. (2025) report ~38% reliability on nuanced instruction variants.
We predict enhanced-prompt EQLR ~10-20% (vs 46% naive, 0% probe-guarded).
The gap between enhanced-prompt and probe is the value proposition of the probe:
  - Enhanced prompt: probabilistic compliance
  - Probe: deterministic abort (no model cooperation required)

Run
---
    python -m evals.null_hypothesis              # n=50
    python -m evals.null_hypothesis --n 10       # quick sample
    python -m evals.null_hypothesis --dry-run    # schema/import check only

Requires: ANTHROPIC_API_KEY (both conditions call Haiku + Opus)
Saves:    evals/null_hypothesis_results.json
"""

from __future__ import annotations
import os, sys, json, argparse, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from credence.context_manager import _UNCERTAINTY_MARKERS

_MODEL_HAIKU = "claude-haiku-4-5-20251001"
_MODEL_OPUS  = "claude-opus-4-7"

_NAIVE_PROMPT = (
    "Summarize this conversation in 2-3 sentences, preserving key facts."
)

_ENHANCED_PROMPT = (
    "Summarize this conversation in 2-3 sentences, preserving key facts. "
    "CRITICAL: You MUST preserve all uncertainty qualifiers, hedging language, "
    "and modal verbs exactly as stated — words like 'might', 'probably', "
    "'I think', 'unconfirmed', 'approximately', 'haven\\'t verified'. "
    "If the original expresses uncertainty about a value, your summary MUST "
    "express the same uncertainty. Do not convert uncertain claims into "
    "confident statements."
)


def _has_uncertainty(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _UNCERTAINTY_MARKERS)


def _call(client, model: str, messages: list[dict], max_tokens: int = 250,
          system: str | None = None) -> str:
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip()


def run(client, conversation: list[dict], callback_question: str, index: int) -> dict:
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)

    # Naive compression
    naive_summary = _call(client, _MODEL_HAIKU, [
        {"role": "user", "content": f"{_NAIVE_PROMPT}\n\n{conv_text}"}
    ])
    naive_qual = _has_uncertainty(naive_summary)
    time.sleep(0.35)

    # Enhanced prompt compression
    enhanced_summary = _call(client, _MODEL_HAIKU, [
        {"role": "user", "content": f"{_ENHANCED_PROMPT}\n\n{conv_text}"}
    ])
    enhanced_qual = _has_uncertainty(enhanced_summary)
    time.sleep(0.35)

    # Downstream answers
    system_ctx = "You are a technical assistant. Answer concisely."
    naive_answer = _call(client, _MODEL_OPUS, [
        {"role": "user", "content": f"Context:\n{naive_summary}\n\nQuestion: {callback_question}"}
    ], system=system_ctx)
    time.sleep(0.35)

    enhanced_answer = _call(client, _MODEL_OPUS, [
        {"role": "user", "content": f"Context:\n{enhanced_summary}\n\nQuestion: {callback_question}"}
    ], system=system_ctx)
    time.sleep(0.35)

    result = {
        "index": index,
        "naive_qualifier_survived": naive_qual,
        "naive_summary": naive_summary,
        "naive_answer": naive_answer,
        "enhanced_qualifier_survived": enhanced_qual,
        "enhanced_summary": enhanced_summary,
        "enhanced_answer": enhanced_answer,
    }

    print(f"  [{index+1:02d}] naive={'+' if naive_qual else '-'}  "
          f"enhanced={'+' if enhanced_qual else '-'}  "
          f"naive_summary={naive_summary[:60].replace(chr(10),' ')!r}")
    return result


def _binom_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    margin = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - margin), min(1.0, p + margin)


def main():
    parser = argparse.ArgumentParser(description="Null hypothesis: does prompting Haiku solve EQL?")
    parser.add_argument("--n",        type=int, default=50, help="Number of scenarios (max 50)")
    parser.add_argument("--dry-run",  action="store_true",  help="Schema check only, no API calls")
    parser.add_argument("--out",      default="evals/null_hypothesis_results.json")
    args = parser.parse_args()

    # Import scenarios from main study
    from evals.compression_faithfulness import SCENARIOS, _build_conversation

    if args.dry_run:
        print(f"✓ Dry run: {len(SCENARIOS)} scenarios available, top {args.n} would run")
        print(f"  Models: compress={_MODEL_HAIKU}  downstream={_MODEL_OPUS}")
        print(f"  Enhanced prompt: {_ENHANCED_PROMPT[:80]}...")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and _ANTHROPIC_AVAILABLE:
        client = anthropic.Anthropic(api_key=api_key)
        print("Using Anthropic API client (Haiku for compression, Opus for downstream)")
    else:
        try:
            from evals.claude_code_client import ClaudeCodeClient
            client = ClaudeCodeClient()
            print(f"Using Claude Code client (no API key needed): {client._version}")
            print("Note: both compression and downstream use the same Claude Code model.")
        except Exception as e:
            print(f"ERROR: No API key and Claude Code client failed: {e}")
            sys.exit(1)

    n = min(args.n, 100)  # SCENARIOS now has 100 entries
    print(f"\nNull Hypothesis Experiment — n={n}")
    print(f"Question: does Haiku + explicit qualifier-preservation instruction achieve EQLR=0%?")
    print(f"Prediction: EQLR ~10-20% (vs 46% naive, 0% probe-guarded)\n")

    results = []
    for i, (stmt, label, question) in enumerate(SCENARIOS[:n]):
        conversation = _build_conversation(stmt)
        r = run(client, conversation, question, i)
        results.append(r)

    n_results = len(results)
    naive_survived     = sum(1 for r in results if r["naive_qualifier_survived"])
    enhanced_survived  = sum(1 for r in results if r["enhanced_qualifier_survived"])

    naive_eqlr    = (n_results - naive_survived)    / n_results
    enhanced_eqlr = (n_results - enhanced_survived) / n_results

    naive_lo,    naive_hi    = _binom_ci(n_results - naive_survived,    n_results)
    enhanced_lo, enhanced_hi = _binom_ci(n_results - enhanced_survived, n_results)

    summary = {
        "n": n_results,
        "naive_qualifier_survival":    round(naive_survived    / n_results, 3),
        "enhanced_qualifier_survival": round(enhanced_survived / n_results, 3),
        "naive_eqlr":    round(naive_eqlr,    3),
        "enhanced_eqlr": round(enhanced_eqlr, 3),
        "naive_eqlr_ci":    [round(naive_lo, 3),    round(naive_hi, 3)],
        "enhanced_eqlr_ci": [round(enhanced_lo, 3), round(enhanced_hi, 3)],
        "probe_eqlr":      0.000,
        "improvement_over_naive":     round(naive_eqlr - enhanced_eqlr, 3),
        "remaining_gap_to_probe":     round(enhanced_eqlr,               3),
    }

    print(f"\n{'='*60}")
    print(f"NULL HYPOTHESIS RESULTS (n={n_results})")
    print(f"{'='*60}")
    print(f"  Naive Haiku EQLR:    {naive_eqlr:.1%} (CI: {naive_lo:.1%}–{naive_hi:.1%})")
    print(f"  Enhanced prompt EQLR:{enhanced_eqlr:.1%} (CI: {enhanced_lo:.1%}–{enhanced_hi:.1%})")
    print(f"  Probe-guarded EQLR:   0.0% (deterministic, from n=50 study)")
    print(f"")
    print(f"  Improvement (naive→enhanced):   {naive_eqlr - enhanced_eqlr:.1%}")
    print(f"  Remaining gap (enhanced→probe): {enhanced_eqlr:.1%}")
    print(f"")
    if enhanced_eqlr < 0.05:
        print("  FINDING: Enhanced prompt achieves near-zero EQLR.")
        print("  Probe value is in determinism and zero-latency pipeline contexts.")
    elif enhanced_eqlr <= 0.20:
        print(f"  FINDING: Enhanced prompt reduces EQLR but does not eliminate it.")
        print(f"  Probe provides deterministic guarantee; prompt does not.")
    else:
        print(f"  FINDING: Enhanced prompt substantially reduces but does not solve EQL.")
        print(f"  Probe is necessary for EQLR=0% guarantee.")

    output = {"summary": summary, "scenarios": results}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

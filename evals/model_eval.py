"""
evals/model_eval.py
===================
FCR evaluation for the fine-tuned Phi-2 LoRA model against EQL-Bench v2.

Compares microsoft/phi-2 (base) vs microsoft/phi-2 + LoRA adapter on the
False Certainty Rate (FCR) metric: does the fine-tuned model strip uncertainty
qualifiers less often when summarising uncertain technical claims?

Designed to run on Kaggle T4 GPU. Has --dry-run mode for local validation
(no model loading, no GPU required).

Usage
-----
    python -m evals.model_eval --dry-run
    python -m evals.model_eval --model-path models/credence-phi-2-dpo/credence-dpo-final --n 20
    python -m evals.model_eval --model-path models/credence-phi-2-dpo/credence-dpo-final --all

Output
------
    evals/model_eval_results.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Optional heavy imports (GPU / transformers) — lazy-loaded at runtime
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

# ---------------------------------------------------------------------------
# Metrics (pure Python, no GPU)
# ---------------------------------------------------------------------------
from evals.kv_cache.metrics import eqlr_token, fcr, aggregate, bootstrap_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EQL_BENCH_V2 = os.path.join(
    os.path.dirname(__file__), "eql_bench", "eql_bench_v2.json"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model_eval_results.json")

PROMPT_TEMPLATE = (
    "Summarize the following technical statement in 1-2 sentences, "
    "preserving any uncertainty:\n\n{uncertain_statement}"
)

MAX_NEW_TOKENS = 150
BASE_MODEL_NAME = "microsoft/phi-2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eql_bench(path: str = EQL_BENCH_V2) -> list[dict]:
    """Load and return scenarios from EQL-Bench v2 JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"EQL-Bench v2 not found at '{path}'. "
            "Run: python -m evals.eql_bench --generate  (or ensure the file exists)."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenarios found in '{path}'.")
    return scenarios


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_tokenizer(base_model: str) -> "AutoTokenizer":
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_base_model(base_model: str, device: str) -> "AutoModelForCausalLM":
    """Load the base Phi-2 model (no LoRA)."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model


def _load_lora_model(
    base_model: str,
    adapter_path: str,
    device: str,
) -> "AutoModelForCausalLM":
    """Load the base model then attach the LoRA adapter."""
    base = _load_base_model(base_model, device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def _generate(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
) -> str:
    """Run greedy generation and return the decoded output (new tokens only)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _score_scenario(scenario: dict, answer: str) -> dict:
    """Score one scenario and return per-scenario result dict."""
    value_frags = scenario.get("value_fragments", [])
    qual_frags  = scenario.get("qualifier_fragments", [])
    eqlr        = eqlr_token(answer, qual_frags)
    fcr_score   = fcr(answer, value_frags, qual_frags)
    return {
        "scenario_id":       scenario.get("scenario_id", "?"),
        "domain":            scenario.get("domain", "?"),
        "qualifier_type":    scenario.get("qualifier_type", "?"),
        "uncertain_statement": scenario.get("uncertain_statement", "")[:120],
        "answer":            answer[:300],
        "eqlr_token":        eqlr,
        "fcr":               fcr_score,
        "value_recalled":    not fcr_score and not eqlr,
        "qualifier_present": not eqlr,
    }


def run_evaluation(
    adapter_path: str,
    base_model:   str = BASE_MODEL_NAME,
    n_scenarios:  int = 20,
    all_scenarios: bool = False,
) -> dict:
    """
    Run the full evaluation: base model vs LoRA model on EQL-Bench v2.

    Returns a results dict ready for JSON serialisation.
    """
    if not _HAS_TORCH or not _HAS_TRANSFORMERS:
        raise ImportError(
            "torch and transformers are required for model evaluation. "
            "Install with: pip install torch transformers peft"
        )
    if not _HAS_PEFT:
        raise ImportError(
            "peft is required for LoRA loading. "
            "Install with: pip install peft"
        )

    # Load dataset
    scenarios = load_eql_bench()
    if not all_scenarios:
        scenarios = scenarios[:n_scenarios]
    total = len(scenarios)

    device = _device()
    print(f"Device: {device}")
    print(f"Evaluating {total} scenarios from EQL-Bench v2")
    print(f"Base model:   {base_model}")
    print(f"Adapter path: {adapter_path}")
    print()

    # Load tokenizer (shared)
    print("Loading tokenizer...")
    tokenizer = _load_tokenizer(base_model)

    # ------------------------------------------------------------------
    # Condition 1: phi2_base (no LoRA)
    # ------------------------------------------------------------------
    print("Loading base model (phi2_base)...")
    t0 = time.time()
    base_mdl = _load_base_model(base_model, device)
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    base_scores: list[dict] = []
    for i, sc in enumerate(scenarios):
        prompt = PROMPT_TEMPLATE.format(uncertain_statement=sc["uncertain_statement"])
        answer = _generate(base_mdl, tokenizer, prompt, device=device)
        scored = _score_scenario(sc, answer)
        scored["condition"] = "phi2_base"
        base_scores.append(scored)
        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"  phi2_base: {i + 1}/{total} done")

    # Free base model memory before loading LoRA
    del base_mdl
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Condition 2: phi2_dpo (with LoRA adapter)
    # ------------------------------------------------------------------
    print(f"\nLoading LoRA model (phi2_dpo) from '{adapter_path}'...")
    t0 = time.time()
    lora_mdl = _load_lora_model(base_model, adapter_path, device)
    print(f"  LoRA model loaded in {time.time() - t0:.1f}s")

    dpo_scores: list[dict] = []
    for i, sc in enumerate(scenarios):
        prompt = PROMPT_TEMPLATE.format(uncertain_statement=sc["uncertain_statement"])
        answer = _generate(lora_mdl, tokenizer, prompt, device=device)
        scored = _score_scenario(sc, answer)
        scored["condition"] = "phi2_dpo"
        dpo_scores.append(scored)
        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"  phi2_dpo:  {i + 1}/{total} done")

    del lora_mdl
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    base_fcr_vals = [1.0 if s["fcr"] else 0.0 for s in base_scores]
    dpo_fcr_vals  = [1.0 if s["fcr"] else 0.0 for s in dpo_scores]

    base_fcr_mean  = sum(base_fcr_vals) / total if total else 0.0
    dpo_fcr_mean   = sum(dpo_fcr_vals)  / total if total else 0.0
    delta_fcr      = base_fcr_mean - dpo_fcr_mean   # positive = DPO improved

    base_ci = bootstrap_ci(base_fcr_vals)
    dpo_ci  = bootstrap_ci(dpo_fcr_vals)

    base_eqlr_vals = [1.0 if s["eqlr_token"] else 0.0 for s in base_scores]
    dpo_eqlr_vals  = [1.0 if s["eqlr_token"] else 0.0 for s in dpo_scores]
    base_eqlr = sum(base_eqlr_vals) / total if total else 0.0
    dpo_eqlr  = sum(dpo_eqlr_vals)  / total if total else 0.0

    per_scenario = []
    for b, d in zip(base_scores, dpo_scores):
        per_scenario.append({
            "scenario_id":        b["scenario_id"],
            "domain":             b["domain"],
            "qualifier_type":     b["qualifier_type"],
            "uncertain_statement": b["uncertain_statement"],
            "phi2_base": {
                "answer":  b["answer"],
                "fcr":     b["fcr"],
                "eqlr":    b["eqlr_token"],
            },
            "phi2_dpo": {
                "answer":  d["answer"],
                "fcr":     d["fcr"],
                "eqlr":    d["eqlr_token"],
            },
        })

    results = {
        "model":         base_model,
        "adapter":       adapter_path,
        "n_scenarios":   total,
        "phi2_base_fcr": round(base_fcr_mean, 4),
        "phi2_dpo_fcr":  round(dpo_fcr_mean,  4),
        "delta_fcr":     round(delta_fcr,      4),
        "phi2_base_fcr_ci": list(base_ci),
        "phi2_dpo_fcr_ci":  list(dpo_ci),
        "phi2_base_eqlr": round(base_eqlr, 4),
        "phi2_dpo_eqlr":  round(dpo_eqlr,  4),
        "delta_eqlr":     round(base_eqlr - dpo_eqlr, 4),
        "per_scenario":  per_scenario,
    }
    return results


# ---------------------------------------------------------------------------
# Dry-run mode (no model loading, validates dataset + prints table header)
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """Validate dataset loading and print a results table with TBD values."""
    print("=== model_eval.py — DRY RUN ===")
    print()

    # Validate EQL-Bench file loads
    try:
        scenarios = load_eql_bench()
        print(f"EQL-Bench v2 loaded: {len(scenarios)} scenarios across "
              f"{len(set(s.get('domain','?') for s in scenarios))} domains.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading EQL-Bench v2: {e}")
        sys.exit(1)

    # Validate metrics imports
    test_answer = "The rate limit is approximately 50 req/min (unconfirmed)."
    val_frags   = ["50", "req"]
    qual_frags  = ["approximately", "unconfirmed", "might"]
    eqlr_result = eqlr_token(test_answer, qual_frags)
    fcr_result  = fcr(test_answer, val_frags, qual_frags)
    print(f"Metrics sanity check — eqlr_token={eqlr_result} (expect False), "
          f"fcr={fcr_result} (expect False).")
    assert eqlr_result is False, "eqlr_token sanity check failed"
    assert fcr_result  is False, "fcr sanity check failed"
    print("Metrics imports: OK")
    print()

    # Print results table with TBD values
    print("Results table (TBD — model not loaded in --dry-run mode):")
    _print_results_table(
        base_model=BASE_MODEL_NAME,
        adapter="<adapter-path>",
        n_scenarios="(first 20 scenarios)",
        phi2_base_fcr="TBD",
        phi2_dpo_fcr="TBD",
        delta_fcr="TBD",
        phi2_base_eqlr="TBD",
        phi2_dpo_eqlr="TBD",
        delta_eqlr="TBD",
    )

    # Validate adapter_config.json check logic
    print()
    print("Adapter validation logic:")
    default_adapter = "models/credence-phi-2-dpo/credence-dpo-final"
    adapter_cfg = os.path.join(default_adapter, "adapter_config.json")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_cfg = os.path.join(repo_root, adapter_cfg)
    if os.path.exists(full_cfg):
        print(f"  adapter_config.json found at: {full_cfg}")
    else:
        print(f"  adapter_config.json NOT found at: {full_cfg}")
        print(f"  (This is expected if the model hasn't been trained yet.)")

    print()
    print("Dry run complete. No GPU or API key required.")
    print(f"To run a real evaluation:")
    print(f"  python -m evals.model_eval \\")
    print(f"    --model-path {default_adapter} \\")
    print(f"    --n 20")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_results_table(
    base_model: str,
    adapter: str,
    n_scenarios,
    phi2_base_fcr,
    phi2_dpo_fcr,
    delta_fcr,
    phi2_base_eqlr,
    phi2_dpo_eqlr,
    delta_eqlr,
) -> None:
    width = 62
    print("┌" + "─" * width + "┐")
    print(f"│{'EQL-Bench v2 — Phi-2 DPO Model Evaluation':^{width}}│")
    print("├" + "─" * width + "┤")
    print(f"│  Base model : {base_model:<{width - 15}}│")
    print(f"│  Adapter    : {str(adapter)[:width - 15]:<{width - 15}}│")
    print(f"│  Scenarios  : {str(n_scenarios):<{width - 15}}│")
    print("├" + "─" * width + "┤")
    print(f"│  {'Metric':<24} {'phi2_base':>8} {'phi2_dpo':>8} {'Δ (base-dpo)':>12} │")
    print("├" + "─" * width + "┤")

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    print(f"│  {'FCR (False Certainty Rate)':<24} {_fmt(phi2_base_fcr):>8} {_fmt(phi2_dpo_fcr):>8} {_fmt(delta_fcr):>12} │")
    print(f"│  {'EQLR (Qualifier Loss Rate)':<24} {_fmt(phi2_base_eqlr):>8} {_fmt(phi2_dpo_eqlr):>8} {_fmt(delta_eqlr):>12} │")
    print("└" + "─" * width + "┘")

    if isinstance(delta_fcr, float):
        if delta_fcr > 0.05:
            print(f"\nResult: LoRA adapter IMPROVED FCR by {delta_fcr:.3f} — DPO training reduced false certainty.")
        elif delta_fcr < -0.05:
            print(f"\nResult: LoRA adapter WORSENED FCR by {abs(delta_fcr):.3f} — DPO training increased false certainty.")
        else:
            print(f"\nResult: No significant FCR difference (Δ = {delta_fcr:.3f}). Models behave similarly.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phi-2 LoRA vs base model on EQL-Bench v2 FCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python -m evals.model_eval --dry-run
              python -m evals.model_eval --model-path models/credence-phi-2-dpo/credence-dpo-final --n 20
              python -m evals.model_eval --model-path models/credence-phi-2-dpo/credence-dpo-final --all
        """),
    )
    parser.add_argument(
        "--model-path", "--adapter-path",
        default="models/credence-phi-2-dpo/credence-dpo-final",
        help="Path to the LoRA adapter directory (default: models/credence-phi-2-dpo/credence-dpo-final)",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_NAME,
        help=f"Base model name (default: {BASE_MODEL_NAME})",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of EQL-Bench scenarios to evaluate (default: 20)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all 370 EQL-Bench v2 scenarios (overrides --n)",
    )
    parser.add_argument(
        "--out", default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset loading and print table header without loading models",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # Validate adapter path before loading models
    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        print(f"WARNING: adapter_config.json not found at '{adapter_cfg}'.")
        print("The adapter may not have been trained yet.")
        print("To train, run: python -m training.dpo_train")
        print("Continuing anyway — will fail at model load if path is invalid.")
        print()

    results = run_evaluation(
        adapter_path=args.model_path,
        base_model=args.base_model,
        n_scenarios=args.n,
        all_scenarios=args.all,
    )

    # Print results table
    print()
    _print_results_table(
        base_model=results["model"],
        adapter=results["adapter"],
        n_scenarios=results["n_scenarios"],
        phi2_base_fcr=results["phi2_base_fcr"],
        phi2_dpo_fcr=results["phi2_dpo_fcr"],
        delta_fcr=results["delta_fcr"],
        phi2_base_eqlr=results["phi2_base_eqlr"],
        phi2_dpo_eqlr=results["phi2_dpo_eqlr"],
        delta_eqlr=results["delta_eqlr"],
    )
    if results.get("phi2_base_fcr_ci"):
        lo_b, hi_b = results["phi2_base_fcr_ci"]
        lo_d, hi_d = results["phi2_dpo_fcr_ci"]
        print(f"\n95% CI — base: [{lo_b:.3f}, {hi_b:.3f}]  dpo: [{lo_d:.3f}, {hi_d:.3f}]")

    # Save results
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()

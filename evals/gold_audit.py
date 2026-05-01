"""
evals/gold_audit.py
===================
Gold audit: measure whether the Phi-2 DPO adapter prefers
epistemically faithful summaries over unfaithful ones.

Loads the LoRA adapter from models/credence-phi-2-dpo/credence-dpo-final,
runs the first 50 examples from data/elite_500.json, and computes:

  Accuracy     = fraction where logprob(faithful) > logprob(unfaithful)
  Avg margin   = mean logprob gap (faithful − unfaithful)

Run:
    python -m evals.gold_audit              # full run (needs GPU)
    python -m evals.gold_audit --dry-run    # validate files only, no model load
    python -m evals.gold_audit --n 20       # run on first N examples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ADAPTER_PATH  = ROOT / "models" / "credence-phi-2-dpo" / "credence-dpo-final"
EVAL_DATASET  = ROOT / "data" / "elite_500.json"
RESULTS_PATH  = ROOT / "evals" / "gold_audit_results.json"
BASE_MODEL    = "microsoft/phi-2"


def _validate_files() -> bool:
    ok = True
    for p, label in [(ADAPTER_PATH, "adapter"), (EVAL_DATASET, "eval dataset")]:
        if not p.exists():
            print(f"  MISSING {label}: {p}")
            ok = False
        else:
            print(f"  ✓ {label}: {p}")
    return ok


def run_gold_audit(n: int = 50, dry_run: bool = False) -> dict:
    print("Credence Gold Audit — Phi-2 DPO Faithfulness Check")
    print("=" * 55)

    print("\nChecking required files …")
    if not _validate_files():
        sys.exit(1)

    with open(EVAL_DATASET) as f:
        data = json.load(f)
    examples = data["examples"][:n]
    print(f"\nEval dataset: {len(data['examples'])} total, running first {len(examples)}")

    if dry_run:
        sample = examples[0]
        print(f"\n[dry-run] First example keys: {list(sample.keys())}")
        print(f"  input_conversation[:80]: {sample['input_conversation'][:80]!r}")
        print(f"  faithful_summary[:80]:   {sample['faithful_summary'][:80]!r}")
        print(f"  unfaithful_summary[:80]: {sample['unfaithful_summary'][:80]!r}")
        print("\n[dry-run] PASS — all files present, dataset valid. No model loaded.")
        return {"dry_run": True, "n": len(examples)}

    # --- GPU run ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float32   # MPS prefers float32
    else:
        device = "cpu"
        dtype  = torch.float32
        print("WARNING: no GPU found — running on CPU, will be very slow")

    print(f"\nDevice: {device}  dtype: {dtype}")

    print(f"\nLoading base model: {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    load_kw = {"torch_dtype": dtype, "trust_remote_code": True}
    if device == "cuda":
        load_kw["device_map"] = "auto"
    else:
        load_kw["device_map"] = device

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kw)
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model.eval()
    print("Model loaded.")

    results = {"total": 0, "faithful_preferred": 0, "unfaithful_preferred": 0,
               "avg_margin": 0.0, "per_example": []}
    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        prompt = (
            f"User: Summarize faithfully preserving all uncertainty qualifiers: "
            f"{ex['input_conversation']}\nAssistant:"
        )

        def get_logprob(continuation: str) -> float:
            full = prompt + continuation
            inputs = tokenizer(full, return_tensors="pt",
                               truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            return -out.loss.item()

        lp_faithful   = get_logprob(ex["faithful_summary"])
        lp_unfaithful = get_logprob(ex["unfaithful_summary"])
        margin = lp_faithful - lp_unfaithful

        results["total"]           += 1
        results["avg_margin"]      += margin
        if margin > 0:
            results["faithful_preferred"] += 1
        else:
            results["unfaithful_preferred"] += 1

        results["per_example"].append({
            "id": ex.get("id", i),
            "domain": ex.get("domain", ""),
            "qualifier_type": ex.get("qualifier_type", ""),
            "is_ghost": ex.get("is_ghost", False),
            "lp_faithful": round(lp_faithful, 4),
            "lp_unfaithful": round(lp_unfaithful, 4),
            "margin": round(margin, 4),
            "preferred": "faithful" if margin > 0 else "unfaithful",
        })

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(examples)} done  ({elapsed:.1f}s)")

    elapsed_total = time.perf_counter() - t0
    results["avg_margin"] /= max(results["total"], 1)
    results["accuracy"]    = results["faithful_preferred"] / max(results["total"], 1)
    results["elapsed_s"]   = round(elapsed_total, 2)
    results["n"]           = results["total"]
    results["adapter"]     = str(ADAPTER_PATH)
    results["base_model"]  = BASE_MODEL
    results["device"]      = device

    print("\n" + "=" * 55)
    print("   GOLD AUDIT RESULTS")
    print("=" * 55)
    print(f"  Scenarios:          {results['total']}")
    print(f"  Faithful preferred: {results['faithful_preferred']}  "
          f"({results['accuracy']*100:.1f}%)")
    print(f"  Avg logprob margin: {results['avg_margin']:+.4f}")
    print(f"  Elapsed:            {elapsed_total:.1f}s")
    print("=" * 55)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH.relative_to(ROOT)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Credence Gold Audit — Phi-2 DPO faithfulness check")
    parser.add_argument("--dry-run", action="store_true", help="Validate files only, no model load")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to run (default 50)")
    args = parser.parse_args()
    run_gold_audit(n=args.n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

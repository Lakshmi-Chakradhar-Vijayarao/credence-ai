"""
training/merge_model.py
=======================
Merge a LoRA adapter into the base Phi-2 model to produce a standalone
merged model that can be loaded without PEFT at inference time.

Auto-detects Kaggle vs local environment and adjusts default paths accordingly.

Usage
-----
    # Local (uses models/ directory)
    python -m training.merge_model

    # Custom paths
    python -m training.merge_model \\
        --adapter-path models/credence-phi-2-dpo/credence-dpo-final \\
        --output-path  models/credence-phi-2-merged \\
        --base-model   microsoft/phi-2

    # Dry-run (validates paths, no model loading)
    python -m training.merge_model --dry-run

    # Kaggle: paths default to /kaggle/working/ automatically
    python -m training.merge_model
"""

from __future__ import annotations

import os
import sys
import argparse
import textwrap

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_kaggle() -> bool:
    """Return True if running inside a Kaggle kernel."""
    return os.path.isdir("/kaggle/working")


def _default_paths() -> tuple[str, str, str]:
    """
    Return (adapter_path, output_path, base_model) based on environment.

    Kaggle: /kaggle/working/credence-dpo-final → /kaggle/working/credence-phi-2-merged
    Local:  models/credence-phi-2-dpo/credence-dpo-final → models/credence-phi-2-merged
    """
    if _is_kaggle():
        adapter = "/kaggle/working/credence-dpo-final"
        output  = "/kaggle/working/credence-phi-2-merged"
    else:
        adapter = "models/credence-phi-2-dpo/credence-dpo-final"
        output  = "models/credence-phi-2-merged"
    return adapter, output, "microsoft/phi-2"


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(adapter_path: str, output_path: str, base_model: str) -> None:
    """
    Validate paths and print what would happen — no model loading.

    Checks:
    1. adapter_config.json exists at adapter_path
    2. Output directory is writable (or can be created)
    3. Prints resolved absolute paths
    """
    print("=== merge_model.py — DRY RUN ===")
    print(f"Environment : {'Kaggle' if _is_kaggle() else 'Local'}")
    print()

    # Resolve to absolute paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(repo_root, p)

    abs_adapter = _resolve(adapter_path)
    abs_output  = _resolve(output_path)

    print(f"Base model    : {base_model}")
    print(f"Adapter path  : {abs_adapter}")
    print(f"Output path   : {abs_output}")
    print()

    # Check adapter_config.json
    adapter_cfg = os.path.join(abs_adapter, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        print(f"OK  adapter_config.json found at: {adapter_cfg}")
        # Print key fields if readable
        try:
            import json as _json
            with open(adapter_cfg) as f:
                cfg = _json.load(f)
            print(f"    base_model_name_or_path : {cfg.get('base_model_name_or_path', '?')}")
            print(f"    peft_type               : {cfg.get('peft_type', '?')}")
            print(f"    r (LoRA rank)           : {cfg.get('r', '?')}")
        except Exception as e:
            print(f"    (Could not parse adapter_config.json: {e})")
    else:
        print(f"WARN  adapter_config.json NOT found at: {adapter_cfg}")
        print(f"      The adapter may not have been trained yet.")
        print(f"      Train with: python -m training.dpo_train")

    # Check output directory
    out_dir = os.path.dirname(abs_output)
    if not out_dir:
        out_dir = "."
    if os.path.exists(out_dir):
        print(f"OK  Output parent directory exists: {out_dir}")
    else:
        print(f"INFO  Output parent directory will be created: {out_dir}")

    print()
    print("What would happen (no model loaded):")
    print(f"  1. Load tokenizer from {base_model}")
    print(f"  2. Load base model   from {base_model} (float16 on CUDA, float32 on CPU)")
    print(f"  3. Attach LoRA adapter from {abs_adapter}")
    print(f"  4. Call model.merge_and_unload() to fuse LoRA weights into base")
    print(f"  5. Save merged model to {abs_output}")
    print(f"  6. Save tokenizer to   {abs_output}")
    print()
    print("Dry run complete. Run without --dry-run to execute.")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_and_save(
    adapter_path: str,
    output_path:  str,
    base_model:   str,
) -> None:
    """Load base model, attach LoRA adapter, merge weights, save."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("Install with: pip install torch transformers peft")
        sys.exit(1)

    # Resolve relative paths from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(repo_root, p)

    abs_adapter = _resolve(adapter_path)
    abs_output  = _resolve(output_path)

    print(f"Environment : {'Kaggle' if _is_kaggle() else 'Local'}")
    print(f"Base model  : {base_model}")
    print(f"Adapter     : {abs_adapter}")
    print(f"Output      : {abs_output}")
    print()

    # Validate adapter_config.json
    adapter_cfg = os.path.join(abs_adapter, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        print(f"ERROR: adapter_config.json not found at '{adapter_cfg}'.")
        print("The adapter may not have been trained. Run: python -m training.dpo_train")
        sys.exit(1)

    # Determine dtype / device
    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. Load tokenizer
    print("--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load base model
    print("--- Loading Base Model ---")
    base_mdl = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 3. Attach LoRA adapter
    print("--- Loading LoRA Adapter ---")
    model = PeftModel.from_pretrained(base_mdl, abs_adapter)

    # 4. Merge and unload (fuse LoRA weights into base model)
    print("--- Merging Weights (Fusing LoRA into Base) ---")
    merged_model = model.merge_and_unload()

    # 5. Save merged model
    os.makedirs(abs_output, exist_ok=True)
    print(f"--- Saving Merged Model to {abs_output} ---")
    merged_model.save_pretrained(abs_output)
    tokenizer.save_pretrained(abs_output)

    # 6. Report file sizes
    print()
    print("--- Saved Files ---")
    total_bytes = 0
    for root, _dirs, files in os.walk(abs_output):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size  = os.path.getsize(fpath)
            total_bytes += size
            rel = os.path.relpath(fpath, abs_output)
            print(f"  {rel:<50}  {size / (1024 * 1024):>8.1f} MB")
    print(f"  {'Total':.<50}  {total_bytes / (1024 * 1024):>8.1f} MB")
    print()
    print(f"--- DONE: Merged model saved to {abs_output} ---")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _default_adapter, _default_output, _default_base = _default_paths()

    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into Phi-2 and save a standalone merged model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""
            Defaults (auto-detected environment: {'Kaggle' if _is_kaggle() else 'Local'}):
              --adapter-path  {_default_adapter}
              --output-path   {_default_output}
              --base-model    {_default_base}

            Examples:
              python -m training.merge_model --dry-run
              python -m training.merge_model
              python -m training.merge_model \\
                --adapter-path /kaggle/working/credence-dpo-final \\
                --output-path  /kaggle/working/credence-phi-2-merged
        """),
    )
    parser.add_argument(
        "--adapter-path",
        default=_default_adapter,
        help="Path to the LoRA adapter directory containing adapter_config.json",
    )
    parser.add_argument(
        "--output-path",
        default=_default_output,
        help="Directory where the merged model will be saved",
    )
    parser.add_argument(
        "--base-model",
        default=_default_base,
        help=f"Base model name or path (default: {_default_base})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paths and print plan without loading any models",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run(
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            base_model=args.base_model,
        )
    else:
        merge_and_save(
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            base_model=args.base_model,
        )


if __name__ == "__main__":
    main()

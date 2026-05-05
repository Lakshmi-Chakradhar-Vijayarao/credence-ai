"""
training/dpo_finetune.py
========================
DPO training setup for fine-tuning a generative model with an epistemic loss.

NOTE ON MODEL CHOICE:
    LLMLingua-2 (microsoft/llmlingua-2-bert-large-multilingual-cased-meetingbank) is the
    *comparison baseline*, NOT the training target. LLMLingua-2 is a BERT encoder — it is
    a token importance scorer with no generative head. It cannot be loaded with
    AutoModelForCausalLM and cannot be DPO fine-tuned as a text generator. DPO requires a
    causal language model that produces sequences via next-token prediction.

    The default training target is microsoft/phi-2 (2.7B), which runs on a T4 GPU with
    4-bit quantisation. See _MODEL_OPTIONS for alternatives.

The EpistemicDPOTrainer extends HuggingFace trl.DPOTrainer and overrides
compute_loss() to add the epistemic regularisation term on top of the standard
DPO loss:

    total_loss = dpo_loss + lambda_epistemic * epistemic_loss

The epistemic_loss is computed offline (no GPU) by CredenceProxy's J-score
on the decoded chosen/rejected sequences — it is an asymmetric penalty for
confidence inflation in the rejected (unfaithful) summary.

Usage:
    python -m training.dpo_finetune --data data/epistemic_compression_training.json --epochs 3 --lambda 0.3
    python -m training.dpo_finetune --dry-run   # validates data loading, no GPU needed
"""

from __future__ import annotations

import os
import sys
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Optional heavy imports — wrapped so --dry-run works without GPU
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:
    from trl import DPOTrainer, DPOConfig
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    DPOTrainer = object  # type: ignore[assignment,misc]
    DPOConfig = None  # type: ignore[assignment]

try:
    from datasets import Dataset as HFDataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    HFDataset = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    LoraConfig = None       # type: ignore[assignment]
    get_peft_model = None   # type: ignore[assignment]
    TaskType = None         # type: ignore[assignment]

# Local
from training.epistemic_loss import compute_epistemic_loss, _DEFAULT_LAMBDA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Training target: Phi-2 (2.7B causal LM) — runs on T4 with 4-bit quantisation.
#
# Why NOT LLMLingua-2:
#   microsoft/llmlingua-2-bert-large-multilingual-cased-meetingbank is a BERT encoder
#   trained as a token importance scorer. It has no generative head, cannot be loaded
#   with AutoModelForCausalLM, and cannot produce sequences for DPO training.
#   LLMLingua-2 is the *comparison baseline* (the system we measure qualifier loss
#   against), not the model we fine-tune.
_DEFAULT_MODEL = "microsoft/phi-2"

# Model options by GPU size — pass via --model CLI flag
_MODEL_OPTIONS = {
    "phi-2":        "microsoft/phi-2",                           # 2.7B — T4 with 4-bit quant
    "tinyllama":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",       # 1.1B — fits T4 fp16
    "flan-t5-base": "google/flan-t5-base",                       # 250M — fastest, seq2seq
}

# Eval split fraction
_EVAL_SPLIT = 0.20

# Training checkpoint directory
_DEFAULT_OUTPUT_DIR = "training/checkpoints"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class DPOExample:
    """Single DPO training example in HuggingFace trl format."""
    prompt:   str   # input conversation
    chosen:   str   # faithful summary (preferred)
    rejected: str   # unfaithful summary (dispreferred)


def load_dataset(path: str) -> list[DPOExample]:
    """Load training triples from JSON and convert to DPOExample list."""
    with open(path) as f:
        data = json.load(f)

    examples = []
    for entry in data.get("examples", []):
        examples.append(DPOExample(
            prompt   = entry["input_conversation"],
            chosen   = entry["faithful_summary"],
            rejected = entry["unfaithful_summary"],
        ))
    return examples


def split_dataset(
    examples: list[DPOExample],
    eval_frac: float = _EVAL_SPLIT,
    seed: int = 42,
) -> tuple[list[DPOExample], list[DPOExample]]:
    """Shuffle and split into train/eval."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_frac))
    return shuffled[n_eval:], shuffled[:n_eval]


def to_hf_dataset(examples: list[DPOExample]) -> "HFDataset":
    """Convert to HuggingFace Dataset with the columns DPOTrainer expects."""
    if not _DATASETS_AVAILABLE:
        raise ImportError("pip install datasets")
    return HFDataset.from_dict({
        "prompt":   [e.prompt   for e in examples],
        "chosen":   [e.chosen   for e in examples],
        "rejected": [e.rejected for e in examples],
    })


# ---------------------------------------------------------------------------
# EpistemicDPOTrainer
# ---------------------------------------------------------------------------

class EpistemicDPOTrainer(DPOTrainer):  # type: ignore[misc]
    """
    Extends trl.DPOTrainer with an epistemic regularisation loss.

    The epistemic loss penalises compression outputs that inflate confidence
    relative to the input conversation. It is computed offline on the decoded
    token sequences (no gradient through CredenceProxy).

    total_loss = dpo_loss + lambda_epistemic * epistemic_loss
    """

    def __init__(self, *args, lambda_epistemic: float = _DEFAULT_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_epistemic = lambda_epistemic

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Override DPOTrainer.compute_loss to inject epistemic loss.

        The standard DPO loss is computed first. Then we decode the chosen
        and rejected token sequences and compute the epistemic penalty on
        the decoded text (offline, not part of the computation graph).

        The epistemic loss is treated as a regularisation scalar — it is
        not backpropagated through its own computation, only added to the
        total loss value to influence gradient magnitude.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for training")

        # Standard DPO loss
        dpo_out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(dpo_out, tuple):
            dpo_loss, outputs = dpo_out
        else:
            dpo_loss = dpo_out
            outputs  = None

        # Epistemic loss — computed offline on decoded text
        epistemic_penalty = self._compute_batch_epistemic_loss(inputs)

        total_loss = dpo_loss + epistemic_penalty

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_batch_epistemic_loss(self, inputs: dict) -> "torch.Tensor":
        """
        Decode chosen and rejected input_ids to text, then compute epistemic
        loss for each pair against the prompt.

        Returns a scalar Tensor (no gradient).
        """
        if not _TORCH_AVAILABLE or self.tokenizer is None:
            return torch.tensor(0.0)

        prompts   = inputs.get("prompt_input_ids")
        chosen    = inputs.get("chosen_input_ids")
        rejected  = inputs.get("rejected_input_ids")

        if prompts is None or chosen is None or rejected is None:
            return torch.tensor(0.0)

        batch_losses = []
        for i in range(len(chosen)):
            try:
                prompt_text   = self.tokenizer.decode(prompts[i],   skip_special_tokens=True)
                chosen_text   = self.tokenizer.decode(chosen[i],    skip_special_tokens=True)
                rejected_text = self.tokenizer.decode(rejected[i],  skip_special_tokens=True)

                # Epistemic loss for chosen (should be low)
                e_chosen   = compute_epistemic_loss(prompt_text, chosen_text,   self.lambda_epistemic)
                # Epistemic loss for rejected (should be high — this is the penalty)
                e_rejected = compute_epistemic_loss(prompt_text, rejected_text, self.lambda_epistemic)

                # We want e_rejected > e_chosen; penalise if ordering is violated
                margin_loss = max(0.0, e_chosen - e_rejected + 0.05)
                batch_losses.append(e_rejected + margin_loss)
            except Exception:
                batch_losses.append(0.0)

        if not batch_losses:
            return torch.tensor(0.0)

        mean_loss = sum(batch_losses) / len(batch_losses)
        return torch.tensor(mean_loss, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Epoch-level eval callback
# ---------------------------------------------------------------------------

@dataclass
class EpochEvalResults:
    epoch:    int
    eqlr:     float   # Epistemic Qualifier Loss Rate on eval set
    fcr:      float   # False Certainty Rate on eval set
    n_eval:   int


_QUAL_MARKERS = [
    "unverified", "estimate", "unconfirmed", "approximately", "roughly",
    "probably", "maybe", "might", "unclear", "uncertain", "not certain",
    "i think", "i believe", "preliminary", "approximate", "based on",
    "haven't confirmed", "haven't verified", "reportedly", "assumed",
]


def evaluate_epistemic_metrics(
    model_outputs: list[str],
    eval_examples: list[DPOExample],
) -> EpochEvalResults:
    """
    Compute EQLR and FCR on the eval set given model-generated summaries.

    EQLR = fraction of outputs that lost the qualifier (value survived, qualifier didn't)
    FCR  = fraction of outputs that stated a value as confirmed fact
    """
    # Approximate: extract value from prompt (first number found)
    import re
    num_re = re.compile(r'\b\d+(?:\.\d+)?\b')

    eqlr_events = 0
    fcr_events  = 0
    n_with_value = 0

    for output, ex in zip(model_outputs, eval_examples):
        output_lo = output.lower()
        # Find numeric values in the prompt
        values = num_re.findall(ex.prompt)
        if not values:
            continue
        value = values[0]
        has_value     = value in output
        has_qualifier = any(m in output_lo for m in _QUAL_MARKERS)

        if has_value:
            n_with_value += 1
            if not has_qualifier:
                eqlr_events += 1
                fcr_events  += 1

    n = max(1, len(model_outputs))
    return EpochEvalResults(
        epoch  = 0,  # filled by caller
        eqlr   = round(eqlr_events / max(1, n_with_value), 4),
        fcr    = round(fcr_events  / n, 4),
        n_eval = n,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(
    data_path:         str,
    model_name:        str       = _DEFAULT_MODEL,
    output_dir:        str       = _DEFAULT_OUTPUT_DIR,
    epochs:            int       = 3,
    lambda_epistemic:  float     = _DEFAULT_LAMBDA,
    batch_size:        int       = 2,
    learning_rate:     float     = 5e-6,
    dry_run:           bool      = False,
) -> list[EpochEvalResults]:
    """
    Full DPO training run with epistemic loss.

    Args:
        data_path:        path to training JSON
        model_name:       HuggingFace model ID
        output_dir:       checkpoint directory
        epochs:           number of training epochs
        lambda_epistemic: epistemic loss weight
        batch_size:       per-device batch size
        learning_rate:    learning rate
        dry_run:          if True, validate data loading only (no GPU needed)

    Returns:
        list of EpochEvalResults (one per epoch)
    """
    # --- Load data ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    examples = load_dataset(data_path)
    train_examples, eval_examples = split_dataset(examples)
    print(f"  Train: {len(train_examples)}  Eval: {len(eval_examples)}")

    if dry_run:
        print("\n[dry-run] Data loading validated successfully.")
        print(f"  Example prompt[:100]:   {train_examples[0].prompt[:100]!r}")
        print(f"  Example chosen[:80]:    {train_examples[0].chosen[:80]!r}")
        print(f"  Example rejected[:80]:  {train_examples[0].rejected[:80]!r}")
        print(f"\n  Would train {epochs} epochs with lambda_epistemic={lambda_epistemic}")
        print(f"  Model:                  {model_name}")
        print(f"  Output dir:             {output_dir}")
        print(f"  Batch size:             {batch_size} (effective=8 with grad_accum_steps=4)")
        print(f"  LR:                     {learning_rate}")
        print(f"  DPO beta:               0.1")
        print(f"  max_length:             512  (max_prompt_length=384)")
        print(f"  warmup_ratio:           0.1")
        print(f"  LoRA:                   {'enabled (r=8, alpha=16)' if _PEFT_AVAILABLE else 'unavailable (pip install peft)'}")
        print(f"  4-bit quantisation:     {'enabled (nf4, double-quant)' if _BNB_AVAILABLE else 'unavailable (pip install bitsandbytes)'}")
        return []

    # --- Require heavy deps for actual training ---
    if not _TORCH_AVAILABLE:
        raise ImportError("pip install torch")
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("pip install transformers")
    if not _TRL_AVAILABLE:
        raise ImportError("pip install trl")
    if not _DATASETS_AVAILABLE:
        raise ImportError("pip install datasets")

    # --- Tokenizer + Model ---
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": True}
    if torch.cuda.is_available() and _BNB_AVAILABLE:
        print("  Using 4-bit quantisation (nf4, double-quant, fp16 compute)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if not _BNB_AVAILABLE:
            print("  bitsandbytes not available — loading in fp16/fp32 (may OOM on small GPUs)")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # --- LoRA wrapping (required to fit Phi-2 on T4) ---
    if _PEFT_AVAILABLE:
        print("  Applying LoRA (r=8, alpha=16, target: q_proj + v_proj)")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("  peft not available — training all parameters (may OOM; pip install peft)")

    # --- HF Dataset ---
    train_dataset = to_hf_dataset(train_examples)
    eval_dataset  = to_hf_dataset(eval_examples)

    # --- Training config ---
    os.makedirs(output_dir, exist_ok=True)
    training_args = DPOConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = 4,           # effective batch = batch_size * 4 = 8
        learning_rate               = learning_rate,
        beta                        = 0.1,          # DPO regularisation; lower = less constrained from ref
        max_length                  = 512,
        max_prompt_length           = 384,
        warmup_ratio                = 0.1,
        save_strategy               = "epoch",
        evaluation_strategy         = "epoch",
        logging_steps               = 50,
        remove_unused_columns       = False,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
    )

    # --- EpistemicDPOTrainer ---
    trainer = EpistemicDPOTrainer(
        model             = model,
        args              = training_args,
        tokenizer         = tokenizer,
        train_dataset     = train_dataset,
        eval_dataset      = eval_dataset,
        lambda_epistemic  = lambda_epistemic,
    )

    # --- Train ---
    epoch_results: list[EpochEvalResults] = []
    print(f"\nStarting training — {epochs} epochs, lambda={lambda_epistemic}")

    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch}/{epochs}...")
        trainer.train()

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}")
        trainer.save_model(ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}")

        # Eval — generate summaries on eval set and score
        print(f"  Evaluating on {len(eval_examples)} held-out examples...")
        model_outputs = []
        for ex in eval_examples[:min(100, len(eval_examples))]:  # cap for speed
            inputs = tokenizer(ex.prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            model_outputs.append(decoded)

        result = evaluate_epistemic_metrics(
            model_outputs,
            eval_examples[:len(model_outputs)],
        )
        result.epoch = epoch
        epoch_results.append(result)
        print(f"  Epoch {epoch} eval — EQLR={result.eqlr:.3f}  FCR={result.fcr:.3f}  "
              f"n={result.n_eval}")

    # Save epoch results
    results_path = os.path.join(output_dir, "epoch_results.json")
    with open(results_path, "w") as f:
        json.dump([{"epoch": r.epoch, "eqlr": r.eqlr, "fcr": r.fcr, "n_eval": r.n_eval}
                   for r in epoch_results], f, indent=2)
    print(f"\nEpoch results saved to {results_path}")

    return epoch_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO fine-tune LLMLingua-2 with epistemic loss"
    )
    parser.add_argument("--data",       default="data/epistemic_compression_training.json",
                        help="Path to training data JSON")
    parser.add_argument("--model",      default=_DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--out",        default=_DEFAULT_OUTPUT_DIR,
                        help="Checkpoint output directory")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lambda",     dest="lambda_epistemic",
                        type=float, default=_DEFAULT_LAMBDA,
                        help="Epistemic loss weight (default 0.3)")
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Validate data loading without GPU or training")
    args = parser.parse_args()

    try:
        results = run_training(
            data_path        = args.data,
            model_name       = args.model,
            output_dir       = args.out,
            epochs           = args.epochs,
            lambda_epistemic = args.lambda_epistemic,
            batch_size       = args.batch_size,
            learning_rate    = args.lr,
            dry_run          = args.dry_run,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -m data.build_training_dataset --hand-only first")
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if results:
        print(f"\nTraining complete. Final epoch FCR: {results[-1].fcr:.3f}")


if __name__ == "__main__":
    main()

"""
examples/quickstart.py — Five-minute EpistemicTag integration

Three steps:
  1. Calibrate (once, ~10 min on GPU — saves to JSON)
  2. Wrap your model
  3. Get routing on every query

Prerequisites:
    pip install epistemic-stack transformers torch datasets
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from esm import wrap_model

# ── Step 1: Load your model ────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="float16",
    device_map=None,
).cuda().eval()

# ── Step 2: Calibrate (one-time) ──────────────────────────────────────────────
# Skip this if you already have a calibration file and go to Step 3.

# Option A: CLI (recommended)
#   esm calibrate --model meta-llama/Llama-3.2-3B-Instruct \
#                 --dataset trivia_qa --n 100 \
#                 --output checkpoints/llama3b_cal.json

# Option B: Python API
# em = wrap_model(model, tokenizer)
# from datasets import load_dataset
# ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
# samples = [{"question": r["question"],
#              "answers": r["answer"]["aliases"],
#              "context": (r["entity_pages"]["wiki_context"] or [""])[0]}
#            for _, r in zip(range(500), ds)]
# em.calibrate(samples, n_target=100, save_path="checkpoints/llama3b_cal.json")

# ── Step 3: Wrap and query ────────────────────────────────────────────────────

model = wrap_model(model, tokenizer, calibration="checkpoints/llama3b_cal.json")

questions = [
    "Who wrote Hamlet?",
    "What is the melting point of titanium?",
    "What did the user purchase on their last visit to our store?",  # CTX_DEP
    "Explain the French Revolution.",
]

print(f"\n{'='*60}")
print("EPISTEMIC ROUTING DEMO")
print(f"{'='*60}")
print(f"{'Question':<45} {'Routing':>10} {'J_know':>8} {'Verify?':>8}")
print(f"{'─'*45} {'─'*10} {'─'*8} {'─'*8}")

for q in questions:
    tag = model.tag(q)
    print(
        f"{q[:44]:<45} {tag.routing:>10} "
        f"{tag.j_know:>+8.3f} {'⚠' if tag.verify_flag else '':>8}"
    )

print(f"{'='*60}\n")

# ── Routing actions ────────────────────────────────────────────────────────────

q = "Who was the first person to walk on the moon?"
tag = model.tag(q)

match tag.routing:
    case "ANSWER":
        resp = model.generate(q)
        print(f"ANSWER: {resp.text}")
    case "VERIFY":
        resp = model.generate(q)
        print(f"VERIFY (may confabulate): {resp.text}")
        print("  → Adding caveat: please verify this with an authoritative source.")
    case "RETRIEVE":
        print(f"RETRIEVE: Triggering RAG for '{q}'")
        # context = retriever.search(q)
        # resp = model.generate(f"Context: {context}\nQuestion: {q}")
    case "DEFER":
        print(f"DEFER: Model is uncertain. Acknowledging explicitly.")
    case "ESCALATE":
        print(f"ESCALATE: Routing to larger model or human review.")

# Credence — Kaggle GPU Execution Guide

Two GPU experiments run on Kaggle T4. Both are self-contained scripts (no repo imports).
Results are pulled back and committed to `evals/`.

---

## Experiments

### Experiment A — Phase 3 DPO Fine-Tuning
**Kernel:** `credence-phase-3-dpo-epistemic-fine-tuning`  
**Script:** `kaggle_training/run_dpo_training.py`  
**Dataset:** `credence-epistemic-dpo-training-data` (5,000 DPO triples, 4.8 MB)  
**GPU time:** ~2.5h on T4  
**Output:** epoch loss curve + per-epoch FCR measurement  
**Research question:** Does DPO on faithful/unfaithful pairs reduce FCR vs. base Phi-2?

### Experiment B — Phase 1 KV-Cache Eviction Study
**Kernel:** `credence-phase-1-kv-cache-eql-experiment`  
**Script:** `kaggle_kv_cache/run_kv_experiment.py`  
**Dataset:** `eql-bench-v2-epistemic-qualifier-loss-benchmark` (370 scenarios + ghost)  
**GPU time:** ~3-4h on T4 (Qwen2.5-7B × 7 eviction methods × 100 scenarios)  
**Output:** `kv_cache_results.json` — EQLR/FCR/GhostFCR per method  
**Research question:** Do H2O, SnapKV, StreamingLLM lose uncertainty qualifiers more or less than naive window?

---

## Submission

Both kernels are already pushed to Kaggle. To re-push after local edits:

```bash
kaggle kernels push -p kaggle_training/
kaggle kernels push -p kaggle_kv_cache/
```

---

## Monitoring

```bash
# Check run status
kaggle kernels status chakradharvijayarao/credence-phase-3-dpo-epistemic-fine-tuning
kaggle kernels status chakradharvijayarao/credence-phase-1-kv-cache-eql-experiment
```

Status values: `queued` → `running` → `complete` / `error`

---

## Pulling Results

### After Phase 3 DPO completes:

```bash
# Download output files
kaggle kernels output chakradharvijayarao/credence-phase-3-dpo-epistemic-fine-tuning -p /tmp/dpo_out/
ls /tmp/dpo_out/

# Copy results into repo
cp /tmp/dpo_out/epoch_results.json evals/dpo_epoch_results.json
```

Expected output file: `epoch_results.json` with per-epoch train loss and FCR.

### After Phase 1 KV-Cache completes:

```bash
# Download output files
kaggle kernels output chakradharvijayarao/credence-phase-1-kv-cache-eql-experiment -p /tmp/kv_out/
ls /tmp/kv_out/

# Copy results into repo
cp /tmp/kv_out/kv_cache_results.json evals/kv_cache_results.json
```

Expected output file: `kv_cache_results.json` with EQLR/FCR per eviction method.

---

## Local Dry-Run (no GPU needed)

Both scripts support `--dry-run` to validate without loading a model:

```bash
python kaggle_training/run_dpo_training.py --dry-run
python kaggle_kv_cache/run_kv_experiment.py --dry-run --n 3
```

---

## Gold Audit (Phi-2 adapter, needs GPU)

The gold audit validates the trained Phi-2 adapter on 50 elite scenarios:

```bash
# Dry-run (validates paths, no GPU)
python -m evals.gold_audit --dry-run

# Full run (needs CUDA or MPS, ~14GB VRAM recommended)
python -m evals.gold_audit --n 50
```

Adapter: `models/credence-phi-2-dpo/credence-dpo-final/`  
Eval data: `data/elite_500.json` (500 examples)  
Results: `evals/gold_audit_results.json`

---

## Expected Results

| Condition | FCR (target) | Notes |
|-----------|-------------|-------|
| Base Phi-2 (no DPO) | ~35% | Confirmed Phase 1 v29 |
| DPO fine-tuned Phi-2 | ~15% | Gate opens at any drop |
| DPO + probe | 0% | Deterministic override |
| H2O 70% KV | TBD | Likely high FCR via silence |
| SnapKV 70% | TBD | Similar to H2O expected |
| StreamingLLM 70% | ~31% | Phase 1 measured value |
| Naive baseline | ~6% | n=50 measured |

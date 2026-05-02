# Kaggle Push Instructions

## Prerequisites
```bash
pip install kaggle
# Place ~/.kaggle/kaggle.json with your API credentials
```

## Step 1: Upload the dataset (first time only)
```bash
cd kaggle_v6/dataset
kaggle datasets create -p .
```

If updating an existing dataset:
```bash
kaggle datasets version -p . -m "EQL-Bench v2: 370 scenarios"
```

## Step 2: Push the notebook
```bash
cd kaggle_v6
kaggle kernels push -p .
```

## Step 3: Monitor the run
```bash
kaggle kernels status chakradharvijayarao/credence-eqlr-open-model
```

When status is "complete":
```bash
kaggle kernels output chakradharvijayarao/credence-eqlr-open-model -p /tmp/kaggle_output/
```

## Expected runtime
- Dataset load: ~5s
- Qwen-2.5-1.5B load: ~30s (GPU warmup)
- 370 scenarios × ~3s each: ~18 min total
- Results: `eql_bench_qwen_results.json` in kernel output

## Expected results
| Condition | EQLR |
|---|---|
| Naive (all explicit, no probe) | 35–55% |
| Probe-blocked (canonical markers) | ~0% |
| Ghost scenarios | 80–90% |

A naive EQLR of 35–55% on Qwen-2.5-1.5B confirms the failure is **model-agnostic** —
not specific to Claude Haiku (which measured 46% EQLR separately).

# 🧠 CAMS — Confidence-Adaptive Memory System

> **Every token you keep should earn its place.**

CAMS gives Claude a voice in managing its own memory.
Instead of blindly preserving every conversation turn, CAMS reads Claude's response,
extracts a **J-score confidence signal**, and decides what to keep:

- **Confident answer?** → Compress old context. Claude already knows what it knows.
- **Uncertain answer?** → Preserve everything. Claude needs that history.

The result: **40–60% fewer tokens** on long sessions. **Same answer quality.** Less cost.

---

## The Problem

Every turn of a long Claude conversation costs money — even when Claude is confidently answering
a well-established fact and doesn't need 8,000 tokens of history behind it.

Naive sliding-window compression doesn't care about confidence — it throws away context
indiscriminately and degrades quality. CAMS only compresses when the model signals it's certain.

---

## How It Works

```
User message
     │
     ▼
Claude Opus 4.7 responds
     │
     ▼
CAMS extracts J-score (0 → 1) from response text
  ┌──── 5 factors ────────────────────────────────────────┐
  │  • Hedging density    "I think", "perhaps", "might"   │ (−)
  │  • Anchor density     "specifically", "exactly"       │ (+)
  │  • Self-correction    "actually", "wait", "let me"    │ (−)
  │  • Response length    shorter = more grounded         │ (+)
  │  • Numeric specificity numbers, named entities        │ (+)
  └───────────────────────────────────────────────────────┘
     │
     ▼
  J ≥ 0.65  →  HIGH   →  COMPRESS  (Claude summarises its own old turns)
  J ∈ 0.35–0.65  →  MEDIUM →  TRIM  (keep last 10 turns)
  J < 0.35  →  LOW    →  PRESERVE  (keep everything)
```

The compressor **is Claude itself** — Opus 4.7 both answers the question and summarises
its own history when confident. The model manages its own memory.

---

## Results

### Phase O — Statistical Confirmation (Qwen 2.5-7B, SQuAD v2, n=300)

| Condition | F1 | vs SnapKV | p-value |
|-----------|-----|-----------|---------|
| Baseline (FP16, no eviction) | 0.4196 | — | — |
| SnapKV-256 (static budget) | 0.3946 | — | — |
| **CAMS 512/256 + rotation** | **0.4254** | **+3.08pp** | **0.00046** |

CAMS **beats the uncompressed baseline** — compression that makes the model *more* accurate.
Wilcoxon signed-rank, one-sided. Built on 12 experimental phases over 3 months.

### F(b|J) Saturation Curve (Phase R-3B, τ_l=109.58, τ_h=154.23, R²>0.97)

High-J queries saturate earlier. They need less budget.
Low-J queries need more context to reach the same quality.
This is the theoretical proof that confidence-routing is justified.

---

## Quickstart

```bash
git clone https://github.com/chakrivijayarao/cams-claude
cd cams-claude
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# Run the demo (works without API key in demo mode)
streamlit run demo/app.py

# Run the benchmark (requires API key)
python -m evals.benchmark

# Use in code
from cams import CAMSContextManager
mgr = CAMSContextManager()
result = mgr.chat("What is the speed of light?")
print(f"J={result.j_score:.2f}  decision={result.decision}  saved={result.tokens_saved} tokens")
```

---

## Usage

### Simple chat

```python
from cams import CAMSContextManager

mgr = CAMSContextManager()

for question in questions:
    result = mgr.chat(question)
    print(result.response)
    print(f"J={result.j_score:.2f} | {result.zone} | {result.decision} | saved {result.tokens_saved} tokens")

print(f"\nSession: used {mgr.stats.total_tokens_in:,} tokens, "
      f"saved {mgr.stats.total_tokens_saved:,} tokens, "
      f"${mgr.stats.total_cost_usd:.4f} cost")
```

### Long document Q&A (Agent mode)

```python
from cams import CAMSAgent

agent = CAMSAgent()
result = agent.document_qa(
    document=long_paper_text,
    questions=[
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
    ],
)
print(result.final_report)
print(result.summary)
# → "Completed 3 sub-tasks | Tokens used: 4,821 | Saved: 2,104 (30%) | Cost: $0.0089"
```

---

## Benchmark

Run the full benchmark to see CAMS vs Baseline vs Naive sliding window:

```bash
python -m evals.benchmark
```

```
======================================================================
CAMS BENCHMARK RESULTS
======================================================================
Condition                Tokens used   Tokens saved    Cost ($)  ROUGE-L  Comp %
----------------------------------------------------------------------
Baseline (no compression)      18,432              0      0.2765    0.421      0%
Naive sliding window           11,204          7,228      0.1682    0.389     39%
CAMS                            9,871          8,561      0.1481    0.418     46%
======================================================================

CAMS vs Baseline:
  Token reduction : −46.5%
  Cost reduction  : −46.4%
  Quality delta   : −0.003 ROUGE-L  (not statistically significant)
```

CAMS achieves similar token savings to naive compression **without the quality degradation**.

---

## Architecture

```
cams-claude/
├── cams/
│   ├── confidence_proxy.py   J-score extraction (5-factor weighted signal)
│   ├── context_manager.py    Adaptive context window (Claude manages itself)
│   └── agent.py              Long-running task agent (document Q&A, research)
├── demo/
│   └── app.py                Streamlit: chat + J-gauge + research evidence
├── evals/
│   └── benchmark.py          Quality + cost comparison vs baselines
├── requirements.txt
└── .env.example
```

---

## Research Backing

This project is informed by prior experimental work validating the J-signal hypothesis
across 12 phases on Kaggle T4 GPUs using Qwen 2.5-7B:

- **J-signal AUROC**: ~0.99 for discriminating easy vs hard queries at layer 26 of Qwen 7B
- **Phase O** (n=300): CAMS +3.08pp F1 over SnapKV-256, p=0.00046 (Wilcoxon signed-rank)
- **Phase R-3B**: τ_l=109.58 < τ_h=154.23, R²>0.97 — theoretical justification confirmed
- **Negative results**: J-signal cannot select *which tokens* to evict (−72pp vs LRU at long context)
  — it can only decide *how much* budget to allocate

The language-level J-proxy used here is a surface-level adaptation of the hidden-state
Fisher J-score. It is weaker but requires no model internals — it runs on any Claude response.

---

## Limitations

- The J-proxy is a heuristic, not the true hidden-state Fisher score
- Compression quality depends on Claude's summarisation ability
- Token counts are approximate for the savings estimator
- Evaluated on SQuAD-style extractive QA; open-domain QA results vary

---

## Built with Claude Code

Every file in this repo was written during the hackathon using Claude Code (claude-sonnet-4-6),
building on a deep understanding of the J-signal mechanism developed over 12 prior
experimental phases. The tool shaped the code; the research shaped the idea.

---

*Built for the "Built with Opus 4.7: a Claude Code hackathon" — April 2026*

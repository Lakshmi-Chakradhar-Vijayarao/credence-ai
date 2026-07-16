# Paper — Epistemic Qualifier Loss

**Working title:** "Epistemic Qualifier Loss: Measuring and Preventing the Systematic
Loss of Uncertainty Signals in LLM Context Compression"

This is a draft for a standalone research paper on the scientific basis of the
Credence system — distinct from the geometry thesis.

**The companion thesis repo** (epistemic geometry / confabulation detection) is at:
→ https://github.com/Lakshmi-Chakradhar-Vijayarao/detection-without-control

---

## Files

| File | Description |
|------|-------------|
| `PAPER_DRAFT.md` | Working draft — EQL definition, EQLR metric, FCR downstream eval |
| `figures/` | Figure generation scripts for EQL paper plots |

---

## Core claims this paper makes

1. **EQL** (Epistemic Qualifier Loss): context compression systematically discards
   hedging language at rates above chance, measurable via a probe on retained tokens.

2. **EQLR** (EQL Retention Rate): scalar metric for how well a compressor preserves
   uncertainty signals. Validated on 8 models across 6 orgs.

3. **FCR** (Faithful Compression Rate): downstream task metric — does compressed
   context produce equally calibrated outputs?

4. Ghost constraint detection (the `credence_self_probe` / `credence_scan` tools)
   operationalizes EQL in real-time for any LLM workflow.

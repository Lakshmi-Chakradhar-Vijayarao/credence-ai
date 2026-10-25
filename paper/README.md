# Paper — Epistemic Qualifier Loss

**Working title:** "Epistemic Qualifier Loss: Measuring and Preventing the Systematic Loss of Uncertainty Signals in LLM Context Compression"

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

1. **EQL** (Epistemic Qualifier Loss): context compression in LLM pipelines systematically
   removes user-stated uncertainty markers ("I think", "roughly", "the vendor says") while
   keeping the value they qualify. Measured by checking whether any of 423 uncertainty
   markers survives next to the value in the compressed context.

2. **EQLR** (Epistemic Qualifier Loss Rate): the fraction of scenarios in which the value
   survives compression but its qualifier does not. **Higher is worse.** Measured at
   0.41–0.75 across 8 models from 6 organisations under standard compression; the
   Credence faithfulness probe reduces it to 0.0 on the scenarios it covers.

3. **FCR** (False Certainty Rate): downstream metric — the fraction of callbacks in which
   the model states the value with no qualifier while the value is numerically present.
   **Higher is worse.**

4. **Ghost constraints**: uncertain values stated *without* hedging language. A
   domain-keyword heuristic on user statements (paper §5) recovers them; marker-based
   detection alone cannot.

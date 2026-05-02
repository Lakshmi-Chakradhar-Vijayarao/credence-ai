# Archive Log

Files moved here during the open-source cleanup pass (2026-05-02).
Nothing is deleted — everything here is recoverable. Each section explains
why the files were moved and what would need to happen to bring them back.

---

## `api_dependent/` — Modules requiring an LLM API key

These are real, working modules. They were moved because the core product
vision is zero-config (no API key). They belong in a future `credence[plus]`
or paid tier once we have a model-routing layer.

| File | What it does | Why archived |
|------|-------------|--------------|
| `credence/agent.py` | `CredenceAgent` — full Claude Opus agent with epistemic enforcement on every turn | Requires `ANTHROPIC_API_KEY`, Claude-specific, not zero-config |
| `credence/behavioral_signal.py` | `BehavioralConsistencyProbe` — N=5 Haiku re-completions → ROUGE-L variance as uncertainty signal | Requires API key (Haiku calls), good Tier 2 signal |
| `credence/claim_extractor.py` | `ClaimExtractor` — Haiku-based ghost constraint extraction from user messages | Requires API key, catches implicit uncertain claims with no surface markers |
| `credence/semantic_entropy.py` | `SemanticEntropyProbe` — Kuhn et al. 2023 semantic entropy at API surface (N=3 completions + NLI clustering) | Requires API key, ~$0.0005/probe |
| `credence/envelope.py` | `CredenceEnvelope` — epistemic provenance wrapper for cross-agent propagation | No API key needed but adds complexity without product traction yet |
| `credence/adaptive/bandit.py` | Thompson sampling bandit for per-session-type threshold learning | Research prototype, not wired into production flow |
| `credence/adaptive/marker_weights.py` | Bayesian online learning of per-marker precision/recall | Research prototype, not wired into production flow |
| `credence/experimental/dpo_proxy.py` | `DPOConfidenceProxy` — neural faithfulness scoring via Phi-2 LoRA adapter | Requires PyTorch + HuggingFace + trained weights |
| `tests/test_claims.py` | Unit tests for `ClaimExtractor` | Archived with the module |
| `tests/unit/test_adaptive.py` | Unit tests for Thompson sampling bandit + marker weight learner | Archived with the modules |

**To restore:** Uncomment from `credence/__init__.py`, re-add to `pyproject.toml` optional deps.

---

## `research_evals/` — Academic-only evaluation scripts

These evals were written to validate research claims for the arXiv paper.
Most require the Anthropic API key and/or are multi-trial studies that
take hours to run. They are correct and reusable for future paper submissions
but are not part of the open-source product's test suite.

| File | What it tests | Status |
|------|--------------|--------|
| `calibration_curve.py` + `_results.json` | Expected Calibration Error of J-score as compression predictor | Results committed, rerun needs API |
| `ce_false_enforcement_rate.py` + `_results.json` | Consistency Enforcer false-enforcement rate (FER) | Results: 0.012 FER on 200 benign queries |
| `manifest_survival.py` + `_results.json` | Whether structured `<EPISTEMIC_MANIFEST>` XML survives Haiku compression better than natural language | Results committed |
| `comparison_table.py` + `_results.json` | Side-by-side comparison vs naive window, LLMLingua-sim, full-context baseline | Results committed, paper table |
| `cross_session_eval.py` | Cross-session FCR eval (no_memory vs naive_summary vs credence_memory) | Blocked: needs API key |
| `model_eval.py` | FCR eval of fine-tuned Phi-2 LoRA on EQL-Bench v2 | Blocked: needs API key + LoRA weights |
| `gold_audit.py` | Does Phi-2 DPO prefer epistemically faithful summaries? | Blocked: needs API key + LoRA weights |
| `ghost_detector_ablation.py` | Comparison: no_detection vs haiku_extract vs opus_ghost vs full_credence | Blocked: needs API key |
| `e6_repeated.py` + `_results.json` | 23-trial E6 bootstrap CI study | Results committed |
| `e6_ablation.py` + `_results.json` | 4-condition E6 ablation (faithfulness probe alone vs Truth Buffer alone vs both) | Results committed |
| `long_session_eval.py` | 50-turn sessions with 5 planted constraints, 3+ compression cycles | Blocked: needs API key |
| `null_hypothesis.py` + `_results.json` | Null hypothesis baseline — does prompt injection alone explain results? | Results committed |
| `agent_propagation_eval.py` + `_results.json` | Cross-agent epistemic propagation fidelity | Results committed |
| `conversation_benchmark.py` + `_results.json` | 10 multi-turn sessions × 3 conditions: constraint_recall, chain_complete, hallucination_rate | Results committed |
| `claim_gauntlet.py` | Per-claim FCR benchmark (30 claims × qualifier_type breakdown) | Blocked: needs API key |
| `flagship/` | 3-scenario × 3-condition flagship experiment | Results committed |
| `evals/kv_cache/` | KV-cache positional sensitivity study (Phase 1 research) | Good analysis, no product integration path currently |

**Note on `evals/kv_cache/`**: This is genuinely interesting research — shows that KV-cache attention patterns have positional sensitivity that correlates with epistemic importance. Filed as "good loose end." Could become a future product feature if we get access to attention weights.

---

## `kaggle/` — Kaggle-specific staging artifacts

| File | Notes |
|------|-------|
| `KAGGLE_WORKFLOW.md` | Step-by-step guide for running DPO training on Kaggle T4 |
| `requirements_kaggle.txt` | Kaggle-specific requirements (peft, trl, bitsandbytes, etc.) |
| `scripts/pull_kv_results.sh` | Shell script to pull KV-cache results from Kaggle output |
| `kv_cache_results.json` | Root-level KV-cache results (duplicate of evals/kv_cache/) |

---

## `demo_extra/` — Demo files requiring API key or for web presentation

| File | Notes |
|------|-------|
| `demo/app.py` | Full Streamlit 4-tab demo app (needs API key for live chat tab) |
| `demo/live_demo.py` | Interactive terminal demo with 7 checkpoints (needs API key) |
| `demo/cover.html` | Web presentation cover slide |
| `demo/intro.html` | Web presentation intro slide |
| `demo/presentation.html` | Full web slide deck |

**Kept in `demo/`**: `gate_demo.py`, `gate_demo.gif`, `gate_demo.svg`, `gate_demo.cast` — these demonstrate the Rust gate working without any API key, which is the right product demo.

---

## `docs_internal/` — Internal docs not needed in the public repo

| File | Notes |
|------|-------|
| `docs/SUBMISSION_SUMMARY.md` | Hackathon submission summary (superseded by SUBMISSION.md) |
| `docs/DEMO_SCRIPT.md` | Scripted demo walkthrough for a specific presentation session |

---

## Loose ends worth noting

These were identified as "good analysis but no current product path":

1. **KV-cache positional sensitivity** (`evals/kv_cache/`): Showed that attention entropy correlates with whether positions contain uncertain constraints. Hypothesis: models attend more to hedging language than to values. This could become a future zero-API-key signal if we get logit/attention access (e.g. via Ollama local models).

2. **Thompson sampling bandit** (`adaptive/bandit.py`): The idea of per-session-type threshold learning is sound. Debug sessions genuinely have different optimal theta_high than design sessions. Revisit once we have telemetry on real user sessions.

3. **Bayesian marker weight learning** (`adaptive/marker_weights.py`): Online Bayesian precision/recall tracking per-marker. Currently we have 108 markers with uniform weight. With 1000+ real sessions we could drop low-precision markers and boost high-precision ones. Data-flywheel opportunity.

4. **CredenceEnvelope** (`credence/envelope.py`): JSON-serializable provenance wrapper with trust-score decay. The concept is correct — downstream agents need to know epistemically where information came from. Held back from v1 because it adds surface area without proven uptake. Revisit for v2 multi-agent launch.

5. **DPO Phi-2 proxy** (`experimental/dpo_proxy.py`): Neural faithfulness scoring via LoRA. Epoch-2 checkpoint achieves FCR=19.1% vs 31.2% baseline. Not zero-config (needs PyTorch + weights). Candidate for a `credence[neural]` optional install tier once weights are on HuggingFace Hub.

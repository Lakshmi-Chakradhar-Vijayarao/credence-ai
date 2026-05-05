# Evals

Validation studies for Credence. All result JSON files are committed alongside the scripts that produced them.

## Offline (no API key required)

| Script | What it measures |
|--------|-----------------|
| `adversarial_tests.py` | Robustness of the faithfulness probe against adversarial inputs |
| `false_positive_rate.py` | False-positive rate of the gate on benign developer messages |
| `latency_report.py` | P50/P95/P99 latency for all enforcement checkpoints |
| `precision_eval.py` | Precision of the uncertainty marker classifier |
| `stress_test.py` | Registry throughput under high constraint volume |
| `null_hypothesis.py` | Baseline: what happens with no enforcement at all |
| `ghost_gauntlet.py` | Ghost constraint detection precision/recall |

## Requires API key (`ANTHROPIC_API_KEY`)

| Script | What it measures |
|--------|-----------------|
| `compression_faithfulness.py` | FCR before/after Credence on Claude Haiku (n=50) |
| `end_to_end_compression.py` | End-to-end FCR with full enforcement stack |
| `gauntlet.py` | 50-scenario epistemic enforcement gauntlet |
| `eql_bench.py` | EQL Benchmark across multiple models |
| `eql_bench_qwen_analysis.py` | Qwen-specific EQL analysis |
| `calibration.py` | J-score calibration vs. human labels |
| `cross_session_test.py` | Cross-session constraint persistence |
| `experiments.py` | Exploratory experiments |
| `claude_code_client.py` | Claude Code integration tests |

## Result files

All `*_results.json` files are committed. They are the evidence behind the claims in [docs/TECHNICAL_REPORT.md](../docs/TECHNICAL_REPORT.md).

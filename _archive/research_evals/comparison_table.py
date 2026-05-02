"""
evals/comparison_table.py
==========================
Generates the final comparison table for the Credence paper.

Table rows (methods) × columns (metrics):

    | Method                        | EQLR | FCR | Ghost FCR | QAR  |
    |-------------------------------|------|-----|-----------|------|
    | Naive Haiku summarization     |      |     |           |  N/A |
    | LLMLingua-2 baseline          |      |     |           |  N/A |
    | H2O KV eviction               |      |     |           |      |
    | SnapKV eviction               |      |     |           |      |
    | StreamingLLM eviction         |      |     |           |      |
    | Credence (block-based)        |      |     |           |  N/A |
    | LLMLingua-2 + epistemic loss  |      |     |           |  N/A |

Metric definitions:
    EQLR (Epistemic Qualifier Loss Rate):
        Fraction of uncertain values whose qualifier was dropped in output.
        Computed over a corpus of text compression calls.
        Source: compression_faithfulness.py results.

    FCR (False Certainty Rate):
        Fraction of uncertain values stated as confirmed facts without qualification.
        Source: compression_faithfulness.py, claim_gauntlet.py, or ghost_gauntlet.py.

    Ghost FCR:
        FCR restricted to ghost constraints — claims with no surface hedging markers.
        Source: claim_gauntlet.py or ghost_gauntlet.py.

    QAR (Question-Answer Recall):
        Downstream QA performance on contexts that went through compression.
        Only applicable to methods that maintain a full conversation context.
        "N/A" for pure compression methods.
        Source: benchmark.py ROUGE-L results.

Usage:
    python -m evals.comparison_table --results-dir results/ --out comparison_table.json
    python -m evals.comparison_table --demo   # print placeholder table with column definitions
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import random
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Maps method name → result JSON file basename(s) to look for
_METHOD_FILE_MAP: dict[str, list[str]] = {
    "Naive Haiku summarization":    [
        "compression_faithfulness_results.json",
        "compression_faithfulness_n50_results.json",
    ],
    "LLMLingua-2 baseline":         [
        "llmlingua_results.json",
        "claim_benchmark_results.json",
    ],
    "H2O KV eviction":              [
        "kv_eviction_h2o_results.json",
        "kv/h2o_results.json",
    ],
    "SnapKV eviction":              [
        "kv_eviction_snapkv_results.json",
        "kv/snapkv_results.json",
    ],
    "StreamingLLM eviction":        [
        "kv_eviction_streamingllm_results.json",
        "kv/streamingllm_results.json",
    ],
    "Credence (block-based)":       [
        "manifest_survival_results.json",
        "ghost_gauntlet_results.json",
        "claim_gauntlet_results.json",
    ],
    "LLMLingua-2 + epistemic loss": [
        "fine_tuned_results.json",
        "dpo_eval_results.json",
    ],
}

# Methods where QAR is not applicable (pure compression, no full context)
_QAR_NA_METHODS = {
    "Naive Haiku summarization",
    "LLMLingua-2 baseline",
    "Credence (block-based)",
    "LLMLingua-2 + epistemic loss",
}

# Column display order
_COLUMNS = ["EQLR", "FCR", "Ghost FCR", "QAR"]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name:         str
    eqlr:         Optional[float]   = None
    fcr:          Optional[float]   = None
    ghost_fcr:    Optional[float]   = None
    qar:          Optional[float]   = None   # None = N/A

    # 95% bootstrap CI (lo, hi) for each metric
    eqlr_ci:      Optional[tuple[float, float]] = None
    fcr_ci:       Optional[tuple[float, float]] = None
    ghost_fcr_ci: Optional[tuple[float, float]] = None
    qar_ci:       Optional[tuple[float, float]] = None

    source_files: list[str] = field(default_factory=list)
    notes:        str       = ""


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci:     float = 0.95,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for the mean.

    Args:
        values: sample values
        n_boot: number of bootstrap resamples
        ci:     confidence level (default 0.95)

    Returns:
        (lo, hi) CI bounds
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(42)
    n   = len(values)
    boot_means = []
    for _ in range(n_boot):
        resample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(resample) / n)

    boot_means.sort()
    alpha  = (1.0 - ci) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_boot)))
    hi_idx = min(n_boot - 1, int(math.ceil((1 - alpha) * n_boot)))
    return (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


# ---------------------------------------------------------------------------
# Result file discovery
# ---------------------------------------------------------------------------

def find_results_files(results_dir: str) -> dict[str, list[str]]:
    """
    Scan results_dir (and subdirectories) for JSON files that match
    the expected result file names for each method.

    Returns:
        dict mapping method_name → list of found file paths
    """
    found: dict[str, list[str]] = {m: [] for m in _METHOD_FILE_MAP}

    for method, candidates in _METHOD_FILE_MAP.items():
        for candidate in candidates:
            # Direct path
            direct = os.path.join(results_dir, candidate)
            if os.path.exists(direct):
                found[method].append(direct)
                continue
            # Nested path (already includes subdir)
            nested = os.path.join(results_dir, *candidate.split("/"))
            if os.path.exists(nested):
                found[method].append(nested)

    return found


def _scan_evals_dir(evals_dir: str) -> dict[str, list[str]]:
    """
    Also check the evals/ directory itself for result files.
    """
    found: dict[str, list[str]] = {m: [] for m in _METHOD_FILE_MAP}

    for method, candidates in _METHOD_FILE_MAP.items():
        for candidate in candidates:
            basename = os.path.basename(candidate)
            path = os.path.join(evals_dir, basename)
            if os.path.exists(path):
                if path not in found[method]:
                    found[method].append(path)

    return found


# ---------------------------------------------------------------------------
# JSON result parsers for each known format
# ---------------------------------------------------------------------------

def _parse_compression_faithfulness(path: str) -> dict:
    """
    Parse compression_faithfulness_results.json.
    Expected keys: naive_haiku_eqlr, naive_haiku_fcr, credence_eqlr, credence_fcr, ...
    """
    with open(path) as f:
        data = json.load(f)

    out = {}

    # Try common shapes
    if "naive_haiku" in data:
        nh = data["naive_haiku"]
        out["Naive Haiku summarization"] = {
            "eqlr":      nh.get("eqlr",  nh.get("qualifier_loss_rate")),
            "fcr":       nh.get("fcr",   nh.get("false_certainty_rate")),
            "ghost_fcr": nh.get("ghost_fcr"),
        }
    if "credence" in data:
        cr = data["credence"]
        out["Credence (block-based)"] = {
            "eqlr":      cr.get("eqlr",  cr.get("qualifier_loss_rate")),
            "fcr":       cr.get("fcr",   cr.get("false_certainty_rate")),
            "ghost_fcr": cr.get("ghost_fcr"),
        }

    # Flat format: conditions list
    if "conditions" in data:
        for cond in data["conditions"]:
            name = cond.get("name", "")
            if "naive" in name or "haiku" in name:
                out["Naive Haiku summarization"] = {
                    "eqlr":      cond.get("eqlr"),
                    "fcr":       cond.get("fcr"),
                    "ghost_fcr": cond.get("ghost_fcr"),
                }
            elif "credence" in name or "locked" in name:
                out["Credence (block-based)"] = {
                    "eqlr":      cond.get("eqlr"),
                    "fcr":       cond.get("fcr"),
                    "ghost_fcr": cond.get("ghost_fcr"),
                }

    # Top-level scalar format
    for prefix, method in [("naive", "Naive Haiku summarization"),
                            ("llmlingua", "LLMLingua-2 baseline"),
                            ("credence", "Credence (block-based)")]:
        eqlr_key = next((k for k in data if prefix in k.lower() and "eqlr" in k.lower()), None)
        fcr_key  = next((k for k in data if prefix in k.lower() and "fcr" in k.lower()), None)
        if eqlr_key or fcr_key:
            out[method] = {
                "eqlr":      data.get(eqlr_key),
                "fcr":       data.get(fcr_key),
                "ghost_fcr": data.get(f"{prefix}_ghost_fcr"),
            }

    return out


def _parse_ghost_gauntlet(path: str) -> dict:
    """
    Parse ghost_gauntlet_results.json.
    """
    with open(path) as f:
        data = json.load(f)

    out = {}
    if "conditions" in data:
        for cond in data["conditions"]:
            name = cond.get("name", cond.get("condition", ""))
            metrics = {
                "eqlr":      cond.get("eqlr"),
                "fcr":       cond.get("fcr", cond.get("both_rate_complement")),
                "ghost_fcr": cond.get("ghost_fcr", cond.get("fcr")),
            }
            if "naive" in name.lower():
                out["Naive Haiku summarization"] = metrics
            elif "credence" in name.lower() or "epistemic" in name.lower():
                out["Credence (block-based)"] = metrics
    elif "summary" in data:
        # Flat summary format from ghost_gauntlet
        s = data["summary"]
        for key, method in [("naive_window", "Naive Haiku summarization"),
                             ("credence_eg2", "Credence (block-based)")]:
            if key in s:
                v = s[key]
                out[method] = {
                    "eqlr":      v.get("eqlr", 1.0 - v.get("both_rate", 0.0)),
                    "fcr":       v.get("fcr"),
                    "ghost_fcr": v.get("ghost_fcr", v.get("fcr")),
                }

    return out


def _parse_manifest_survival(path: str) -> dict:
    """
    Parse manifest_survival_results.json.
    """
    with open(path) as f:
        data = json.load(f)

    out = {}
    conditions = data.get("conditions", [])
    for cond in conditions:
        name = cond.get("name", "")
        metrics = {
            "eqlr":      cond.get("eqlr"),
            "fcr":       cond.get("fcr"),
            "ghost_fcr": cond.get("ghost_fcr"),
        }
        if name == "no_injection":
            out["Naive Haiku summarization"] = metrics
        elif name in ("locked_manifest", "repeated_injection"):
            out["Credence (block-based)"] = metrics

    return out


def _parse_kv_results(path: str, method_name: str) -> dict:
    """
    Parse KV eviction result files (H2O, SnapKV, StreamingLLM).
    """
    with open(path) as f:
        data = json.load(f)

    out = {}
    metrics = {
        "eqlr":      data.get("eqlr", data.get("qualifier_loss_rate")),
        "fcr":       data.get("fcr",  data.get("false_certainty_rate")),
        "ghost_fcr": data.get("ghost_fcr"),
        "qar":       data.get("qar",  data.get("rouge_l", data.get("qa_recall"))),
    }
    out[method_name] = metrics
    return out


def _parse_dpo_results(path: str) -> dict:
    """
    Parse DPO fine-tuned model results.
    """
    with open(path) as f:
        data = json.load(f)

    out = {}
    metrics = {
        "eqlr":      data.get("eqlr"),
        "fcr":       data.get("fcr"),
        "ghost_fcr": data.get("ghost_fcr"),
    }
    out["LLMLingua-2 + epistemic loss"] = metrics
    return out


# ---------------------------------------------------------------------------
# Aggregate results from all found files
# ---------------------------------------------------------------------------

def aggregate_results(
    found_files: dict[str, list[str]],
) -> dict[str, MethodResult]:
    """
    Load and parse all found result files, returning one MethodResult per method.
    """
    results: dict[str, MethodResult] = {
        m: MethodResult(name=m, source_files=found_files.get(m, []))
        for m in _METHOD_FILE_MAP
    }

    for method, paths in found_files.items():
        for path in paths:
            try:
                _load_file_into_results(path, method, results)
            except Exception as e:
                results[method].notes += f" [parse error: {e}]"

    # Mark QAR as N/A for methods where it doesn't apply
    for method, result in results.items():
        if method in _QAR_NA_METHODS:
            result.qar = None  # will render as N/A

    return results


def _load_file_into_results(
    path:    str,
    method:  str,
    results: dict[str, MethodResult],
) -> None:
    """Parse a single result file and update the results dict."""
    basename = os.path.basename(path)

    if "compression_faithfulness" in basename:
        parsed = _parse_compression_faithfulness(path)
    elif "ghost_gauntlet" in basename or "claim_gauntlet" in basename:
        parsed = _parse_ghost_gauntlet(path)
    elif "manifest_survival" in basename:
        parsed = _parse_manifest_survival(path)
    elif "h2o" in basename.lower():
        parsed = _parse_kv_results(path, "H2O KV eviction")
    elif "snapkv" in basename.lower():
        parsed = _parse_kv_results(path, "SnapKV eviction")
    elif "streamingllm" in basename.lower():
        parsed = _parse_kv_results(path, "StreamingLLM eviction")
    elif "dpo" in basename.lower() or "fine_tuned" in basename.lower():
        parsed = _parse_dpo_results(path)
    elif "llmlingua" in basename.lower() or "claim_benchmark" in basename.lower():
        with open(path) as f:
            data = json.load(f)
        parsed = {"LLMLingua-2 baseline": {
            "eqlr":      data.get("eqlr"),
            "fcr":       data.get("fcr"),
            "ghost_fcr": data.get("ghost_fcr"),
        }}
    else:
        return

    # Merge parsed into results
    for parsed_method, metrics in parsed.items():
        if parsed_method not in results:
            continue
        r = results[parsed_method]
        if metrics.get("eqlr") is not None and r.eqlr is None:
            r.eqlr = metrics["eqlr"]
        if metrics.get("fcr") is not None and r.fcr is None:
            r.fcr = metrics["fcr"]
        if metrics.get("ghost_fcr") is not None and r.ghost_fcr is None:
            r.ghost_fcr = metrics["ghost_fcr"]
        if metrics.get("qar") is not None and r.qar is None:
            r.qar = metrics["qar"]


# ---------------------------------------------------------------------------
# Bootstrap CI over per-claim arrays (if available)
# ---------------------------------------------------------------------------

def _try_add_ci(result: MethodResult, path: str) -> None:
    """
    If the result file contains per-claim arrays, compute bootstrap CI.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return

    # Look for per-item arrays
    eqlr_vals      = data.get("eqlr_per_claim",      data.get("eqlr_values"))
    fcr_vals       = data.get("fcr_per_claim",        data.get("fcr_values"))
    ghost_fcr_vals = data.get("ghost_fcr_per_claim")
    qar_vals       = data.get("qar_per_claim",        data.get("rouge_l_per_turn"))

    if eqlr_vals and isinstance(eqlr_vals, list):
        result.eqlr_ci = bootstrap_ci(eqlr_vals)
    if fcr_vals and isinstance(fcr_vals, list):
        result.fcr_ci = bootstrap_ci(fcr_vals)
    if ghost_fcr_vals and isinstance(ghost_fcr_vals, list):
        result.ghost_fcr_ci = bootstrap_ci(ghost_fcr_vals)
    if qar_vals and isinstance(qar_vals, list):
        result.qar_ci = bootstrap_ci(qar_vals)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], ci: Optional[tuple[float, float]] = None) -> str:
    """Format a metric value with optional CI."""
    if value is None:
        return "  TBD"
    s = f"{value:.1%}"
    if ci:
        s += f" [{ci[0]:.1%}–{ci[1]:.1%}]"
    return s


def _fmt_qar(value: Optional[float], method: str,
             ci: Optional[tuple[float, float]] = None) -> str:
    """Format QAR, marking N/A methods."""
    if method in _QAR_NA_METHODS:
        return "  N/A"
    return _fmt(value, ci)


def print_comparison_table(results: dict[str, MethodResult]) -> None:
    """Print the comparison table to stdout."""
    col_w = [32, 10, 10, 12, 10]
    headers = ["Method", "EQLR", "FCR", "Ghost FCR", "QAR"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    def row(*cells):
        return "|" + "|".join(
            f" {str(c):<{w-1}}" for c, w in zip(cells, col_w)
        ) + "|"

    print()
    print(sep)
    print(row(*headers))
    print(sep)

    for method in _METHOD_FILE_MAP:
        r = results.get(method, MethodResult(name=method))
        cells = [
            method,
            _fmt(r.eqlr,      r.eqlr_ci),
            _fmt(r.fcr,       r.fcr_ci),
            _fmt(r.ghost_fcr, r.ghost_fcr_ci),
            _fmt_qar(r.qar, method, r.qar_ci),
        ]
        print(row(*cells))

    print(sep)
    print()
    print("  Metric definitions:")
    print("  EQLR      = Epistemic Qualifier Loss Rate")
    print("             Fraction of uncertain values whose qualifier was dropped")
    print("  FCR       = False Certainty Rate")
    print("             Fraction stated as confirmed fact without qualification")
    print("  Ghost FCR = FCR restricted to ghost constraints (no surface hedging)")
    print("  QAR       = Question-Answer Recall (ROUGE-L on held-out QA pairs)")
    print("             N/A for pure compression methods")
    print()
    print("  CI = 95% bootstrap confidence interval (when per-item data available)")
    print()


def print_demo_table() -> None:
    """Print placeholder table with column definitions (no data needed)."""
    print("\n  COMPARISON TABLE — DEMO MODE (TBD = not yet measured)")
    print()

    # Build placeholder results
    demo_results: dict[str, MethodResult] = {}
    for method in _METHOD_FILE_MAP:
        demo_results[method] = MethodResult(name=method)

    print_comparison_table(demo_results)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def results_to_json(results: dict[str, MethodResult]) -> dict:
    """Serialise results to a JSON-compatible dict."""
    def ci_fmt(ci):
        return {"lo": ci[0], "hi": ci[1]} if ci else None

    return {
        "columns": _COLUMNS,
        "methods": [
            {
                "name":         m,
                "eqlr":         r.eqlr,
                "fcr":          r.fcr,
                "ghost_fcr":    r.ghost_fcr,
                "qar":          r.qar if m not in _QAR_NA_METHODS else None,
                "qar_na":       m in _QAR_NA_METHODS,
                "eqlr_ci":      ci_fmt(r.eqlr_ci),
                "fcr_ci":       ci_fmt(r.fcr_ci),
                "ghost_fcr_ci": ci_fmt(r.ghost_fcr_ci),
                "qar_ci":       ci_fmt(r.qar_ci),
                "source_files": r.source_files,
                "notes":        r.notes,
            }
            for m, r in results.items()
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Credence comparison table")
    parser.add_argument("--results-dir", default="evals/",
                        help="Directory to scan for result JSON files")
    parser.add_argument("--out",         default="evals/comparison_table.json",
                        help="Output path for JSON table")
    parser.add_argument("--demo",        action="store_true",
                        help="Print placeholder table with TBD values (no data needed)")
    parser.add_argument("--no-save",     action="store_true",
                        help="Print table only, do not save JSON")
    args = parser.parse_args()

    if args.demo:
        print_demo_table()
        return

    # Scan for result files
    evals_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    found = find_results_files(args.results_dir)
    # Also check the evals dir itself
    found_evals = _scan_evals_dir(evals_dir)
    for method, paths in found_evals.items():
        for p in paths:
            if p not in found[method]:
                found[method].append(p)

    n_found = sum(len(v) for v in found.values())
    print(f"\nComparison Table — found {n_found} result file(s) across "
          f"{sum(1 for v in found.values() if v)} method(s)")
    for method, paths in found.items():
        if paths:
            print(f"  {method}: {[os.path.basename(p) for p in paths]}")

    results = aggregate_results(found)

    # Try to add per-claim CIs
    for method, paths in found.items():
        for path in paths:
            _try_add_ci(results[method], path)

    print_comparison_table(results)

    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

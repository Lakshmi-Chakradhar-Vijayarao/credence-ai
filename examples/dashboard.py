"""
examples/dashboard.py — EpistemicTag observability dashboard

Live visualization of:
  - Routing distribution (pie/bar chart)
  - J_know scatter plot (PARAM vs CTX_DEP geometry)
  - J_velocity trajectory (commitment evolution per query)
  - VERIFY heatmap (which queries trigger confabulation fingerprint)
  - Audit log table (recent queries with routing labels)

Reads from a JSONL audit log (produced by compliance_logging.py or any
EpistemicRuntime integration that writes audit records).

Can also run in live mode: wraps a model and processes queries in real-time.

Usage:
    # Offline mode — read from existing audit log
    python examples/dashboard.py --log logs/epistemic_audit.jsonl

    # Live mode — load model and process queries interactively
    python examples/dashboard.py --live \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --calibration checkpoints/llama3b_cal.json

Prerequisites:
    pip install epistemic-stack gradio plotly pandas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


# ── Data loading ──────────────────────────────────────────────────────────────

def load_audit_log(path: str, n_max: int = 1000) -> List[dict]:
    """Load records from a JSONL audit log."""
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-n_max:]


# ── Plot builders ─────────────────────────────────────────────────────────────

def routing_distribution_chart(records: List[dict]):
    """Routing distribution bar chart."""
    import plotly.graph_objects as go
    from collections import Counter

    if not records:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    routing_counts = Counter(r.get("routing", "UNKNOWN") for r in records)
    routes  = ["ANSWER", "VERIFY", "RETRIEVE", "DEFER", "ESCALATE"]
    colors  = ["#2ecc71", "#e67e22", "#3498db", "#95a5a6", "#e74c3c"]
    counts  = [routing_counts.get(r, 0) for r in routes]

    fig = go.Figure(go.Bar(
        x=routes, y=counts,
        marker_color=colors,
        text=counts, textposition="auto",
    ))
    fig.update_layout(
        title="Routing Distribution",
        xaxis_title="Route",
        yaxis_title="Count",
        template="plotly_dark",
        height=350,
    )
    return fig


def j_know_scatter(records: List[dict]):
    """J_know scatter: x=j_know, y=j_velocity, color=routing."""
    import plotly.express as px
    import pandas as pd

    if not records:
        import plotly.graph_objects as go
        return go.Figure().add_annotation(text="No data yet", showarrow=False)

    df = pd.DataFrame([{
        "j_know":     r.get("j_know", 0.0),
        "j_velocity": r.get("j_velocity", 0.0),
        "routing":    r.get("routing", "UNKNOWN"),
        "verify_flag": r.get("verify_flag", False),
    } for r in records])

    color_map = {
        "ANSWER":   "#2ecc71",
        "VERIFY":   "#e67e22",
        "RETRIEVE": "#3498db",
        "DEFER":    "#95a5a6",
        "ESCALATE": "#e74c3c",
    }
    fig = px.scatter(
        df, x="j_know", y="j_velocity",
        color="routing", color_discrete_map=color_map,
        symbol="verify_flag",
        title="Epistemic Geometry (J_know × J_velocity)",
        labels={"j_know": "J_know (PARAM → positive)", "j_velocity": "J_velocity (commitment growth)"},
        template="plotly_dark",
        height=400,
    )
    fig.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.3,
                  annotation_text="PARAM/CTX_DEP boundary")
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    return fig


def verify_heatmap(records: List[dict], n_bins: int = 20):
    """Heatmap of verify_flag rate across J_know × entropy space."""
    import plotly.graph_objects as go
    import numpy as np

    if len(records) < 10:
        return go.Figure().add_annotation(text="Need ≥ 10 records for heatmap", showarrow=False)

    j_vals  = [r.get("j_know", 0.0) for r in records]
    ent_vals = [r.get("entropy", 0.0) for r in records]
    verify  = [1.0 if r.get("verify_flag", False) else 0.0 for r in records]

    j_bins   = np.linspace(min(j_vals), max(j_vals), n_bins + 1)
    ent_bins = np.linspace(min(ent_vals), max(ent_vals), n_bins + 1)

    heat = np.zeros((n_bins, n_bins))
    cnt  = np.zeros((n_bins, n_bins))

    for j, e, v in zip(j_vals, ent_vals, verify):
        ji = min(np.searchsorted(j_bins, j) - 1, n_bins - 1)
        ei = min(np.searchsorted(ent_bins, e) - 1, n_bins - 1)
        ji = max(0, ji); ei = max(0, ei)
        heat[ei, ji] += v
        cnt[ei, ji]  += 1

    with np.errstate(invalid="ignore"):
        rate = np.where(cnt > 0, heat / cnt, np.nan)

    fig = go.Figure(go.Heatmap(
        z=rate,
        x=[(j_bins[i] + j_bins[i+1]) / 2 for i in range(n_bins)],
        y=[(ent_bins[i] + ent_bins[i+1]) / 2 for i in range(n_bins)],
        colorscale="Oranges",
        colorbar=dict(title="VERIFY rate"),
    ))
    fig.update_layout(
        title="VERIFY Flag Heatmap (J_know × Entropy)",
        xaxis_title="J_know",
        yaxis_title="Entropy",
        template="plotly_dark",
        height=400,
    )
    return fig


def audit_table(records: List[dict], n_recent: int = 20):
    """DataFrame of recent audit records for Gradio Dataframe component."""
    import pandas as pd

    if not records:
        return pd.DataFrame()

    rows = []
    for r in records[-n_recent:][::-1]:
        rows.append({
            "Timestamp":   r.get("timestamp_utc", "")[:19],
            "Routing":     r.get("routing", "—"),
            "Verify?":     "⚠" if r.get("verify_flag") else "",
            "J_know":      f"{r.get('j_know', 0.0):+.3f}",
            "J_velocity":  f"{r.get('j_velocity', 0.0):+.3f}",
            "Entropy":     f"{r.get('entropy', 0.0):.2f}",
            "Latency (ms)": f"{r.get('latency_ms', 0.0):.1f}",
            "Action":      r.get("action_taken", "—"),
        })
    return pd.DataFrame(rows)


def summary_stats(records: List[dict]) -> str:
    """Markdown string of summary statistics."""
    if not records:
        return "_No data yet._"

    from collections import Counter
    n = len(records)
    routing_dist = Counter(r.get("routing") for r in records)
    verify_rate  = sum(1 for r in records if r.get("verify_flag")) / n * 100
    blocked_rate = sum(1 for r in records if not r.get("response_served", True)) / n * 100
    mean_j       = sum(r.get("j_know", 0.0) for r in records) / n
    mean_lat     = sum(r.get("latency_ms", 0.0) for r in records) / n

    lines = [
        f"**{n} queries total**",
        f"",
        f"| Route | Count | % |",
        f"|-------|-------|---|",
    ]
    for route in ["ANSWER", "VERIFY", "RETRIEVE", "DEFER", "ESCALATE"]:
        c = routing_dist.get(route, 0)
        lines.append(f"| {route} | {c} | {c/n*100:.1f}% |")

    lines += [
        f"",
        f"VERIFY rate: **{verify_rate:.1f}%**  ",
        f"Blocked rate: **{blocked_rate:.1f}%**  ",
        f"Mean J_know: **{mean_j:+.3f}**  ",
        f"Mean latency: **{mean_lat:.1f} ms**  ",
    ]
    return "\n".join(lines)


# ── Gradio interface ──────────────────────────────────────────────────────────

def build_gradio_app(log_path: str, live_model=None):
    """Build the Gradio dashboard."""
    import gradio as gr

    def refresh(_):
        records = load_audit_log(log_path)
        return (
            routing_distribution_chart(records),
            j_know_scatter(records),
            verify_heatmap(records),
            audit_table(records),
            summary_stats(records),
        )

    def live_query(prompt, model_wrapper):
        if model_wrapper is None:
            return "No model loaded (offline mode)", None
        tag  = model_wrapper.tag(prompt)
        resp = model_wrapper.generate(prompt)

        import json
        tag_json = json.dumps(tag.to_dict(), indent=2)
        return resp.text, tag_json

    with gr.Blocks(
        title="Epistemic Telemetry Dashboard",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1400px; }",
    ) as app:
        gr.Markdown("# Epistemic Telemetry Dashboard")
        gr.Markdown(
            f"**Audit log:** `{log_path}`  "
            "| Routing signals extracted from residual stream at gen-step-1. "
            "corr(J_know, entropy) = 0.0039."
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row():
                    stats_md = gr.Markdown("_Loading..._")
                with gr.Row():
                    bar_chart = gr.Plot(label="Routing Distribution")
                refresh_btn = gr.Button("Refresh", variant="primary")
                refresh_btn.click(
                    fn=refresh, inputs=[refresh_btn],
                    outputs=[bar_chart, gr.Plot(), gr.Plot(), gr.Dataframe(), stats_md]
                )

            with gr.Tab("Epistemic Geometry"):
                scatter_plot = gr.Plot(label="J_know × J_velocity scatter")
                heat_plot    = gr.Plot(label="VERIFY Heatmap")
                refresh_btn2 = gr.Button("Refresh geometry", variant="secondary")

                def refresh_geo(_):
                    records = load_audit_log(log_path)
                    return j_know_scatter(records), verify_heatmap(records)

                refresh_btn2.click(fn=refresh_geo, inputs=[refresh_btn2],
                                   outputs=[scatter_plot, heat_plot])

            with gr.Tab("Audit Log"):
                table = gr.Dataframe(label="Recent Queries")
                refresh_btn3 = gr.Button("Refresh table", variant="secondary")

                def refresh_table(_):
                    return audit_table(load_audit_log(log_path))

                refresh_btn3.click(fn=refresh_table, inputs=[refresh_btn3], outputs=[table])

            if live_model is not None:
                with gr.Tab("Live Query"):
                    prompt_in = gr.Textbox(label="Query", placeholder="Enter a question...")
                    with gr.Row():
                        answer_out = gr.Textbox(label="Answer", lines=5)
                        tag_out    = gr.Code(label="EpistemicTag JSON", language="json")
                    query_btn = gr.Button("Query", variant="primary")
                    query_btn.click(
                        fn=lambda p: live_query(p, live_model),
                        inputs=[prompt_in],
                        outputs=[answer_out, tag_out],
                    )

        # Auto-load on startup
        app.load(fn=refresh, inputs=[gr.State(None)],
                 outputs=[bar_chart, scatter_plot, heat_plot, table, stats_md])

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Epistemic telemetry observability dashboard")
    parser.add_argument("--log",  default="logs/epistemic_audit.jsonl",
                        help="JSONL audit log path (default: logs/epistemic_audit.jsonl)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live query mode (loads model)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model ID for live mode")
    parser.add_argument("--calibration", default="checkpoints/llama3b_cal.json",
                        help="Calibration path for live mode")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    live_model = None
    if args.live:
        print(f"Loading model: {args.model}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from esm import wrap_model

        tokenizer  = AutoTokenizer.from_pretrained(args.model)
        hf_model   = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=None,
        ).cuda().eval()
        live_model = wrap_model(hf_model, tokenizer, calibration=args.calibration)
        print("Model loaded.")

    app = build_gradio_app(args.log, live_model=live_model)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

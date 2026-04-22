"""
demo/app.py
===========
CAMS — Confidence-Adaptive Memory System for Claude
Streamlit demo: three tabs

  Tab 1  Live Chat      — real-time J-score + savings on every turn
  Tab 2  Benchmark      — side-by-side CAMS vs Baseline quality + cost
  Tab 3  Evidence       — J-proxy calibration, per-turn decision log, zone validation

Run:
    streamlit run demo/app.py
"""

import os
import sys
import json
import math
import time

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cams.confidence_proxy import ConfidenceProxy, ConfidenceResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CAMS — Confidence-Adaptive Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #6366f1;
    }
    .decision-high   { color: #22c55e; font-weight: 700; }
    .decision-medium { color: #f59e0b; font-weight: 700; }
    .decision-low    { color: #ef4444; font-weight: 700; }
    .chat-user     { background: #1e3a5f; border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0; }
    .chat-assistant{ background: #1e293b; border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0; }
    .savings-banner {
        background: linear-gradient(90deg, #064e3b, #065f46);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Demo mode — pre-computed turns when API key is absent
# ---------------------------------------------------------------------------

DEMO_TURNS = [
    {
        "user": "What is the boiling point of water at sea level?",
        "assistant": "The boiling point of water at sea level is exactly 100°C (212°F), or 373.15 Kelvin. This is defined as the standard boiling point at 1 atmosphere (101.325 kPa) of pressure.",
        "j_score": 0.81, "zone": "HIGH",  "decision": "COMPRESS",
        "tokens_in": 312, "tokens_out": 48, "tokens_saved": 198,
        "reasoning": "J=0.81 (HIGH): anchored specific claims; numeric/entity grounded; concise response",
    },
    {
        "user": "What are the long-term effects of social media on mental health?",
        "assistant": "This is an area where the research is still evolving and I should note there's significant debate. Some studies suggest possible correlations with anxiety and depression, particularly among adolescents, but causality remains unclear. It might be that people with existing challenges use social media more. Effects likely vary by age, usage patterns, and individual factors. I'm genuinely uncertain about strong conclusions here.",
        "j_score": 0.24, "zone": "LOW", "decision": "PRESERVE",
        "tokens_in": 498, "tokens_out": 89, "tokens_saved": 0,
        "reasoning": "J=0.24 (LOW): heavy hedging language; self-corrects mid-response",
    },
    {
        "user": "Who wrote the play Hamlet?",
        "assistant": "Hamlet was written by William Shakespeare, around 1600–1601. It's one of his most celebrated tragedies.",
        "j_score": 0.87, "zone": "HIGH", "decision": "COMPRESS",
        "tokens_in": 289, "tokens_out": 32, "tokens_saved": 241,
        "reasoning": "J=0.87 (HIGH): anchored specific claims; numeric/entity grounded; concise response",
    },
    {
        "user": "What will AI look like in 20 years?",
        "assistant": "It's genuinely difficult to predict with any certainty. I think we'll likely see continued progress in reasoning and multimodal capabilities, but the specific form is highly uncertain. Perhaps systems that can maintain longer context, though it's unclear how far reasoning will advance. Some researchers believe AGI might emerge, others are much more skeptical. The landscape could change dramatically in ways that are hard to foresee from today's vantage point.",
        "j_score": 0.21, "zone": "LOW", "decision": "PRESERVE",
        "tokens_in": 543, "tokens_out": 94, "tokens_saved": 0,
        "reasoning": "J=0.21 (LOW): heavy hedging language",
    },
    {
        "user": "What is the chemical symbol for gold?",
        "assistant": "The chemical symbol for gold is Au, from the Latin word 'aurum'. Its atomic number is 79.",
        "j_score": 0.89, "zone": "HIGH", "decision": "COMPRESS",
        "tokens_in": 241, "tokens_out": 28, "tokens_saved": 187,
        "reasoning": "J=0.89 (HIGH): anchored specific claims; numeric/entity grounded; concise response",
    },
]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "messages":         [],
        "turn_log":         [],
        "session_tokens_used":   0,
        "session_tokens_saved":  0,
        "session_cost":          0.0,
        "session_savings":       0.0,
        "demo_idx":              0,
        "api_mode":              False,
        "cams_mgr":              None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

proxy = ConfidenceProxy()

def _j_color(zone: str) -> str:
    return {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(zone, "#94a3b8")

def _decision_label(decision: str) -> str:
    icons = {"COMPRESS": "⚡ COMPRESS", "TRIM": "✂️ TRIM", "PRESERVE": "🔒 PRESERVE"}
    return icons.get(decision, decision)

def _format_usd(v: float) -> str:
    if v < 0.001:
        return f"${v*1000:.2f}m"
    return f"${v:.4f}"

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.markdown('<p class="main-title">🧠 CAMS</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Confidence-Adaptive Memory System for Claude — '
    'every token you keep should earn its place</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key detection
# ---------------------------------------------------------------------------

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
LIVE_MODE = bool(api_key)

if not LIVE_MODE:
    st.info(
        "🎭 **Demo mode** — running pre-computed examples. "
        "Set `ANTHROPIC_API_KEY` to enable live Claude API calls.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["💬 Live Chat", "📊 Benchmark", "🔬 Research"])

# ===========================================================================
# TAB 1 — LIVE CHAT
# ===========================================================================

with tab1:
    col_chat, col_panel = st.columns([3, 2], gap="large")

    with col_panel:
        st.markdown("### CAMS Signal Panel")

        # J-score gauge
        j_placeholder   = st.empty()
        zone_placeholder = st.empty()
        reason_placeholder = st.empty()

        st.divider()

        # Session savings
        st.markdown("**Session Stats**")
        m1, m2 = st.columns(2)
        tokens_used_ph   = m1.empty()
        tokens_saved_ph  = m2.empty()
        m3, m4 = st.columns(2)
        cost_ph          = m3.empty()
        savings_ph       = m4.empty()

        st.divider()
        st.markdown("**Decision Log**")
        log_placeholder = st.empty()

        thinking_ph = st.empty()   # thinking utilization indicator

        def _refresh_panel(j: float, zone: str, reasoning: str,
                           thinking_util: float = 0.0,
                           thinking_budget: int = 0,
                           drift_state: bool = False):
            color = _j_color(zone)
            j_placeholder.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:3rem;font-weight:900;color:{color}'>{j:.2f}</div>"
                f"<div style='font-size:0.8rem;color:#94a3b8'>J-SCORE</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            drift_badge = (
                " &nbsp;<span style='background:#7f1d1d;color:#fca5a5;"
                "font-size:0.7rem;padding:2px 6px;border-radius:4px;"
                "font-weight:700'>⚠ DRIFT</span>"
                if drift_state else ""
            )
            zone_placeholder.markdown(
                f"<div style='text-align:center;font-size:1.3rem;font-weight:700;"
                f"color:{color}'>{zone} CONFIDENCE{drift_badge}</div>",
                unsafe_allow_html=True,
            )
            reason_placeholder.caption(reasoning)
            # Thinking budget + utilization — shown when thinking budget was allocated
            if thinking_budget > 0:
                bar_color = "#ef4444" if thinking_util > 0.50 else "#f59e0b"
                override_note = " → zone override to MEDIUM" if thinking_util > 0.50 else ""
                thinking_ph.markdown(
                    f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:4px'>"
                    f"🧠 Thinking budget: <span style='color:#94a3b8'>{thinking_budget} tok</span>"
                    f" &nbsp;|&nbsp; utilization: "
                    f"<span style='color:{bar_color};font-weight:700'>{thinking_util:.0%}</span>"
                    f"{override_note}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                thinking_ph.empty()

        def _refresh_stats():
            su = st.session_state.session_tokens_used
            ss = st.session_state.session_tokens_saved
            tokens_used_ph.metric("Tokens used",  f"{su:,}")
            tokens_saved_ph.metric("Tokens saved", f"{ss:,}")
            cost_ph.metric("Cost",    _format_usd(st.session_state.session_cost))
            savings_ph.metric("Saved", _format_usd(st.session_state.session_savings))

        def _refresh_log():
            log = st.session_state.turn_log
            if not log:
                log_placeholder.caption("No turns yet.")
                return
            rows = []
            for e in reversed(log[-8:]):
                c = _j_color(e["zone"])
                drift_tag = (
                    " <span style='color:#fca5a5;font-size:0.68rem'>⚠drift</span>"
                    if e.get("drift_state") else ""
                )
                rows.append(
                    f"<div style='font-size:0.78rem;padding:2px 0'>"
                    f"<b>T{e['turn']}</b> "
                    f"<span style='color:{c}'>J={e['j_score']:.2f}</span>"
                    f"{drift_tag} "
                    f"→ <b>{_decision_label(e['decision'])}</b> "
                    f"<span style='color:#94a3b8'>(-{e['tokens_saved']} tok)</span>"
                    f"</div>"
                )
            log_placeholder.markdown("\n".join(rows), unsafe_allow_html=True)

        # Initial render
        _refresh_panel(0.0, "—", "Send a message to see the signal.", 0.0, 0, False)
        _refresh_stats()
        _refresh_log()

    with col_chat:
        # Render chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                role_label = "You" if msg["role"] == "user" else "Claude"
                css_cls    = "chat-user" if msg["role"] == "user" else "chat-assistant"
                st.markdown(
                    f"<div class='{css_cls}'><b>{role_label}:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )

        # Input
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            user_input = col_input.text_input(
                "Message", placeholder="Ask anything…", label_visibility="collapsed"
            )
            submitted = col_btn.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})

            if LIVE_MODE:
                # ---- Live API call ----
                try:
                    from cams.context_manager import CAMSContextManager
                    if st.session_state.cams_mgr is None:
                        st.session_state.cams_mgr = CAMSContextManager(
                            api_key=api_key, max_tokens=512
                        )
                    mgr    = st.session_state.cams_mgr
                    result = mgr.chat(user_input)

                    assistant_text    = result.response
                    j_score          = result.j_score
                    zone             = result.zone
                    decision         = result.decision
                    t_saved          = result.tokens_saved
                    reasoning        = result.reasoning
                    tokens_in        = result.tokens_in
                    tokens_out       = result.tokens_out
                    cost_usd         = result.cost_usd
                    sav_usd          = result.savings_usd
                    thinking_util    = result.thinking_utilization
                    thinking_budget  = result.thinking_budget_used
                    drift_state      = result.drift_state

                except Exception as e:
                    assistant_text = f"[API error: {e}]"
                    j_score = zone = decision = reasoning = "—"
                    t_saved = tokens_in = tokens_out = 0
                    cost_usd = sav_usd = 0.0
                    thinking_util = 0.0
                    thinking_budget = 0
                    drift_state = False

            else:
                # ---- Demo mode ----
                idx  = st.session_state.demo_idx % len(DEMO_TURNS)
                demo = DEMO_TURNS[idx]
                st.session_state.demo_idx += 1

                assistant_text   = demo["assistant"]
                j_score         = demo["j_score"]
                zone            = demo["zone"]
                decision        = demo["decision"]
                t_saved         = demo["tokens_saved"]
                reasoning       = demo["reasoning"]
                tokens_in       = demo["tokens_in"]
                tokens_out      = demo["tokens_out"]
                cost_usd        = (tokens_in * 15 + tokens_out * 75) / 1_000_000
                sav_usd         = t_saved * 15 / 1_000_000
                thinking_util   = 0.0
                thinking_budget = 0
                drift_state     = False

            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

            # Update stats
            st.session_state.session_tokens_used  += tokens_in + tokens_out
            st.session_state.session_tokens_saved += t_saved
            st.session_state.session_cost         += cost_usd
            st.session_state.session_savings      += sav_usd

            turn_idx = len(st.session_state.turn_log) + 1
            st.session_state.turn_log.append({
                "turn": turn_idx, "j_score": j_score, "zone": zone,
                "decision": decision, "tokens_saved": t_saved,
                "drift_state": drift_state,
            })

            _refresh_panel(j_score, zone, reasoning, thinking_util,
                           thinking_budget, drift_state)
            _refresh_stats()
            _refresh_log()
            st.rerun()

        # Reset
        if st.button("🔄 New session"):
            for k in ["messages", "turn_log", "session_tokens_used",
                      "session_tokens_saved", "session_cost",
                      "session_savings", "demo_idx", "cams_mgr"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else \
                                      (0.0 if isinstance(st.session_state[k], float) else 0)
            st.session_state.cams_mgr = None
            st.rerun()

# ===========================================================================
# TAB 2 — BENCHMARK
# ===========================================================================

with tab2:
    st.markdown("### CAMS vs Baselines — Quality & Cost")
    st.caption(
        "30 questions across 3 domains (factual / reasoning / uncertain). "
        "CAMS adapts compression to confidence; the others don't."
    )

    # Try to load pre-computed results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evals", "results.json"
    )
    bench_data = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            bench_data = json.load(f)

    if bench_data:
        # ---- Show pre-computed results ----
        names   = [r["condition"]        for r in bench_data]
        tokens  = [r["total_tokens_used"] for r in bench_data]
        saved   = [r["total_tokens_saved"] for r in bench_data]
        costs   = [r["total_cost_usd"]    for r in bench_data]
        rouges  = [r["mean_rouge_l"]      for r in bench_data]
        ratios  = [r["compression_ratio"] for r in bench_data]

        c1, c2, c3, c4 = st.columns(4)
        cams_idx = next((i for i, n in enumerate(names) if n == "CAMS"), 0)
        base_idx = next((i for i, n in enumerate(names) if "Baseline" in n), 0)
        tok_save_pct = (tokens[base_idx] - tokens[cams_idx]) / max(tokens[base_idx], 1) * 100
        cost_save_pct = (costs[base_idx] - costs[cams_idx]) / max(costs[base_idx], 1e-9) * 100
        quality_delta = rouges[cams_idx] - rouges[base_idx]

        auarcs = [r.get("auarc", 0.0) for r in bench_data]
        rds    = [r.get("reasoning_density_per_kdollar", 0.0) for r in bench_data]

        c1.metric("CAMS token reduction",    f"{tok_save_pct:.0f}%",   delta="vs baseline")
        c2.metric("CAMS cost reduction",     f"{cost_save_pct:.0f}%",  delta="vs baseline")
        c3.metric("Quality delta (ROUGE-L)", f"{quality_delta:+.3f}",  delta="vs baseline")
        c4.metric("CAMS compression ratio",  f"{ratios[cams_idx]*100:.0f}%")

        st.divider()

        # Charts
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0f172a")
        COLORS = ["#6366f1", "#f59e0b", "#22c55e"]
        for ax in axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")

        short = [n.replace("(no compression)", "").replace("sliding window", "sliding\nwindow").strip() for n in names]

        # Tokens
        bars = axes[0].bar(short, tokens, color=COLORS[:len(names)], width=0.5, edgecolor="#0f172a")
        axes[0].set_title("Tokens Used", color="white", fontsize=10, pad=8)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        for bar, val in zip(bars, tokens):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tokens)*0.01,
                        f"{val/1000:.1f}k", ha="center", va="bottom", color="white", fontsize=8)

        # Cost
        bars = axes[1].bar(short, costs, color=COLORS[:len(names)], width=0.5, edgecolor="#0f172a")
        axes[1].set_title("Cost (USD)", color="white", fontsize=10, pad=8)
        for bar, val in zip(bars, costs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                        f"${val:.4f}", ha="center", va="bottom", color="white", fontsize=8)

        # ROUGE-L
        bars = axes[2].bar(short, rouges, color=COLORS[:len(names)], width=0.5, edgecolor="#0f172a")
        axes[2].set_title("Answer Quality (ROUGE-L)", color="white", fontsize=10, pad=8)
        axes[2].set_ylim(0, 1)
        for bar, val in zip(bars, rouges):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(
            "Benchmark results not yet computed. "
            "Run: `python -m evals.benchmark` to generate them."
        )
        st.info(
            "The benchmark runs 10 questions through all 3 conditions "
            "using real Claude API calls, then saves results to `evals/results.json`."
        )

        # Show what the benchmark measures
        st.markdown("""
        **What the benchmark tests:**
        | Condition | Strategy | Expected |
        |-----------|----------|---------|
        | Baseline | Full context every turn | High cost, high quality |
        | Naive sliding window | Drop turns older than N | Lower cost, some quality loss |
        | **CAMS** | Compress only when J ≥ 0.65 | **Lower cost, quality preserved** |
        """)

# ===========================================================================
# TAB 3 — EVIDENCE
# ===========================================================================

with tab3:
    st.markdown("### CAMS Evidence — All Results From This Project")
    st.caption(
        "Every number on this page is generated by running "
        "`python -m evals.benchmark`. No external data."
    )

    # ── Section 1: Benchmark results (from evals/results.json) ──────────────
    st.markdown("#### Benchmark Results")

    if bench_data:
        names  = [r["condition"]         for r in bench_data]
        tokens = [r["total_tokens_used"] for r in bench_data]
        costs  = [r["total_cost_usd"]    for r in bench_data]
        rouges = [r["mean_rouge_l"]      for r in bench_data]
        ratios = [r["compression_ratio"] for r in bench_data]

        cams_idx = next((i for i, n in enumerate(names) if n == "CAMS"), 0)
        base_idx = next((i for i, n in enumerate(names) if "Baseline" in n), 0)

        tok_save_pct  = (tokens[base_idx] - tokens[cams_idx]) / max(tokens[base_idx], 1) * 100
        cost_save_pct = (costs[base_idx]  - costs[cams_idx])  / max(costs[base_idx], 1e-9) * 100
        quality_delta = rouges[cams_idx]  - rouges[base_idx]

        auarcs = [r.get("auarc", 0.0) for r in bench_data]
        rds    = [r.get("reasoning_density_per_kdollar", 0.0) for r in bench_data]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Token reduction",    f"{tok_save_pct:.1f}%",       delta="vs Baseline")
        m2.metric("Cost reduction",     f"{cost_save_pct:.1f}%",      delta="vs Baseline")
        m3.metric("Quality Δ ROUGE-L",  f"{quality_delta:+.4f}",      delta="vs Baseline")
        m4.metric("Compression ratio",  f"{ratios[cams_idx]*100:.0f}%")

        m5, m6 = st.columns(2)
        m5.metric(
            "AUARC (proxy calibration)",
            f"{auarcs[cams_idx]:.4f}",
            delta=f"{auarcs[cams_idx] - auarcs[base_idx]:+.4f} vs Baseline",
            help="Area Under Abstention-Risk Curve: higher = J-proxy correctly identifies uncertain answers",
        )
        m6.metric(
            "Reasoning Density (ROUGE/$K)",
            f"{rds[cams_idx]:.4f}",
            delta=f"{rds[cams_idx] - rds[base_idx]:+.4f} vs Baseline",
            help="ROUGE-L per $0.001 spent — quality-per-dollar metric",
        )

        # ── Zone quality: do HIGH-J answers score better? ────────────────────
        cams_result = bench_data[cams_idx]
        high_rl = [t["rouge_l"] for t in cams_result["turns"] if t["zone"] == "HIGH"]
        med_rl  = [t["rouge_l"] for t in cams_result["turns"] if t["zone"] == "MEDIUM"]
        low_rl  = [t["rouge_l"] for t in cams_result["turns"] if t["zone"] == "LOW"]

        st.divider()
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("**Cost & Token Savings**")
            short = [n.replace("(no compression)","").replace("sliding window","sliding\nwindow").strip()
                     for n in names]
            fig1, axes = plt.subplots(1, 2, figsize=(7, 3.5), facecolor="#0f172a")
            COLORS = ["#6366f1", "#f59e0b", "#22c55e"]
            for ax in axes:
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")

            axes[0].bar(short, tokens, color=COLORS[:len(names)], width=0.5, edgecolor="#0f172a")
            axes[0].set_title("Tokens Used", color="white", fontsize=9, pad=6)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
            axes[0].grid(True, axis="y", color="#334155", alpha=0.4)

            axes[1].bar(short, costs, color=COLORS[:len(names)], width=0.5, edgecolor="#0f172a")
            axes[1].set_title("Cost (USD)", color="white", fontsize=9, pad=6)
            axes[1].grid(True, axis="y", color="#334155", alpha=0.4)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig1)
            plt.close(fig1)

        with col_b:
            st.markdown("**J-Zone vs Answer Quality**")
            st.caption("Does HIGH confidence actually mean better answers? ↓")

            zones_present = []
            means_present = []
            colors_present = []
            zone_map = [("HIGH", high_rl, "#22c55e"),
                        ("MEDIUM", med_rl, "#f59e0b"),
                        ("LOW",    low_rl, "#ef4444")]
            for z, rl_list, col in zone_map:
                if rl_list:
                    zones_present.append(z)
                    means_present.append(sum(rl_list) / len(rl_list))
                    colors_present.append(col)

            fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor="#0f172a")
            ax2.set_facecolor("#1e293b")
            ax2.tick_params(colors="#94a3b8", labelsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#334155")

            bars = ax2.bar(zones_present, means_present,
                           color=colors_present, width=0.5, edgecolor="#0f172a")
            ax2.set_title("Mean ROUGE-L by J-Zone", color="white", fontsize=10)
            ax2.set_ylabel("ROUGE-L", color="#94a3b8")
            ax2.set_ylim(0, 1)
            ax2.grid(True, axis="y", color="#334155", alpha=0.4)
            for bar, val in zip(bars, means_present):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom",
                         color="white", fontsize=9)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(
                "HIGH-zone = confident responses. "
                "If this bar is tallest, the J-proxy is calibrated correctly."
            )

        # ── Per-turn decision breakdown ──────────────────────────────────────
        st.divider()
        st.markdown("**Per-Turn Decision Log (CAMS)**")
        turns = cams_result["turns"]
        cols  = st.columns(min(len(turns), 5))
        for i, (col, t) in enumerate(zip(cols, turns[:5])):
            c = _j_color(t["zone"])
            col.markdown(
                f"<div style='background:#1e293b;border-radius:8px;padding:0.6rem;"
                f"border-top:3px solid {c}'>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>Turn {i+1}</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{c}'>"
                f"J={t['j_score']:.2f}</div>"
                f"<div style='font-size:0.75rem;color:white'>{t['decision']}</div>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>-{t['tokens_saved']} tok</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:
        st.info(
            "**No benchmark results yet.** "
            "Run `python -m evals.benchmark` to generate all evidence charts. "
            "Results auto-load here once `evals/results.json` exists.",
            icon="📊",
        )
        st.markdown("""
        **What the benchmark measures — all from this project:**

        | Metric | What it proves |
        |--------|---------------|
        | Token reduction (CAMS vs Baseline) | Cost savings are real |
        | ROUGE-L delta (CAMS vs Baseline) | Quality is preserved |
        | ROUGE-L by J-zone (HIGH vs LOW) | The confidence signal is calibrated |
        | Per-turn decision log | The routing logic fires correctly |
        | Net savings (after compression cost) | The accounting is honest |
        """)

    # ── Section 2: J-proxy live validation (no external data needed) ────────
    st.divider()
    st.markdown("#### J-Proxy Signal: Live Validation")
    st.caption(
        "Paste any text and see how CAMS reads its confidence. "
        "This runs entirely in your browser — no API call."
    )

    col_ex1, col_ex2 = st.columns(2)
    if col_ex1.button("Load high-confidence example"):
        st.session_state["probe_text"] = (
            "The speed of light in a vacuum is exactly 299,792,458 metres per second. "
            "This value is a defined constant in the International System of Units."
        )
    if col_ex2.button("Load low-confidence example"):
        st.session_state["probe_text"] = (
            "I think the effects might vary considerably, and it's quite difficult to say "
            "with certainty. Perhaps some researchers believe one thing while others might "
            "argue differently. It's possibly the case that it depends on context."
        )

    test_input = st.text_area(
        "Response text to probe:",
        value=st.session_state.get("probe_text",
            "The speed of light in a vacuum is exactly 299,792,458 metres per second."),
        height=110,
        key="probe_area",
    )

    if st.button("Compute J-score", key="compute_probe"):
        result = proxy.compute(test_input)
        color  = _j_color(result.zone)

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:3rem;font-weight:900;color:{color}'>"
            f"{result.j_score:.3f}</div>"
            f"<div style='color:#94a3b8;font-size:0.8rem'>J-SCORE</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:2rem;font-weight:700;color:{color}'>"
            f"{result.zone}</div>"
            f"<div style='color:#94a3b8;font-size:0.8rem'>CONFIDENCE ZONE</div></div>",
            unsafe_allow_html=True,
        )
        action = {"HIGH": "⚡ COMPRESS history",
                  "MEDIUM": "✂️ TRIM to window",
                  "LOW": "🔒 PRESERVE history"}[result.zone]
        c3.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>"
            f"{action}</div>"
            f"<div style='color:#94a3b8;font-size:0.8rem'>CAMS DECISION</div></div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**5-Factor Breakdown**")
        scalar_factors = {k: v for k, v in result.factors.items()
                          if isinstance(v, (int, float))}
        fac_cols = st.columns(len(scalar_factors))
        for col, (name, val) in zip(fac_cols, scalar_factors.items()):
            col.metric(name.capitalize(), f"{val:.3f}")

        if result.content_type != "text":
            st.warning(
                f"**Type Prior active:** content classified as `{result.content_type}` — "
                f"J capped to prevent compression of structured content.",
                icon="⚠️",
            )
        st.caption(result.reasoning)

"""
demo/app.py
===========
Credence — Streamlit Demo

4 tabs:
  Tab 1  The Failure     — side-by-side: naive window drops uncertain constraints → propagation error
  Tab 2  The Fix         — Credence preserves uncertain constraints, zero propagation errors
  Tab 3  Live Chat       — real-time J-gauge, zone badge, decision log, session stats
  Tab 4  Evidence        — benchmark results, calibration data, per-experiment breakdown

Run:
    streamlit run demo/app.py
"""

import os
import sys
import json
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from credence.confidence_proxy import CredenceProxy, CredenceResult

LIVE_MODE = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credence",
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
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #6366f1;
    }
    .failure-box {
        background: #3b1313;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #0f291a;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .uncertain-highlight { background: #7c3aed33; border-radius: 4px; padding: 0 4px; font-weight: 600; }
    .certain-error { background: #ef444433; border-radius: 4px; padding: 0 4px; font-weight: 600; color: #ef4444; }
    .preserved { background: #22c55e22; border-radius: 4px; padding: 0 4px; font-weight: 600; color: #22c55e; }
    .zone-high   { color: #22c55e; font-weight: 700; }
    .zone-medium { color: #f59e0b; font-weight: 700; }
    .zone-low    { color: #ef4444; font-weight: 700; }
    .chat-user     { background: #1e3a5f; border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0; }
    .chat-assistant{ background: #1e293b; border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0; }
    .j-bar-outer { background: #334155; border-radius: 8px; height: 18px; width: 100%; }
    .decision-badge {
        display: inline-block; border-radius: 20px; padding: 2px 12px;
        font-size: 0.85rem; font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Demo data (shown when API key not present)
# ---------------------------------------------------------------------------

FAILURE_DEMO = {
    "scenario": "API Integration — rate limit and token expiry uncertain",
    "seed": [
        {"role": "user",      "j": "LOW",
         "content": "I checked Stripe's docs — the rate limit is **either 100 or 50 req/min** depending on sandbox vs production. Not confirmed yet.",
         "preserved": True},
        {"role": "assistant", "j": "LOW",
         "content": "Got it — uncertain rate limit (50 vs 100 req/min). Also need to confirm the token expiry?",
         "preserved": True},
        {"role": "user",      "j": "LOW",
         "content": "Right, token expiry is **either 3600 or 86400 seconds** depending on grant type. Haven't verified.",
         "preserved": True},
        {"role": "assistant", "j": "LOW",
         "content": "Two open constraints: rate limit (50 vs 100) and token expiry (3600 vs 86400). Both need verification.",
         "preserved": True},
    ],
    "filler": [
        {"role": "user",      "j": "HIGH", "content": "Let's set up the HTTP client with connection pooling."},
        {"role": "assistant", "j": "HIGH", "content": "Set pool_maxsize=10, pool_connections=5, max_retries=3 with exponential backoff."},
        {"role": "user",      "j": "HIGH", "content": "How should we structure the webhook endpoint?"},
        {"role": "assistant", "j": "HIGH", "content": "Validate the HMAC-SHA256 signature on X-Signature-256. Return 200 immediately, process async."},
        {"role": "user",      "j": "HIGH", "content": "Correct status code for duplicate payment?"},
        {"role": "assistant", "j": "HIGH", "content": "Return 409 Conflict with error.code = 'idempotency_conflict'."},
    ],
    "callback_q": "Before we write retry logic — what were the two API constraints we still needed to verify?",
    "naive_answer": {
        "text": "The rate limit is **100 requests per minute** and the token expiry is **3600 seconds**.",
        "recall": 0.20,
        "prop_error": True,
        "explanation": "Naive window dropped the LOW-J uncertain seed turns. Only saw HIGH-J filler. Presented uncertain values as confirmed facts.",
    },
    "em_answer": {
        "text": "We had two unverified constraints: the **rate limit (50 vs 100 req/min, uncertain — sandbox vs production)** and the **token expiry (3600 vs 86400 sec, depends on grant type)**. Both still need confirmation.",
        "recall": 0.90,
        "prop_error": False,
        "explanation": "Credence preserved the LOW-J seed turns verbatim despite 6 HIGH-J filler turns. Uncertain constraints survived.",
    },
}

E6_RESULT  = {"credence": {"recall": 1.00, "halluc": 0.00}, "naive": {"recall": 0.00, "halluc": 1.00}}
E7_RESULT  = {"credence": {"hops": 3},   "naive": {"hops": 0}}
E8_RESULT  = {"credence": {"recall": 1.000}, "naive": {"recall": 1.000}, "baseline": {"recall": 0.944}}
CONV_BENCH = {"credence": {"recall": 0.818, "chain": 0.80}, "naive": {"recall": 0.657, "chain": 0.20}}
CF_RESULT  = {"naive": {"qual_survival": 0.433, "false_certainty": 0.267},
              "probe": {"block_rate": 1.00, "false_certainty": 0.00}}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

proxy = CredenceProxy()


def _j_bar(j: float, zone: str) -> str:
    pct  = int(j * 100)
    color = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(zone, "#6366f1")
    return (
        f'<div class="j-bar-outer">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:8px;'
        f'transition:width 0.4s;"></div></div>'
    )


def _zone_badge(zone: str) -> str:
    color = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(zone, "#94a3b8")
    bg    = {"HIGH": "#0f291a", "MEDIUM": "#2d1f00", "LOW": "#2d0a0a"}.get(zone, "#1e293b")
    return (f'<span class="decision-badge" style="background:{bg};color:{color};border:1px solid {color}">'
            f'{zone}</span>')


def _decision_label(d: str) -> str:
    labels = {"COMPRESS": "🗜 COMPRESS", "TRIM": "✂ TRIM", "PRESERVE": "🔒 PRESERVE"}
    return labels.get(d, d)


def _conf_tier_badge(eff_conf: float) -> str:
    """Return a coloured tier badge for a GTS hit."""
    if eff_conf < 0.20:
        return '<span style="background:#3b0a0a;color:#ef4444;border:1px solid #ef4444;border-radius:12px;padding:1px 8px;font-size:0.75rem;font-weight:700">⚠⚠ HIGH RISK</span>'
    elif eff_conf < 0.40:
        return '<span style="background:#2d1f00;color:#f59e0b;border:1px solid #f59e0b;border-radius:12px;padding:1px 8px;font-size:0.75rem;font-weight:700">⚠ UNVERIFIED</span>'
    else:
        return '<span style="background:#1e293b;color:#94a3b8;border:1px solid #475569;border-radius:12px;padding:1px 8px;font-size:0.75rem;font-weight:700">CHECK</span>'


def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "mgr" not in st.session_state:
        st.session_state.mgr = None
    if "registry" not in st.session_state:
        st.session_state.registry = None
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "turns": 0, "compressed": 0, "trimmed": 0, "preserved": 0,
            "tokens_used": 0, "tokens_saved": 0,
            "faithfulness_blocks": 0,
            "fcr_prevented": 0,      # turns where GTS annotated ≥1 unverified value
            "high_risk_hits": 0,     # HIGH RISK (conf < 0.20) annotations
            "enforcement_fires": 0,  # turns where Consistency Enforcer fired
            "ghost_detections": 0,   # implicit uncertain constraints caught by Opus ghost detector
        }


# ---------------------------------------------------------------------------
# Tab 1 — The Failure
# ---------------------------------------------------------------------------

def render_failure_tab():
    st.markdown("""
    <div class="main-title">The Failure</div>
    <div class="sub-title">
    Naive context compression drops uncertain constraints — the model then presents uncertain values as confirmed facts.
    </div>
    """, unsafe_allow_html=True)

    demo = FAILURE_DEMO

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Seed turns (uncertain constraints)")
        st.caption("These are LOW-J turns. Naive window will drop them when filler pushes them out.")
        for t in demo["seed"]:
            role_label = "You" if t["role"] == "user" else "Claude"
            j_class = "zone-low" if t["j"] == "LOW" else "zone-high"
            st.markdown(
                f'<div class="{"chat-user" if t["role"] == "user" else "chat-assistant"}">'
                f'<small><b>{role_label}</b> &nbsp; <span class="{j_class}">J={t["j"]}</span></small><br>'
                f'{t["content"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### 6 HIGH-J filler turns")
        st.caption("These push the seed turns out of the naive window.")
        for t in demo["filler"]:
            role_label = "You" if t["role"] == "user" else "Claude"
            st.markdown(
                f'<div class="{"chat-user" if t["role"] == "user" else "chat-assistant"}" style="opacity:0.65">'
                f'<small><b>{role_label}</b> &nbsp; <span class="zone-high">J=HIGH</span></small><br>'
                f'{t["content"]}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### Callback question")
        st.info(f'**Q:** "{demo["callback_q"]}"')

        st.markdown("#### Naive window response")
        nr = demo["naive_answer"]
        st.markdown(
            f'<div class="failure-box">'
            f'<b>Answer:</b> {nr["text"]}<br><br>'
            f'<b>Recall:</b> {nr["recall"]:.0%} &nbsp;&nbsp; '
            f'<span class="certain-error">PROPAGATION ERROR ⚠</span><br>'
            f'<small style="color:#94a3b8">{nr["explanation"]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### What should have happened")
        er = demo["em_answer"]
        st.markdown(
            f'<div class="success-box">'
            f'<b>Answer:</b> {er["text"]}<br><br>'
            f'<b>Recall:</b> {er["recall"]:.0%} &nbsp;&nbsp; '
            f'<span class="preserved">✓ NO PROPAGATION ERROR</span><br>'
            f'<small style="color:#94a3b8">{er["explanation"]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Why this matters")
        st.markdown("""
        When the naive window drops uncertain seed turns, the next agent sees only the
        HIGH-J filler context. It has no record that these values were unconfirmed.
        It answers with full confidence — presenting a guess as a fact.

        In a multi-step pipeline, that confident-wrong answer flows downstream.
        Each hop amplifies it further. This is the failure mode FAIL-CHAIN documented.
        """)


# ---------------------------------------------------------------------------
# Tab 2 — The Fix
# ---------------------------------------------------------------------------

def render_fix_tab():
    st.markdown("""
    <div class="main-title">The Fix</div>
    <div class="sub-title">
    Credence conditions compression on J-score.
    Only HIGH-J (resolved) content is compressed. LOW-J uncertain turns are preserved verbatim.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### J-selective memory policy")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                    '<div class="zone-low" style="font-size:1.3rem">🔒 PRESERVE</div>'
                    '<b>J &lt; 0.45</b><br>'
                    'Full history. No compression.'
                    '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                    '<div class="zone-medium" style="font-size:1.3rem">✂ TRIM</div>'
                    '<b>0.45 ≤ J &lt; 0.70</b><br>'
                    'Keep last N turns. LOW/MEDIUM-J turns always survive.'
                    '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                    '<div class="zone-high" style="font-size:1.3rem">🗜 COMPRESS</div>'
                    '<b>J ≥ 0.70</b><br>'
                    'Haiku summarises. Only HIGH-J eligible. Faithfulness probe before compress.'
                    '</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Experimental results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Compression Faithfulness (30 scenarios)")
        st.caption("Haiku compresses uncertain context. Does it strip the qualifiers?")
        c1, c2 = st.columns(2)
        c1.metric("Naive qualifier survival", f"{CF_RESULT['naive']['qual_survival']:.0%}",
                  delta=f"−{(1-CF_RESULT['naive']['qual_survival']):.0%} stripped", delta_color="inverse")
        c2.metric("Naive false certainty", f"{CF_RESULT['naive']['false_certainty']:.0%}",
                  delta=f"probe: {CF_RESULT['probe']['false_certainty']:.0%}", delta_color="inverse")

        st.markdown("#### E6 — Negative Needle")
        st.caption("Uncertain constraint planted in T3. 6 HIGH-J filler turns follow. Callback at T12.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Credence recall", f"{E6_RESULT['credence']['recall']:.0%}", "vs naive 0%")
        c2.metric("Credence hallucination", f"{E6_RESULT['credence']['halluc']:.0%}", "vs naive 100%")
        c3.metric("Chain complete", "YES", "naive: NO")

        st.markdown("#### E7 — Multi-Hop Reasoning")
        st.caption("3-hop reasoning chain (Project Falcon → Nexus config → Python ≥3.10). Naive drops T3-T5.")
        c1, c2 = st.columns(2)
        c1.metric("Credence hops recalled", f"{E7_RESULT['credence']['hops']}/3")
        c2.metric("Naive hops recalled", f"{E7_RESULT['naive']['hops']}/3")

    with col2:
        st.markdown("#### E8 — Real Debugging Session")
        st.caption("Uncertain hypothesis at T4, 6 HIGH-J filler, callback at T12. Single-trial result — stochastic.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Credence recall", f"{E8_RESULT['credence']['recall']:.3f}")
        c2.metric("Naive recall", f"{E8_RESULT['naive']['recall']:.3f}")
        c3.metric("Baseline recall", f"{E8_RESULT['baseline']['recall']:.3f}")

        st.markdown("#### Conversation Benchmark (10 sessions)")
        st.caption("3 debugging + 3 design + 2 code review + 2 research. Chain complete = all callbacks ≥ 60%.")
        c1, c2 = st.columns(2)
        c1.metric("Credence chain complete", f"{CONV_BENCH['credence']['chain']:.0%}")
        c2.metric("Naive chain complete", f"{CONV_BENCH['naive']['chain']:.0%}")

    st.markdown("---")
    st.markdown("### Guard rails preventing unsafe compression")
    g1, g2, g3 = st.columns(3)
    g1.info("**Attention sink protection**\n\nFirst 2 turns never compressed — they establish conversation identity.")
    g2.info("**Type Prior**\n\nCode blocks, error traces, math get a J floor. Code max J = 0.64 → MEDIUM zone, never COMPRESS.")
    g3.info("**Faithfulness probe**\n\nBefore every Haiku compress: scan old segment for uncertainty markers. If found → abort compress → PRESERVE.")


# ---------------------------------------------------------------------------
# Tab 3 — Live Chat
# ---------------------------------------------------------------------------

def render_live_tab():
    _init_state()

    st.markdown("""
    <div class="main-title">Live Chat</div>
    <div class="sub-title">
    Chat with Credence active. Watch J-score and compression decisions in real time.
    </div>
    """, unsafe_allow_html=True)

    if not LIVE_MODE:
        st.warning("Set `ANTHROPIC_API_KEY` to use live mode. Showing static demo below.")
        _render_live_demo()
        return

    # Ghost detector toggle — off by default (adds ~3s per turn; enable for demos)
    ghost_on = st.sidebar.toggle(
        "👻 Ghost Detector (Opus)",
        value=False,
        help="Detects implicit uncertain claims with no hedging markers. "
             "Uses a live Opus call (~3s per turn). Off by default.",
    )

    # Init Credence manager and registry
    if st.session_state.mgr is None or st.session_state.get("ghost_setting") != ghost_on:
        from credence.context_manager import ContextManager
        from credence.registry import CredenceRegistry
        reg = CredenceRegistry(":memory:")
        st.session_state.registry = reg
        st.session_state.ghost_setting = ghost_on
        st.session_state.mgr = ContextManager(
            theta_high=0.70, theta_low=0.45,
            registry=reg,
            session_id="live_demo",
            use_ghost_detector=ghost_on,
        )

    mgr = st.session_state.mgr

    col_chat, col_meta = st.columns([3, 2], gap="large")

    with col_chat:
        # Chat history
        for item in st.session_state.history:
            if item["role"] == "user":
                st.markdown(f'<div class="chat-user"><b>You</b><br>{item["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                zone = item.get("zone", "MEDIUM")
                decision = item.get("decision", "")
                j = item.get("j_score", 0.5)
                drift = item.get("drift", False)
                faith = item.get("uncertainty_preserved", False)
                enforced = item.get("enforcement_active", False)
                tb_count = item.get("truth_buffer_count", 0)
                scan_hits = item.get("scan_hits", [])

                ghost_count = item.get("ghost_detections", 0)
                drift_badge = ' <span style="color:#f59e0b">⚠ DRIFT</span>' if drift else ""
                faith_badge = ' <span style="color:#a855f7;font-weight:700">🔒 FAITHFULNESS</span>' if faith else ""
                ce_badge = ' <span style="color:#ef4444;font-weight:700">⚡ ENFORCED</span>' if enforced else ""
                tb_badge = (f' <span style="color:#6366f1;font-size:0.78rem">TB:{tb_count}</span>'
                            if tb_count > 0 else "")
                ghost_badge = (f' <span style="color:#f97316;font-weight:700">👻 GHOST×{ghost_count}</span>'
                               if ghost_count > 0 else "")

                st.markdown(
                    f'<div class="chat-assistant">'
                    f'<small><b>Claude</b> &nbsp; {_zone_badge(zone)} &nbsp; '
                    f'J={j:.3f} &nbsp; {_decision_label(decision)}'
                    f'{drift_badge}{faith_badge}{ce_badge}{tb_badge}{ghost_badge}</small><br><br>'
                    f'{item["content"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_j_bar(j, zone), unsafe_allow_html=True)

                # Show GTS hits as expandable panel
                if scan_hits:
                    high_risk = [h for h in scan_hits if h.get("eff_conf", 1.0) < 0.20]
                    label = (f"⚠⚠ {len(high_risk)} HIGH RISK literal(s) annotated"
                             if high_risk else
                             f"⚠ {len(scan_hits)} unverified literal(s) annotated")
                    with st.expander(label, expanded=bool(high_risk)):
                        for h in scan_hits:
                            ec = h.get("eff_conf", 0.30)
                            st.markdown(
                                f'{_conf_tier_badge(ec)} &nbsp; '
                                f'<code>{h["value"]}</code> — '
                                f'<em>{h["constraint_text"][:70]}{"…" if len(h["constraint_text"]) > 70 else ""}</em>',
                                unsafe_allow_html=True,
                            )

        user_input = st.chat_input("Send a message…")
        if user_input:
            st.session_state.history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                result = mgr.chat(user_input)

            faith_preserved = getattr(result, "uncertainty_preserved", False)
            scan_hits = getattr(result, "scan_hits", []) or []
            enforcement_active = getattr(result, "enforcement_active", False)
            truth_buffer_count = getattr(result, "truth_buffer_count", 0)
            ghost_detections = getattr(result, "ghost_detections", 0)

            st.session_state.history.append({
                "role": "assistant",
                "content": result.response,
                "j_score": result.j_score,
                "zone": result.zone,
                "decision": result.decision,
                "drift": getattr(result, "drift_state", False),
                "uncertainty_preserved": faith_preserved,
                "enforcement_active": enforcement_active,
                "truth_buffer_count": truth_buffer_count,
                "scan_hits": scan_hits,
                "ghost_detections": ghost_detections,
            })
            s = st.session_state.stats
            s["turns"] += 1
            s["tokens_used"] += result.tokens_in + result.tokens_out
            s["tokens_saved"] += result.tokens_saved
            if result.decision == "COMPRESS":  s["compressed"] += 1
            elif result.decision == "TRIM":    s["trimmed"] += 1
            else:                               s["preserved"] += 1
            if faith_preserved:                s["faithfulness_blocks"] += 1
            if scan_hits:                      s["fcr_prevented"] += 1
            high_risk = [h for h in scan_hits if h.get("eff_conf", 1.0) < 0.20]
            s["high_risk_hits"] += len(high_risk)
            if enforcement_active:             s["enforcement_fires"] += 1
            s["ghost_detections"] += ghost_detections
            st.rerun()

    with col_meta:
        st.markdown("#### Session Stats")
        s = st.session_state.stats
        c1, c2 = st.columns(2)
        c1.metric("Turns", s["turns"])
        c2.metric("Tokens used", f"{s['tokens_used']:,}")
        c1.metric("Tokens saved", f"{s['tokens_saved']:,}")
        ratio = (s["tokens_saved"] / max(s["tokens_used"] + s["tokens_saved"], 1))
        c2.metric("Savings ratio", f"{ratio:.1%}")

        st.markdown("#### Epistemic enforcement")
        total = max(s["turns"], 1)
        fcr = s.get("fcr_prevented", 0)
        hr  = s.get("high_risk_hits", 0)
        ce  = s.get("enforcement_fires", 0)
        fb  = s.get("faithfulness_blocks", 0)
        gd  = s.get("ghost_detections", 0)

        if gd > 0:
            st.markdown(
                f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0;border-left:3px solid #f97316">'
                f'<span style="color:#f97316;font-weight:700">👻 Ghost constraints</span> &nbsp; '
                f'<b>{gd}</b> implicit uncertain claim(s) detected by Opus'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0">'
            f'<span style="color:#ef4444;font-weight:700">FCR prevented</span> &nbsp; '
            f'<b>{fcr}</b> turns with GTS annotations'
            f'{"&nbsp; — &nbsp;<span style=\'color:#ef4444\'>⚠⚠ " + str(hr) + " HIGH RISK</span>" if hr else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0">'
            f'<span style="color:#a855f7;font-weight:700">CE fires</span> &nbsp; '
            f'<b>{ce}</b> imperative enforcements'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#1e293b;border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0">'
            f'<span style="color:#6366f1;font-weight:700">Faithfulness blocks</span> &nbsp; '
            f'<b>{fb}</b> compressions prevented'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Live Registry — shows every registered unverified constraint in real time
        reg = st.session_state.registry
        if reg is not None:
            st.markdown("#### Live Registry")
            try:
                uncertain = reg.list_uncertain("live_demo")
                if uncertain:
                    for c in uncertain[:8]:
                        vs = c.get("validation_status", "unverified")
                        src = c.get("source", "user_stated")
                        eff_conf = c.get("j_score", 0.30)
                        if vs == "disputed":
                            color, icon = "#ef4444", "⚠⚠"
                        elif src == "ghost_detector":
                            color, icon = "#f97316", "👻"
                        elif eff_conf < 0.35:
                            color, icon = "#ef4444", "⚠"
                        else:
                            color, icon = "#f59e0b", "?"
                        snippet = c["content"][:60] + ("…" if len(c["content"]) > 60 else "")
                        src_label = {"ghost_detector": "Opus ghost", "scout": "Scout",
                                     "user_stated": "user", "auto_extracted": "auto"}.get(src, src)
                        st.markdown(
                            f'<div style="background:#1e293b;border-radius:6px;padding:0.4rem 0.7rem;'
                            f'margin:0.2rem 0;border-left:3px solid {color};font-size:0.82rem">'
                            f'<span style="color:{color};font-weight:700">{icon}</span> '
                            f'<em>{snippet}</em> '
                            f'<span style="color:#64748b;font-size:0.75rem">[{src_label}, conf={eff_conf:.2f}]</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No unverified constraints registered yet."
                               " Say something unconfirmed and watch it appear here.")
            except Exception:
                pass

        st.markdown("#### Decision breakdown")
        st.markdown(f'<span class="zone-high">🗜 COMPRESS</span> {s["compressed"]}/{total}', unsafe_allow_html=True)
        st.markdown(f'<span class="zone-medium">✂ TRIM</span> {s["trimmed"]}/{total}', unsafe_allow_html=True)
        st.markdown(f'<span class="zone-low">🔒 PRESERVE</span> {s["preserved"]}/{total}', unsafe_allow_html=True)

        if st.button("Reset session"):
            st.session_state.history = []
            st.session_state.mgr = None
            st.session_state.registry = None
            st.session_state.stats = {
                "turns": 0, "compressed": 0, "trimmed": 0, "preserved": 0,
                "tokens_used": 0, "tokens_saved": 0,
                "faithfulness_blocks": 0,
                "fcr_prevented": 0,
                "high_risk_hits": 0,
                "enforcement_fires": 0,
                "ghost_detections": 0,
            }
            st.rerun()

        st.markdown("#### J-score analyser")
        sample = st.text_area("Paste any text to compute J-score:", height=100,
                               placeholder="Enter a response to analyse…")
        if sample:
            cr = proxy.compute(sample)
            st.markdown(f"{_zone_badge(cr.zone)} J = **{cr.j_score:.3f}**", unsafe_allow_html=True)
            st.markdown(_j_bar(cr.j_score, cr.zone), unsafe_allow_html=True)
            st.caption(cr.reasoning)
            with st.expander("Factor breakdown"):
                for k, v in cr.factors.items():
                    if k not in ("content_type", "j_floor"):
                        st.markdown(f"**{k}**: {v:.3f}")

        # Registry panel
        reg = st.session_state.get("registry")
        if reg is not None:
            st.markdown("#### Epistemic Registry")
            uncertain = reg.list_uncertain("live_demo")
            all_constraints = reg.get_all("live_demo")
            verified = [c for c in all_constraints if c.get("verified")]
            if all_constraints:
                st.caption(f"{len(uncertain)} unverified · {len(verified)} verified")
                for c in all_constraints[:8]:
                    icon = "✓" if c.get("verified") else "?"
                    color = "#22c55e" if c.get("verified") else "#f59e0b"
                    st.markdown(
                        f'<div style="background:#1e293b;border-left:3px solid {color};'
                        f'border-radius:6px;padding:0.4rem 0.7rem;margin:0.2rem 0;font-size:0.85rem">'
                        f'<span style="color:{color}">{icon}</span> '
                        f'{c["content"][:80]}{"…" if len(c["content"])>80 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No constraints registered yet. Say something uncertain to see tracking.")


def _render_live_demo():
    """Static demo turns shown when API key is not available."""
    demo_turns = [
        {"role": "user", "content": "I think the rate limit might be 100 req/min but I'm not sure.", "j_score": 0.35, "zone": "LOW", "decision": "PRESERVE"},
        {"role": "assistant", "content": "Got it — uncertain rate limit, possibly 100 req/min. Worth confirming before we write the retry logic.", "j_score": 0.38, "zone": "LOW", "decision": "PRESERVE"},
        {"role": "user", "content": "How do I set up connection pooling?", "j_score": 0.72, "zone": "HIGH", "decision": "COMPRESS"},
        {"role": "assistant", "content": "Set pool_maxsize=10, pool_connections=5, max_retries=3 with exponential backoff. Use requests.Session() for all calls.", "j_score": 0.85, "zone": "HIGH", "decision": "COMPRESS"},
    ]
    for t in demo_turns:
        if t["role"] == "user":
            st.markdown(f'<div class="chat-user"><b>You</b><br>{t["content"]}</div>', unsafe_allow_html=True)
        else:
            j, zone, dec = t["j_score"], t["zone"], t["decision"]
            st.markdown(
                f'<div class="chat-assistant">'
                f'<small><b>Claude</b> &nbsp; {_zone_badge(zone)} &nbsp; J={j:.3f} &nbsp; {_decision_label(dec)}</small><br><br>'
                f'{t["content"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_j_bar(j, zone), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4 — Evidence
# ---------------------------------------------------------------------------

def render_evidence_tab():
    st.markdown("""
    <div class="main-title">Evidence</div>
    <div class="sub-title">
    Benchmark results, calibration data, and proxy ceiling characterisation.
    </div>
    """, unsafe_allow_html=True)

    # ── E7 Hero Result — highlighted first ───────────────────────────────
    st.markdown("### E7 — Multi-Hop Chain Preservation (Hero Result)")
    st.markdown("""
    **3-hop dependency chain**: Project Falcon → Nexus config → CVE/v5 → Python ≥3.10.
    Six filler turns force naive window to drop T3–T5 entirely.
    """)

    col_label, col_cred, col_naive, col_base = st.columns([2, 2, 2, 2])
    col_label.markdown("**Condition**")
    col_cred.markdown("**Credence**")
    col_naive.markdown("**Naive window**")
    col_base.markdown("**Baseline**")

    rows_e7 = [
        ("Hops recalled", "3 / 3", "0 / 3", "3 / 3"),
        ("Chain complete", "✓", "✗  (chain destroyed)", "✓"),
    ]
    for label, cred, naive, base in rows_e7:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        c1.caption(label)
        c2.markdown(f'<span class="zone-high">{cred}</span>', unsafe_allow_html=True)
        c3.markdown(f'<span class="zone-low">{naive}</span>', unsafe_allow_html=True)
        c4.markdown(f'<span class="zone-medium">{base}</span>', unsafe_allow_html=True)

    st.markdown("""
    > **Why this matters**: This result is categorical, not probabilistic. Naive window
    > failed completely (0/3). The difference cannot be explained by noise or sample size.
    > Credence's selective J-routing preserved the dependency chain that naive window
    > dropped from context. Run: `python -m evals.experiments --exp E7`
    """)

    st.markdown("---")

    # ── Enforcement layer latency ─────────────────────────────────────────
    st.markdown("### Deterministic Layer Latency (JIT Buffer Design)")
    st.markdown("""
    All deterministic enforcement layers run in **under 1ms total** with zero API calls.
    """)

    latency_rows = [
        ("Faithfulness probe", "frozenset keyword lookup", "~0.07 ms", "✓ no API call"),
        ("Registry lookup + decay", "SQLite + arithmetic", "~0.37 ms", "✓ no API call"),
        ("Consistency Enforcer", "synonym-expanded token overlap", "~0.03 ms", "✓ no API call"),
        ("GTS scan (code + prose)", "regex + value map", "~0.08 ms", "✓ no API call"),
        ("**Total deterministic**", "all layers combined", "**~0.56 ms**", "✓ no API calls"),
        ("Haiku compression call", "LLM — only when needed (HIGH-J turns)", "~400–900 ms", "optional"),
    ]
    for name, mechanism, latency, note in latency_rows:
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
        c1.markdown(name)
        c2.caption(mechanism)
        c3.markdown(latency)
        c4.caption(note)

    st.markdown("""
    > JIT buffer design: enforcement fires only when uncertain constraints are
    > **present** and **queried**. Sessions with no registered uncertain constraints
    > incur zero overhead. Run `python quickstart.py` to see live latency measurements.
    """)

    st.markdown("---")

    # ── Key experiments summary ──────────────────────────────────────────
    st.markdown("### All Experiments")

    rows = [
        ("E6 — Negative Needle",   "Uncertain constraint → 8 filler → callback",
         "100% recall / 0% halluc", "0% recall / 100% halluc"),
        ("E7 — Multi-Hop Chain",   "3-hop chain, 6 filler force naive drop",
         "3/3 hops ✓", "0/3 hops ✗"),
        ("E8 — Real Debugging",    "Uncertain hypothesis, 8 HIGH-J filler",
         "0.944 recall", "0.522 recall"),
        ("E4 — Causal Validation", "Credence vs random J routing",
         "0.938", "0.750 (random: 0.875)"),
        ("Conv. Benchmark",        "10 sessions × 3 conditions, chain integrity",
         "80% chain-complete", "20% chain-complete"),
        ("Compression Faithfulness", "n=30 Haiku compressions, qualifier survival",
         "0% FCR (probe active)", "36.7% FCR (naive Haiku)"),
        ("Ghost Gauntlet",         "Implicit uncertainty, n=5 sessions",
         "BothRate 1.000", "BothRate 0.067"),
    ]

    for exp, desc, credence_r, naive_r in rows:
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
        c1.markdown(f"**{exp}**")
        c2.caption(desc)
        c3.markdown(f'<span class="zone-high">Credence: {credence_r}</span>', unsafe_allow_html=True)
        c4.markdown(f'<span class="zone-low">Naive: {naive_r}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Conversation benchmark breakdown ─────────────────────────────────
    conv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "evals", "conv_results_full.json")
    if os.path.exists(conv_path):
        with open(conv_path) as f:
            conv_data = json.load(f)

        st.markdown("### Conversation Benchmark — Full Results")

        # sessions is a flat list: each entry = {session_id, condition, constraint_recall, chain_complete, ...}
        sessions = conv_data.get("sessions", [])
        if sessions:
            rows_data = []
            for sess in sessions:
                rows_data.append({
                    "Session": sess.get("session_id", ""),
                    "Condition": sess.get("condition", ""),
                    "Recall": round(sess.get("constraint_recall", 0), 3),
                    "Chain": "YES" if sess.get("chain_complete") else "NO",
                    "Halluc": f"{sess.get('hallucination_rate', 0):.0%}",
                })
            if rows_data:
                try:
                    import pandas as pd
                    df = pd.DataFrame(rows_data)
                    st.dataframe(df, use_container_width=True)
                except ImportError:
                    for r in rows_data:
                        st.write(r)

        summary = conv_data.get("summary", {})
        if summary:
            st.markdown("**Summary:**")
            c1, c2, c3 = st.columns(3)
            recalls  = summary.get("mean_constraint_recall", {})
            chains   = summary.get("mean_chain_complete", {})
            for cond, col in [("baseline", c1), ("credence", c2), ("naive_window", c3)]:
                col.metric(
                    cond,
                    f"recall={recalls.get(cond, 0):.3f}",
                    f"chain={chains.get(cond, 0):.0%}",
                )

    # ── Flagship results (if run) ─────────────────────────────────────────
    flagship_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "experiments", "flagship", "flagship_results.json")
    if os.path.exists(flagship_path):
        with open(flagship_path) as f:
            flagship = json.load(f)

        st.markdown("---")
        st.markdown("### Flagship Experiment Results")
        from collections import defaultdict
        by_cond: dict = defaultdict(list)
        for t in flagship:
            for cond in ["baseline", "naive_window", "epistemic_memory"]:
                r = t.get(cond, {})
                by_cond[cond].append({
                    "recall": r.get("mean_recall", 0),
                    "prop_rate": r.get("propagation_rate", 0),
                    "chain": r.get("chain_complete", False),
                })

        c1, c2, c3 = st.columns(3)
        for cond, col in [("baseline", c1), ("epistemic_memory", c2), ("naive_window", c3)]:
            vals = by_cond[cond]
            if vals:
                mr = sum(v["recall"] for v in vals) / len(vals)
                pr = sum(v["prop_rate"] for v in vals) / len(vals)
                cc = sum(1 for v in vals if v["chain"]) / len(vals)
                col.metric(cond, f"recall={mr:.3f}", f"prop_rate={pr:.3f} | chain={cc:.0%}")

    # ── Proxy ceiling ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### J-Proxy Ceiling Characterisation")
    st.markdown("""
    The linguistic J-score cannot catch **confident-wrong** content — factual errors
    stated without hedging. These score HIGH-J and would be eligible for compression.

    | Category | Cases | HIGH-J rate | Notes |
    |---|---|---|---|
    | Confident-wrong (wrong facts stated confidently) | 3 | 3/3 = 100% | Documented ceiling |
    | Soft-implicit (uncertainty from context, not words) | 4 | 1/4 = 25% | Partial coverage |
    | Hedged-control (explicit hedging words) | 3 | 0/3 = 0% | Proxy works as designed |

    **Ceiling fix:** Tier 2 (behavioral consistency via N=5 Haiku samples) partially addresses
    confident-wrong cases — an uncertain fact produces *different* answers across samples,
    yielding low consistency → lower fused score → less likely to compress.
    """)

    # ── Calibration ───────────────────────────────────────────────────────
    cal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "evals", "calibration.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)

        st.markdown("---")
        st.markdown("### Threshold Calibration")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("θ_high", cal.get("theta_high", 0.70))
        c2.metric("θ_low", cal.get("theta_low", 0.45))
        c3.metric("AUARC", f"{cal.get('auarc', 0):.4f}")
        c4.metric("OOF accuracy", f"{cal.get('oof_accuracy', 0):.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-title" style="text-align:center">Credence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="text-align:center">'
                'Memory allocation conditioned on epistemic state — '
                'compress only what is resolved, preserve what is uncertain.'
                '</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 The Failure",
        "🟢 The Fix",
        "💬 Live Chat",
        "📊 Evidence",
    ])

    with tab1: render_failure_tab()
    with tab2: render_fix_tab()
    with tab3: render_live_tab()
    with tab4: render_evidence_tab()


if __name__ == "__main__":
    main()

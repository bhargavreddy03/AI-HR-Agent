"""
app.py  ─  AI HR Agent · Streamlit Dashboard
Run:  streamlit run app.py
"""

import os
import sys
import io
import re
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from hr_agent import HRAgent, VALID_TRANSITIONS
from utils    import (load_csv_candidates, generate_demo_candidates, score_badge,
                      priority_badge, summarise_results,
                      graphs_exist, get_graph_path, GRAPH_FILES, GRAPHS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        pass
    return pdf_bytes.decode("utf-8", errors="ignore")


def guess_name_from_text(text: str) -> str:
    m = re.search(r'(?:name\s*[:\-]?\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', text, re.I)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$', line):
            return line
    return "Unknown Candidate"


def guess_experience_from_text(text: str) -> float:
    # FIX-9: updated regex to capture decimal years (e.g. "1.5 years of experience")
    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\s*of\s*experience',
        r'experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*years?',
        r'(\d+(?:\.\d+)?)\s*years?\s*(?:work|professional|industry)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return float(m.group(1))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI HR Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0f17;
    color: #e2e4ef;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111428 0%, #0d0f17 100%);
    border-right: 1px solid #1e2140;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] .stRadio > label  { display: none !important; }
[data-testid="stSidebar"] .stRadio > div {
    display: flex; flex-direction: column; gap: 2px; padding: 0 8px;
}
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important; align-items: center !important; gap: 10px !important;
    padding: 0 14px !important; border-radius: 10px !important;
    border: 1px solid transparent !important; cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #7b84a8 !important; background: transparent !important;
    position: relative !important; margin: 0 !important; user-select: none !important;
    height: 42px !important; min-height: 42px !important; max-height: 42px !important;
    width: 100% !important; box-sizing: border-box !important;
    justify-content: flex-start !important; white-space: nowrap !important;
    overflow: hidden !important; text-overflow: ellipsis !important;
    line-height: 1 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #1a1f3a !important; border-color: #2a2f55 !important;
    transform: scale(1.015) !important; color: #c5caf0 !important;
    box-shadow: 0 0 0 1px #4361ee22 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:active { transform: scale(0.98) !important; }
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked),
[data-testid="stSidebar"] .stRadio > div [data-checked="true"] {
    background: linear-gradient(135deg, #1e2d6e 0%, #1a1535 100%) !important;
    border-color: #4361ee44 !important; color: #e0e4f8 !important;
    box-shadow: 0 0 0 1px #4361ee33, inset 0 0 20px #7209b711 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked)::before {
    content: '' !important; position: absolute !important; left: 0 !important;
    top: 20% !important; bottom: 20% !important; width: 3px !important;
    background: linear-gradient(180deg, #4361ee, #7209b7) !important;
    border-radius: 0 3px 3px 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] .stRadio > div > label > div:last-child,
[data-testid="stSidebar"] .stRadio > div > label > p {
    color: inherit !important; font-size: 13px !important; font-weight: 500 !important;
    line-height: 1 !important; white-space: nowrap !important;
    overflow: hidden !important; text-overflow: ellipsis !important; max-width: 100% !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div,
[data-testid="stSidebar"] .stRadio > div > label > div > div {
    height: 100% !important; display: flex !important; align-items: center !important;
    margin: 0 !important; padding: 0 !important; box-sizing: border-box !important;
}
[data-testid="stSidebar"] .stRadio > div > label p {
    margin: 0 !important; padding: 0 !important; line-height: 1 !important;
}

.nav-section-label {
    font-size: 9.5px; font-weight: 700; letter-spacing: 1.3px; color: #3b4270;
    text-transform: uppercase; padding: 10px 20px 3px; margin: 0;
}
[data-testid="stSidebar"] hr {
    border: none; border-top: 1px solid #1e2140; margin: 10px 0;
}

/* ── Cards ── */
.hr-card {
    background: #141729; border: 1px solid #1e2140;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1f3a 0%, #141729 100%);
    border: 1px solid #2a2f55; border-radius: 12px;
    padding: 18px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #4cc9f0; }
.metric-lbl { font-size: 0.78rem; color: #7b83a8; margin-top: 4px; }
.metric-sub { font-size: 0.65rem; color: #3b4270; margin-top: 2px; }

/* ── Typography ── */
.page-title {
    font-size: 2rem; font-weight: 700; color: #f0f2ff;
    letter-spacing: -0.5px; margin-bottom: 4px;
}
.page-sub   { font-size: 0.9rem; color: #6b74a4; margin-bottom: 24px; }

.section-hd {
    font-size: 1.05rem; font-weight: 600; color: #c5caf0;
    border-left: 3px solid #4361ee; padding-left: 10px; margin: 20px 0 12px;
}

/* ── Badges ── */
.badge        { display:inline-block; padding:3px 10px; border-radius:20px;
                font-size:0.75rem; font-weight:600; margin:2px; }
.badge-blue   { background:#1e3a8a; color:#93c5fd; }
.badge-purple { background:#4c1d95; color:#c4b5fd; }
.badge-green  { background:#064e3b; color:#6ee7b7; }
.badge-red    { background:#7f1d1d; color:#fca5a5; }
.badge-amber  { background:#78350f; color:#fde68a; }

/* ── Score bar ── */
.score-bar  { height:6px; border-radius:4px; background:#1e2140; margin:4px 0 12px; }
.score-fill { height:100%; border-radius:4px; }

/* ── Rank card ── */
.rank-card { background:#141729; border:1px solid #1e2140; border-radius:10px;
             padding:14px 18px; margin-bottom:10px; display:flex;
             justify-content:space-between; align-items:center; }
.rank-num  { font-size:1.6rem; font-weight:700; color:#4361ee; min-width:40px; }

/* ── Question card ── */
.q-card { background:#111428; border-left:3px solid #7209b7;
          border-radius:8px; padding:12px 16px; margin-bottom:8px; }
.q-text { font-size:0.88rem; color:#c8cde8; }

/* ── Buttons & inputs ── */
div.stButton > button {
    background: linear-gradient(135deg, #4361ee, #7209b7);
    color: white; border: none; border-radius: 8px; font-weight: 600;
    padding: 0.5rem 1.5rem; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }
div.stTextArea textarea { background:#141729; color:#e2e4ef; border:1px solid #2a2f55; }
div.stSelectbox div[data-baseweb="select"] { background:#141729; }
div[data-testid="stFileUploader"] {
    background:#141729; border:1px dashed #2a2f55; border-radius:8px; padding:8px;
}
.stDataFrame { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "agent":          None,
    "train_results":  None,
    "best_model":     None,
    "ranked":         [],
    "ranked_full":    [],
    "last_ranked_jd": "",
    "questions":      [],
    "scheduled":      [],
    "pdf_candidates": [],
    # frozen watermark: total resumes screened — never decreases as candidates progress
    "total_screened": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.agent is None:
    with st.spinner("Loading HR Agent…"):
        st.session_state.agent = HRAgent()

agent: HRAgent = st.session_state.agent

# Integration: single shared SkillExtractor instance reused across all pages
extractor = agent.ranker.skill_extractor

CSV_PATH = "resume_dataset_1200.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:22px 0 14px;'>
        <div style='width:44px;height:44px;border-radius:12px;
                    background:linear-gradient(135deg,#4361ee33,#7209b733);
                    border:1px solid #4361ee44;display:flex;align-items:center;
                    justify-content:center;font-size:20px;margin:0 auto 10px;'>
            🤖
        </div>
        <div style='font-size:1.05rem;font-weight:700;color:#e0e4f8;letter-spacing:0.3px;'>
            AI HR Agent
        </div>
        <div style='font-size:0.68rem;color:#5b6490;margin-top:3px;'>
            Intelligent Talent Platform
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #1e2140;margin:0 0 8px;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Main</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Overview",
            "🧠  Train ML Models",
            "📊  Analytics Dashboard",
            "📄  Resume Screening",
            "🗓  Interview Scheduling",
            "❓  Question Generator",
            "🔄  Pipeline Manager",
            "📤  Export Results",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid #1e2140;margin:10px 8px;'>
    <div style='font-size:0.68rem;color:#3b4270;text-align:center;padding-bottom:8px;'>
        v2.0 · AI HR Agent
    </div>
    """, unsafe_allow_html=True)

page = page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: show graph
# ─────────────────────────────────────────────────────────────────────────────
def show_graph(key: str, caption: str = ""):
    path = get_graph_path(key)
    if path:
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info("📊 Graph not generated yet — run the training pipeline first.")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline sync helper — called after scheduling
# ─────────────────────────────────────────────────────────────────────────────
def sync_pipeline_after_scheduling(scheduled_records: list):
    """
    After interview scheduling:
      • Candidates with a confirmed slot       → interview_scheduled
      • Candidates still in early-stage states → rejected
    Only downgrades candidates in 'applied' or 'shortlisted' to avoid
    overwriting any later stage set by manual override.
    """
    if not st.session_state.ranked:
        return

    scheduled_ids = {s["candidate_id"] for s in scheduled_records}
    early_states  = {"applied", "shortlisted"}

    for cand in st.session_state.ranked:
        cid     = str(cand["id"])
        current = agent.pipeline.get_status(cid)
        if cid in scheduled_ids:
            agent.pipeline.upsert_state(cid, "interview_scheduled")
        elif current is None or current in early_states:
            agent.pipeline.upsert_state(cid, "rejected")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-transition: interview_scheduled → interviewed (if slot time has passed)
# ─────────────────────────────────────────────────────────────────────────────
def auto_update_interviewed():
    """
    For every scheduled interview whose slot datetime is in the past,
    transition the candidate from 'interview_scheduled' → 'interviewed'.
    Uses pipeline.transition() to respect VALID_TRANSITIONS.
    Safe to call on every page load — fully idempotent.
    """
    now = datetime.now()
    for record in st.session_state.scheduled:
        cid  = record.get("candidate_id")
        slot = record.get("slot")
        if not cid or not slot:
            continue
        try:
            slot_dt = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if slot_dt < now and agent.pipeline.get_status(cid) == "interview_scheduled":
            agent.pipeline.transition(cid, "interviewed")


# ─────────────────────────────────────────────────────────────────────────────
# Stage Summary renderer — correct counts
# ─────────────────────────────────────────────────────────────────────────────
def render_stage_summary(state_colors: dict):
    """
    Render Stage Summary metric cards with accurate counts:

      applied              → st.session_state.total_screened
                             Frozen watermark set once when resumes are screened.
                             Equals the total number of resumes received/ranked.
                             Never changes as candidates progress through later stages.

      shortlisted          → live count from pipeline dict
      interview_scheduled  → live count from pipeline dict (upcoming interviews only)
      interviewed          → live count from pipeline dict (completed interviews)
      selected             → live count from pipeline dict
      rejected             → live count from pipeline dict

    This mirrors a real ATS: 'applied' is a permanent historical record;
    every other stage is a live snapshot of the current pipeline state.
    """
    statuses = agent.pipeline.all_statuses()
    live     = pd.Series(statuses).value_counts() if statuses else pd.Series(dtype=int)

    stage_order = [
        "applied",
        "shortlisted",
        "interview_scheduled",
        "interviewed",
        "selected",
        "rejected",
    ]
    sublabels = {}

    display_counts = {
        "applied":             st.session_state.total_screened,       # ← frozen watermark
        "shortlisted":         int(live.get("shortlisted",         0)),
        "interview_scheduled": int(live.get("interview_scheduled", 0)),
        "interviewed":         int(live.get("interviewed",         0)),
        "selected":            int(live.get("selected",            0)),
        "rejected":            int(live.get("rejected",            0)),
    }

    metric_cols = st.columns(3)
    for idx, stage in enumerate(stage_order):
        count    = display_counts[stage]
        color    = state_colors.get(stage, "#888")
        sublabel = sublabels.get(stage, "")
        with metric_cols[idx % 3]:
            st.markdown(
                f'<div class="metric-card" style="padding:12px;">'
                f'<div class="metric-val" style="color:{color};font-size:1.5rem;">{count}</div>'
                f'<div class="metric-lbl">{stage}</div>'
                + ""
                + "</div>",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">🤖 AI HR Agent Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Intelligent Talent Acquisition & HR Automation Platform</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    kpis = [
        (len(st.session_state.ranked),    "#4cc9f0", "Candidates Ranked"),
        (len(st.session_state.scheduled), "#4361ee", "Interviews Scheduled"),
        (len(st.session_state.questions), "#7209b7", "Questions Generated"),
    ]
    for col, (val, color, label) in zip([c1, c2, c3], kpis):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:{color};">{val}</div>'
                f'<div class="metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("📄 Resume Screening",
         "Upload PDF resumes — auto-extract name & skills, then rank against any job description."),
        ("🧠 ML Analytics",
         "6 regularised classifiers (no data leakage) with realistic 90–96 % accuracy on holdout set."),
        ("📊 Visual Analytics",
         "16 professional graphs: accuracy, ROC, confusion matrix, feature importance & more."),
        ("🗓 Smart Scheduling",
         "Conflict-free interview slot assignment with real-time availability tracking."),
        ("❓ Question Gen",
         "Skill-specific technical, scenario, and behavioral questions — no generic templates."),
        ("🔄 Pipeline Manager",
         "Track and transition candidates through every stage of the hiring pipeline."),
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div class="hr-card"><div class="section-hd">{title}</div>'
                f'<p style="font-size:0.85rem;color:#7b84a8;">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    if st.session_state.train_results:
        st.markdown('<div class="section-hd">🏆 Best Model Performance</div>', unsafe_allow_html=True)
        best = st.session_state.best_model
        res  = st.session_state.train_results[best]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Accuracy", "ROC-AUC", "CV Score", "MAE"],
            [res["test_accuracy"], res["roc_auc"], res["cv_mean"], res["mae"]],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{val:.3f}</div>'
                    f'<div class="metric-lbl">{label} · {best}</div></div>',
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Train ML Models
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Train ML Models":
    st.markdown('<div class="page-title">🧠 Train ML Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Train 6 regularised ML Models.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="hr-card">', unsafe_allow_html=True)
        st.markdown(
            "**Models:** RandomForest · LogisticRegression · LinearSVC · "
            "KNN · GradientBoosting · MLP"
        )
        st.markdown(
            "**Pipeline:** Load CSV → Impute → Scale → SMOTE → Train/Evaluate → Save graphs"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        csv_exists = os.path.exists(CSV_PATH)
        st.markdown(
            f'<div class="hr-card" style="text-align:center">'
            f'<div style="font-size:2rem;">{"✅" if csv_exists else "❌"}</div>'
            f'<div style="color:#7b84a8;font-size:0.82rem;">Dataset: {CSV_PATH}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if not csv_exists:
        st.error(f"Dataset not found: `{CSV_PATH}`. Place the CSV in the same directory as app.py.")
    else:
        if st.button("🚀 Start Training Pipeline", use_container_width=True):
            with st.spinner("Training models — this may take 60-90 seconds…"):
                try:
                    from train_models import train_pipeline
                    bar = st.progress(0, text="Initialising…")
                    results, best = train_pipeline(CSV_PATH)
                    bar.progress(100, text="Done!")
                    st.session_state.train_results = results
                    st.session_state.best_model    = best
                    st.success(f"✅ Training complete! Best model: **{best}**")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.exception(e)

    if st.session_state.train_results:
        st.markdown('<div class="section-hd">📋 Model Performance Summary</div>', unsafe_allow_html=True)
        df   = summarise_results(st.session_state.train_results)
        best = st.session_state.best_model

        def highlight_best(row):
            return [
                "background-color:#1a2040;font-weight:bold" if row["Model"] == best else ""
                for _ in row
            ]

        st.dataframe(
            df.style.apply(highlight_best, axis=1).format({
                "Test Accuracy":  "{:.4f}",
                "Train Accuracy": "{:.4f}",
                "ROC-AUC":        "{:.4f}",
                "Precision":      "{:.4f}",
                "Recall":         "{:.4f}",
                "F1 Score":       "{:.4f}",
                "FNR":            "{:.4f}",
                "FPR":            "{:.4f}",
                "RMSE":           "{:.4f}",
                "MSE":            "{:.4f}",
            }),
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Analytics Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Analytics Dashboard":
    st.markdown('<div class="page-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">All 16 ML performance graphs — train models first to populate</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🎯 Accuracy", "📐 Classification", "⚠️ Error Rates", "📉 ROC & AUC",
        "🔲 Confusion Matrix", "🔁 Cross-Validation", "📏 Error Metrics",
        "🌟 Feature Importance", "🔮 Predictions", "📊 Full Dashboard",
    ])

    with tabs[0]:
        st.markdown('<div class="section-hd">Train Accuracy vs Test Accuracy</div>', unsafe_allow_html=True)
        show_graph("train_vs_test_accuracy")

    with tabs[1]:
        st.markdown('<div class="section-hd">Precision, Recall & F1 — Combined</div>', unsafe_allow_html=True)
        show_graph("precision_recall_f1")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hd">Individual Metric Comparisons</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p style="text-align:center;color:#9ba4c7;font-size:0.82rem;font-weight:600;margin-bottom:6px;">Precision</p>', unsafe_allow_html=True)
            show_graph("precision_comparison")
        with c2:
            st.markdown('<p style="text-align:center;color:#9ba4c7;font-size:0.82rem;font-weight:600;margin-bottom:6px;">Recall</p>', unsafe_allow_html=True)
            show_graph("recall_comparison")
        with c3:
            st.markdown('<p style="text-align:center;color:#9ba4c7;font-size:0.82rem;font-weight:600;margin-bottom:6px;">F1 Score</p>', unsafe_allow_html=True)
            show_graph("f1_score_comparison")

    with tabs[2]:
        st.markdown('<div class="section-hd">FPR vs FNR — Combined</div>', unsafe_allow_html=True)
        show_graph("fpr_fnr_comparison")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hd">Individual Error Rate Comparisons</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="text-align:center;color:#9ba4c7;font-size:0.82rem;font-weight:600;margin-bottom:6px;">False Positive Rate (FPR)</p>', unsafe_allow_html=True)
            show_graph("fpr_comparison")
        with c2:
            st.markdown('<p style="text-align:center;color:#9ba4c7;font-size:0.82rem;font-weight:600;margin-bottom:6px;">False Negative Rate (FNR)</p>', unsafe_allow_html=True)
            show_graph("fnr_comparison")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-hd">ROC Curves — All Models</div>', unsafe_allow_html=True)
            show_graph("roc_curves")
        with c2:
            st.markdown('<div class="section-hd">ROC-AUC Bar Comparison</div>', unsafe_allow_html=True)
            show_graph("roc_auc_comparison")

    with tabs[4]:
        st.markdown('<div class="section-hd">Confusion Matrix — Best Model</div>', unsafe_allow_html=True)
        col_cm, _ = st.columns([1, 1])
        with col_cm:
            show_graph("confusion_matrix", "Confusion Matrix — Best Model")

    with tabs[5]:
        st.markdown('<div class="section-hd">Cross-Validation Mean vs Test Accuracy</div>', unsafe_allow_html=True)
        show_graph("cv_comparison")

    with tabs[6]:
        st.markdown('<div class="section-hd">Error Metrics — MSE / RMSE / MAE</div>', unsafe_allow_html=True)
        show_graph("error_metrics", "MSE / RMSE / MAE Comparison")

    with tabs[7]:
        st.markdown('<div class="section-hd">Top-20 Feature Importances — RandomForest</div>', unsafe_allow_html=True)
        show_graph("feature_importance", "Top-20 Feature Importances")

    with tabs[8]:
        st.markdown('<div class="section-hd">Actual vs Predicted — Best Model (Probability)</div>', unsafe_allow_html=True)
        show_graph("actual_vs_predicted", "Actual vs Predicted")

    with tabs[9]:
        st.markdown('<div class="section-hd">Full Metric Dashboard — All Models Overview</div>', unsafe_allow_html=True)
        show_graph("metric_dashboard")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Resume Screening
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Resume Screening":
    st.markdown('<div class="page-title">📄 Resume Screening & Ranking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload PDF resumes — name & skills extracted automatically</div>',
        unsafe_allow_html=True,
    )

    screening_tabs = st.tabs(["📤 Upload Resumes", "📋 Dataset Candidates", "🏆 Rankings"])

    # ── Tab 0: PDF Upload ─────────────────────────────────────────────────────
    with screening_tabs[0]:
        st.markdown('<div class="section-hd">Upload Resume PDFs</div>', unsafe_allow_html=True)
        st.caption("Upload one or more PDF resumes. Name and skills will be extracted automatically.")

        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_candidates = []
            for uf in uploaded_files:
                pdf_bytes = uf.read()
                text      = extract_text_from_pdf(pdf_bytes)
                # FIX-8: reject empty/unreadable PDFs before processing
                if not text.strip():
                    st.error(
                        f"❌ Could not extract text from **{uf.name}**. "
                        "The PDF may be scanned, image-only, or corrupted. "
                        "Please upload a text-based PDF."
                    )
                    continue
                name      = guess_name_from_text(text)
                exp       = guess_experience_from_text(text)
                # Integration: use shared SkillExtractor instead of ranker.extract_skills()
                skills    = extractor.extract(text)

                cand = {
                    "id":               f"PDF_{uf.name.replace('.pdf', '')}",
                    "name":             name,
                    "resume_text":      text,
                    "experience_years": exp,
                    "source":           "pdf_upload",
                    "filename":         uf.name,
                }
                new_candidates.append(cand)

                skill_badges = (
                    "".join(f'<span class="badge badge-purple">{s}</span>' for s in skills)
                    if skills else '<span style="color:#f72585;">No known skills detected</span>'
                )
                st.markdown(
                    f'<div class="hr-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div><strong style="color:#e2e4ef;font-size:1rem;">👤 {name}</strong>'
                    f'<span style="color:#6b74a4;font-size:0.8rem;margin-left:12px;">{uf.name}</span></div>'
                    f'<span class="badge badge-blue">{exp:.0f} yrs exp</span></div>'
                    f'<div style="margin-top:10px;">'
                    f'<strong style="font-size:0.8rem;color:#9ba4c7;">Extracted Skills:</strong>'
                    f'<br>{skill_badges}</div></div>',
                    unsafe_allow_html=True,
                )

                with st.expander(f"✏️ Edit extracted info for {uf.name}"):
                    corrected_name = st.text_input(
                        f"Name ({uf.name})", value=name, key=f"name_{uf.name}"
                    )
                    corrected_exp = st.number_input(
                        f"Experience years ({uf.name})",
                        value=float(exp), min_value=0.0, max_value=50.0, step=0.5,
                        key=f"exp_{uf.name}",
                    )
                    manual_skills = st.text_input(
                        f"Override skills ({uf.name})",
                        value=", ".join(skills),
                        key=f"skills_{uf.name}",
                    )
                    cand["name"]             = corrected_name
                    cand["experience_years"] = corrected_exp
                    if manual_skills.strip():
                        cand["resume_text"]  = text + " " + manual_skills

            if st.button("💾 Save Uploaded Candidates", use_container_width=True):
                st.session_state.pdf_candidates = new_candidates
                st.success(
                    f"✅ {len(new_candidates)} candidate(s) saved. "
                    "Go to Rankings tab to screen them."
                )

    # ── Tab 1: Dataset Candidates ─────────────────────────────────────────────
    with screening_tabs[1]:
        st.markdown('<div class="section-hd">Load from Dataset</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            n_candidates = st.slider("Number of candidates to load", 5, 50, 10)
        with col2:
            use_demo = st.checkbox(
                "Use demo candidates instead",
                value=not os.path.exists(CSV_PATH),
            )

        if st.button("📂 Load Candidates", use_container_width=True):
            if use_demo or not os.path.exists(CSV_PATH):
                candidates = generate_demo_candidates(n_candidates)
                st.info("Using demo candidates (CSV not found).")
            else:
                candidates = load_csv_candidates(CSV_PATH, n_candidates)
            st.session_state.pdf_candidates = candidates
            st.success(f"✅ Loaded {len(candidates)} candidates from dataset.")

        if st.session_state.pdf_candidates:
            preview = pd.DataFrame([
                {
                    "Name":             c["name"],
                    "Exp (yrs)":        c.get("experience_years", 0),
                    "Source":           c.get("source", "dataset"),
                    # Integration: use shared SkillExtractor for dataset preview
                    "Skills (preview)": ", ".join(
                        extractor.extract(c.get("resume_text", ""))[:4]
                    ) or "—",
                }
                for c in st.session_state.pdf_candidates
            ])
            st.dataframe(preview, use_container_width=True, hide_index=True)

    # ── Tab 2: Rankings ───────────────────────────────────────────────────────
    with screening_tabs[2]:
        st.markdown('<div class="section-hd">Screen & Rank Candidates</div>', unsafe_allow_html=True)

        jd_text = st.text_area(
            "Job Description",
            height=120,
            value=(
                "Looking for a senior Python engineer with Machine Learning experience, "
                "proficient in Docker, AWS, and REST APIs. Minimum 3 years experience."
            ),
            key="jd_input",
        )

        last_jd    = st.session_state.get("last_ranked_jd", "")
        jd_changed = (
            last_jd != ""
            and jd_text.strip() != last_jd.strip()
            and st.session_state.ranked
        )
        if jd_changed:
            st.warning("⚠️ Job description has changed — click **Screen & Rank** again to update results.")

        # Integration: use shared SkillExtractor for JD skill preview
        jd_skills_preview = extractor.extract(jd_text)
        if jd_skills_preview:
            badges_jd = "".join(
                f'<span class="badge badge-amber">{s}</span>' for s in jd_skills_preview
            )
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<span style="font-size:0.78rem;color:#9ba4c7;">Skills detected in JD: </span>'
                f'{badges_jd}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "⚠️ No standard skills detected in JD — add explicit skill names "
                "(e.g. 'Python', 'Docker') for better matching."
            )

        cands_available = st.session_state.pdf_candidates
        if not cands_available:
            st.warning("No candidates loaded. Upload PDFs in Tab 1 or load from dataset in Tab 2 first.")
        else:
            st.caption(f"{len(cands_available)} candidates ready for screening.")
            if st.button("🔍 Screen & Rank All Candidates", use_container_width=True):
                # FIX-1/2: Validate JD BEFORE ranking — stop immediately if no valid skills
                jd_skills_check = extractor.extract(jd_text.strip())
                if not jd_text.strip():
                    st.error(
                        "❌ Job Description is empty. "
                        "Please enter a job description before ranking."
                    )
                    st.stop()
                if not jd_skills_check:
                    st.error(
                        "❌ No valid skills detected in the Job Description. "
                        "Ranking requires at least one recognisable skill "
                        "(e.g. 'Python', 'Docker', 'Machine Learning'). "
                        "Please revise the JD and try again."
                    )
                    st.stop()

                with st.spinner("Ranking candidates…"):
                    try:
                        result = agent.screen_resumes(cands_available, jd_text)
                    except ValueError as e:
                        # FIX-2: rank_candidates raises ValueError for empty jd_skills
                        st.error(f"❌ Ranking aborted: {e}")
                        st.stop()

                    st.session_state.ranked         = result["ranked_candidates"]
                    st.session_state.last_ranked_jd = jd_text
                    try:
                        st.session_state.ranked_full = agent.ranker.rank_candidates(
                            cands_available, jd_text
                        )
                    except ValueError:
                        st.session_state.ranked_full = []

                    # ── Freeze total_screened: permanent watermark of resumes received ──
                    st.session_state.total_screened = len(result["ranked_candidates"])

                    # FIX-4: add_candidate in HRAgent.screen_resumes is idempotent —
                    # existing entries are NOT overwritten, preventing duplicates
                    # on repeated "Screen & Rank" clicks.

                # FIX-3: warn if all candidates have identical match scores
                scores = [c["match_score"] for c in st.session_state.ranked]
                if scores and len(set(scores)) == 1:
                    st.warning(
                        "⚠️ All candidates have identical scores. "
                        "This usually means the Job Description is too generic. "
                        "Try adding more specific skill requirements."
                    )

                st.success(
                    f"✅ Ranked {len(st.session_state.ranked)} candidates · "
                    "All added to pipeline at 'applied' stage."
                )

        if st.session_state.ranked:
            ranked_jd = st.session_state.get("last_ranked_jd", "")
            if ranked_jd:
                st.markdown(
                    f'<div style="background:#111428;border:1px solid #2a2f55;border-radius:8px;'
                    f'padding:8px 14px;margin-bottom:16px;font-size:0.8rem;color:#6b74a4;">'
                    f'📋 Rankings based on: '
                    f'<em>{ranked_jd[:120]}{"…" if len(ranked_jd) > 120 else ""}</em>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-hd">🏆 Ranked Results</div>', unsafe_allow_html=True)
            full_ranked = st.session_state.get("ranked_full", st.session_state.ranked)

            for cand in full_ranked:
                score = cand["match_score"]
                pct   = int(score * 100)
                if pct >= 75:
                    bar_color, badge_class, badge_label = "#06d6a0", "badge-green",  "Excellent"
                elif pct >= 55:
                    bar_color, badge_class, badge_label = "#4cc9f0", "badge-blue",   "Good"
                elif pct >= 35:
                    bar_color, badge_class, badge_label = "#f4a261", "badge-amber",  "Fair"
                else:
                    bar_color, badge_class, badge_label = "#f72585", "badge-red",    "Poor"

                # ── Skill display: matched skills first, then others ──────────
                # Backend skills list is preserved in full for scoring.
                # UI shows up to MAX_DISPLAY_SKILLS, always prioritising matched ones.
                MAX_DISPLAY_SKILLS = 10
                all_skills = cand.get("skills", [])
                jd_s       = set(s.lower() for s in cand.get("jd_skills", []))

                matched_skills = [s for s in all_skills if s.lower() in jd_s]
                other_skills   = [s for s in all_skills if s.lower() not in jd_s]

                # Fill display slots: matched first, then others up to cap
                display_skills = (
                    matched_skills
                    + other_skills[: max(0, MAX_DISPLAY_SKILLS - len(matched_skills))]
                )[:MAX_DISPLAY_SKILLS]

                skill_badges = "".join(
                    f'<span class="badge '
                    f'{"badge-green" if s.lower() in jd_s else "badge-purple"}">{s}</span>'
                    for s in display_skills
                )

                hidden_count = len(all_skills) - len(display_skills)
                overflow_note = (
                    f"&nbsp;·&nbsp;<span style='color:#7b84a8;'>"
                    f"+{hidden_count} more skill{'s' if hidden_count != 1 else ''} not shown</span>"
                    if hidden_count > 0 else ""
                )

                cov_pct = int(cand.get("jd_coverage", 0) * 100)
                exp_pct = int(cand.get("experience_score", 0) * 100)
                kw_pct  = int(cand.get("keyword_sim", cand.get("kw_sim", 0)) * 100)
                breakdown = (
                    f'<div style="display:flex;gap:12px;margin-top:6px;'
                    f'font-size:0.72rem;color:#7b84a8;">'
                    f'<span>🎯 JD Coverage: <strong style="color:#4cc9f0;">{cov_pct}%</strong></span>'
                    f'<span>📅 Experience: <strong style="color:#7209b7;">{exp_pct}%</strong></span>'
                    f'<span>🔑 Keywords: <strong style="color:#f4a261;">{kw_pct}%</strong></span>'
                    f'</div>'
                )

                st.markdown(
                    f"""
                    <div class="rank-card"
                         style="flex-direction:column;align-items:flex-start;gap:6px;">
                      <div style="display:flex;justify-content:space-between;
                                  width:100%;align-items:center;">
                        <div style="display:flex;align-items:center;gap:12px;">
                          <span class="rank-num">#{cand['rank']}</span>
                          <strong style="color:#e2e4ef;font-size:1rem;">{cand['name']}</strong>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:1.4rem;font-weight:700;
                                      color:{bar_color};">{pct}%</div>
                          <span class="badge {badge_class}"
                                style="font-size:0.68rem;">{badge_label}</span>
                        </div>
                      </div>
                      <div>{skill_badges}</div>
                      <div style="font-size:0.72rem;color:#6b74a4;margin-top:2px;">
                        ✅ Matched Skills: <strong style="color:#6ee7b7;">{len(matched_skills)}</strong>
                        / {len(jd_s)}{overflow_note}
                      </div>
                      <div style="width:100%;">
                        <div class="score-bar" style="margin:4px 0 0;">
                          <div class="score-fill"
                               style="width:{pct}%;background:{bar_color};"></div>
                        </div>
                      </div>
                      {breakdown}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Interview Scheduling
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Interview Scheduling":
    st.markdown('<div class="page-title">🗓 Interview Scheduling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Assign conflict-free interview slots · Pipeline auto-updates on schedule</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-hd">Available Slots</div>', unsafe_allow_html=True)
        avail = agent.scheduler.available_slots()
        if avail:
            for s in avail:
                st.markdown(f'<span class="badge badge-green">🕐 {s}</span>', unsafe_allow_html=True)
        else:
            st.warning("No slots remaining.")

    with col2:
        st.markdown('<div class="section-hd">Schedule Candidates</div>', unsafe_allow_html=True)

        if st.session_state.ranked:
            options  = [f"{c['name']} (#{c['rank']})" for c in st.session_state.ranked]
            sel      = st.multiselect(
                "Select candidates to schedule",
                options,
                default=options[:min(3, len(options))],
            )
            cand_ids = [str(st.session_state.ranked[options.index(s)]["id"]) for s in sel]
        else:
            ids_input = st.text_input("Candidate IDs (comma-separated)", "C001,C002,C003")
            cand_ids  = [x.strip() for x in ids_input.split(",") if x.strip()]

        if st.button("📅 Schedule Interviews", use_container_width=True):
            if not cand_ids:
                st.warning("Please select at least one candidate.")
            else:
                with st.spinner("Scheduling interviews and syncing pipeline…"):
                    result = agent.schedule_interviews(cand_ids)
                    st.session_state.scheduled = result["interviews_scheduled"]
                    sync_pipeline_after_scheduling(st.session_state.scheduled)
                    auto_update_interviewed()   # promote any already-past slots immediately

                n_scheduled = len(st.session_state.scheduled)
                n_conflicts  = len(result["conflicts"])
                n_synced     = len(st.session_state.ranked)

                if n_conflicts:
                    st.warning(
                        f"⚠️ {n_conflicts} candidate(s) could not be scheduled — no available slots."
                    )
                st.success(
                    f"✅ {n_scheduled} interview(s) scheduled · "
                    f"Pipeline auto-synced for {n_synced} ranked candidate(s)."
                )

    if st.session_state.scheduled:
        st.markdown('<div class="section-hd">📋 Scheduled Interviews</div>', unsafe_allow_html=True)
        df_sched = pd.DataFrame(st.session_state.scheduled)
        if st.session_state.ranked:
            id_to_name = {str(c["id"]): c["name"] for c in st.session_state.ranked}
            df_sched.insert(
                1, "Candidate Name",
                df_sched["candidate_id"].map(lambda cid: id_to_name.get(cid, "—"))
            )
        st.dataframe(df_sched, use_container_width=True, hide_index=True)

    if st.session_state.ranked and st.session_state.scheduled:
        st.markdown('<div class="section-hd">🔄 Pipeline State Snapshot</div>', unsafe_allow_html=True)
        STATUS_COLORS = {
            "applied":             "#4cc9f0",
            "shortlisted":         "#4361ee",
            "interview_scheduled": "#f4a261",
            "interviewed":         "#7209b7",
            "selected":            "#06d6a0",
            "rejected":            "#f72585",
        }
        statuses = agent.pipeline.all_statuses()
        cols     = st.columns(3)
        for idx, cand in enumerate(st.session_state.ranked):
            cid    = str(cand["id"])
            state  = statuses.get(cid, "unknown")
            color  = STATUS_COLORS.get(state, "#888")
            with cols[idx % 3]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;padding:8px 12px;'
                    f'background:#141729;border-radius:8px;margin-bottom:6px;">'
                    f'<span style="width:9px;height:9px;border-radius:50%;'
                    f'background:{color};display:inline-block;margin-right:10px;flex-shrink:0;"></span>'
                    f'<span style="flex:1;color:#c5caf0;font-size:0.82rem;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{cand["name"]}</span>'
                    f'<span class="badge" style="background:{color}22;color:{color};'
                    f'font-size:0.68rem;margin-left:6px;">{state}</span></div>',
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Question Generator
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Question Generator":
    st.markdown('<div class="page-title">❓ Interview Question Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Skill-specific technical, scenario, and behavioral questions</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_input = st.text_area(
            "Resume Text (or paste extracted PDF text)",
            height=160,
            value=(
                "Expert in Python, Machine Learning, TensorFlow, Docker, AWS, "
                "REST APIs, and SQL with 5 years of experience."
            ),
        )
        # Integration: use shared SkillExtractor instead of a standalone ResumeRanker
        skills_from_resume = extractor.extract(resume_input)
        if skills_from_resume:
            badges = "".join(
                f'<span class="badge badge-purple">{s}</span>' for s in skills_from_resume
            )
            st.markdown(f"**Extracted Skills:**<br>{badges}", unsafe_allow_html=True)

    with col2:
        manual_skills = st.text_input(
            "Or enter skills manually (comma-separated)",
            ", ".join(skills_from_resume[:5]),
        )
        final_skills = [s.strip() for s in manual_skills.split(",") if s.strip()]
        n_per_skill  = st.slider(
            "Questions per skill",
            min_value=1, max_value=5, value=3,
            help="1 → technical only  |  2 → + behavioral  |  3 → + scenario  |  4-5 → more of each",
        )

        experience_years = st.number_input(
            "Candidate experience (years)",
            min_value=0.0, max_value=40.0, value=0.0, step=0.5,
        )
        candidate_name = st.text_input("Candidate name (optional)", value="Candidate")

    if st.button("⚡ Generate Questions", use_container_width=True):
        if not final_skills:
            st.warning("Please provide skills.")
        else:
            # Integration: use agent.generate_questions() with experience-aware args
            jd_context = st.session_state.get("last_ranked_jd", "")
            qs = agent.generate_questions(
                final_skills,
                n_per_skill=n_per_skill,
                experience_years=experience_years,
                jd_text=jd_context,
                name=candidate_name,
            )
            st.session_state.questions = qs
            st.success(f"✅ {len(qs)} questions generated for {len(final_skills)} skills")

    if st.session_state.questions:
        type_filter = st.multiselect(
            "Filter by type",
            ["conceptual", "coding", "scenario", "system_design", "behavioral",
             "technical"],  # support both old and new type labels
            default=["conceptual", "coding", "scenario", "system_design", "behavioral"],
        )
        filtered    = [q for q in st.session_state.questions if q["type"] in type_filter]
        type_colors = {
            "conceptual":   "#4361ee",
            "coding":       "#4cc9f0",
            "scenario":     "#7209b7",
            "system_design":"#f4a261",
            "behavioral":   "#06d6a0",
            "technical":    "#4361ee",  # legacy label fallback
        }
        for i, q in enumerate(filtered, 1):
            color = type_colors.get(q["type"], "#4cc9f0")
            difficulty = q.get("difficulty", "")
            diff_badge = (
                f'<span class="badge badge-amber" style="font-size:0.62rem;">'
                f'{difficulty}</span> '
                if difficulty else ""
            )
            follow_up = q.get("follow_up", "")
            follow_up_html = (
                f'<div style="margin-top:6px;font-size:0.78rem;color:#7b84a8;">'
                f'↳ <em>{follow_up}</em></div>'
                if follow_up else ""
            )
            st.markdown(
                f"""
                <div class="q-card" style="border-left-color:{color};">
                  <div style="display:flex;gap:8px;margin-bottom:4px;">
                    <span style="font-size:0.68rem;font-weight:700;color:{color};
                                 letter-spacing:1px;">{q['type'].upper()}</span>
                    <span class="badge badge-blue" style="font-size:0.65rem;">{q['skill']}</span>
                    {diff_badge}
                  </div>
                  <div class="q-text"><strong>Q{i}.</strong> {q['question']}</div>
                  {follow_up_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Pipeline Manager
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Manager":
    st.markdown('<div class="page-title">🔄 HR Pipeline Manager</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Manage candidate state transitions — manual overrides always available</div>',
        unsafe_allow_html=True,
    )

    # ── Auto-promote past-slot candidates to "interviewed" on every page load ─
    auto_update_interviewed()

    STATE_COLORS = {
        "applied":             "#4cc9f0",
        "shortlisted":         "#4361ee",
        "interview_scheduled": "#f4a261",
        "interviewed":         "#7209b7",
        "selected":            "#06d6a0",
        "rejected":            "#f72585",
    }

    col1, col2 = st.columns(2)

    with col1:

        # ── Add candidate ─────────────────────────────────────────────────────
        st.markdown('<div class="section-hd">Add Candidate</div>', unsafe_allow_html=True)
        new_id = st.text_input("Candidate ID", "CAND_001")
        if st.button("➕ Add to Pipeline"):
            result = agent.pipeline.add_candidate(new_id)
            if result["state"] == "applied":
                st.success(f"Added **{new_id}** → applied")
            else:
                st.info(f"**{new_id}** already in pipeline (state: {result['state']})")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Mark Interview Completed (manual override) ────────────────────────
        scheduled_cids = [
            cid for cid, state in agent.pipeline.all_statuses().items()
            if state == "interview_scheduled"
        ]
        if scheduled_cids:
            st.markdown(
                '<div class="section-hd" style="font-size:0.9rem;">✅ Mark Interview Completed</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Select candidates whose interview is done — "
                "moves them from interview_scheduled → interviewed."
            )
            id_to_name   = (
                {str(c["id"]): c["name"] for c in st.session_state.ranked}
                if st.session_state.ranked else {}
            )
            display_opts = [
                f"{id_to_name.get(cid, cid)}  ({cid})" for cid in scheduled_cids
            ]
            selected_labels = st.multiselect(
                "Candidates to mark as interviewed",
                display_opts,
                key="mark_interviewed_select",
            )
            if st.button("🎯 Mark as Interviewed", key="btn_mark_interviewed", use_container_width=True):
                if not selected_labels:
                    st.warning("Select at least one candidate.")
                else:
                    moved, failed = [], []
                    for label in selected_labels:
                        cid = label.rsplit("(", 1)[-1].rstrip(")").strip()
                        res = agent.pipeline.transition(cid, "interviewed")
                        if res["success"]:
                            moved.append(id_to_name.get(cid, cid))
                        else:
                            failed.append(f"{cid}: {res.get('error', 'unknown error')}")
                    if moved:
                        st.success(f"✅ Moved to **interviewed**: {', '.join(moved)}")
                    if failed:
                        st.error("Some transitions failed:\n" + "\n".join(failed))
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Manual state transition ───────────────────────────────────────────
        st.markdown('<div class="section-hd">Manual State Transition</div>', unsafe_allow_html=True)
        st.caption("💡 Tip: Rejected candidates can be moved back to **shortlisted** for re-consideration.")

        all_cands = list(agent.pipeline.all_statuses().keys())
        if all_cands:
            sel_cand   = st.selectbox("Select Candidate", all_cands)
            curr       = agent.pipeline.get_status(sel_cand)
            allowed    = VALID_TRANSITIONS.get(curr, [])
            curr_color = STATE_COLORS.get(curr, "#888")
            st.markdown(
                f'Current state: <span class="badge" '
                f'style="background:{curr_color}22;color:{curr_color};">{curr}</span>',
                unsafe_allow_html=True,
            )
            if allowed:
                new_st = st.selectbox("Transition to", allowed)
                if st.button("⚡ Apply Transition", use_container_width=True):
                    res = agent.pipeline.transition(sel_cand, new_st)
                    if res["success"]:
                        st.success(f"✅ **{sel_cand}**: {res['previous']} → {res['current_state']}")
                        st.rerun()
                    else:
                        st.error(res["error"])
            else:
                st.info(f"**{sel_cand}** is in a terminal state ({curr}). No further transitions available.")
        else:
            st.info("No candidates in pipeline yet — run Resume Screening or add one above.")

    with col2:

        # ── Pipeline status board ─────────────────────────────────────────────
        st.markdown('<div class="section-hd">Pipeline Status Board</div>', unsafe_allow_html=True)
        statuses = agent.pipeline.all_statuses()

        if statuses:
            from collections import defaultdict
            by_state: dict = defaultdict(list)
            for cid, state in statuses.items():
                by_state[state].append(cid)

            stage_order = [
                "applied", "shortlisted", "interview_scheduled",
                "interviewed", "selected", "rejected",
            ]
            id_to_name = (
                {str(c["id"]): c["name"] for c in st.session_state.ranked}
                if st.session_state.ranked else {}
            )
            for stage in stage_order:
                members = by_state.get(stage, [])
                if not members:
                    continue
                color = STATE_COLORS.get(stage, "#888")
                st.markdown(
                    f'<div style="margin-bottom:4px;">'
                    f'<span class="badge" style="background:{color}22;color:{color};">'
                    f'{stage} ({len(members)})</span></div>',
                    unsafe_allow_html=True,
                )
                for cid in members:
                    display = id_to_name.get(cid, cid)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;padding:6px 12px;'
                        f'background:#141729;border-radius:8px;margin-bottom:4px;margin-left:12px;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;'
                        f'background:{color};display:inline-block;margin-right:10px;flex-shrink:0;"></span>'
                        f'<span style="flex:1;color:#c5caf0;font-size:0.82rem;">{display}</span>'
                        f'<span style="font-size:0.7rem;color:#4a5280;">{cid}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No candidates in pipeline.")

        # ── Stage Summary — accurate counts with frozen 'applied' ─────────────
        if statuses:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-hd">Stage Summary</div>', unsafe_allow_html=True)
            render_stage_summary(STATE_COLORS)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Export Results
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Export Results":
    st.markdown('<div class="page-title">📤 Export Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Download full candidate pipeline results as Excel (.xlsx)</div>',
        unsafe_allow_html=True,
    )

    pipeline_statuses = agent.pipeline.all_statuses()

    if not st.session_state.ranked:
        st.info(
            "No ranked candidates yet — run **Resume Screening** first, "
            "then return here to export."
        )
    else:
        source = st.session_state.get("ranked_full") or st.session_state.ranked

        rows = []
        for cand in source:
            cid    = str(cand["id"])
            skills = cand.get("skills", [])
            rows.append({
                "Rank":                 cand.get("rank", "—"),
                "Candidate ID":         cid,
                "Name":                 cand.get("name", "—"),
                "Skills":               ", ".join(skills) if skills else "—",
                "Match Score (%)":      round(cand.get("match_score", 0) * 100, 1),
                "JD Coverage (%)":      round(cand.get("jd_coverage",  0) * 100, 1),
                "Experience Score (%)": round(cand.get("experience_score", 0) * 100, 1),
                "Keyword Sim (%)":      round(cand.get("keyword_sim",   0) * 100, 1),
                "Pipeline Status":      pipeline_statuses.get(cid, "not_in_pipeline"),
            })

        df_export = pd.DataFrame(rows).sort_values("Rank").reset_index(drop=True)

        STATUS_EMOJI = {
            "applied":             "🔵 applied",
            "shortlisted":         "🟣 shortlisted",
            "interview_scheduled": "🟡 interview_scheduled",
            "interviewed":         "🟠 interviewed",
            "selected":            "🟢 selected",
            "rejected":            "🔴 rejected",
            "not_in_pipeline":     "⚪ not in pipeline",
        }
        df_display = df_export.copy()
        df_display["Pipeline Status"] = df_display["Pipeline Status"].map(
            lambda s: STATUS_EMOJI.get(s, s)
        )

        st.markdown('<div class="section-hd">📋 Candidate Pipeline Summary</div>', unsafe_allow_html=True)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-hd">📊 Status Breakdown</div>', unsafe_allow_html=True)
        STATUS_COLORS = {
            "selected":            "#06d6a0",
            "interview_scheduled": "#f4a261",
            "shortlisted":         "#4361ee",
            "applied":             "#4cc9f0",
            "rejected":            "#f72585",
            "interviewed":         "#7209b7",
            "not_in_pipeline":     "#888888",
        }
        status_counts = df_export["Pipeline Status"].value_counts()
        metric_cols   = st.columns(min(len(status_counts), 6))
        for col, (status, count) in zip(metric_cols, status_counts.items()):
            color = STATUS_COLORS.get(status, "#888")
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val" style="color:{color};">{count}</div>'
                    f'<div class="metric-lbl">{status}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hd">⬇️ Download Excel Report</div>', unsafe_allow_html=True)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_export.to_excel(writer, sheet_name="Candidate Pipeline", index=False)

            if st.session_state.scheduled:
                df_sched = pd.DataFrame(st.session_state.scheduled)
                if st.session_state.ranked:
                    id_to_name = {str(c["id"]): c["name"] for c in st.session_state.ranked}
                    df_sched.insert(
                        1, "Candidate Name",
                        df_sched["candidate_id"].map(lambda cid: id_to_name.get(cid, "—"))
                    )
                df_sched.to_excel(writer, sheet_name="Scheduled Interviews", index=False)

            if st.session_state.train_results:
                summarise_results(st.session_state.train_results).to_excel(
                    writer, sheet_name="ML Model Results", index=False
                )

            if st.session_state.questions:
                pd.DataFrame(st.session_state.questions).to_excel(
                    writer, sheet_name="Interview Questions", index=False
                )

        excel_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Full Report (.xlsx)",
            data=excel_buffer,
            file_name="hr_agent_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        sheets_included = ["✅ Candidate Pipeline"]
        sheets_included.append("✅ Scheduled Interviews" if st.session_state.scheduled
                                else "⬜ Scheduled Interviews (no data)")
        sheets_included.append("✅ ML Model Results" if st.session_state.train_results
                                else "⬜ ML Model Results (not trained)")
        sheets_included.append("✅ Interview Questions" if st.session_state.questions
                                else "⬜ Interview Questions (not generated)")

        st.caption("**Sheets in workbook:**  " + "  ·  ".join(sheets_included))
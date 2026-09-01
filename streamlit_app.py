"""SentixAI Analytics: Explainable Deep Learning Dashboard for Sentiment & Emotion Analysis.

Powered by BiLSTM with Additive Attention, LIME, and SHAP.
Ready for deployment on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sentixai_inference import SavedTextClassifier
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation

# Root paths resolution
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SentixAI · Explainable AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive modern styling
st.markdown(
    """
    <style>
    /* Metric & Card Styling */
    .sentix-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .sentix-hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sentix-hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 9999px;
        margin-right: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .status-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .pred-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .pred-positive {
        border-left: 5px solid #10b981;
    }
    .pred-negative {
        border-left: 5px solid #ef4444;
    }
    .pred-emotion {
        border-left: 5px solid #8b5cf6;
    }
    .attention-box {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        line-height: 2.2;
        font-size: 1.15rem;
        margin: 12px 0 20px 0;
    }
    .attn-chip {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 4px;
        border-radius: 6px;
        font-weight: 500;
        transition: transform 0.15s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .attn-chip:hover {
        transform: scale(1.08);
    }
    .info-note {
        font-size: 0.85rem;
        color: #94a3b8;
        font-style: italic;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading PyTorch BiLSTM Sentiment Model...")
def get_sentiment_model() -> SavedTextClassifier:
    """Load sentiment BiLSTM attention model with caching."""
    artifact_dir = ROOT / "outputs/artifacts"
    if not (artifact_dir / "bilstm_attention.pt").exists():
        artifact_dir = ROOT
    return SavedTextClassifier(artifact_dir, "bilstm_attention.pt")


@st.cache_resource(show_spinner="Loading PyTorch BiLSTM Emotion Model...")
def get_emotion_model() -> SavedTextClassifier:
    """Load emotion BiLSTM attention model with caching."""
    artifact_dir = ROOT / "outputs/emotion/artifacts"
    if not (artifact_dir / "emotion_bilstm_attention.pt").exists():
        artifact_dir = ROOT
    return SavedTextClassifier(artifact_dir, "emotion_bilstm_attention.pt")


@st.cache_data
def load_evaluation_data() -> dict[str, Any]:
    """Load test evaluation reports and training history."""
    data = {}
    sentiment_path = ROOT / "outputs/results/evaluation.json"
    if not sentiment_path.exists():
        sentiment_path = ROOT / "evaluation.json"
    if sentiment_path.exists():
        try:
            data["sentiment"] = json.loads(sentiment_path.read_text(encoding="utf-8"))
        except Exception:
            data["sentiment"] = None

    emotion_path = ROOT / "outputs/emotion/results/evaluation.json"
    if emotion_path.exists():
        try:
            data["emotion"] = json.loads(emotion_path.read_text(encoding="utf-8"))
        except Exception:
            data["emotion"] = None
    return data


# Header Hero Section
st.markdown(
    """
    <div class="sentix-hero">
        <div>
            <span class="badge-pill">Deep Learning</span>
            <span class="badge-pill">BiLSTM + Attention</span>
            <span class="badge-pill">Explainable AI</span>
            <span class="badge-pill">PyTorch 2.x</span>
        </div>
        <h1>SentixAI Analytics</h1>
        <p>Explainable Deep Learning Dashboard for Sentiment & Emotion Analysis with Additive Attention, LIME, and SHAP interpretability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("SentixAI Controls")
    st.caption("v4.0 · B.Tech CSE-AIML Project")
    st.divider()

    st.subheader("Model Status")
    try:
        sent_clf = get_sentiment_model()
        st.success("Sentiment Model: Ready (IMDB · 90.02% Acc)")
    except Exception as exc:
        st.error(f"Sentiment Model: Error ({exc})")
        sent_clf = None

    try:
        emo_clf = get_emotion_model()
        st.success("Emotion Model: Ready (GoEmotions · 73.44% Acc)")
    except Exception as exc:
        st.error(f"Emotion Model: Error ({exc})")
        emo_clf = None

    st.divider()
    st.subheader("XAI Explainability Settings")
    include_emotion = st.checkbox("Include Emotion Model", value=True)
    show_attention = st.checkbox("Additive Attention Heatmap", value=True)
    show_lime = st.checkbox("LIME Local Perturbations", value=True)
    show_shap = st.checkbox("SHAP Shapley Values (Slower)", value=False)

    lime_samples = st.slider(
        "LIME Perturbation Samples",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Higher samples yield more stable local word importance but take slightly longer.",
    )

    if show_shap:
        shap_samples = st.slider(
            "Kernel SHAP Samples",
            min_value=40,
            max_value=200,
            value=80,
            step=20,
            help="Kernel SHAP evaluates combinations against masked baseline; bounded for interactive responsiveness.",
        )
    else:
        shap_samples = 80

    st.divider()
    st.markdown(
        """
        **Sharable Link Note**:
        This dashboard runs seamlessly on **Streamlit Community Cloud** with 1-click GitHub deployment, persistent URLs, and zero cold-start sleeps.
        """
    )


# Main Tabs Navigation
tab_predict, tab_batch, tab_analytics, tab_architecture, tab_deploy = st.tabs([
    "🔍 Real-time Analysis & XAI",
    "📁 Batch CSV Processing",
    "📊 Model Analytics & Curves",
    "🧠 Architecture & Theory",
    "🚀 Sharable Link & Deployment",
])


# ==========================================
# TAB 1: SINGLE TEXT INFERENCE & XAI
# ==========================================
with tab_predict:
    st.subheader("Interactive Sentiment & Emotion Assessment")
    st.markdown("Analyze any sentence or review to inspect predictions and model evidence simultaneously.")

    # Sample Presets
    sample_presets = {
        "Custom Input": "",
        "Mixed Contrast": "The camera is excellent but the battery is terrible.",
        "High Joy & Praise": "I am absolutely thrilled and amazed by this wonderful experience! Highly recommended.",
        "Severe Disappointment": "The flight was delayed for 8 hours and customer support was completely rude and dismissive.",
        "Fear & Anxiety": "I was completely terrified when the car engine started making alarming loud noises on the highway.",
        "Surprise": "Wait, did they really announce the brand new release date today?!",
    }

    selected_sample = st.selectbox(
        "Choose an example preset or write your own text:",
        options=list(sample_presets.keys()),
        index=1,
    )

    default_text = sample_presets[selected_sample] if selected_sample != "Custom Input" else "The camera is excellent but the battery is terrible."
    user_text = st.text_area(
        "Enter text to analyze:",
        value=default_text,
        height=100,
        placeholder="Type or paste a product review, feedback comment, or social post...",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

    if analyze_clicked or user_text:
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        elif not sent_clf:
            st.error("Sentiment model could not be loaded. Please check checkpoint files.")
        else:
            with st.spinner("Running BiLSTM inference and calculating explainability..."):
                try:
                    # Run Sentiment Prediction
                    sent_pred = sent_clf.predict(user_text)

                    # Run Emotion Prediction if enabled
                    emo_pred = emo_clf.predict(user_text) if (include_emotion and emo_clf) else None

                    # Predictions Display Columns
                    col_sent, col_emo = st.columns(2)

                    with col_sent:
                        sent_label = sent_pred["label"].upper()
                        sent_conf = sent_pred["confidence"]
                        card_class = "pred-positive" if sent_label == "POSITIVE" else "pred-negative"
                        badge_color = "#10b981" if sent_label == "POSITIVE" else "#ef4444"

                        st.markdown(
                            f"""
                            <div class="pred-card {card_class}">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="color:#94a3b8; font-weight:700; font-size:0.85rem; letter-spacing:0.05em;">SENTIMENT PREDICTION</span>
                                    <span style="background:{badge_color}; color:white; padding:3px 12px; border-radius:9999px; font-weight:700; font-size:0.85rem;">{sent_label}</span>
                                </div>
                                <h2 style="margin:10px 0 4px 0; font-size:2rem; font-weight:800;">{sent_label}</h2>
                                <p style="margin:0; color:#cbd5e1;">Model Confidence: <strong style="color:white;">{sent_conf*100:.2f}%</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Probability Distribution
                        st.markdown("##### Sentiment Class Probabilities")
                        sent_probs = sent_pred["probabilities"]
                        probs_df = pd.DataFrame({
                            "Class": [c.capitalize() for c in sent_probs.keys()],
                            "Probability": list(sent_probs.values()),
                            "Percentage": [f"{v*100:.2f}%" for v in sent_probs.values()],
                        })
                        chart_sent = (
                            alt.Chart(probs_df)
                            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                            .encode(
                                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                                y=alt.Y("Class:N", sort="-x", title=""),
                                color=alt.Color(
                                    "Class:N",
                                    scale=alt.Scale(
                                        domain=["Positive", "Negative"],
                                        range=["#10b981", "#ef4444"],
                                    ),
                                    legend=None,
                                ),
                                tooltip=["Class", "Percentage"],
                            )
                            .properties(height=110)
                        )
                        st.altair_chart(chart_sent, use_container_width=True)
                        st.markdown(
                            "<p class='info-note'>* Binary sentiment trained on IMDB (genuine classes: positive, negative). Neutral is intentionally marked not trained.</p>",
                            unsafe_allow_html=True,
                        )

                    if emo_pred:
                        with col_emo:
                            emo_label = emo_pred["label"].upper()
                            emo_conf = emo_pred["confidence"]
                            st.markdown(
                                f"""
                                <div class="pred-card pred-emotion">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <span style="color:#94a3b8; font-weight:700; font-size:0.85rem; letter-spacing:0.05em;">EMOTION PREDICTION</span>
                                        <span style="background:#8b5cf6; color:white; padding:3px 12px; border-radius:9999px; font-weight:700; font-size:0.85rem;">{emo_label}</span>
                                    </div>
                                    <h2 style="margin:10px 0 4px 0; font-size:2rem; font-weight:800;">{emo_label}</h2>
                                    <p style="margin:0; color:#cbd5e1;">Model Confidence: <strong style="color:white;">{emo_conf*100:.2f}%</strong></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Emotion Probability Distribution
                            st.markdown("##### Emotion Class Probabilities (7 Classes)")
                            emo_probs = emo_pred["probabilities"]
                            emo_df = pd.DataFrame({
                                "Emotion": [c.capitalize() for c in emo_probs.keys()],
                                "Probability": list(emo_probs.values()),
                                "Percentage": [f"{v*100:.2f}%" for v in emo_probs.values()],
                            })
                            chart_emo = (
                                alt.Chart(emo_df)
                                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                                .encode(
                                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                                    y=alt.Y("Emotion:N", sort="-x", title=""),
                                    color=alt.Color(
                                        "Probability:Q",
                                        scale=alt.Scale(scheme="purples"),
                                        legend=None,
                                    ),
                                    tooltip=["Emotion", "Percentage"],
                                )
                                .properties(height=220)
                            )
                            st.altair_chart(chart_emo, use_container_width=True)
                            st.markdown(
                                "<p class='info-note'>* Filtered GoEmotions labels: anger, disappointment, fear, joy, love, sadness, surprise.</p>",
                                unsafe_allow_html=True,
                            )

                    # -------------------------------------------------------------
                    # EXPLAINABLE AI (XAI) SECTION
                    # -------------------------------------------------------------
                    st.divider()
                    st.subheader("💡 Explainable AI (XAI) Interpretability")
                    st.markdown(
                        "Inspect how the BiLSTM attention weights, LIME surrogate linear model, and Kernel SHAP attribution highlight word importance."
                    )

                    # 1. BiLSTM Additive Attention
                    if show_attention:
                        st.markdown("#### 1. BiLSTM Additive Attention Mechanism")
                        attn_tokens = sent_pred.get("attention_tokens", [])
                        if attn_tokens:
                            max_weight = max((item["weight"] for item in attn_tokens), default=0.001)

                            # HTML chips rendering with intensity
                            chips_html = []
                            for item in attn_tokens:
                                w = item["weight"]
                                norm_w = w / (max_weight + 1e-6)
                                # Alpha background intensity
                                bg_alpha = max(0.12, min(0.95, norm_w * 0.95))
                                text_color = "#ffffff" if norm_w > 0.4 else "#cbd5e1"
                                chip = (
                                    f'<span class="attn-chip" style="background: rgba(99, 102, 241, {bg_alpha:.2f}); '
                                    f'color: {text_color};" title="Weight: {w:.4f}">'
                                    f'{item["token"]} <small style="opacity:0.75; font-size:0.75em;">({w:.3f})</small></span>'
                                )
                                chips_html.append(chip)

                            st.markdown(
                                f"""
                                <div class="attention-box">
                                    {' '.join(chips_html)}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                "Hover or inspect token chips above. Background intensity indicates the normalized attention weight (softmax scores over BiLSTM hidden states)."
                            )

                            # Top attention ranked chart
                            col_attn_chart, col_attn_table = st.columns([3, 2])
                            with col_attn_chart:
                                top_attn_df = pd.DataFrame(attn_tokens).sort_values("weight", ascending=False).head(10)
                                attn_bar = (
                                    alt.Chart(top_attn_df)
                                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#6366f1")
                                    .encode(
                                        x=alt.X("weight:Q", title="Attention Weight (Softmax)"),
                                        y=alt.Y("token:N", sort="-x", title="Token"),
                                        tooltip=["token", "weight"],
                                    )
                                    .properties(height=260, title="Top 10 Focused Tokens (Attention Layer)")
                                )
                                st.altair_chart(attn_bar, use_container_width=True)

                            with col_attn_table:
                                st.markdown("###### Attention Tokens Ranking")
                                st.dataframe(
                                    top_attn_df.rename(columns={"token": "Word", "weight": "Attention Score"}),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    # 2. LIME Explanations
                    if show_lime:
                        st.markdown("#### 2. LIME (Local Interpretable Model-agnostic Explanations)")
                        with st.spinner("Computing LIME word perturbation contributions..."):
                            lime_res = lime_explanation(sent_clf, user_text, num_features=10, num_samples=lime_samples)
                            lime_words = lime_res.get("important_words", [])

                            if lime_words:
                                lime_df = pd.DataFrame(lime_words)
                                lime_df["direction"] = np.where(lime_df["contribution"] >= 0, "Supports Class", "Opposes Class")
                                lime_df["color"] = np.where(lime_df["contribution"] >= 0, "#10b981", "#ef4444")
                                lime_df["abs_contrib"] = lime_df["contribution"].abs()

                                col_l1, col_l2 = st.columns([3, 2])
                                with col_l1:
                                    lime_chart = (
                                        alt.Chart(lime_df)
                                        .mark_bar(cornerRadius=4)
                                        .encode(
                                            x=alt.X("contribution:Q", title="Local Feature Weight (LIME)"),
                                            y=alt.Y("token:N", sort=alt.EncodingSortField(field="abs_contrib", order="descending"), title="Token"),
                                            color=alt.Color(
                                                "direction:N",
                                                scale=alt.Scale(
                                                    domain=["Supports Class", "Opposes Class"],
                                                    range=["#10b981", "#ef4444"],
                                                ),
                                                legend=alt.Legend(title="Local Impact"),
                                            ),
                                            tooltip=["token", "contribution", "direction"],
                                        )
                                        .properties(height=280, title=f"LIME Word Contributions toward '{sent_pred['label'].upper()}'")
                                    )
                                    st.altair_chart(lime_chart, use_container_width=True)

                                with col_l2:
                                    st.markdown(f"###### LIME Feature Scores ({lime_samples} samples)")
                                    st.dataframe(
                                        lime_df[["token", "contribution", "direction"]].rename(
                                            columns={"token": "Token", "contribution": "Weight", "direction": "Effect"}
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                st.caption(lime_res.get("note", ""))

                    # 3. SHAP Explanations
                    if show_shap:
                        st.markdown("#### 3. Kernel SHAP (SHapley Additive exPlanations)")
                        with st.spinner("Computing Kernel SHAP values against word-masked baseline..."):
                            try:
                                shap_res = shap_explanation(sent_clf, user_text, max_tokens=16, nsamples=shap_samples)
                                shap_words = shap_res.get("important_words", [])

                                if shap_words:
                                    shap_df = pd.DataFrame(shap_words).head(10)
                                    shap_df["direction"] = np.where(shap_df["contribution"] >= 0, "Positive Impact", "Negative Impact")
                                    shap_df["abs_contrib"] = shap_df["contribution"].abs()

                                    col_s1, col_s2 = st.columns([3, 2])
                                    with col_s1:
                                        shap_chart = (
                                            alt.Chart(shap_df)
                                            .mark_bar(cornerRadius=4)
                                            .encode(
                                                x=alt.X("contribution:Q", title="Shapley Value (SHAP)"),
                                                y=alt.Y("token:N", sort=alt.EncodingSortField(field="abs_contrib", order="descending"), title="Token"),
                                                color=alt.Color(
                                                    "direction:N",
                                                    scale=alt.Scale(
                                                        domain=["Positive Impact", "Negative Impact"],
                                                        range=["#38bdf8", "#f43f5e"],
                                                    ),
                                                    legend=alt.Legend(title="Impact"),
                                                ),
                                                tooltip=["token", "contribution", "direction"],
                                            )
                                            .properties(height=280, title=f"Kernel SHAP Impact on '{sent_pred['label'].upper()}'")
                                        )
                                        st.altair_chart(shap_chart, use_container_width=True)

                                    with col_s2:
                                        st.markdown(f"###### SHAP Attribution ({shap_samples} evaluations)")
                                        st.dataframe(
                                            shap_df[["token", "contribution", "direction"]].rename(
                                                columns={"token": "Token", "contribution": "Shapley Value", "direction": "Effect"}
                                            ),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    st.caption(shap_res.get("note", ""))
                            except Exception as shap_err:
                                st.warning(f"Could not compute SHAP explanation: {shap_err}")

                except Exception as err:
                    st.error(f"Prediction failed: {err}")


# ==========================================
# TAB 2: BATCH CSV PROCESSING
# ==========================================
with tab_batch:
    st.subheader("📁 Batch CSV Analysis")
    st.markdown(
        "Upload a `.csv` file containing a **`review`** column to classify multiple reviews with both Sentiment and Emotion models simultaneously."
    )

    sample_csv_path = ROOT / "sample_batch_reviews.csv"
    if not sample_csv_path.exists():
        sample_csv_path = ROOT / "data/sample_batch_reviews.csv"

    col_upload, col_sample = st.columns([3, 2])
    with col_sample:
        st.markdown("###### Need sample data?")
        if sample_csv_path.exists():
            sample_data = sample_csv_path.read_text(encoding="utf-8")
            st.download_button(
                label="📥 Download Sample Batch CSV",
                data=sample_data,
                file_name="sample_batch_reviews.csv",
                mime="text/csv",
                use_container_width=True,
            )
            load_sample_btn = st.button("📄 Load Sample CSV Automatically", use_container_width=True)
        else:
            load_sample_btn = False

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV File (Max 5,000 reviews):", type=["csv"])

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_sample_btn and sample_csv_path.exists():
        input_df = pd.read_csv(sample_csv_path)
        st.info("Loaded sample batch reviews dataset.")

    if input_df is not None:
        if "review" not in input_df.columns:
            st.error("The uploaded CSV must contain a column named **'review'**.")
        elif input_df.empty:
            st.warning("The uploaded CSV has no rows.")
        else:
            st.write(f"Loaded **{len(input_df)}** reviews.")
            with st.expander("Preview Uploaded CSV", expanded=False):
                st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("⚡ Run Batch Classification", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                clean_reviews = input_df["review"].fillna("").astype(str).str.strip().tolist()
                valid_reviews = [r for r in clean_reviews if r]

                if not valid_reviews:
                    st.error("All review rows appear to be empty.")
                else:
                    status_text.text("Running Sentiment BiLSTM Batch Inference...")
                    progress_bar.progress(20)
                    sent_results = sent_clf.predict_many(valid_reviews, batch_size=128)

                    status_text.text("Running Emotion BiLSTM Batch Inference...")
                    progress_bar.progress(60)
                    emo_results = emo_clf.predict_many(valid_reviews, batch_size=128) if emo_clf else [{"label": "N/A", "confidence": 0.0}] * len(valid_reviews)

                    progress_bar.progress(90)
                    status_text.text("Compiling results and summary analytics...")

                    results_df = pd.DataFrame({
                        "review": valid_reviews,
                        "sentiment": [item["label"] for item in sent_results],
                        "sentiment_confidence": [round(item["confidence"], 4) for item in sent_results],
                        "emotion": [item["label"] for item in emo_results],
                        "emotion_confidence": [round(item["confidence"], 4) for item in emo_results],
                    })

                    progress_bar.progress(100)
                    status_text.success("Batch classification complete!")

                    # Summary Metrics
                    st.divider()
                    st.subheader("Batch Summary Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    tot = len(results_df)
                    pos_count = (results_df["sentiment"] == "positive").sum()
                    neg_count = (results_df["sentiment"] == "negative").sum()
                    top_emotion = results_df["emotion"].mode()[0] if not results_df["emotion"].empty else "N/A"

                    m1.metric("Total Reviews", tot)
                    m2.metric("Positive Sentiment", f"{pos_count} ({pos_count/tot*100:.1f}%)")
                    m3.metric("Negative Sentiment", f"{neg_count} ({neg_count/tot*100:.1f}%)")
                    m4.metric("Top Emotion", top_emotion.capitalize())

                    # Charts
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.markdown("###### Sentiment Distribution")
                        sent_counts = results_df["sentiment"].value_counts().reset_index()
                        sent_counts.columns = ["Sentiment", "Count"]
                        chart_s = (
                            alt.Chart(sent_counts)
                            .mark_bar(cornerRadius=6)
                            .encode(
                                x=alt.X("Count:Q", title="Reviews"),
                                y=alt.Y("Sentiment:N", sort="-x", title=""),
                                color=alt.Color(
                                    "Sentiment:N",
                                    scale=alt.Scale(domain=["positive", "negative"], range=["#10b981", "#ef4444"]),
                                    legend=None,
                                ),
                                tooltip=["Sentiment", "Count"],
                            )
                            .properties(height=180)
                        )
                        st.altair_chart(chart_s, use_container_width=True)

                    with col_c2:
                        st.markdown("###### Emotion Distribution")
                        emo_counts = results_df["emotion"].value_counts().reset_index()
                        emo_counts.columns = ["Emotion", "Count"]
                        chart_e = (
                            alt.Chart(emo_counts)
                            .mark_bar(cornerRadius=6)
                            .encode(
                                x=alt.X("Count:Q", title="Reviews"),
                                y=alt.Y("Emotion:N", sort="-x", title=""),
                                color=alt.Color("Count:Q", scale=alt.Scale(scheme="purples"), legend=None),
                                tooltip=["Emotion", "Count"],
                            )
                            .properties(height=220)
                        )
                        st.altair_chart(chart_e, use_container_width=True)

                    # Table preview and download
                    st.divider()
                    st.subheader("Processed Reviews Preview")
                    st.dataframe(results_df, use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Processed CSV Results",
                        data=csv_data,
                        file_name="sentixai_batch_results.csv",
                        mime="text/csv",
                        type="primary",
                    )


# ==========================================
# TAB 3: MODEL ANALYTICS & CURVES
# ==========================================
with tab_analytics:
    st.subheader("📊 Training & Evaluation Analytics")
    st.markdown("Metrics, curves, and confusion matrices saved from genuine held-out test evaluations.")

    eval_data = load_evaluation_data()
    sent_eval = eval_data.get("sentiment")
    emo_eval = eval_data.get("emotion")

    # High-level comparison KPIs
    st.markdown("##### Held-out Test Set Performance")
    kpi_cols = st.columns(4)
    if sent_eval:
        s_acc = sent_eval["test_metrics"]["accuracy"] * 100
        s_f1 = sent_eval["test_metrics"]["f1_weighted"] * 100
        kpi_cols[0].metric("Sentiment Accuracy", f"{s_acc:.2f}%", help="IMDB 4,959 held-out test reviews")
        kpi_cols[1].metric("Sentiment Weighted F1", f"{s_f1:.2f}%")
    if emo_eval:
        e_acc = emo_eval["test_metrics"]["accuracy"] * 100
        e_f1 = emo_eval["test_metrics"]["f1_weighted"] * 100
        kpi_cols[2].metric("Emotion Accuracy", f"{e_acc:.2f}%", help="GoEmotions 738 held-out test examples")
        kpi_cols[3].metric("Emotion Weighted F1", f"{e_f1:.2f}%")

    st.divider()

    # Training Curves
    st.subheader("Training vs Validation Curves")
    col_hist1, col_hist2 = st.columns(2)

    if sent_eval and "history" in sent_eval:
        s_hist = sent_eval["history"]
        epochs = list(range(1, len(s_hist["train_loss"]) + 1))
        sent_curve_df = pd.DataFrame({
            "Epoch": epochs * 2,
            "Accuracy": s_hist["train_accuracy"] + s_hist["validation_accuracy"],
            "Loss": s_hist["train_loss"] + s_hist["validation_loss"],
            "Split": ["Train"] * len(epochs) + ["Validation"] * len(epochs),
        })

        with col_hist1:
            st.markdown("###### Sentiment: Accuracy across Epochs")
            c_acc = (
                alt.Chart(sent_curve_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Epoch:O", title="Epoch"),
                    y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.7, 1.0]), title="Accuracy"),
                    color=alt.Color("Split:N", scale=alt.Scale(domain=["Train", "Validation"], range=["#3b82f6", "#f97316"])),
                    tooltip=["Epoch", "Split", "Accuracy"],
                )
                .properties(height=240)
            )
            st.altair_chart(c_acc, use_container_width=True)

        with col_hist2:
            st.markdown("###### Sentiment: Loss across Epochs")
            c_loss = (
                alt.Chart(sent_curve_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Epoch:O", title="Epoch"),
                    y=alt.Y("Loss:Q", title="Cross-Entropy Loss"),
                    color=alt.Color("Split:N", scale=alt.Scale(domain=["Train", "Validation"], range=["#3b82f6", "#f97316"])),
                    tooltip=["Epoch", "Split", "Loss"],
                )
                .properties(height=240)
            )
            st.altair_chart(c_loss, use_container_width=True)

    # Confusion Matrices
    st.divider()
    st.subheader("Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)

    sent_cm_path = ROOT / "outputs/results/confusion_matrix.png"
    if not sent_cm_path.exists():
        sent_cm_path = ROOT / "confusion_matrix.png"

    emo_cm_path = ROOT / "outputs/emotion/results/confusion_matrix.png"

    with col_cm1:
        st.markdown("###### Sentiment Confusion Matrix (IMDB)")
        if sent_cm_path.exists():
            st.image(str(sent_cm_path), caption="Sentiment BiLSTM + Attention Confusion Matrix", use_container_width=True)
        else:
            st.info("Sentiment confusion matrix image not found.")

    with col_cm2:
        st.markdown("###### Emotion Confusion Matrix (GoEmotions)")
        if emo_cm_path.exists():
            st.image(str(emo_cm_path), caption="Emotion BiLSTM + Attention Confusion Matrix", use_container_width=True)
        else:
            st.info("Emotion confusion matrix image not found.")

    # Classification Reports
    st.divider()
    st.subheader("Detailed Classification Reports")
    col_cr1, col_cr2 = st.columns(2)

    with col_cr1:
        if sent_eval and "classification_report" in sent_eval:
            st.markdown("###### Sentiment Model (Test Split)")
            report = sent_eval["classification_report"]
            rows = []
            for cls in ["negative", "positive", "macro avg", "weighted avg"]:
                if cls in report:
                    d = report[cls]
                    rows.append({
                        "Class": cls.capitalize(),
                        "Precision": f"{d['precision']*100:.2f}%",
                        "Recall": f"{d['recall']*100:.2f}%",
                        "F1-Score": f"{d['f1-score']*100:.2f}%",
                        "Support": int(d.get("support", 0)),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_cr2:
        if emo_eval and "classification_report" in emo_eval:
            st.markdown("###### Emotion Model (Test Split)")
            report = emo_eval["classification_report"]
            rows = []
            for cls in emo_eval["class_names"] + ["macro avg", "weighted avg"]:
                if cls in report:
                    d = report[cls]
                    rows.append({
                        "Class": cls.capitalize(),
                        "Precision": f"{d['precision']*100:.2f}%",
                        "Recall": f"{d['recall']*100:.2f}%",
                        "F1-Score": f"{d['f1-score']*100:.2f}%",
                        "Support": int(d.get("support", 0)),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==========================================
# TAB 4: ARCHITECTURE & THEORY
# ==========================================
with tab_architecture:
    st.subheader("🧠 Model Architecture & Deep Learning Foundations")

    st.markdown(
        """
        ### Neural Network Pipeline
        SentixAI implements a custom **Bidirectional Long Short-Term Memory (BiLSTM)** network enhanced with an **Additive Attention Mechanism**.
        """
    )

    st.code(
        """
[Input Tokens]
      ↓
[Embedding Layer]        (Vocab Size → 128-dim vectors + Dropout 0.35)
      ↓
[Bidirectional LSTM]     (Forward LSTM + Backward LSTM → 256-dim concatenated states)
      ↓
[Additive Attention]     (Masked Bahdanau Attention → Normalized Token Weights α_t)
      ↓
[Context Pooling]        (Context Vector c = Σ α_t · h_t)
      ↓
[Dropout & Dense Layer]  (Linear Projection: 256 → Num Classes)
      ↓
[Softmax Normalization]  (Yields calibrated class probabilities during inference)
        """,
        language="text",
    )

    st.markdown(
        """
        ### Additive Attention Formulation
        Unlike vanilla sequence pooling (which takes only the final hidden state $h_T$ or simple mean pooling), the additive attention module learns to allocate importance dynamically:

        1. **Score Calculation**:
        $$e_t = v^T \\tanh(W h_t + b)$$
        Where $h_t \\in \\mathbb{R}^{2d}$ is the concatenated BiLSTM hidden state at position $t$, $W \\in \\mathbb{R}^{a \\times 2d}$ is the projection weight, and $v \\in \\mathbb{R}^a$ is the attention vector.

        2. **Padding Masking & Softmax Normalization**:
        $$\\alpha_t = \\frac{\\exp(e_t)}{\\sum_{k=1}^T \\exp(e_k)}$$
        Padding tokens ($\text{<PAD>}$) are masked with $-10^9$ prior to softmax so zero probability mass is allocated to padding.

        3. **Context Vector Generation**:
        $$c = \\sum_{t=1}^T \\alpha_t h_t$$

        ### Explainability Frameworks Comparison
        | Method | Category | How it Works | Pros / Scope |
        | :--- | :--- | :--- | :--- |
        | **Additive Attention** | Model-Intrinsic | Normalizes internal weights across BiLSTM hidden states $\\alpha_t$. | Zero inference overhead; exact internal representation. |
        | **LIME** | Model-Agnostic Surrogate | Samples word-removal perturbations and fits a local ridge regressor. | Identifies both positive (supporting) and negative (opposing) words. |
        | **Kernel SHAP** | Cooperative Game Theory | Computes Shapley values comparing token subsets against masked baselines. | Fair attribution with mathematical efficiency and consistency guarantees. |
        """
    )


# ==========================================
# TAB 5: SHARABLE LINK & DEPLOYMENT GUIDE
# ==========================================
with tab_deploy:
    st.subheader("🚀 How to Get a Permanent Sharable Link")
    st.markdown(
        """
        ### Why Streamlit Community Cloud is Superior for Deep Learning

        | Feature | **Streamlit Community Cloud** (Recommended) | **Render Free Tier** | **Vercel Serverless** |
        | :--- | :--- | :--- | :--- |
        | **Cost** | 100% Free | Free Tier | Free Tier |
        | **PyTorch Support** | ✅ Full Native PyTorch, Torch CPU, LIME, SHAP | ⚠️ Strict 512MB RAM (OOM crashes) | ❌ Exceeds 250MB Serverless bundle limit |
        | **Cold Starts** | ✅ Always available, zero 50-second spin-downs | ❌ 50+ second freeze after inactivity | ❌ Lambda cold start & 10s-60s timeout |
        | **Sharable Link** | `https://<your-username>-sentixai.streamlit.app` | `https://sentixai-xxxx.onrender.com` | `https://sentixai.vercel.app` (frontend only) |
        | **Setup Time** | 2 minutes (1 click from GitHub) | Requires Docker build & memory tuning | Requires splitting frontend & backend |

        ---

        ### Step-by-Step: Deploy to Streamlit Cloud in 3 Steps

        1. **Push your code to GitHub**:
        ```bash
        git add .
        git commit -m "Add production Streamlit dashboard for Sentiment & Emotion XAI"
        git push origin main
        ```

        2. **Connect to Streamlit Community Cloud**:
           - Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
           - Click **"Create app"** (or **"New app"**).
           - Select repository: **`govindshaw67/SentixAI`**
           - Select branch: **`main`**
           - Select main file path: **`streamlit_app.py`**

        3. **Launch & Share**:
           - Click **"Deploy!"**
           - Streamlit Cloud will install `requirements.txt` and launch your live app with a public, sharable link:
           - **`https://sentixai.streamlit.app`** (or your custom subdomain under App Settings)!
        """
    )

st.divider()
st.caption(
    "SentixAI Analytics · B.Tech CSE-AIML Project Demonstration · Explainable BiLSTM Attention Neural Intelligence"
)

"""SentixAI Platform: Explainable Deep Learning System for Sentiment & Emotion Analysis.

Architecture: Bidirectional LSTM with Masked Additive Attention, LIME, and Kernel SHAP.
Configured for production deployment on Streamlit Community Cloud.
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
from vae_inference import VAEAnomalyDetector

# Path definitions
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SentixAI | Explainable Deep Learning Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS
st.markdown(
    """
    <style>
    .sentix-hero {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .sentix-hero h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .sentix-hero p {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 8px;
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .pred-card {
        background: #111827;
        border-radius: 6px;
        padding: 16px 20px;
        border: 1px solid #1f2937;
        margin-bottom: 16px;
    }
    .pred-positive {
        border-left: 4px solid #10b981;
    }
    .pred-negative {
        border-left: 4px solid #ef4444;
    }
    .pred-emotion {
        border-left: 4px solid #6366f1;
    }
    .attention-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 16px 18px;
        line-height: 2.1;
        font-size: 1.05rem;
        margin: 10px 0 16px 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .attn-chip {
        display: inline-block;
        padding: 2px 7px;
        margin: 2px 3px;
        border-radius: 4px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .info-note {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Sentiment Classifier...")
def get_sentiment_model() -> SavedTextClassifier:
    """Load sentiment BiLSTM attention model with caching."""
    artifact_dir = ROOT / "outputs/artifacts"
    if not (artifact_dir / "bilstm_attention.pt").exists():
        artifact_dir = ROOT
    return SavedTextClassifier(artifact_dir, "bilstm_attention.pt")


@st.cache_resource(show_spinner="Loading Emotion Classifier...")
def get_emotion_model() -> SavedTextClassifier:
    """Load emotion BiLSTM attention model with caching."""
    artifact_dir = ROOT / "outputs/emotion/artifacts"
    if not (artifact_dir / "emotion_bilstm_attention.pt").exists():
        artifact_dir = ROOT
    return SavedTextClassifier(artifact_dir, "emotion_bilstm_attention.pt")


@st.cache_data
def load_evaluation_data() -> dict[str, Any]:
    """Load held-out evaluation reports and training history."""
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


# Header Section
st.markdown(
    """
    <div class="sentix-hero">
        <div>
            <span class="badge-pill">Deep Learning</span>
            <span class="badge-pill">BiLSTM + Attention</span>
            <span class="badge-pill">Explainable AI</span>
            <span class="badge-pill">PyTorch Inference</span>
        </div>
        <h1>SentixAI Analytics</h1>
        <p>Explainable deep learning platform for text sentiment and emotion classification with attention mapping, LIME, and Kernel SHAP attribution.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### System Controls")
    st.caption("SentixAI Platform Version 4.0")
    st.divider()

    st.markdown("#### Model Status")
    try:
        sent_clf = get_sentiment_model()
        st.success("Sentiment Model: Active (IMDB, 90.02% Accuracy)")
    except Exception as exc:
        st.error(f"Sentiment Model: Offline ({exc})")
        sent_clf = None

    try:
        emo_clf = get_emotion_model()
        st.success("Emotion Model: Active (GoEmotions, 73.44% Accuracy)")
    except Exception as exc:
        st.error(f"Emotion Model: Offline ({exc})")
        emo_clf = None

    st.divider()
    st.markdown("#### Explainability Configuration")
    include_emotion = st.checkbox("Include Emotion Classification", value=True)
    show_attention = st.checkbox("Additive Attention Heatmap", value=True)
    show_lime = st.checkbox("LIME Local Perturbations", value=True)
    show_shap = st.checkbox("Kernel SHAP Attribution", value=False)

    lime_samples = st.slider(
        "LIME Sample Count",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Higher sample counts produce more stable perturbation weights with moderate compute time.",
    )

    if show_shap:
        shap_samples = st.slider(
            "Kernel SHAP Evaluation Budget",
            min_value=40,
            max_value=200,
            value=80,
            step=20,
            help="Evaluates feature combinations against an all-words-masked baseline.",
        )
    else:
        shap_samples = 80

    st.divider()
    st.markdown("#### Deployment Information")
    st.caption(
        "Standard deployment target: Streamlit Community Cloud (Python 3.11, CPU execution, persistent container)."
    )


# Tabs Navigation
tab_predict, tab_batch, tab_analytics, tab_architecture, tab_deploy, tab_vae = st.tabs([
    "Inference and Interpretability",
    "Batch CSV Processing",
    "Evaluation and Analytics",
    "Architecture and Methodology",
    "Deployment Documentation",
    "Unusual Review Detection",
])


# ==========================================
# TAB 1: SINGLE TEXT INFERENCE & XAI
# ==========================================
with tab_predict:
    st.markdown("### Text Classification & Interpretability")
    st.markdown(
        "Submit input text to generate sentiment and emotion predictions alongside intrinsic attention weights and local post-hoc explanations."
    )

    sample_presets = {
        "Custom Input": "",
        "Contrastive Sentence": "The camera is excellent but the battery is terrible.",
        "Positive Evaluation": "I am thoroughly impressed by the overall build quality and system performance.",
        "Service Complaint": "The delivery was delayed by several days and support provided no resolution.",
        "Safety Alert": "The device overheated suddenly and produced an alarming noise during operation.",
        "Unexpected Finding": "The experimental results were surprisingly different from the published baseline figures.",
    }

    selected_sample = st.selectbox(
        "Benchmark Examples:",
        options=list(sample_presets.keys()),
        index=1,
    )

    default_text = (
        sample_presets[selected_sample]
        if selected_sample != "Custom Input"
        else "The camera is excellent but the battery is terrible."
    )

    user_text = st.text_area(
        "Input Text:",
        value=default_text,
        height=100,
        placeholder="Enter English text for inference...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("Run Inference", type="primary", use_container_width=True)

    if analyze_clicked or user_text:
        if not user_text.strip():
            st.warning("Input text cannot be empty.")
        elif not sent_clf:
            st.error("Sentiment model checkpoint is unavailable.")
        else:
            with st.spinner("Executing neural inference and generating feature attributions..."):
                try:
                    sent_pred = sent_clf.predict(user_text)
                    emo_pred = emo_clf.predict(user_text) if (include_emotion and emo_clf) else None

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
                                    <span style="color:#94a3b8; font-weight:600; font-size:0.8rem; letter-spacing:0.05em;">SENTIMENT CLASSIFICATION</span>
                                    <span style="background:{badge_color}; color:white; padding:2px 10px; border-radius:4px; font-weight:600; font-size:0.8rem;">{sent_label}</span>
                                </div>
                                <h3 style="margin:10px 0 4px 0; font-size:1.6rem; font-weight:700; color:#f8fafc;">{sent_label}</h3>
                                <p style="margin:0; color:#94a3b8; font-size:0.92rem;">Confidence: <strong style="color:#f8fafc;">{sent_conf*100:.2f}%</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### Sentiment Class Probabilities")
                        sent_probs = sent_pred["probabilities"]
                        probs_df = pd.DataFrame({
                            "Class": [c.capitalize() for c in sent_probs.keys()],
                            "Probability": list(sent_probs.values()),
                            "Percentage": [f"{v*100:.2f}%" for v in sent_probs.values()],
                        })
                        chart_sent = (
                            alt.Chart(probs_df)
                            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                            .encode(
                                x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                                y=alt.Y("Class:N", sort="-x", title=""),
                                color=alt.Color(
                                    "Class:N",
                                    scale=alt.Scale(domain=["Positive", "Negative"], range=["#10b981", "#ef4444"]),
                                    legend=None,
                                ),
                                tooltip=["Class", "Percentage"],
                            )
                            .properties(height=100)
                        )
                        st.altair_chart(chart_sent, use_container_width=True)
                        st.markdown(
                            "<p class='info-note'>* Trained on IMDB (binary labels: positive, negative). Neutral is not part of the training vocabulary.</p>",
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
                                        <span style="color:#94a3b8; font-weight:600; font-size:0.8rem; letter-spacing:0.05em;">EMOTION CLASSIFICATION</span>
                                        <span style="background:#6366f1; color:white; padding:2px 10px; border-radius:4px; font-weight:600; font-size:0.8rem;">{emo_label}</span>
                                    </div>
                                    <h3 style="margin:10px 0 4px 0; font-size:1.6rem; font-weight:700; color:#f8fafc;">{emo_label}</h3>
                                    <p style="margin:0; color:#94a3b8; font-size:0.92rem;">Confidence: <strong style="color:#f8fafc;">{emo_conf*100:.2f}%</strong></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("##### Emotion Class Probabilities")
                            emo_probs = emo_pred["probabilities"]
                            emo_df = pd.DataFrame({
                                "Emotion": [c.capitalize() for c in emo_probs.keys()],
                                "Probability": list(emo_probs.values()),
                                "Percentage": [f"{v*100:.2f}%" for v in emo_probs.values()],
                            })
                            chart_emo = (
                                alt.Chart(emo_df)
                                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
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
                                .properties(height=210)
                            )
                            st.altair_chart(chart_emo, use_container_width=True)
                            st.markdown(
                                "<p class='info-note'>* Filtered GoEmotions classes: anger, disappointment, fear, joy, love, sadness, surprise.</p>",
                                unsafe_allow_html=True,
                            )

                    # Explainability Section
                    st.divider()
                    st.markdown("### Explainable AI (XAI) Attributions")
                    st.markdown(
                        "Comparative analysis of model-internal attention weights and local post-hoc interpretability techniques."
                    )

                    # 1. Additive Attention
                    if show_attention:
                        st.markdown("#### 1. BiLSTM Additive Attention Layer")
                        attn_tokens = sent_pred.get("attention_tokens", [])
                        if attn_tokens:
                            max_weight = max((item["weight"] for item in attn_tokens), default=0.001)

                            chips_html = []
                            for item in attn_tokens:
                                w = item["weight"]
                                norm_w = w / (max_weight + 1e-6)
                                bg_alpha = max(0.12, min(0.92, norm_w * 0.92))
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
                                "Token background intensity represents normalized softmax attention weights allocated during sequence context pooling."
                            )

                            col_attn_chart, col_attn_table = st.columns([3, 2])
                            with col_attn_chart:
                                top_attn_df = pd.DataFrame(attn_tokens).sort_values("weight", ascending=False).head(10)
                                attn_bar = (
                                    alt.Chart(top_attn_df)
                                    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#6366f1")
                                    .encode(
                                        x=alt.X("weight:Q", title="Attention Weight (Softmax)"),
                                        y=alt.Y("token:N", sort="-x", title="Token"),
                                        tooltip=["token", "weight"],
                                    )
                                    .properties(height=260, title="Top Token Attention Weights")
                                )
                                st.altair_chart(attn_bar, use_container_width=True)

                            with col_attn_table:
                                st.markdown("###### Ranked Token Weights")
                                st.dataframe(
                                    top_attn_df.rename(columns={"token": "Token", "weight": "Weight"}),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    # 2. LIME
                    if show_lime:
                        st.markdown("#### 2. LIME (Local Interpretable Model-agnostic Explanations)")
                        with st.spinner("Computing local surrogate perturbations..."):
                            lime_res = lime_explanation(sent_clf, user_text, num_features=10, num_samples=lime_samples)
                            lime_words = lime_res.get("important_words", [])

                            if lime_words:
                                lime_df = pd.DataFrame(lime_words)
                                lime_df["direction"] = np.where(
                                    lime_df["contribution"] >= 0, "Supports Prediction", "Opposes Prediction"
                                )
                                lime_df["abs_contrib"] = lime_df["contribution"].abs()

                                col_l1, col_l2 = st.columns([3, 2])
                                with col_l1:
                                    lime_chart = (
                                        alt.Chart(lime_df)
                                        .mark_bar(cornerRadius=3)
                                        .encode(
                                            x=alt.X("contribution:Q", title="Local Regression Weight"),
                                            y=alt.Y(
                                                "token:N",
                                                sort=alt.EncodingSortField(field="abs_contrib", order="descending"),
                                                title="Token",
                                            ),
                                            color=alt.Color(
                                                "direction:N",
                                                scale=alt.Scale(
                                                    domain=["Supports Prediction", "Opposes Prediction"],
                                                    range=["#10b981", "#ef4444"],
                                                ),
                                                legend=alt.Legend(title="Effect"),
                                            ),
                                            tooltip=["token", "contribution", "direction"],
                                        )
                                        .properties(
                                            height=270,
                                            title=f"LIME Word Weights toward '{sent_pred['label'].upper()}'",
                                        )
                                    )
                                    st.altair_chart(lime_chart, use_container_width=True)

                                with col_l2:
                                    st.markdown(f"###### Feature Weights ({lime_samples} samples)")
                                    st.dataframe(
                                        lime_df[["token", "contribution", "direction"]].rename(
                                            columns={"token": "Token", "contribution": "Weight", "direction": "Effect"}
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                st.caption(lime_res.get("note", ""))

                    # 3. Kernel SHAP
                    if show_shap:
                        st.markdown("#### 3. Kernel SHAP (SHapley Additive exPlanations)")
                        with st.spinner("Computing Shapley values against masked baseline..."):
                            try:
                                shap_res = shap_explanation(sent_clf, user_text, max_tokens=16, nsamples=shap_samples)
                                shap_words = shap_res.get("important_words", [])

                                if shap_words:
                                    shap_df = pd.DataFrame(shap_words).head(10)
                                    shap_df["direction"] = np.where(
                                        shap_df["contribution"] >= 0, "Positive Impact", "Negative Impact"
                                    )
                                    shap_df["abs_contrib"] = shap_df["contribution"].abs()

                                    col_s1, col_s2 = st.columns([3, 2])
                                    with col_s1:
                                        shap_chart = (
                                            alt.Chart(shap_df)
                                            .mark_bar(cornerRadius=3)
                                            .encode(
                                                x=alt.X("contribution:Q", title="Shapley Value"),
                                                y=alt.Y(
                                                    "token:N",
                                                    sort=alt.EncodingSortField(field="abs_contrib", order="descending"),
                                                    title="Token",
                                                ),
                                                color=alt.Color(
                                                    "direction:N",
                                                    scale=alt.Scale(
                                                        domain=["Positive Impact", "Negative Impact"],
                                                        range=["#38bdf8", "#f43f5e"],
                                                    ),
                                                    legend=alt.Legend(title="Attribution"),
                                                ),
                                                tooltip=["token", "contribution", "direction"],
                                            )
                                            .properties(
                                                height=270,
                                                title=f"Kernel SHAP Attribution toward '{sent_pred['label'].upper()}'",
                                            )
                                        )
                                        st.altair_chart(shap_chart, use_container_width=True)

                                    with col_s2:
                                        st.markdown(f"###### Shapley Values ({shap_samples} evaluations)")
                                        st.dataframe(
                                            shap_df[["token", "contribution", "direction"]].rename(
                                                columns={"token": "Token", "contribution": "Value", "direction": "Impact"}
                                            ),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    st.caption(shap_res.get("note", ""))
                            except Exception as shap_err:
                                st.warning(f"Kernel SHAP computation failed: {shap_err}")

                except Exception as err:
                    st.error(f"Inference execution failed: {err}")


# ==========================================
# TAB 2: BATCH CSV PROCESSING
# ==========================================
with tab_batch:
    st.markdown("### Batch Review Classification")
    st.markdown(
        "Upload a structured CSV containing a **`review`** column to execute batch inference across both sentiment and emotion networks."
    )

    sample_csv_path = ROOT / "sample_batch_reviews.csv"
    if not sample_csv_path.exists():
        sample_csv_path = ROOT / "data/sample_batch_reviews.csv"

    col_upload, col_sample = st.columns([3, 2])
    with col_sample:
        st.markdown("###### Reference Dataset")
        if sample_csv_path.exists():
            sample_data = sample_csv_path.read_text(encoding="utf-8")
            st.download_button(
                label="Download Sample CSV",
                data=sample_data,
                file_name="sample_batch_reviews.csv",
                mime="text/csv",
                use_container_width=True,
            )
            load_sample_btn = st.button("Load Sample CSV", use_container_width=True)
        else:
            load_sample_btn = False

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV (maximum 5,000 rows):", type=["csv"])

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
    elif load_sample_btn and sample_csv_path.exists():
        input_df = pd.read_csv(sample_csv_path)
        st.info("Loaded reference benchmark dataset.")

    if input_df is not None:
        if "review" not in input_df.columns:
            st.error("Input CSV must contain a header named 'review'.")
        elif input_df.empty:
            st.warning("Input CSV contains no rows.")
        else:
            st.write(f"Total rows identified: **{len(input_df)}**")
            with st.expander("Preview Input Records", expanded=False):
                st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("Execute Batch Inference", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                clean_reviews = input_df["review"].fillna("").astype(str).str.strip().tolist()
                valid_reviews = [r for r in clean_reviews if r]

                if not valid_reviews:
                    st.error("All rows in the review column are empty.")
                else:
                    status_text.text("Processing Sentiment BiLSTM batches...")
                    progress_bar.progress(25)
                    sent_results = sent_clf.predict_many(valid_reviews, batch_size=128)

                    status_text.text("Processing Emotion BiLSTM batches...")
                    progress_bar.progress(65)
                    emo_results = (
                        emo_clf.predict_many(valid_reviews, batch_size=128)
                        if emo_clf
                        else [{"label": "N/A", "confidence": 0.0}] * len(valid_reviews)
                    )

                    progress_bar.progress(90)
                    status_text.text("Compiling summary distributions...")

                    results_df = pd.DataFrame({
                        "review": valid_reviews,
                        "sentiment": [item["label"] for item in sent_results],
                        "sentiment_confidence": [round(item["confidence"], 4) for item in sent_results],
                        "emotion": [item["label"] for item in emo_results],
                        "emotion_confidence": [round(item["confidence"], 4) for item in emo_results],
                    })

                    progress_bar.progress(100)
                    status_text.success("Batch classification completed.")

                    # Summary Metrics
                    st.divider()
                    st.markdown("#### Aggregate Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    tot = len(results_df)
                    pos_count = int((results_df["sentiment"] == "positive").sum())
                    neg_count = int((results_df["sentiment"] == "negative").sum())
                    top_emotion = results_df["emotion"].mode()[0] if not results_df["emotion"].empty else "N/A"

                    m1.metric("Total Records", tot)
                    m2.metric("Positive Sentiment", f"{pos_count} ({pos_count/tot*100:.1f}%)")
                    m3.metric("Negative Sentiment", f"{neg_count} ({neg_count/tot*100:.1f}%)")
                    m4.metric("Predominant Emotion", top_emotion.capitalize())

                    # Distributions
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.markdown("###### Sentiment Distribution")
                        sent_counts = results_df["sentiment"].value_counts().reset_index()
                        sent_counts.columns = ["Sentiment", "Count"]
                        chart_s = (
                            alt.Chart(sent_counts)
                            .mark_bar(cornerRadius=3)
                            .encode(
                                x=alt.X("Count:Q", title="Count"),
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
                            .mark_bar(cornerRadius=3)
                            .encode(
                                x=alt.X("Count:Q", title="Count"),
                                y=alt.Y("Emotion:N", sort="-x", title=""),
                                color=alt.Color("Count:Q", scale=alt.Scale(scheme="purples"), legend=None),
                                tooltip=["Emotion", "Count"],
                            )
                            .properties(height=210)
                        )
                        st.altair_chart(chart_e, use_container_width=True)

                    st.divider()
                    st.markdown("#### Output Table")
                    st.dataframe(results_df, use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Processed Results (CSV)",
                        data=csv_data,
                        file_name="sentixai_batch_results.csv",
                        mime="text/csv",
                        type="primary",
                    )


# ==========================================
# TAB 3: MODEL ANALYTICS & CURVES
# ==========================================
with tab_analytics:
    st.markdown("### Model Training and Evaluation Metrics")
    st.markdown(
        "Quantitative benchmarks, convergence curves, and confusion matrices evaluated on independent held-out test splits."
    )

    eval_data = load_evaluation_data()
    sent_eval = eval_data.get("sentiment")
    emo_eval = eval_data.get("emotion")

    st.markdown("##### Held-out Test Set Performance")
    kpi_cols = st.columns(4)
    if sent_eval:
        s_acc = sent_eval["test_metrics"]["accuracy"] * 100
        s_f1 = sent_eval["test_metrics"]["f1_weighted"] * 100
        kpi_cols[0].metric("Sentiment Accuracy", f"{s_acc:.2f}%", help="Evaluated on 4,959 test reviews")
        kpi_cols[1].metric("Sentiment Weighted F1", f"{s_f1:.2f}%")
    if emo_eval:
        e_acc = emo_eval["test_metrics"]["accuracy"] * 100
        e_f1 = emo_eval["test_metrics"]["f1_weighted"] * 100
        kpi_cols[2].metric("Emotion Accuracy", f"{e_acc:.2f}%", help="Evaluated on 738 test samples")
        kpi_cols[3].metric("Emotion Weighted F1", f"{e_f1:.2f}%")

    st.divider()

    st.markdown("#### Convergence History")
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
            st.markdown("###### Sentiment: Accuracy per Epoch")
            c_acc = (
                alt.Chart(sent_curve_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Epoch:O", title="Epoch"),
                    y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.7, 1.0]), title="Accuracy"),
                    color=alt.Color(
                        "Split:N",
                        scale=alt.Scale(domain=["Train", "Validation"], range=["#3b82f6", "#f97316"]),
                    ),
                    tooltip=["Epoch", "Split", "Accuracy"],
                )
                .properties(height=240)
            )
            st.altair_chart(c_acc, use_container_width=True)

        with col_hist2:
            st.markdown("###### Sentiment: Cross-Entropy Loss per Epoch")
            c_loss = (
                alt.Chart(sent_curve_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Epoch:O", title="Epoch"),
                    y=alt.Y("Loss:Q", title="Loss"),
                    color=alt.Color(
                        "Split:N",
                        scale=alt.Scale(domain=["Train", "Validation"], range=["#3b82f6", "#f97316"]),
                    ),
                    tooltip=["Epoch", "Split", "Loss"],
                )
                .properties(height=240)
            )
            st.altair_chart(c_loss, use_container_width=True)

    st.divider()
    st.markdown("#### Test Set Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)

    sent_cm_path = ROOT / "outputs/results/confusion_matrix.png"
    if not sent_cm_path.exists():
        sent_cm_path = ROOT / "confusion_matrix.png"

    emo_cm_path = ROOT / "outputs/emotion/results/confusion_matrix.png"

    with col_cm1:
        st.markdown("###### Sentiment Network (IMDB Test Set)")
        if sent_cm_path.exists():
            st.image(str(sent_cm_path), caption="Normalized Confusion Matrix (Binary Sentiment)", use_container_width=True)
        else:
            st.info("Sentiment confusion matrix graphic is not available.")

    with col_cm2:
        st.markdown("###### Emotion Network (GoEmotions Test Set)")
        if emo_cm_path.exists():
            st.image(str(emo_cm_path), caption="Normalized Confusion Matrix (7-Class Emotion)", use_container_width=True)
        else:
            st.info("Emotion confusion matrix graphic is not available.")

    st.divider()
    st.markdown("#### Detailed Classification Reports")
    col_cr1, col_cr2 = st.columns(2)

    with col_cr1:
        if sent_eval and "classification_report" in sent_eval:
            st.markdown("###### Sentiment Model Metrics")
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
            st.markdown("###### Emotion Model Metrics")
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
    st.markdown("### Model Architecture and Theory")

    st.markdown(
        """
        #### Network Pipeline
        The classification pipeline couples a bidirectional recurrent structure with a parameterised additive attention layer to pool variable-length token representations into fixed-dimensional context vectors.
        """
    )

    st.code(
        """
Input Tokens (x_1, ..., x_T)
        |
Embedding Layer           (Vocabulary Dimension -> 128-d, Dropout = 0.35)
        |
Bidirectional LSTM        (Hidden Dimension = 128; Forward + Backward = 256-d states h_t)
        |
Masked Additive Attention (Computes alignment scores e_t and normalized weights alpha_t)
        |
Context Vector Pooling    (c = SUM(alpha_t * h_t))
        |
Linear Dense Projection   (256-d -> Number of Target Classes)
        |
Softmax Normalization     (Inference probability distribution)
        """,
        language="text",
    )

    st.markdown(
        """
        #### Mathematical Formulation
        1. **Attention Energy Calculation**:
        $$e_t = v^T \\tanh(W h_t + b)$$
        Where $h_t \\in \\mathbb{R}^{2d}$ represents the bidirectional state at token index $t$, $W \\in \\mathbb{R}^{a \\times 2d}$ denotes the trainable projection matrix, and $v \\in \\mathbb{R}^a$ is the attention scoring vector.

        2. **Masked Softmax Normalization**:
        $$\\alpha_t = \\frac{\\exp(e_t)}{\\sum_{k=1}^T \\exp(e_k)}$$
        Padding tokens ($\\text{<PAD>}$) are assigned a large negative mask ($-10^9$) prior to softmax to ensure zero attention mass is allocated to padding.

        3. **Context Aggregation**:
        $$c = \\sum_{t=1}^T \\alpha_t h_t$$

        ---

        #### Interpretability Methodologies Summary
        | Framework | Paradigm | Mechanism | Interpretative Scope |
        | :--- | :--- | :--- | :--- |
        | **Additive Attention** | Model-Intrinsic | Normalizes pooling weights directly inside network computation graph. | Token-level focus during context aggregation. |
        | **LIME** | Model-Agnostic Surrogate | Fits an interpretable local ridge regressor on perturbed text samples. | Directional feature weights (supporting vs. opposing). |
        | **Kernel SHAP** | Cooperative Game Theory | Computes Shapley values against a marginal masked background distribution. | Fair attribution satisfying efficiency and consistency axioms. |
        """
    )


# ==========================================
# TAB 5: DEPLOYMENT DOCUMENTATION
# ==========================================
with tab_deploy:
    st.markdown("### Production Deployment Guide")
    st.markdown(
        """
        #### Hosting Comparison

        | Specification | **Streamlit Community Cloud** (Recommended) | **Render (Free Tier)** | **Vercel Serverless** |
        | :--- | :--- | :--- | :--- |
        | **Cost** | Free tier available | Free tier available | Free tier available |
        | **PyTorch Execution** | Supported (Native PyTorch, Scikit-Learn, LIME, SHAP) | Limited (512 MB memory constraint prone to OOM) | Unsupported (Exceeds 250 MB bundle ceiling) |
        | **Container Persistence** | Persistent container; no sleep timeout | Spins down after 15 minutes of inactivity | Function timeouts capped at 10–15 seconds |
        | **Public URL** | `https://<organization>-sentixai.streamlit.app` | `https://sentixai.onrender.com` | `https://sentixai.vercel.app` (Static UI only) |
        | **Deployment Workflow** | Automated GitHub integration (1-click) | Dockerfile compilation required | Requires decoupled API backend |

        ---

        #### Deployment Steps for Streamlit Community Cloud

        1. **Commit and push changes to GitHub**:
        ```bash
        git add .
        git commit -m "Configure production Streamlit dashboard"
        git push origin main
        ```

        2. **Deploy on Streamlit Cloud**:
           - Log in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
           - Click **New app**.
           - Select the repository: **`govindshaw67/SentixAI`**
           - Set branch: **`main`**
           - Set entrypoint file path: **`streamlit_app.py`**
           - Specify custom app URL subdomain under **Advanced settings** if desired.

        3. **Verification**:
           - Streamlit Cloud will parse `requirements.txt`, load the committed checkpoints, and generate your live URL.
        """
    )


# ==========================================
# TAB 6: UNUSUAL REVIEW DETECTION (VAE)
# Module 5 – Generative Models
# ==========================================

@st.cache_resource(show_spinner="Loading VAE anomaly detector...")
def get_vae_detector():
    """Load the trained ReviewVAE checkpoint if available."""
    vae_dir = ROOT / "outputs/vae/artifacts"
    required = ["vae_model.pt", "vae_vectorizer.pkl", "vae_scaler.pkl"]
    if all((vae_dir / f).exists() for f in required):
        return VAEAnomalyDetector(vae_dir)
    return None


with tab_vae:
    st.markdown("### Unusual Review Detection")
    st.markdown(
        "This tab uses a **Variational Autoencoder (VAE)** trained on review text to detect "
        "anomalous or unusual reviews. The VAE learns a compressed latent representation of "
        "normal review language. Reviews that the decoder cannot reconstruct well receive a "
        "high anomaly score and are flagged as *unusual*."
    )

    # Architecture explainer
    with st.expander("How the VAE works (Module 5 – Generative Models)", expanded=False):
        st.code(
            """
Input Review
      |
  TF-IDF Vectorisation  (up to 5,000 token features, sublinear TF)
      |
  Min-Max Normalisation  → feature vector x ∈ [0, 1]^d
      |
 ┌──────────────────────────────────────┐
 │  ENCODER                             │
 │  Linear → LayerNorm → ReLU           │
 │  Linear → ReLU                       │
 │      ↓               ↓               │
 │   FC_µ(x)       FC_logσ²(x)         │
 └──────────────────────────────────────┘
      |
  Reparameterisation:  z = µ + ε·σ   (ε ~ N(0,I))
      |
 ┌──────────────────────────────────────┐
 │  DECODER                             │
 │  Linear → ReLU                       │
 │  Linear → LayerNorm → ReLU           │
 │  Linear → Sigmoid → x̂ ∈ [0,1]^d    │
 └──────────────────────────────────────┘
      |
  Reconstruction Error = MSE(x̂, x)   ← Anomaly Score
  Loss = Recon Loss + β · KL(N(µ,σ²) ‖ N(0,I))
            """,
            language="text",
        )
        st.markdown(
            """
**Key concepts demonstrated:**
| Concept | Implementation |
|---------|----------------|
| Generative model | ReviewVAE generates plausible review feature vectors |
| Encoder & Decoder | Two-layer MLP with LayerNorm |
| Latent space | 32-dimensional Gaussian latent variable z |
| Reparameterisation | `z = µ + ε·σ` enables gradient flow through sampling |
| Reconstruction loss | Mean-squared error between input and reconstruction |
| KL divergence | Regularises latent space toward N(0, I) |
| Anomaly detection | High reconstruction error → review is *unusual* |
"""
        )

    # ── Load model ───────────────────────────────────────────────────────────
    vae_detector = get_vae_detector()

    if vae_detector is None:
        st.warning(
            "**VAE checkpoint not found.**  "
            "Train the model first with the command below, then restart the app."
        )
        st.code(
            "python train_vae.py --csv data/sample_batch_reviews.csv --epochs 30",
            language="bash",
        )
        st.info(
            "The training script produces `outputs/vae/artifacts/vae_model.pt` "
            "along with the fitted vectoriser and scaler. "
            "Re-open this tab after training completes."
        )
    else:
        threshold = vae_detector.threshold

        st.success(
            f"VAE loaded successfully · "
            f"Anomaly threshold: **{threshold:.6f}** "
            f"(top {vae_detector.anomaly_top_k_pct:.1f}% of training scores)"
        )
        st.divider()

        # ── Single review scoring ─────────────────────────────────────────
        st.markdown("#### Single Review Anomaly Check")

        sample_presets_vae = {
            "Custom Input": "",
            "Normal positive": "Great product, works exactly as expected.",
            "Normal negative": "Battery life is disappointing and does not last a day.",
            "Spam-like": "BUY NOW!!! BEST DEAL EVER 99% OFF LIMITED OFFER CLICK HERE!!!",
            "Off-topic / Gibberish": "asdfgh jkl qwerty zxcvbn lorem ipsum dolor sit amet.",
            "Adversarial": "This product is simultaneously the best and the worst thing I ever bought.",
        }

        selected_vae_preset = st.selectbox(
            "Example reviews:",
            options=list(sample_presets_vae.keys()),
            index=2,
            key="vae_preset",
        )
        default_vae_text = sample_presets_vae[selected_vae_preset] or ""

        vae_input = st.text_area(
            "Review text:",
            value=default_vae_text,
            height=90,
            placeholder="Paste any review here …",
            key="vae_single_input",
        )

        if st.button("Check for anomaly", type="primary", key="vae_single_btn"):
            if not vae_input.strip():
                st.warning("Please enter a review.")
            else:
                result = vae_detector.score_single(vae_input.strip())
                score = result["anomaly_score"]
                is_unusual = result["is_unusual"]

                col_score, col_badge = st.columns([3, 1])
                with col_score:
                    st.metric(
                        label="Reconstruction Error (Anomaly Score)",
                        value=f"{score:.6f}",
                        delta=f"threshold: {threshold:.6f}",
                        delta_color="off",
                    )
                with col_badge:
                    if is_unusual:
                        st.markdown(
                            "<div style='background:#ef4444;color:white;padding:14px 18px;"
                            "border-radius:8px;font-weight:700;font-size:1.1rem;margin-top:12px;"
                            "text-align:center'>⚠ UNUSUAL</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div style='background:#10b981;color:white;padding:14px 18px;"
                            "border-radius:8px;font-weight:700;font-size:1.1rem;margin-top:12px;"
                            "text-align:center'>✓ NORMAL</div>",
                            unsafe_allow_html=True,
                        )

                st.caption(
                    "Score = mean squared error between the original TF-IDF vector and "
                    "the VAE reconstruction.  Higher score → less typical review."
                )

        st.divider()

        # ── Batch CSV scoring ────────────────────────────────────────────
        st.markdown("#### Batch Anomaly Detection")
        st.markdown(
            "Upload a CSV containing a **`review`** column.  "
            "Every review will be scored and the results (with anomaly flags) can be downloaded."
        )

        col_upload_vae, col_sample_vae = st.columns([3, 2])
        with col_sample_vae:
            sample_csv_vae = ROOT / "data/sample_batch_reviews.csv"
            if not sample_csv_vae.exists():
                sample_csv_vae = ROOT / "sample_batch_reviews.csv"
            if sample_csv_vae.exists():
                st.download_button(
                    "Download sample CSV",
                    sample_csv_vae.read_text(encoding="utf-8"),
                    file_name="sample_batch_reviews.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                load_sample_vae = st.button("Load sample CSV", use_container_width=True, key="vae_load_sample")
            else:
                load_sample_vae = False

        with col_upload_vae:
            uploaded_vae = st.file_uploader(
                "Upload CSV (max 10,000 rows):",
                type=["csv"],
                key="vae_uploader",
            )

        batch_df_vae = None
        if uploaded_vae is not None:
            try:
                batch_df_vae = pd.read_csv(uploaded_vae)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        elif load_sample_vae and sample_csv_vae.exists():
            batch_df_vae = pd.read_csv(sample_csv_vae)
            st.info("Loaded sample CSV.")

        if batch_df_vae is not None:
            if "review" not in batch_df_vae.columns:
                st.error("CSV must contain a column named **`review`**.")
            elif batch_df_vae.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                reviews_vae = (
                    batch_df_vae["review"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                reviews_vae = [r for r in reviews_vae if r]

                if st.button("Run anomaly detection", type="primary", key="vae_batch_btn"):
                    with st.spinner("Scoring reviews …"):
                        scored_df = vae_detector.score_batch(reviews_vae)

                    n_unusual = int(scored_df["is_unusual"].sum())
                    pct_unusual = n_unusual / len(scored_df) * 100

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Reviews", len(scored_df))
                    m2.metric("Flagged Unusual", n_unusual)
                    m3.metric("Unusual %", f"{pct_unusual:.1f}%")
                    m4.metric("Threshold", f"{threshold:.5f}")

                    st.divider()

                    # Score distribution chart
                    st.markdown("##### Anomaly Score Distribution")
                    import altair as alt
                    hist_df = scored_df[["anomaly_score", "is_unusual"]].copy()
                    hist_df["Category"] = hist_df["is_unusual"].map(
                        {True: "Unusual", False: "Normal"}
                    )
                    chart_hist = (
                        alt.Chart(hist_df)
                        .mark_bar(opacity=0.75, binSpacing=0)
                        .encode(
                            x=alt.X(
                                "anomaly_score:Q",
                                bin=alt.Bin(maxbins=40),
                                title="Reconstruction Error",
                            ),
                            y=alt.Y("count()", title="Count"),
                            color=alt.Color(
                                "Category:N",
                                scale=alt.Scale(
                                    domain=["Normal", "Unusual"],
                                    range=["#10b981", "#ef4444"],
                                ),
                            ),
                            tooltip=["Category", "count()"],
                        )
                        .properties(height=240, title="Score distribution (red = flagged)")
                    )
                    threshold_rule = (
                        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
                        .mark_rule(color="#f97316", strokeDash=[6, 3], strokeWidth=2)
                        .encode(x="threshold:Q")
                    )
                    st.altair_chart(chart_hist + threshold_rule, use_container_width=True)

                    # Flagged reviews table
                    st.markdown("##### Flagged Unusual Reviews")
                    unusual_shown = scored_df[scored_df["is_unusual"]].head(50).copy()
                    unusual_shown.index = range(1, len(unusual_shown) + 1)
                    st.dataframe(unusual_shown, use_container_width=True)

                    # Full results download
                    csv_out = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download full scored results (CSV)",
                        csv_out,
                        file_name="vae_anomaly_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

st.divider()
st.caption(
    "SentixAI Platform · Explainable Deep Learning System for Sentiment and Emotion Classification"
)

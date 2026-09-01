# SentixAI Analytics

**Explainable Deep Learning for Sentiment and Emotion Analysis**

**Live demo:** [https://sentixai-01cn.onrender.com](https://sentixai-01cn.onrender.com)

## Abstract

SentixAI is a local web application for English sentiment and emotion analysis. It uses independently trained BiLSTM-attention models, shows confidence and class probabilities, and offers attention, LIME, and SHAP explanations. It also supports batch CSV processing and a reproducible neural-model comparison study.

## Problem statement and objectives

Text classifiers often return a label without explaining it. SentixAI predicts sentiment/emotion, surfaces model evidence, supports batch analysis, and preserves metrics/experiments for a B.Tech CSE-AIML demonstration.

## Features

- Saved-model inference: the dashboard never retrains a model.
- Single-review and CSV batch analysis.
- Confidence scores, real model probabilities, and downloadable batch results.
- Attention, optional LIME, and optional SHAP explanations.
- Training curves, real confusion matrices, and saved evaluation reports.
- MLP, LSTM, BiLSTM, and BiLSTM-attention comparison runner.

## System architecture

```text
Browser dashboard / CSV upload
        ↓
FastAPI (app.py)
        ↓
Shared preprocessing + saved vocabulary
        ↓
Sentiment checkpoint ─→ label, probability, explanation
Emotion checkpoint   ─→ label, probability, explanation
        ↓
Dashboard cards, charts, and downloadable CSV
```

## Dataset

### Sentiment

The IMDB CSV uses `review` and `sentiment` columns. Duplicate/empty rows are removed; the recorded run contained 49,582 usable reviews. Its genuine labels are `positive` and `negative` only.

### Emotion

The emotion model uses filtered public GoEmotions data. Only exact, single-label source classes are retained: `anger`, `disappointment`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

## Data preprocessing

Text is lowercased, HTML entities/tags and URLs are removed, and word-like English tokens are extracted. A vocabulary with `<PAD>`/`<UNK>` is fitted only on training text. Sequences are padded or truncated to a fixed length.

## Neural network architecture

```text
Tokens → Embedding → BiLSTM → Additive Attention → Dropout → Dense logits → Softmax
```

`CrossEntropyLoss` receives logits while softmax is used only to produce inference probabilities.

## BiLSTM and attention mechanism

The BiLSTM reads token sequences in both directions. Additive attention gives each non-padding token a normalized weight and pools the BiLSTM states into a context vector. These actual weights are exposed in inference output.

## Emotion detection

Emotion is an independently trained model with its own vocabulary, checkpoint, and evaluation report; it is not trained from IMDB sentiment labels.

## Explainable AI: Attention, LIME, and SHAP

- **Attention** displays token weights produced by the model.
- **LIME** perturbs words, runs the saved model, and estimates local word contributions.
- **SHAP** optionally estimates word contributions versus an all-words-masked baseline.

Attention is evidence of model focus, not a causal guarantee. LIME and SHAP are local approximations; Kernel SHAP is intentionally bounded because it is slower.

## Model comparison and hyperparameter experiments

`run_experiments.py` trains MLP, LSTM, BiLSTM, and BiLSTM-attention models on the same stratified split and training vocabulary. `experiment_config.json` controls learning rate, batch size, epochs, dropout, hidden units, embedding dimensions, optimizer, and BiLSTM-attention variants. The best run is selected by **macro-F1**, rather than accuracy alone.

## Evaluation metrics

Each run saves accuracy, precision, recall, weighted F1, macro-F1, classification report, confusion matrix, training curves, training time, parameter count, model size, and checkpoint.

## Results

These are actual held-out test results from completed training runs:

| Model | Dataset | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BiLSTM + Attention sentiment | IMDB | 90.02% | 90.02% | 90.02% | 90.02% |
| BiLSTM + Attention emotion | Filtered GoEmotions | 73.44% | 76.77% | 73.44% | 74.37% |

The controlled architecture-comparison study is configured but has not yet produced checked-in results. Run it before claiming a best architecture.

## Dashboard

The FastAPI-served dashboard provides text prediction, batch CSV prediction, probabilities, explanations, confusion matrices, and training/evaluation charts.

## Project structure

```text
app.py                    Dashboard/API and batch-upload routes
static/                   Dashboard HTML, CSS, JavaScript
sentixai_data.py          Cleaning, tokenization, vocabulary, datasets
sentixai_model.py         Production BiLSTM-attention model
sentixai_inference.py     Saved-checkpoint inference used by API/batch jobs
sentixai_xai.py           Attention, LIME, SHAP explanations
train_sentixai.py         Reproducible sentiment/emotion training
prepare_goemotions.py     GoEmotions preparation utility
comparison_models.py      MLP, LSTM, BiLSTM, attention architectures
run_experiments.py        Controlled comparison runner
outputs/                  Ignored generated model/results/download files
data/                     Prepared emotion data and sample batch CSV
```

## Installation

```bash
cd /Users/govindprasadshaw/Desktop/DL_Projects
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

Train sentiment:

```bash
python train_sentixai.py --csv "/Users/govindprasadshaw/Downloads/DATASETS/IMDB Dataset.csv" --output-dir outputs
```

Prepare and train emotion:

```bash
python prepare_goemotions.py --output data/goemotions_target_emotions.csv
python train_sentixai.py --csv data/goemotions_target_emotions.csv --text-column text --label-column emotion --output-dir outputs/emotion --model-name emotion_bilstm_attention.pt
```

Run the controlled comparison study:

```bash
python run_experiments.py --config experiment_config.json --csv "/Users/govindprasadshaw/Downloads/DATASETS/IMDB Dataset.csv"
```

It writes report-ready results to `outputs/experiments/`, including comparison CSV/JSON files, charts, checkpoints, and per-run confusion matrices.

## Testing

```bash
python predict_sentixai.py --text "This movie was excellent and enjoyable." --model-dir outputs/artifacts
python predict_sentixai_full.py --text "The service was slow and the product arrived damaged."
python explain_sentixai.py --text "The camera is excellent but the battery is terrible." --model sentiment --methods attention lime shap
```

## Running the application

### Option 1: Streamlit Dashboard (Recommended for Live Demo & Sharing)

Run the interactive Explainable AI dashboard with token attention heatmaps, LIME, SHAP, batch CSV analysis, and model curves:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Option 2: FastAPI Backend & Classic Dashboard

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Deployment & Sharable Link (Streamlit Cloud vs Render vs Vercel)

### 1. Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is 100% free, runs native PyTorch without memory/bundle limits, and provides a permanent public URL (e.g. `https://sentixai.streamlit.app`):

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Add Streamlit Explainable AI dashboard"
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New App"** (or **"Create app"**).
4. Select:
   - **Repository:** `govindshaw67/SentixAI`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"** Your public sharable link will be live in 1-2 minutes!

### 2. Why Not Vercel for PyTorch Deep Learning?

- **Bundle Size Limit:** Vercel Serverless Functions enforce a strict **250MB uncompressed limit**. PyTorch, SciPy, Scikit-learn, LIME, and SHAP together exceed 1.2GB, causing serverless build failure.
- **Execution Timeouts:** XAI computations like LIME and SHAP can take 5-15 seconds, hitting Vercel's strict 10s-15s serverless timeout caps.
- **Vercel Static Hosting:** If desired, Vercel can host the static HTML/CSS/JS frontend (`vercel.json` is provided in the repository) while the models run on a dedicated Python service. However, Streamlit Cloud hosts the full model + UI seamlessly in one place at zero cost.

## Security and reproducibility

The project requires no API keys. `.gitignore` excludes virtual environments, outputs, and `.env` files. Do not commit credentials or private datasets. Training seeds are fixed and deterministic operations are requested where supported by the PyTorch backend.

## Limitations

- Sentiment supports only positive/negative; Neutral is deliberately marked not trained.
- Emotion supports seven source-labelled classes; Frustration is deliberately marked not trained.
- Models are English-focused and reflect IMDB/GoEmotions domains.
- Customer-service complaints can produce unintuitive emotion labels because they are outside the emotion model's main domain.
- LIME/SHAP add inference time; SHAP is bounded.
- Architecture-comparison metrics are pending until the configured study is run.

## Future scope

- Train real three-class sentiment data including neutral.
- Add frustration-labelled customer-support data.
- Add labelled multilingual data and multilingual models.
- Add aspect-based sentiment labels/models.
- Run repeated-seed/cross-validation comparison experiments.
- Add authentication, rate limiting, storage, and monitoring for non-local deployment.

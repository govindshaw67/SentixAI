# SentixAI Analytics

**An Explainable Deep Learning Platform for Sentiment Classification, Emotion Detection, and Out-of-Distribution Review Anomaly Detection**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Web Application:** [https://sentixai-01cn.onrender.com](https://sentixai-01cn.onrender.com)

---

## 1. Abstract & Overview

**SentixAI** is an end-to-end, explainable deep learning platform designed to classify, interpret, and audit textual data. Moving beyond opaque "black-box" classifiers, SentixAI couples state-of-the-art recurrent and convolutional neural architectures with multi-tiered Explainable AI (XAI) frameworks—including **Model-Intrinsic Additive Attention**, **LIME (Local Interpretable Model-agnostic Explanations)**, and **Kernel SHAP (SHapley Additive exPlanations)**.

The platform provides a dual-model sentiment analysis engine allowing direct architectural comparisons between **Bidirectional LSTM with Attention** and a high-throughput **1D Text Convolutional Neural Network (TextCNN)**, alongside an independent **7-class Emotion Classifier** and an unsupervised **Variational Autoencoder (VAE)** for anomaly and outlier review detection.

---

## 2. Key Features

- **Multi-Model Sentiment Engine**: Switch seamlessly between **BiLSTM + Additive Attention** (accuracy-optimized: **90.02%**) and **TextCNN** (latency-optimized: **86.73%**, **~7.6x faster**).
- **Independent Emotion Classification**: 7-class emotion detection (*joy, sadness, anger, fear, surprise, disappointment, love*) trained on filtered GoEmotions text.
- **Triple-Layer Explainability (XAI)**:
  - *Model-Intrinsic Attention*: Real-time token highlighting driven by softmax attention weights.
  - *Local Surrogates (LIME)*: Perturbation-based local ridge regressors to uncover directional word impact.
  - *Game-Theoretic Attribution (Kernel SHAP)*: Evaluates Shapley values against an all-words-masked background.
- **Unsupervised Anomaly Detection (VAE)**: Variational Autoencoder that flags unusual, spammy, or out-of-distribution text reviews via reconstruction loss.
- **Dual User Interfaces**:
  - *Interactive Streamlit Dashboard*: Full multi-tab analytical suite for interactive inference, XAI inspection, and model benchmarks.
  - *FastAPI REST API & HTML Web UI*: Production-grade endpoints with Swagger documentation and CSV batch inference.
- **Saved Checkpoint Inference**: Guaranteed zero re-training during live inference; memory-efficient PyTorch batch processing.

---

## 3. System Architecture

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    User Interfaces                     │
                               │  - Streamlit Dashboard (streamlit_app.py:8501)         │
                               │  - HTML / Vanilla JS Dashboard (static/index.html:8000)│
                               └───────────────────────────┬────────────────────────────┘
                                                           │ HTTP / JSON Payload
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │               FastAPI Backend (app.py)                 │
                               │        Routing, Validation, Thread Concurrency         │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       Preprocessing Pipeline (sentixai_data.py)        │
                               │  Tokenization, RegEx Cleaning, Vocabulary Indexing,    │
                               │  Length Truncation & Zero-Padding (Max Length = 256)   │
                               └───────────┬───────────────┬────────────────┬───────────┘
                                           │               │                │
                   ┌───────────────────────┘               │                └───────────────────────┐
                   ▼                                       ▼                                        ▼
    ┌─────────────────────────────┐        ┌──────────────────────────────┐         ┌──────────────────────────────┐
    │     BiLSTM + Attention      │        │       TextCNN (1D-CNN)       │         │        Emotion BiLSTM        │
    │  (bilstm_attention.pt)      │        │        (textcnn.pt)          │         │(emotion_bilstm_attention.pt) │
    ├─────────────────────────────┤        ├──────────────────────────────┤         ├──────────────────────────────┤
    │ • Embedding (128-d)         │        │ • Embedding (128-d)          │         │ • Embedding (128-d)          │
    │ • BiLSTM (2x128=256-d state)│        │ • Conv1D (Kernels 2, 3, 4)   │         │ • BiLSTM (256-d state)       │
    │ • Additive Masked Attention │        │ • ReLU + 1D Global Max-Pool  │         │ • Additive Masked Attention  │
    │ • Dense -> Softmax (2-class)│        │ • Concat (192-d) + Dropout   │         │ • Dense -> Softmax (7-class) │
    │ • Accuracy: 90.02%          │        │ • Dense -> Softmax (2-class) │         │ • Accuracy: 73.44%           │
    │ • Latency: ~0.061 ms/sample │        │ • Latency: ~0.008 ms/sample  │         │ • Classes: joy, sadness, ... │
    └──────────────┬──────────────┘        └──────────────┬───────────────┘         └──────────────┬───────────────┘
                   │                                      │                                        │
                   └──────────────────┬───────────────────┴────────────────────────────────────────┘
                                      │
                                      ▼
                   ┌────────────────────────────────────────────────────────┐
                   │             Explainable AI (sentixai_xai.py)           │
                   │  - Attention Softmax Weights (BiLSTM)                  │
                   │  - LIME Local Regression Perturbations                 │
                   │  - Kernel SHAP Marginal Shapley Values                 │
                   └────────────────────────────────────────────────────────┘
```

---

## 4. Deep Learning Model Architectures

### 1. BiLSTM with Additive Masked Attention
- **Input**: Token sequence indices $\mathbf{x} = (x_1, \dots, x_T)$ bounded to sequence length $T=256$.
- **Embedding Layer**: Projects tokens into dense $\mathbb{R}^{128}$ space with embedding dropout ($p=0.35$).
- **Bidirectional LSTM**: Processes sequences forward and backward simultaneously:
  $$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$
  $$h_t = \left[\overrightarrow{h}_t \,\Vert\, \overleftarrow{h}_t\right] \in \mathbb{R}^{256}$$
- **Additive Attention Mechanism**:
  $$e_t = \mathbf{v}^T \tanh\left(\mathbf{W} h_t + \mathbf{b}\right)$$
  $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)} \quad (\text{with } e_k = -10^9 \text{ for } \langle\text{PAD}\rangle)$$
  $$\mathbf{c} = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{256}$$
- **Classification Projection**: Fully connected layer mapping $\mathbf{c} \to \mathbb{R}^2$ followed by log-softmax.

---

### 2. TextCNN (Parallel 1D Convolutions)
- **Input Transpose**: Embedding output reshaped to channel-first representation $[\text{Batch}, C=128, T=256]$.
- **Parallel 1D Convolutional Filter Banks**: 64 filters each across kernel sizes $k \in \{2, 3, 4\}$ to capture bigram, trigram, and 4-gram phrasal patterns:
  $$\mathbf{c}_i^{(k)} = \text{ReLU}\left(\mathbf{W}^{(k)} \ast \mathbf{x}_{i:i+k-1} + b^{(k)}\right)$$
- **Global Max-Over-Time Pooling**:
  $$\hat{c}_j^{(k)} = \max_{1 \le t \le T} c_{j, t}^{(k)}$$
- **Feature Concatenation**: Combines $3 \times 64 = 192$ feature activations into a dense unified vector:
  $$\mathbf{z} = \left[\hat{\mathbf{c}}^{(k=2)} \,\Vert\, \hat{\mathbf{c}}^{(k=3)} \,\Vert\, \hat{\mathbf{c}}^{(k=4)}\right] \in \mathbb{R}^{192}$$
- **Regularization & Classifier**: Dropout ($p=0.5$) $\to$ Dense ($192 \to 64$, ReLU) $\to$ Output ($64 \to 2$ classes).

---

### 3. Variational Autoencoder (VAE) for Anomaly Detection
- **Objective**: Identifies outliers, gibberish, spam, or atypical reviews without manual supervision.
- **Encoder**: Projects TF-IDF review representations into a latent Gaussian distribution parameterized by mean $\boldsymbol{\mu}$ and log-variance $\log \boldsymbol{\sigma}^2$.
- **Reparameterization Trick**: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
- **Decoder**: Reconstructs the input feature vector from latent space $\mathbf{z}$.
- **Anomaly Scoring**: Reviews exhibiting abnormally high Mean Squared Error (MSE) reconstruction loss $+ \beta \cdot D_{\text{KL}}$ are flagged as anomalies.

---

## 5. Empirical Performance Comparison (Held-Out Test Set)

Both sentiment models were evaluated on the **exact same 4,959 held-out test reviews** from the 50,000 IMDB dataset (`seed=42`, stratified 80/10/10 split):

| Evaluation Metric | BiLSTM + Additive Attention | TextCNN (1D-CNN) | Relative Difference / Trade-off |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **90.02%** | **86.73%** | +3.29% for BiLSTM |
| **Test Macro F1** | **90.02%** | **86.68%** | +3.34% for BiLSTM |
| **Test Weighted F1** | **90.02%** | **86.68%** | +3.34% for BiLSTM |
| **Test Weighted Precision** | **90.02%** | **87.21%** | +2.81% for BiLSTM |
| **Test Weighted Recall** | **90.02%** | **86.73%** | +3.29% for BiLSTM |
| **Trainable Parameters** | 4,137,730 | **2,646,402** | **36.0% fewer parameters** in TextCNN |
| **Checkpoint Size (Disk)**| 15.8 MB | **10.1 MB** | **36.1% lighter** checkpoint |
| **Inference Latency** | ~0.061 ms / sample | **~0.008 ms / sample** | **~7.6x faster inference** in TextCNN |
| **Training Duration** | ~180.0 s | **70.39 s** | **~2.5x faster training** in TextCNN |
| **Model Interpretability**| Attention Heatmaps + LIME + SHAP | LIME + Kernel SHAP | BiLSTM exposes native attention |

### Engineering Takeaways:
1. **Choose BiLSTM + Attention** when peak classification accuracy, complex document nuances, and native attention heatmaps are mission-critical.
2. **Choose TextCNN** for high-throughput streaming pipelines, low-latency microservices, edge devices, or resource-constrained deployments (7.6x faster inference with 36% less memory).

---

## 6. Project Structure

```text
SentixAI/
├── app.py                            # FastAPI backend with multi-model prediction and batch routes
├── streamlit_app.py                  # Production Streamlit UI (tabs: XAI, Batch, Analytics, Arch, VAE)
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Multi-stage production container configuration
├── render.yaml                       # Render blueprint definition
│
├── static/                           # Classic Web Interface
│   ├── index.html                    # Model selection dropdown and review input forms
│   └── app.js                        # Client-side API interactions and chart rendering
│
├── sentixai_model.py                 # PyTorch BiLSTM + Additive Attention architecture
├── sentixai_inference.py             # Inference class for BiLSTM models
├── train_sentixai.py                 # Training script for BiLSTM sentiment and emotion models
│
├── textcnn_model.py                  # PyTorch TextCNN architecture (Kernels 2, 3, 4)
├── textcnn_inference.py              # Inference class for TextCNN model
├── train_textcnn.py                  # Training pipeline for TextCNN on IMDB
├── predict_textcnn.py                # Command-line prediction tool for TextCNN
├── evaluate_comparison.py            # Comparative evaluation script benchmarked on test set
│
├── vae_model.py                      # Variational Autoencoder architecture for anomaly detection
├── vae_inference.py                  # VAE anomaly detector and scoring class
├── train_vae.py                      # VAE training script
│
├── sentixai_data.py                  # Tokenization, cleaning, and vocabulary creation
├── sentixai_xai.py                   # XAI engines: Attention, LIME, and Kernel SHAP
├── prepare_goemotions.py             # Preprocessing pipeline for GoEmotions dataset
│
├── outputs/                          # Saved Checkpoints and Evaluation Artifacts
│   ├── artifacts/
│   │   ├── bilstm_attention.pt       # Trained BiLSTM sentiment weights (15.8 MB)
│   │   └── vocabulary.json           # BiLSTM sentiment vocabulary
│   ├── results/
│   │   ├── evaluation.json           # BiLSTM held-out evaluation report
│   │   └── confusion_matrix.png      # BiLSTM test confusion matrix
│   ├── textcnn/
│   │   ├── artifacts/
│   │   │   ├── textcnn.pt            # Trained TextCNN weights (10.1 MB)
│   │   │   └── vocabulary.json       # TextCNN vocabulary mapping
│   │   └── results/
│   │       ├── evaluation.json       # TextCNN test evaluation report
│   │       ├── model_comparison.json # Side-by-side comparative metrics
│   │       ├── confusion_matrix.png  # TextCNN confusion matrix
│   │       └── training_curves.png   # TextCNN loss and accuracy curves
│   ├── emotion/
│   │   ├── artifacts/
│   │   │   └── emotion_bilstm_attention.pt # Trained Emotion BiLSTM weights
│   │   └── results/
│   │       ├── evaluation.json       # Emotion evaluation report
│   │       └── confusion_matrix.png  # Emotion 7-class confusion matrix
│   └── vae/
│       ├── artifacts/
│       │   ├── vae_model.pt          # Trained VAE weights
│       │   ├── vae_vectorizer.pkl    # Fitted TF-IDF vectorizer
│       │   └── vae_scaler.pkl        # Fitted standard scaler
│       └── results/
│           └── unusual_reviews.csv   # Detected anomalies
```

---

## 7. Installation & Local Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/govindshaw67/SentixAI.git
cd SentixAI

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Running the Application

### Option 1: Streamlit Analytical Dashboard (Recommended)
Launch the comprehensive multi-tab dashboard:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```
- Open **`http://localhost:8501`** in your browser.
- **Tab 1: Inference & XAI**: Select between `BiLSTM + Attention` and `TextCNN`, inspect sentiment probabilities, emotion classification, attention heatmaps, LIME, and SHAP.
- **Tab 2: Batch CSV Processing**: Upload a CSV with a `review` column and select the processing architecture.
- **Tab 3: Model Analytics**: View convergence loss curves, test confusion matrices, and detailed classification reports.
- **Tab 4: Architecture & Theory**: Inspect side-by-side empirical performance comparison tables and network diagrams.
- **Tab 6: Unusual Review Detection**: Scan reviews for anomalies using the Variational Autoencoder.

---

### Option 2: FastAPI Backend & Classic Web Dashboard
Start the production Uvicorn ASGI server:

```bash
source .venv/bin/activate
uvicorn app:app --port 8000 --reload
```
- **Web Interface**: Open **`http://localhost:8000`** in your browser.
- **Interactive Swagger Docs**: Open **`http://localhost:8000/docs`** to test endpoints interactively.

---

### Option 3: Command-Line Interface (CLI)
Run instant predictions from your terminal:

```bash
# Test TextCNN prediction
python predict_textcnn.py "The cinematography was breathtaking and the plot was engaging."

# Re-run empirical evaluation on the 4,959 test set
python evaluate_comparison.py
```

---

## 9. API Reference

### `POST /api/predict`
Predict sentiment and emotion with on-demand explainability.

**Request Payload:**
```json
{
  "text": "The camera is excellent but the battery life is terrible.",
  "model_type": "textcnn",
  "include_emotion": true,
  "methods": ["attention", "lime", "shap"],
  "lime_samples": 60,
  "shap_samples": 40
}
```
*Note: `model_type` accepts `"bilstm"` (default) or `"textcnn"`.*

**Response:**
```json
{
  "sentiment": {
    "label": "negative",
    "confidence": 0.8421,
    "probabilities": {
      "negative": 0.8421,
      "positive": 0.1579
    },
    "model_used": "TextCNN"
  },
  "sentiment_confidence": 0.8421,
  "model_type": "TextCNN",
  "emotion": {
    "label": "disappointment",
    "confidence": 0.8124,
    "probabilities": { ... }
  },
  "explanation": {
    "sentiment": [
      {
        "method": "lime",
        "predicted_class": "negative",
        "important_words": [
          {"token": "terrible", "contribution": 0.412},
          {"token": "excellent", "contribution": -0.218}
        ]
      }
    ]
  }
}
```

### `POST /api/batch`
Upload a CSV file containing a `review` column (up to 5,000 rows). Returns processed classifications and download URL.

### `GET /health`
Returns system status and model readiness:
```json
{
  "status": "ok",
  "sentiment_model_bilstm": true,
  "sentiment_model_textcnn": true,
  "emotion_model": true
}
```

### `GET /api/analytics`
Returns held-out test metrics, class names, and training convergence histories for both models.

---

## 10. Training Pipelines

### Train TextCNN Model
```bash
python train_textcnn.py \
  --csv "/path/to/IMDB Dataset.csv" \
  --output-dir outputs/textcnn \
  --epochs 6 \
  --batch-size 64 \
  --learning-rate 0.001
```

### Train BiLSTM-Attention Model
```bash
python train_sentixai.py \
  --csv "/path/to/IMDB Dataset.csv" \
  --output-dir outputs \
  --epochs 6 \
  --batch-size 64
```

### Train Emotion Classifier
```bash
python prepare_goemotions.py --output data/goemotions_target_emotions.csv
python train_sentixai.py \
  --csv data/goemotions_target_emotions.csv \
  --text-column text \
  --label-column emotion \
  --output-dir outputs/emotion \
  --model-name emotion_bilstm_attention.pt
```

### Train VAE Anomaly Detector
```bash
python train_vae.py \
  --csv "/path/to/IMDB Dataset.csv" \
  --output-dir outputs/vae \
  --epochs 10 \
  --latent-dim 32
```

---

## 11. Deployment Guide

### Deploying to Streamlit Community Cloud (Recommended)
Streamlit Cloud offers seamless native PyTorch execution with persistent containers:
1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"** and configure:
   - **Repository:** `govindshaw67/SentixAI`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **"Deploy"**. The app will be live with a permanent URL.

### Deploying with Docker / Render
The project includes an optimized Dockerfile using CPU-only PyTorch to fit within cloud free tiers:
```bash
docker build -t sentixai .
docker run -p 8000:8000 sentixai
```

---

## 12. Methodological Note: Domain & Sequence Length Awareness

In natural language processing, models inherit characteristics from their training corpora:
- **Long-Form Review Domain**: Both BiLSTM and TextCNN are trained on IMDB movie critiques (average length: ~230 words). They excel at extracting sentiment from complete reviews (e.g., *"The performances were magnificent and the visuals stunned"* $\to$ 99.8% Positive).
- **Short Conversational Fragments**: For isolated personal statements without review context (e.g., *"i am happy man"*), first-person phrasing (`"i am..."`) carries a statistical association with review complaints in the movie domain. 
- **Complementary Roles**: For short personal emotional expressions, users can refer to the **Emotion Classifier** (trained on conversational GoEmotions), which accurately identifies expressions like *"i am happy man"* as **`joy` (97.89% confidence)**.

---

## 13. License & Acknowledgments

This project is released under the **MIT License**.

- **IMDB Dataset**: Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*.
- **GoEmotions Dataset**: Demszky et al. (2020), *GoEmotions: A Dataset of Fine-Grained Emotions*.
- **Convolutional Architecture**: Kim (2014), *Convolutional Neural Networks for Sentence Classification*.
- **Explainability**: Ribeiro et al. (2016) for LIME; Lundberg & Lee (2017) for SHAP.

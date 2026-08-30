# VISTA-AI: Complete Pipeline Walkthrough

A simple, step-by-step guide to run the VISTA-AI multimodal CSAT classification framework (`microsoft/wavlm-base-plus` + `all-mpnet-base-v2` + Cross-Modal Attention) on Apple Silicon (`mps`).

---

## 1. Environment Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -U yt-dlp

# 3. Ensure required directories exist
mkdir -p data/raw data/audio/youtube/captions data/audio/out data/features models
```

---

## 2. The 4 Target CSAT Categories

| Index | Category Name | Emotional & Resolution Criteria |
| :---: | :--- | :--- |
| **0** | **Very Unsatisfied** | Happy / strong emotion, problem solved (`promoter_delighted`) |
| **1** | **Unsatisfied** | Flat emotion, problem not solved (`at_risk_dissatisfied`) |
| **2** | **Satisfied** | Flat emotion, problem solved (`standard_resolved`) |
| **3** | **Very Satisfied** | Strong angry / shouting, problem not solved (`urgent_follow_up`) |

---

## 3. Data Ingestion & Feature Extraction

### Option A: Ingest Real YouTube Playlists (Download + Captions + B-Roll Removal + Features)
```bash
python scripts/ingest_youtube_dataset.py
```
- Downloads 16kHz audio and `.vtt` captions from YouTube.
- Strips B-roll intros, outro promos, and music.
- Extracts 768-d WavLM acoustic prosody + 768-d MPNet text semantics into `data/features/yt_*.pt`.

### Option B: Unified Feature Extraction from All Local Audio (`data/audio/out` + `data/audio/youtube`)
```bash
python scripts/04_extract_features.py
```
- Processes all synthesized dialogues (`data/audio/out/*.wav` $\to$ `data/features/dial_*.pt`).
- Processes all downloaded YouTube calls (`data/audio/youtube/*.wav` $\to$ `data/features/yt_*.pt`).


### (Optional) Benchmark Speech-to-Text Accuracy
```bash
python scripts/evaluate_asr.py
```

---

## 4. Model Training (Side-by-Side V1 & V2 Comparison)

Train both the **V1 Baseline** (Linear + Mean Pooling) and **V2 Upgraded** (`MLPResBlock` + `[CLS]` Token) on Apple Silicon GPU (`mps`) with strict conversation-level isolation:

```bash
python src/train.py --model_version both --epochs 25 --batch_size 16 --lr 2e-4
```
- **Strict Isolation:** Zero-leakage conversation splits (278 Train / 48 Test samples, 5 held-out real-world YouTube videos).
- **Benchmark Results:**
  - V1 Baseline: **100.0% Train Acc**, **93.8% Test Acc**, **40.0% Held-out YouTube Acc** (`models/dual_transformer_v1_weights.pt`).
  - V2 Upgraded: **100.0% Train Acc**, **91.7% Test Acc**, **40.0% Held-out YouTube Acc** (`models/dual_transformer_v2_weights.pt`).


---

## 5. Side-by-Side Inference & Deep Debug Telemetry

### Run CLI Prediction Test with Telemetry
```bash
python scripts/05_test_youtube.py
```
- Concurrently runs both **V1 Baseline** and **V2 Upgraded** models.
- Outputs comparative prediction tables, confidence deltas ($\Delta\%$), and deep telemetry (embedding norms $\|\mathbf{a}\|_2, \|\mathbf{t}\|_2$, raw logits, and segment timestamps).

### Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
- Open `http://localhost:8501`.
- Displays dual-model prediction cards, side-by-side probability charts, and an expandable interactive debug telemetry inspector.





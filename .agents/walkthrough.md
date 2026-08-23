# VISTA-AI: Complete Pipeline Walkthrough

Welcome to the end-to-end guide for running VISTA-AI. This guide will walk you through setting up the environment, synthesizing audio data, extracting multimodal features (ASR + CNN), training the model, and running inference on a real-world YouTube video.

---

## 1. Environment Setup

Before running anything, ensure your virtual environment is activated and dependencies are installed.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt
pip install -U yt-dlp  # Ensure yt-dlp is updated for YouTube extraction

# 3. Create necessary directories
mkdir -p data/raw data/audio/tmp data/audio/out data/features models
```

---

## 2. Generating Synthetic Audio Data

Since we start with text dialogues (`data/raw/dialogues.jsonl`), we first need to convert these transcripts into stereo `.wav` audio files using the ElevenLabs Text-to-Speech API.

> [!IMPORTANT]
> Ensure you have your `ELEVENLABS_API_KEY` set up in a `.env` file in the root of the project.

```bash
python scripts/03_synthesize_audio.py
```
**What happens here:**
- The script reads the simulated dialogues.
- It concurrently calls the TTS API to synthesize both customer and engineer voices.
- It mixes the voices into a stereo `.wav` file where the Customer is on the Left Channel and the Engineer is on the Right Channel.
- The output files will be saved in `data/audio/out/`.

---

## 3. Feature Extraction (Whisper ASR + Mel-Spectrogram)

Now we process the raw `.wav` audio files. We use a **Hybrid CNN Architecture** that extracts spatial-temporal acoustic features and deep linguistic embeddings.

```bash
python scripts/04_extract_features.py
```
**What happens here:**
- **Local ASR:** The script loads the OpenAI Whisper (`small.en`) model.
- **Chunking:** It dynamically transcribes the audio into semantic segments/chunks.
- **Linguistic Extraction:** It extracts a 768-dimensional text embedding for each chunk using `SentenceTransformers`.
- **Acoustic Extraction:** It generates a Log-Mel Spectrogram for each chunk, padded/truncated to 10 seconds.
- The combined tensors are saved as `.pt` files in `data/features/`.

> [!NOTE]
> Whisper extraction can be compute-intensive. On an Apple Silicon M-series chip, this might take ~5 seconds per dialogue.

---

## 4. Model Training

With the chunked features extracted, we can train the **Y-Shaped Hybrid CNN** model.

```bash
python src/train.py
```
**What happens here:**
- The script automatically loads the chunks and dynamically pads the sequences `(Batch, Segments, 1, 128, 312)` using `pad_collate`.
- The `YShapedHybridCNN` processes every segment independently and uses **Mean Pooling** to aggregate the chunks into a final prediction.
- It trains for 10 epochs and saves the model weights to `models/hybrid_cnn_weights.pt`.

---

## 5. End-to-End Validation (YouTube Video)

To prove the robustness of our architecture, we can test it directly on an unseen real-world video. We've prepared a script that downloads an "Angry Customer Complaint" video from YouTube.

```bash
python scripts/05_test_youtube.py
```
**What happens here:**
1. Uses `yt-dlp` to download the audio track from the YouTube URL.
2. Runs the local Whisper model to transcribe the audio and generate chunks.
3. Passes the chunks into the trained model weights.
4. Outputs the final prediction probabilities for the 4 CSAT categories (e.g., `at_risk_dissatisfied`).

> [!TIP]
> If you test with only a few mock examples, the model might not predict anger correctly. Train it on the entire dataset (307+ files) for optimal classification performance!

---

## 6. Interactive Web UI (Demo)

We have built a beautiful **Streamlit Web Application** to interactively demonstrate the full pipeline.

To start the UI:
```bash
streamlit run app.py
```

Then, open your browser to `http://localhost:8501`. 

**In the UI, you can:**
- Paste any YouTube link.
- Watch as it dynamically downloads the audio and extracts Whisper semantic chunks.
- Read the full Whisper transcript in an expandable box.
- View a beautiful probability distribution chart of the CSAT prediction!

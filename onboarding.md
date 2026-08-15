# VISTA-AI Onboarding Guide

Welcome to the VISTA-AI Multimodal Bullying Detection project! This guide will help you set up your local environment, download the dataset, and run our state-of-the-art **Asymmetric Multimodal Inference** training pipeline from scratch.

## Prerequisites
- Python 3.8+
- A Hugging Face account with an Access Token (needed for dataset downloads)

## Step 1: Install Dependencies
Navigate to the project root directory and install the required Python packages:

```bash
# Navigate to the project root
cd /path/to/VISTA-AI

# Install dependencies
pip install -r requirements.txt
```
*(Note: Because we use a local HuggingFace Transformer model, installing PyTorch and Transformers is required and will download a ~1.5GB model file upon first run).*

## Step 2: Set up Environment Variables
We use Hugging Face to host and version-control our real-world audio dataset. You will need your API token to download it.

1. Create a file named `.env` in the root directory (`VISTA-AI/.env`).
2. Add your Hugging Face API token to the file like this:
```env
hugging_face_api_token=hf_your_actual_token_here
```

## Step 3: Download the Dataset
We have a custom synchronization script that automatically queries our private Hugging Face repository and pulls the most recent dataset branch.

Run the following command from the project root:
```bash
python scripts/hf_dataset_sync.py --download
```
*Note: To save massive amounts of bandwidth and storage, this script only downloads the raw, uncut `data/audio/` files. The ephemeral `chunks/` directory is explicitly ignored during synchronization.*

## Step 4: The 5-Step Retraining Pipeline
VISTA-AI uses an **Asymmetric Inference Pipeline**. We chunk the audio into 15-second sliding windows for the Acoustic SVM (Model 1) to pinpoint aggression spikes, but we feed the *entire, uncut video transcript* into a local Transformer (Model 2) to maintain perfect semantic context. The Meta-Classifier (Model 3) then fuses these localized acoustic and global semantic scores.

To train this architecture, you must run the following 5 scripts in order:

```bash
# 1. Generate local acoustic chunks (Since HF download skips them)
python scripts/chunk_audio.py 

# 2. Train the Acoustic SVM Model 1
python src/train.py

# 3. Transcribe the FULL raw video files for Model 2 (Using Whisper)
python scripts/transcribe_audio.py

# 4. Generate Semantic Context Scores (Using Local BART Zero-Shot)
python scripts/text_classifier.py

# 5. Train the Meta-Classifier Fusion Model 3
python src/train_fusion.py
```
*Model artifacts (like the trained SVM, Fusion models, and JSON predictions) will be saved to the `models/` directory.*

## Step 5: Test the Pipeline
You can test the fully trained multimodal architecture in two ways:

### A. Command Line Timeline Testing
Run the testing script on any raw `.mp4`, `.mov`, or `.wav` file. It will automatically extract the audio using `moviepy`, analyze it, and print out a chronological timeline of when bullying occurred:
```bash
python scripts/test_video.py --video /path/to/test_video.mp4
```

### B. Live Streamlit Dashboard
If you prefer a visual interface, you can launch our Streamlit web application. Simply upload your raw video file to the dashboard to see the sliding-window analysis!
```bash
streamlit run app.py
```

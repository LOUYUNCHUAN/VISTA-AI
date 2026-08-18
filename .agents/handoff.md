# VISTA-AI: Project Handoff & Architecture Guide

Welcome to the VISTA-AI repository! This document serves as the master blueprint for the project to help human collaborators and future AI agents immediately understand the architecture, the directory structure, and the sequential execution workflows.

## 1. Project Overview
VISTA-AI is a multimodal PyTorch deep learning framework designed to predict Customer Satisfaction (CSAT) scores by combining **linguistic text semantics** and **acoustic prosody features** directly from conversational audio. 

The architecture is built from scratch and relies on:
- **Linguistic Path (Text):** 768-dimensional sentence embeddings (via `all-mpnet-base-v2`).
- **Acoustic Path (Audio):** 25-dimensional Low-Level Descriptors extracted via `opensmile` (eGeMAPSv02), dynamically processed through a custom Bidirectional LSTM.
- **Multimodal Fusion:** A custom Gated Multimodal Unit (GMU) fusing both modalities, optimized with an auxiliary InfoNCE Contrastive Loss to align the embedding spaces before fusion.

## 2. Directory Structure
The repository has been structured according to strict software engineering standards to separate executable scripts from the core deep learning modules:

```text
VISTA-AI/
├── .agents/                    # Workspace rules and handoff context
├── .env                        # Environment variables (e.g., ELEVENLABS_API_KEY, HF_TOKEN)
├── data/                       # (Auto-generated) Datasets
│   ├── audio/out/              # Final synthesized WAV files (stereo)
│   ├── features/               # Extracted PyTorch tensors (.pt) ready for training
│   └── raw/                    # Raw JSONL transcripts and metadata
├── scripts/                    # Sequential Data Processing Pipeline
│   ├── 01_generate_scripts.py  # Generates initial CSAT dialogues via Gemini
│   ├── 02_augment_scripts.py   # Expands the dataset to 500+ samples
│   ├── 03_synthesize_audio.py  # Converts JSON transcripts to TTS audio via ElevenLabs
│   ├── 04_extract_features.py  # Extracts OpenSMILE & Text embeddings from audio/JSON
│   └── hf_dataset_sync.py      # Hugging Face upload/download utility
└── src/                        # Core PyTorch Framework
    ├── data/
    │   └── dataset.py          # Custom PyTorch Dataset (RealCSATDataset) and collate_fn
    ├── models/
    │   └── architecture.py     # Bi-LSTM, GMU, InfoNCELoss, and the main Classifier
    └── train.py                # Main training loop (Run via `python src/train.py`)
```

## 3. The Execution Workflow
If you are pulling this repository fresh, you do **not** need to generate the synthetic data from scratch. 

### Step 1: Download the Data
The full dataset (JSON, WAV files, and PyTorch Tensors) is hosted securely on Hugging Face. Download it directly into the `data/` folder by running:
```bash
python scripts/hf_dataset_sync.py download --repo-id Vista-AI/CustomerServiceAudio
```
*(Ensure your `HF_TOKEN` in the `.env` file has read access if the repo is private).*

### Step 2: Train the Model
Once the `data/features/` folder is populated with `.pt` files, you can immediately begin iterating on the model architecture and training:
```bash
python src/train.py
```

## 4. Active To-Dos / Next Steps
- The model currently achieves ~99.6% training accuracy on the synthetic dataset, proving the custom Bi-LSTM and GMU architectures converge perfectly.
- **Future Work:** Evaluate generalization on a hidden validation set, tune the contrastive loss `alpha` weighting parameter (currently `0.5`), or experiment with different OpenSMILE functional sets.

import os
import json
import torch
import soundfile as sf
import numpy as np
import whisper
from transformers import AutoFeatureExtractor, WavLMModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

AUDIO_DIR = "data/audio/out"
YOUTUBE_AUDIO_DIR = "data/audio/youtube"
JSON_PATH = "data/raw/dialogues.jsonl"
YOUTUBE_METADATA_PATH = "data/youtube_metadata.jsonl"
OUTPUT_DIR = "data/features"

LABEL_MAP = {
    # Canonical labels (0 to 3)
    "very_unsatisfied": 0,
    "unsatisfied": 1,
    "satisfied": 2,
    "very_satisfied": 3,
    # Legacy aliases
    "urgent_follow_up": 3,
    "at_risk_dissatisfied": 1,
    "standard_resolved": 2,
    "promoter_delighted": 0
}

def extract_features_from_audio(audio_path, sr_target=16000, whisper_model=None, wavlm_processor=None, wavlm_model=None, text_model=None, device="cpu"):
    """
    Transcribes audio with Whisper, segments dialogue turns, and extracts 768-d WavLM + MPNet features.
    """
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim > 1:
        customer_audio = audio_data[:, 0]
    else:
        customer_audio = audio_data
        
    customer_audio_fp32 = customer_audio.astype(np.float32)
    transcription = whisper_model.transcribe(customer_audio_fp32)
    
    audio_embeds = []
    chunked_text_embeds = []
    
    for segment in transcription["segments"]:
        seg_text = segment["text"].strip()
        if not seg_text:
            continue
            
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        seg_audio = customer_audio_fp32[start_sample:end_sample]
        
        if len(seg_audio) < 160:
            continue
            
        # WavLM Acoustic Prosody
        inputs = wavlm_processor(seg_audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = wavlm_model(input_values)
            a_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            
        # Text Semantics
        t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
        
        audio_embeds.append(a_emb)
        chunked_text_embeds.append(t_emb)
        
    if not audio_embeds:
        audio_embeds.append(torch.zeros(768))
        chunked_text_embeds.append(torch.zeros(768))
        
    return torch.stack(audio_embeds), torch.stack(chunked_text_embeds)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select Device (Prioritize Apple Silicon MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU Acceleration (MPS) for feature extraction.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU Acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # 1. Initialize Feature Extractors
    print("Loading Text Encoder (all-mpnet-base-v2 for 768-d embeddings)...")
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("Loading WavLM Model (microsoft/wavlm-base-plus for 768-d acoustic prosody)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    print("Loading Whisper ASR model (small.en)...")
    whisper_model = whisper.load_model("small.en")
    
    # 2. Process Synthesized Audio (data/audio/out/*.wav)
    dataset = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                dataset[data["dialogue_id"]] = LABEL_MAP[data["action_label"]]
                
    syn_wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]) if os.path.exists(AUDIO_DIR) else []
    print(f"\n📂 Step 1: Processing {len(syn_wav_files)} Synthesized Audio Files from '{AUDIO_DIR}'...")
    
    processed_syn = 0
    for wav_file in tqdm(syn_wav_files, desc="Synthesized Audio"):
        dialogue_id = wav_file.replace(".wav", "")
        if dialogue_id not in dataset:
            continue
            
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        audio_emb, text_emb = extract_features_from_audio(
            wav_path, whisper_model=whisper_model, wavlm_processor=wavlm_processor,
            wavlm_model=wavlm_model, text_model=text_model, device=device
        )
        
        label_tensor = torch.tensor(dataset[dialogue_id], dtype=torch.long)
        torch.save({
            "audio_embeds": audio_emb,
            "text_embeds": text_emb,
            "label": label_tensor
        }, os.path.join(OUTPUT_DIR, f"{dialogue_id}.pt"))
        processed_syn += 1
        
    # 3. Process Real-World YouTube Audio (data/audio/youtube/*.wav)
    yt_metadata = {}
    if os.path.exists(YOUTUBE_METADATA_PATH):
        with open(YOUTUBE_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                yt_metadata[rec["video_id"]] = rec["label"]
                
    yt_wav_files = sorted([f for f in os.listdir(YOUTUBE_AUDIO_DIR) if f.endswith(".wav")]) if os.path.exists(YOUTUBE_AUDIO_DIR) else []
    print(f"\n🎥 Step 2: Processing {len(yt_wav_files)} Real-World YouTube Audio Files from '{YOUTUBE_AUDIO_DIR}'...")
    
    processed_yt = 0
    for wav_file in tqdm(yt_wav_files, desc="YouTube Audio"):
        video_id = wav_file.replace(".wav", "")
        if video_id not in yt_metadata:
            continue
            
        wav_path = os.path.join(YOUTUBE_AUDIO_DIR, wav_file)
        audio_emb, text_emb = extract_features_from_audio(
            wav_path, whisper_model=whisper_model, wavlm_processor=wavlm_processor,
            wavlm_model=wavlm_model, text_model=text_model, device=device
        )
        
        label_tensor = torch.tensor(yt_metadata[video_id], dtype=torch.long)
        torch.save({
            "audio_embeds": audio_emb,
            "text_embeds": text_emb,
            "label": label_tensor
        }, os.path.join(OUTPUT_DIR, f"yt_{video_id}.pt"))
        processed_yt += 1
        
    print("\n" + "=" * 75)
    print(f"🎉 Unified Feature Extraction Complete!")
    print(f"  • Synthesized Dialogues Processed: {processed_syn}")
    print(f"  • YouTube Dialogues Processed:     {processed_yt}")
    print(f"  • Total Tensors Saved in:          '{OUTPUT_DIR}' ({processed_syn + processed_yt} files)")
    print("=" * 75)

if __name__ == "__main__":
    main()


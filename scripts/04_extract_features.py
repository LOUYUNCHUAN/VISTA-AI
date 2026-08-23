import os
import json
import torch
import soundfile as sf
import librosa
import numpy as np
import whisper
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

AUDIO_DIR = "data/audio/out"
JSON_PATH = "data/raw/dialogues.jsonl"
OUTPUT_DIR = "data/features"

# Mapping labels to integers
LABEL_MAP = {
    "urgent_follow_up": 0,
    "at_risk_dissatisfied": 1,
    "standard_resolved": 2,
    "promoter_delighted": 3
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Initialize Feature Extractors
    print("Loading Text Encoder (all-mpnet-base-v2 for 768-d embeddings)...")
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("Loading Whisper ASR model (small.en)...")
    whisper_model = whisper.load_model("small.en")
    
    # 2. Read JSON to get text and labels
    dataset = {}
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            dialogue_id = data["dialogue_id"]
            
            # Extract only the customer's text (Path A focus on customer intent)
            customer_text = " ".join([t["text"] for t in data["turns"] if t["speaker"] == "customer"])
            
            dataset[dialogue_id] = {
                "text": customer_text,
                "label": LABEL_MAP[data["action_label"]]
            }
            
    # 3. Process generated WAV files
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} generated audio files. Beginning extraction...")
    
    processed_count = 0
    for wav_file in tqdm(wav_files):
        dialogue_id = wav_file.replace(".wav", "")
        if dialogue_id not in dataset:
            continue
            
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        
        # Read the stereo audio
        audio_data, sr = sf.read(wav_path)
        
        # Isolate the Customer Channel (Left Channel / Channel 0)
        customer_audio = audio_data[:, 0]
        
        # --- PATH C: Semantic Chunking via Whisper ASR ---
        # Whisper expects float32
        customer_audio_fp32 = customer_audio.astype(np.float32)
        transcription = whisper_model.transcribe(customer_audio_fp32)
        
        mel_specs = []
        chunked_text_embeds = []
        MAX_FRAMES = 312 # 10 seconds at sr=16000, hop=512
        
        for segment in transcription["segments"]:
            seg_text = segment["text"].strip()
            if not seg_text:
                continue
                
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            seg_audio = customer_audio_fp32[start_sample:end_sample]
            
            if len(seg_audio) == 0:
                continue
                
            # Log-Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=seg_audio, sr=sr, n_mels=128, hop_length=512)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_tensor = torch.tensor(log_mel_spec, dtype=torch.float32)
            
            # Pad or truncate to 10 seconds
            if mel_tensor.shape[1] > MAX_FRAMES:
                mel_tensor = mel_tensor[:, :MAX_FRAMES]
            else:
                pad_amount = MAX_FRAMES - mel_tensor.shape[1]
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_amount), value=0.0)
                
            # Text Embedding
            t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
            
            mel_specs.append(mel_tensor.unsqueeze(0)) # Shape: (1, 128, 312)
            chunked_text_embeds.append(t_emb)
            
        if not mel_specs:
            mel_specs.append(torch.zeros(1, 128, 312))
            chunked_text_embeds.append(torch.zeros(768))
            
        mel_specs_tensor = torch.stack(mel_specs) # Shape: (S, 1, 128, 312)
        chunked_text_tensor = torch.stack(chunked_text_embeds) # Shape: (S, 768)
        
        # --- Save Features ---
        label_tensor = torch.tensor(dataset[dialogue_id]["label"], dtype=torch.long)
        
        output_data = {
            "mel_spec": mel_specs_tensor.cpu(), # Hybrid CNN
            "chunked_text": chunked_text_tensor.cpu(), # Hybrid CNN
            "label": label_tensor
        }
        
        torch.save(output_data, os.path.join(OUTPUT_DIR, f"{dialogue_id}.pt"))
        processed_count += 1
        
    print(f"\\nSuccessfully extracted and saved features for {processed_count} dialogues in '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    main()

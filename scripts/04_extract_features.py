import os
import json
import torch
import soundfile as sf
import opensmile
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
    
    print("Initializing OpenSMILE (eGeMAPSv02 Low-Level Descriptors)...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
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
        
        # --- PATH A: Linguistic Extraction ---
        text = dataset[dialogue_id]["text"]
        text_embedding = text_model.encode(text, convert_to_tensor=True) # (768,)
        
        # --- PATH B: Acoustic Extraction ---
        # Read the stereo audio
        audio_data, sr = sf.read(wav_path)
        
        # Isolate the Customer Channel (Left Channel / Channel 0)
        customer_audio = audio_data[:, 0]
        
        # OpenSMILE requires a 2D array for signal processing: (channels, samples)
        # But wait, opensmile process_signal expects shape (channels, samples) or (samples,)
        # Let's pass it as a 1D array
        acoustic_df = smile.process_signal(customer_audio, sr)
        
        # Convert DataFrame to PyTorch Tensor (frames, 88)
        acoustic_features = torch.tensor(acoustic_df.values, dtype=torch.float32)
        
        # --- Save Features ---
        label_tensor = torch.tensor(dataset[dialogue_id]["label"], dtype=torch.long)
        
        output_data = {
            "text_embed": text_embedding.cpu(),
            "audio_frames": acoustic_features.cpu(),
            "label": label_tensor
        }
        
        torch.save(output_data, os.path.join(OUTPUT_DIR, f"{dialogue_id}.pt"))
        processed_count += 1
        
    print(f"\\nSuccessfully extracted and saved features for {processed_count} dialogues in '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    main()

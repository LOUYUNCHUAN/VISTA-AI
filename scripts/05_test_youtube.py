import os
import sys
import torch
import librosa
import numpy as np
import whisper
import subprocess
from sentence_transformers import SentenceTransformer

# Add src to path so we can import architecture
sys.path.append(os.path.abspath("src"))
from models.architecture import YShapedHybridCNN

def main():
    youtube_url = "https://youtu.be/gD7xQGXpSBg?si=Ok1BDZTkzom_K__6"
    tmp_audio = "data/audio/tmp/youtube.wav"
    
    os.makedirs("data/audio/tmp", exist_ok=True)
    
    if os.path.exists(tmp_audio):
        print(f"1. Audio already exists at {tmp_audio}, skipping download.")
    else:
        print("1. Downloading YouTube Audio...")
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "-ar 16000 -ac 1",
            "-o", tmp_audio.replace(".wav", ""),
            youtube_url
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("\n[ERROR] YouTube blocked the download (HTTP 429 / Bot Protection).")
            print("Since you are a Google One user, you can bypass this by passing your browser cookies to yt-dlp.")
            print("Try running this manual command in your terminal first:")
            print(f"yt-dlp --cookies-from-browser chrome -x --audio-format wav -o data/audio/tmp/youtube {youtube_url}")
            print("(Replace 'chrome' with 'safari' or 'edge' if you use a different browser.)\n")
            sys.exit(1)
    
    print("2. Loading Models...")
    whisper_model = whisper.load_model("small.en")
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YShapedHybridCNN(num_classes=4, text_dim=768).to(device)
    
    weights_path = "models/hybrid_cnn_weights.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Please train the model first.")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    print("3. Transcribing with Whisper & Chunking...")
    audio_data, sr = librosa.load(tmp_audio, sr=16000)
    audio_fp32 = audio_data.astype(np.float32)
    transcription = whisper_model.transcribe(audio_fp32)
    
    print("\n--- WHISPER TRANSCRIPT ---")
    print(transcription["text"])
    print("--------------------------\n")
    
    mel_specs = []
    chunked_text_embeds = []
    MAX_FRAMES = 312 # 10 seconds
    
    for segment in transcription["segments"]:
        seg_text = segment["text"].strip()
        if not seg_text: continue
            
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        seg_audio = audio_fp32[start_sample:end_sample]
        
        if len(seg_audio) == 0: continue
            
        mel_spec = librosa.feature.melspectrogram(y=seg_audio, sr=sr, n_mels=128, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_tensor = torch.tensor(log_mel_spec, dtype=torch.float32)
        
        if mel_tensor.shape[1] > MAX_FRAMES:
            mel_tensor = mel_tensor[:, :MAX_FRAMES]
        else:
            pad_amount = MAX_FRAMES - mel_tensor.shape[1]
            mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_amount), value=0.0)
            
        t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
        
        mel_specs.append(mel_tensor.unsqueeze(0))
        chunked_text_embeds.append(t_emb)
        
    if not mel_specs:
        print("No valid segments found!")
        return
        
    mel_specs_tensor = torch.stack(mel_specs).unsqueeze(0).to(device) # (1, S, 1, 128, 312)
    chunked_text_tensor = torch.stack(chunked_text_embeds).unsqueeze(0).to(device) # (1, S, 768)
    
    print(f"Extracted {mel_specs_tensor.size(1)} chunks.")
    
    print("4. Running Inference...")
    with torch.no_grad():
        logits, _, _ = model(chunked_text_tensor, mel_specs_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
    classes = ["urgent_follow_up", "at_risk_dissatisfied", "standard_resolved", "promoter_delighted"]
    predicted_class = classes[torch.argmax(probs).item()]
    
    print("\n=== PREDICTION RESULTS ===")
    for c, p in zip(classes, probs):
        print(f"{c}: {p.item()*100:.2f}%")
    print(f"\nFINAL PREDICTION: **{predicted_class}**")

if __name__ == "__main__":
    main()

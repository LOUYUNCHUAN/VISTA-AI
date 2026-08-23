import os
import sys
import torch
import librosa
import numpy as np
import whisper
import subprocess
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add src to path so we can import architecture
sys.path.append(os.path.abspath("src"))
from models.architecture import YShapedHybridCNN

st.set_page_config(page_title="VISTA-AI: Multimodal CSAT Demo", layout="wide")

st.title("🎧 VISTA-AI: Customer Complaint Analyzer")
st.markdown("Analyze YouTube customer calls using our **Y-Shaped Hybrid CNN** and **Local Whisper ASR**.")

# -----------------
# 1. LOAD MODELS (Cached to prevent reloading on every UI interaction)
# -----------------
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # 1. Whisper ASR
    whisper_model = whisper.load_model("small.en")
    
    # 2. Text Encoder
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # 3. Multimodal CNN
    hybrid_cnn = YShapedHybridCNN(num_classes=4, text_dim=768)
    weights_path = "models/hybrid_cnn_weights.pt"
    if os.path.exists(weights_path):
        hybrid_cnn.load_state_dict(torch.load(weights_path, map_location=device))
        hybrid_cnn.eval()
    else:
        st.error(f"Weights not found at {weights_path}. Please run training first.")
        
    return whisper_model, text_model, hybrid_cnn

whisper_model, text_model, hybrid_cnn = load_models()

# -----------------
# 2. UI INPUT
# -----------------
youtube_url = st.text_input("Enter YouTube Video URL:", value="https://youtu.be/gD7xQGXpSBg")
run_button = st.button("Analyze Audio")

def get_video_id(url):
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    elif 'youtube.com' in parsed.netloc:
        return urllib.parse.parse_qs(parsed.query).get('v', [None])[0]
    return "unknown_video"

if run_button and youtube_url:
    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()
        
    tmp_audio = f"data/audio/tmp/{video_id}.wav"
    os.makedirs("data/audio/tmp", exist_ok=True)
    
    st.write("---")
    
    # -----------------
    # 3. PIPELINE
    # -----------------
    with st.spinner("Downloading YouTube Audio..."):
        if not os.path.exists(tmp_audio):
            cmd = [
                "yt-dlp",
                "-x", "--audio-format", "wav",
                "--postprocessor-args", "-ar 16000 -ac 1",
                "-o", tmp_audio.replace(".wav", ""),
                youtube_url
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                st.error("YouTube blocked the download (HTTP 429). Try downloading it manually via terminal passing cookies.")
                st.stop()
        else:
            st.info("Audio already exists in cache. Skipping download.")
            
    with st.spinner("Transcribing with Whisper & Semantic Chunking..."):
        # Whisper expects float32
        import soundfile as sf
        audio_data, sr = sf.read(tmp_audio)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
            
        customer_audio_fp32 = audio_data.astype(np.float32)
        transcription = whisper_model.transcribe(customer_audio_fp32)
        
        mel_specs = []
        chunked_text_embeds = []
        MAX_FRAMES = 312 # 10 seconds
        
        full_transcript = ""
        
        for segment in transcription["segments"]:
            seg_text = segment["text"].strip()
            if not seg_text: continue
            
            full_transcript += f"{seg_text} "
            
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            seg_audio = customer_audio_fp32[start_sample:end_sample]
            
            if len(seg_audio) == 0: continue
            
            # Log-Mel
            mel_spec = librosa.feature.melspectrogram(y=seg_audio, sr=sr, n_mels=128, hop_length=512)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_tensor = torch.tensor(log_mel_spec, dtype=torch.float32)
            
            if mel_tensor.shape[1] > MAX_FRAMES:
                mel_tensor = mel_tensor[:, :MAX_FRAMES]
            else:
                pad_amount = MAX_FRAMES - mel_tensor.shape[1]
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_amount), value=0.0)
                
            # Text Embed
            t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
            
            mel_specs.append(mel_tensor.unsqueeze(0))
            chunked_text_embeds.append(t_emb)
            
        st.success(f"Extracted {len(mel_specs)} dynamic semantic chunks!")
        
    with st.expander("View Whisper Transcript"):
        st.write(full_transcript)
        
    # -----------------
    # 4. INFERENCE
    # -----------------
    with st.spinner("Running Hybrid CNN Inference..."):
        if not mel_specs:
            st.error("No valid audio segments found.")
            st.stop()
            
        mel_specs_tensor = torch.stack(mel_specs).unsqueeze(0) # (1, S, 1, 128, 312)
        chunked_text_tensor = torch.stack(chunked_text_embeds).unsqueeze(0) # (1, S, 768)
        
        with torch.no_grad():
            logits, _, _ = hybrid_cnn(chunked_text_tensor, mel_specs_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        classes = ["Urgent Follow-Up", "At-Risk/Dissatisfied", "Standard/Resolved", "Promoter/Delighted"]
        
        # Build DataFrame for Chart
        df = pd.DataFrame({
            "Probability": [p.item() * 100 for p in probs],
            "Category": classes
        }).set_index("Category")
        
        pred_idx = torch.argmax(probs).item()
        final_prediction = classes[pred_idx]
        
    # -----------------
    # 5. RESULTS DISPLAY
    # -----------------
    st.markdown("### Model Prediction")
    st.metric(label="Final CSAT Status", value=final_prediction)
    
    st.markdown("### Probability Distribution")
    st.bar_chart(df)

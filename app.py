import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os
import sys

# Try loading video tools
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    pass

import whisper
import google.generativeai as genai
from dotenv import load_dotenv
from src.inference import run_asymmetric_inference

load_dotenv()

st.set_page_config(page_title="VISTA-AI Asymmetric Inference", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background-color: #1E2329; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    .safe-row { background-color: rgba(116, 198, 157, 0.2); padding: 5px; border-radius: 5px; margin: 2px 0; }
    .warning-row { background-color: rgba(255, 143, 163, 0.4); padding: 5px; border-radius: 5px; margin: 2px 0; border: 1px solid #ff8fa3; }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Asymmetric Inference Dashboard")
st.write("Upload a full-length raw Audio or Video file. The system will analyze the full conversational context, and run a sliding window to detect physical acoustic aggression.")

@st.cache_resource
def load_models():
    models_dir = 'models'
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    fusion_path = os.path.join(models_dir, 'fusion_model.joblib')
    
    svm_pipeline = joblib.load(svm_path) if os.path.exists(svm_path) else None
    fusion_model = joblib.load(fusion_path) if os.path.exists(fusion_path) else None
    
    whisper_model = None
    try: whisper_model = whisper.load_model("base")
    except: pass
        
    gemini_model = None
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-3.6-flash')
        
    return svm_pipeline, fusion_model, whisper_model, gemini_model

svm_pipeline, fusion_model, whisper_model, gemini_model = load_models()

tab1, tab2 = st.tabs(["Live Test / Upload", "Model Benchmarking"])

with tab1:
    uploaded_file = st.file_uploader("Upload a file (.wav, .mp3, .mp4, .mov)", type=["wav", "mp3", "mp4", "mov"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        is_video = filename.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))
        
        # Save temp file
        temp_path = "temp_upload" + os.path.splitext(filename)[1]
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
            
        audio_path = temp_path
        if is_video:
            st.video(temp_path)
            st.info("Extracting audio from video...")
            audio_path = "temp_audio.wav"
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(temp_path)
                video.audio.write_audiofile(audio_path, logger=None)
            except Exception as e:
                st.error(f"Failed to extract video audio: {e}")
        else:
            st.audio(temp_path)
            
        if st.button("Run Asymmetric Inference Pipeline"):
            with st.spinner("Processing... (this may take a minute)"):
                results = run_asymmetric_inference(
                    audio_path=audio_path,
                    svm_pipeline=svm_pipeline,
                    fusion_model=fusion_model,
                    whisper_model=whisper_model,
                    gemini_model=gemini_model,
                    chunk_duration=15,
                    overlap=5
                )
                
            st.subheader("📝 Semantic Analysis (Global Context)")
            st.info(f'"{results["transcript"]}"')
            st.metric("Global NLP Probability", f"{results['global_text_prob']*100:.1f}%")
            
            st.subheader("⏱️ Acoustic Timeline (Sliding Window)")
            
            detected = False
            for segment in results["timeline"]:
                start_m, start_s = int(segment["start"] // 60), int(segment["start"] % 60)
                end_m, end_s = int(segment["end"] // 60), int(segment["end"] % 60)
                ts = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
                
                f_prob = segment['fusion_prob'] * 100
                a_prob = segment['audio_prob'] * 100
                
                if segment["is_bullying"]:
                    detected = True
                    st.markdown(f'<div class="warning-row">🔴 [{ts}] <b>Bullying Detected</b> | Final Conf: {f_prob:.1f}% | (Acoustic: {a_prob:.1f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-row">🟢 [{ts}] Safe | Final Conf: {f_prob:.1f}% | (Acoustic: {a_prob:.1f}%)</div>', unsafe_allow_html=True)
                    
            if detected:
                st.error("🚨 BULLYING DETECTED IN THIS FILE!")
            else:
                st.success("✅ No bullying detected in this file.")
                
            # Cleanup
            if os.path.exists(temp_path): os.remove(temp_path)
            if is_video and os.path.exists(audio_path): os.remove(audio_path)

with tab2:
    st.write("Model Benchmarking requires loading metrics.json. Please see `src/train.py`.")

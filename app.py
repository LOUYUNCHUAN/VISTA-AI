import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
from src.features import extract_features

load_dotenv()

st.set_page_config(page_title="Bullying & Anomaly Detection", layout="wide")

# Modern UI Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .safe-alert {
        padding: 20px;
        background-color: #1b4332;
        color: #74c69d;
        border-radius: 10px;
        font-size: 24px;
        text-align: center;
        font-weight: bold;
    }
    .warning-alert {
        padding: 20px;
        background-color: #641220;
        color: #ff8fa3;
        border-radius: 10px;
        font-size: 24px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .metric-card {
        background-color: #1E2329;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Acoustic School Bullying & Anomaly Detection System")
st.write("Hybrid Multimodal Anomaly Detection (Acoustic SVM + Semantic NLP)")

# Load models and metrics
@st.cache_resource
def load_models():
    models_dir = 'models'
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    fusion_path = os.path.join(models_dir, 'fusion_model.joblib')
    
    svm_pipeline = joblib.load(svm_path) if os.path.exists(svm_path) else None
    fusion_model = joblib.load(fusion_path) if os.path.exists(fusion_path) else None
    
    whisper_model = None
    try:
        whisper_model = whisper.load_model("base")
    except Exception as e:
        print(f"Whisper load error: {e}")
        
    gemini_model = None
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Gemini configure error: {e}")
        
    return svm_pipeline, fusion_model, whisper_model, gemini_model

@st.cache_data
def load_metrics():
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

svm_pipeline, fusion_model, whisper_model, gemini_model = load_models()
metrics = load_metrics()

tab1, tab2 = st.tabs(["Live Demo (Inference)", "Model Benchmarking"])

with tab1:
    st.header("Live Test / Upload")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    with col2:
        st.write("Or record live audio (requires mic permissions)")
        recorded_audio = st.audio_input("Record Audio")
        
    audio_source = uploaded_file if uploaded_file else recorded_audio
    
    if audio_source is not None:
        st.audio(audio_source)
        
        # Save temp file for librosa and whisper
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_source.read())
            
        st.subheader("Multimodal Real-time Inference")
        
        if svm_pipeline and fusion_model:
            # --- MODEL 1: ACOUSTIC ---
            audio_prob = 0.0
            features = extract_features(temp_audio_path)
            if features is not None:
                scaler = svm_pipeline['scaler']
                svm_model = svm_pipeline['model']
                label_map = svm_pipeline['label_map']
                feat_scaled = scaler.transform([features])
                
                if hasattr(svm_model.steps[-1][1], "predict_proba"):
                    probs = svm_model.predict_proba(feat_scaled)[0]
                    audio_prob = probs[label_map.get('bullying', 1)]
                else:
                    dec = svm_model.decision_function(feat_scaled)[0]
                    audio_prob = 1 / (1 + np.exp(-dec))
            
            # --- MODEL 2: SEMANTIC ---
            text_prob = 0.0
            transcript_text = ""
            
            if whisper_model and gemini_model:
                with st.spinner("Transcribing audio (Whisper)..."):
                    res = whisper_model.transcribe(temp_audio_path, fp16=False)
                    transcript_text = res["text"].strip()
                
                with st.spinner("Analyzing semantics (Gemini)..."):
                    if transcript_text:
                        prompt = f"""
                        Analyze if this speech indicates bullying, violence, harassment, or aggression.
                        Transcript: "{transcript_text}"
                        Output strictly JSON: {{"prediction": 1 or 0, "probability": float between 0.0 and 1.0}}
                        """
                        try:
                            response = gemini_model.generate_content(prompt).text.strip()
                            if response.startswith("```json"): response = response[7:]
                            if response.endswith("```"): response = response[:-3]
                            j = json.loads(response.strip())
                            text_prob = float(j.get("probability", 0.0))
                        except:
                            text_prob = 0.0
            else:
                st.warning("NLP Pipeline offline (Missing Whisper or Gemini API Key).")
                
            # --- MODEL 3: FUSION ---
            X_fusion = np.array([[audio_prob, text_prob]])
            final_prob = fusion_model.predict_proba(X_fusion)[0][1]
            
            # --- UI DISPLAY ---
            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown(f'<div class="metric-card"><h3>🎙️ Model 1: Acoustic</h3><h1 style="color:#4cc9f0">{audio_prob*100:.1f}%</h1><p>SVM Output</p></div>', unsafe_allow_html=True)
            with colB:
                st.markdown(f'<div class="metric-card"><h3>📝 Model 2: Semantic</h3><h1 style="color:#7209b7">{text_prob*100:.1f}%</h1><p>Whisper + Gemini</p></div>', unsafe_allow_html=True)
            with colC:
                st.markdown(f'<div class="metric-card"><h3>⚖️ Model 3: Fusion</h3><h1 style="color:#f72585">{final_prob*100:.1f}%</h1><p>Meta-Classifier</p></div>', unsafe_allow_html=True)
            
            st.write(f"**Transcript:** {transcript_text if transcript_text else '(No speech detected or pipeline offline)'}")
            st.write("---")
            
            if final_prob > 0.5:
                st.markdown(f'<div class="warning-alert">⚠️ WARNING: BULLYING DETECTED (Confidence: {final_prob*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✅ SAFE: Normal Activity (Bullying Prob: {final_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                
        else:
            st.error("Model pipeline not found. Run `src/train.py` and `src/train_fusion.py` first.")
            
        st.subheader("Acoustic Visualization")
        try:
            y, sr = librosa.load(temp_audio_path, sr=22050)
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            fig.patch.set_facecolor('#0E1117')
            
            ax[0].set_facecolor('#0E1117')
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#00b4d8')
            ax[0].set_title("Waveform", color="white")
            ax[0].tick_params(colors='white')
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            ax[1].set_facecolor('#0E1117')
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
            ax[1].set_title("Spectrogram", color="white")
            ax[1].tick_params(colors='white')
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            ax[2].set_facecolor('#0E1117')
            img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax[2], cmap='viridis')
            ax[2].set_title("MFCCs", color="white")
            ax[2].tick_params(colors='white')
            fig.colorbar(img2, ax=ax[2])
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization error: {e}")

with tab2:
    st.header("Model Benchmarking")
    if metrics:
        models = list(metrics.keys())
        f1_scores = [metrics[m]['F1-Score'] for m in models]
        acc_scores = [metrics[m]['Accuracy'] for m in models]
        
        st.write("### Performance Metrics Comparison")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor('#0E1117')
        ax2.set_facecolor('#0E1117')
        
        x = np.arange(len(models))
        width = 0.35
        ax2.bar(x - width/2, f1_scores, width, label='F1-Score', color='#7209b7')
        ax2.bar(x + width/2, acc_scores, width, label='Accuracy', color='#4cc9f0')
        
        ax2.set_ylabel('Scores', color='white')
        ax2.set_title('F1-Score and Accuracy by Model', color='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, color='white')
        ax2.tick_params(colors='white')
        ax2.legend()
        st.pyplot(fig2)
        
        st.write("### Confusion Matrices")
        cols = st.columns(2)
        import seaborn as sns
        for i, (name, m_data) in enumerate(metrics.items()):
            col = cols[i % 2]
            with col:
                fig3, ax3 = plt.subplots(figsize=(4,3))
                sns.heatmap(m_data['Confusion_Matrix'], annot=True, cmap='Blues', fmt='d', ax=ax3, cbar=False)
                ax3.set_title(name)
                st.pyplot(fig3)
    else:
        st.warning("Metrics not found. Train the models first.")

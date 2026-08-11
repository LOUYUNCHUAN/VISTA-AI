import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os
from src.features import extract_features

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
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Acoustic School Bullying & Anomaly Detection System")
st.write("Language-agnostic anomaly detection using classical ML.")

# Load models and metrics
@st.cache_resource
def load_pipeline():
    pipeline_path = 'models/svm_model.joblib'
    if os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    return None

@st.cache_data
def load_metrics():
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

pipeline = load_pipeline()
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
        
        # Save temp file for librosa
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_source.read())
            
        st.subheader("Real-time Feature Visualization")
        try:
            y, sr = librosa.load("temp_audio.wav", sr=22050)
            
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            fig.patch.set_facecolor('#0E1117')
            
            # Waveform
            ax[0].set_facecolor('#0E1117')
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#00b4d8')
            ax[0].set_title("Waveform", color="white")
            ax[0].tick_params(colors='white')
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            ax[1].set_facecolor('#0E1117')
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
            ax[1].set_title("Spectrogram", color="white")
            ax[1].tick_params(colors='white')
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            ax[2].set_facecolor('#0E1117')
            img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax[2], cmap='viridis')
            ax[2].set_title("MFCCs", color="white")
            ax[2].tick_params(colors='white')
            fig.colorbar(img2, ax=ax[2])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Inference
            if pipeline:
                st.subheader("Prediction Output")
                features = extract_features("temp_audio.wav")
                if features is not None:
                    scaler = pipeline['scaler']
                    model = pipeline['model']
                    label_map = pipeline['label_map']
                    rev_map = {v: k for k, v in label_map.items()}
                    
                    feat_scaled = scaler.transform([features])
                    pred_idx = model.predict(feat_scaled)[0]
                    probs = model.predict_proba(feat_scaled)[0]
                    
                    pred_label = rev_map[pred_idx]
                    
                    # Get probability of bullying
                    bullying_prob = probs[label_map['bullying']] if 'bullying' in label_map else 0.0
                    
                    if pred_label == 'bullying' or bullying_prob > 0.5:
                        st.markdown(f'<div class="warning-alert">⚠️ WARNING: BULLYING DETECTED (Confidence: {bullying_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert">✅ SAFE: Normal Activity (Bullying Prob: {bullying_prob*100:.1f}%)</div>', unsafe_allow_html=True)
                        
                    st.write(f"**Prediction:** {'Bullying Case' if pred_label == 'bullying' else 'Not Bullying'}")
            else:
                st.error("Model pipeline not found. Train the models first.")
                
        except Exception as e:
            st.error(f"Error processing audio: {e}")

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
        
        st.write("### Analysis: Why SVM Outperforms")
        st.info("Support Vector Machines (SVM) with an RBF kernel excel in this acoustic context because they map high-dimensional, non-linear audio features (like our MFCCs, ZCR, and Spectral Contasts) into a space where they become linearly separable. Logistic Regression struggles with this non-linearity, and KNN can be susceptible to noise in high dimensions. The MLP requires far more data to generalize effectively compared to SVM's margin-maximization approach.")
        
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

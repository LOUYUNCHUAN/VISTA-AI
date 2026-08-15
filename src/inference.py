import os
import json
import librosa
import numpy as np
from src.features import extract_features_from_array

def run_asymmetric_inference(audio_path, svm_pipeline, fusion_model, whisper_model, nlp_classifier, chunk_duration=15, overlap=5):
    """
    Runs the Asymmetric Inference Engine on a long audio file.
    - Model 2 (NLP) gets the FULL uncut audio context.
    - Model 1 (SVM) gets 15-second sliding windows.
    - Model 3 (Fusion) fuses them per window.
    """
    results = {
        "transcript": "",
        "global_text_prob": 0.0,
        "timeline": []
    }
    
    # ---------------------------------------------------------
    # 1. SEMANTIC PIPELINE (FULL CONTEXT)
    # ---------------------------------------------------------
    if whisper_model and nlp_classifier:
        try:
            print("Transcribing full audio with Whisper...")
            res = whisper_model.transcribe(audio_path, fp16=False)
            transcript_text = res["text"].strip()
            results["transcript"] = transcript_text
            
            if transcript_text:
                print("Analyzing full context with local Transformer...")
                labels = ["aggressive bullying and targeted harassment", "friends joking around and casual conversation"]
                res = nlp_classifier(transcript_text, candidate_labels=labels)
                bully_idx = res["labels"].index("aggressive bullying and targeted harassment")
                prob = res["scores"][bully_idx]
                results["global_text_prob"] = float(prob)
        except Exception as e:
            print(f"Error in Semantic Pipeline: {e}")

    global_text_prob = results["global_text_prob"]

    # ---------------------------------------------------------
    # 2. ACOUSTIC PIPELINE & 3. FUSION (SLIDING WINDOW)
    # ---------------------------------------------------------
    if not svm_pipeline or not fusion_model:
        print("Missing SVM or Fusion models. Cannot run acoustic analysis.")
        return results
        
    print("Running Acoustic Sliding Window Analysis...")
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        print(f"Error loading audio for librosa: {e}")
        return results

    chunk_samples = int(chunk_duration * sr)
    step_samples = int((chunk_duration - overlap) * sr)
    
    scaler = svm_pipeline['scaler']
    svm_model = svm_pipeline['model']
    label_map = svm_pipeline['label_map']
    
    # Process sliding windows
    start_sample = 0
    while start_sample < len(y):
        end_sample = min(start_sample + chunk_samples, len(y))
        
        # If the chunk is too small, break (unless it's the only chunk)
        if (end_sample - start_sample) < (3 * sr) and start_sample > 0:
            break
            
        chunk_y = y[start_sample:end_sample]
        
        # We need a modified extract_features that takes an array, not a file path
        # Let's extract features directly
        features = extract_features_from_array(chunk_y, sr)
        
        audio_prob = 0.0
        if features is not None:
            feat_scaled = scaler.transform([features])
            if hasattr(svm_model.steps[-1][1], "predict_proba"):
                probs = svm_model.predict_proba(feat_scaled)[0]
                audio_prob = probs[label_map.get('bullying', 1)]
            else:
                dec = svm_model.decision_function(feat_scaled)[0]
                audio_prob = 1 / (1 + np.exp(-dec))
                
        # Fusion
        X_fusion = np.array([[audio_prob, global_text_prob]])
        final_prob = fusion_model.predict_proba(X_fusion)[0][1]
        
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        
        results["timeline"].append({
            "start": start_sec,
            "end": end_sec,
            "audio_prob": float(audio_prob),
            "text_prob": float(global_text_prob),
            "fusion_prob": float(final_prob),
            "is_bullying": final_prob > 0.5
        })
        
        start_sample += step_samples

    return results

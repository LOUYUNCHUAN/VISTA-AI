import os
import json
import joblib
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import re

# We can reuse load_data to get the original filenames and labels
from src.train import load_data

def get_original_filename(chunk_filename):
    """
    Extracts the original raw video/audio filename from a chunk filename.
    e.g. 'bullying_fight1_chunk000.wav' -> 'fight1.wav'
    """
    if chunk_filename.startswith("bullying_"):
        base = chunk_filename[len("bullying_"):]
    elif chunk_filename.startswith("not_bullying_"):
        base = chunk_filename[len("not_bullying_"):]
    else:
        base = chunk_filename
        
    match = re.search(r"^(.*)_chunk\d+\.wav$", base)
    if match:
        return match.group(1) + ".wav"
    return chunk_filename

def get_audio_probabilities(svm_pipeline, X):
    """
    Runs the SVM model to get the probability of the positive class (bullying).
    Note: The SVM must have been trained with probability=True. 
    If not, we use a fallback to decision_function mapped to [0,1].
    """
    model = svm_pipeline['model']
    scaler = svm_pipeline['scaler']
    
    X_scaled = scaler.transform(X)
    
    # Try predict_proba first (requires probability=True in SVC)
    if hasattr(model.steps[-1][1], "predict_proba"):
        probs = model.predict_proba(X_scaled)
        # Class 1 is usually bullying
        return probs[:, 1]
    else:
        # Fallback: normalize decision function
        dec = model.decision_function(X_scaled)
        # Sigmoid to convert distance to probability roughly
        probs = 1 / (1 + np.exp(-dec))
        return probs

def build_fusion_dataset(X, y, filenames, svm_pipeline, nlp_preds, label_map):
    """
    Creates the [Audio_Prob, Text_Prob] feature matrix for the Fusion Model.
    """
    audio_probs = get_audio_probabilities(svm_pipeline, X)
    
    X_fusion = []
    y_fusion = []
    valid_filenames = []
    
    # We assume label_map['bullying'] == 1
    
    for i, filename in enumerate(filenames):
        audio_p = audio_probs[i]
        
        # Get NLP probability (default to 0.0 if not found)
        nlp_p = 0.0
        orig_name = get_original_filename(filename)
        if orig_name in nlp_preds:
            nlp_p = nlp_preds[orig_name].get("probability", 0.0)
            
        X_fusion.append([audio_p, nlp_p])
        y_fusion.append(y[i])
        valid_filenames.append(filename)
        
    return np.array(X_fusion), np.array(y_fusion), valid_filenames

def main():
    models_dir = os.path.join(project_root, 'models')
    
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    nlp_path = os.path.join(models_dir, 'nlp_predictions.json')
    
    if not os.path.exists(svm_path) or not os.path.exists(nlp_path):
        print("Error: Missing base models. Ensure train.py and text_classifier.py have been run.")
        sys.exit(1)
        
    print("Loading Base Models...")
    svm_pipeline = joblib.load(svm_path)
    label_map = svm_pipeline['label_map']
    
    with open(nlp_path, 'r') as f:
        nlp_preds = json.load(f)
        
    # Load chunks
    train_dir = os.path.join(project_root, 'data', 'chunks', 'train')
    test_dir = os.path.join(project_root, 'data', 'chunks', 'test')
    
    print("\nExtracting Audio Features...")
    X_train, y_train, _, train_filenames = load_data(train_dir)
    X_test, y_test, _, test_filenames = load_data(test_dir)
    
    print("\nBuilding Fusion Datasets...")
    X_fusion_train, y_train, _ = build_fusion_dataset(X_train, y_train, train_filenames, svm_pipeline, nlp_preds, label_map)
    X_fusion_test, y_test, test_filenames = build_fusion_dataset(X_test, y_test, test_filenames, svm_pipeline, nlp_preds, label_map)
    
    print("\nTraining Meta-Classifier (Logistic Regression)...")
    fusion_model = LogisticRegression(random_state=42)
    fusion_model.fit(X_fusion_train, y_train)
    
    print("Weights Learned:")
    print(f" - Audio Weight: {fusion_model.coef_[0][0]:.4f}")
    print(f" - Text Weight: {fusion_model.coef_[0][1]:.4f}")
    print(f" - Intercept: {fusion_model.intercept_[0]:.4f}")
    
    print("\nEvaluating Fusion Model on Test Set...")
    y_pred = fusion_model.predict(X_fusion_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Fusion Model -> F1: {f1:.4f} | Acc: {acc:.4f}")
    
    print("\n--- Detailed Test Results (Fusion Model) ---")
    
    # Load transcripts to get the recognised words
    transcripts_path = os.path.join(project_root, 'data', 'transcripts.json')
    try:
        with open(transcripts_path, 'r') as f:
            transcripts_dict = json.load(f)
    except Exception:
        transcripts_dict = {}

    reverse_map = {v: k for k, v in label_map.items()}
    misclassified = 0
    for i in range(len(y_test)):
        true_label = reverse_map[y_test[i]]
        pred_label = reverse_map[y_pred[i]]
        audio_p = X_fusion_test[i][0]
        text_p = X_fusion_test[i][1]
        filename = test_filenames[i]
        orig_name = get_original_filename(filename)
        recognised_text = transcripts_dict.get(orig_name, "N/A")
        
        status = "❌ MISCLASSIFIED" if y_test[i] != y_pred[i] else "✅ CORRECT"
        if y_test[i] != y_pred[i]:
            misclassified += 1
            
        print(f"{status}: {filename}")
        print(f"   True: {true_label} | Predicted: {pred_label}")
        print(f"   Model 1 (Audio Prob): {audio_p:.2f}")
        print(f"   Model 2 (Text Prob):  {text_p:.2f}")
        print(f"   Recognised Words:     \"{recognised_text}\"\n")
            
    if misclassified == 0:
        print("✅ No chunks were misclassified by the Fusion Model!")
    else:
        print(f"Total Misclassified: {misclassified} out of {len(y_test)}")
        
    # Save the fusion model
    fusion_path = os.path.join(models_dir, 'fusion_model.joblib')
    joblib.dump(fusion_model, fusion_path)
    print(f"\nSaved Fusion Model to {fusion_path}")

if __name__ == "__main__":
    main()

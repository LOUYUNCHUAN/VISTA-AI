import os
import joblib
import librosa
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.features import extract_features

def main():
    models_dir = os.path.join(project_root, 'models')
    joblib_path = os.path.join(models_dir, 'svm_model.joblib')
    
    if not os.path.exists(joblib_path):
        print(f"Error: Could not find model at {joblib_path}")
        return
        
    print("Loading best SVM model pipeline...")
    pipeline = joblib.load(joblib_path)
    scaler = pipeline['scaler']
    model = pipeline['model']
    label_map = pipeline['label_map']
    
    # Reverse label map to get class names
    reverse_map = {v: k for k, v in label_map.items()}
    
    test_dir = os.path.join(project_root, 'data', 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Directory {test_dir} does not exist.")
        return
        
    files = sorted([f for f in os.listdir(test_dir) if not f.startswith('.')])
    
    print("\n--- INFERENCE ON USER TEST DATA ---\n")
    for file in files:
        file_path = os.path.join(test_dir, file)
        
        # Determine expected ground truth from filename
        expected = "Unknown"
        lower_name = file.lower()
        if "fight" in lower_name or "riot" in lower_name or "bully" in lower_name or "argument" in lower_name:
            expected = "bullying_conflict"
        elif "quiet" in lower_name or "control" in lower_name or "normal" in lower_name:
            expected = "ambient_noise / playful_banter"
            
        print(f"Processing: {file}")
        print(f"Expected  : {expected}")
        
        try:
            feat = extract_features(file_path)
            if feat is not None:
                feat_scaled = scaler.transform([feat])
                pred_idx = model.predict(feat_scaled)[0]
                prediction = reverse_map[pred_idx]
                
                # Highlight if correct/incorrect conceptually
                status = "✅" if prediction in expected or (expected == "Unknown") else "❌"
                print(f"Predicted : {prediction} {status}")
            else:
                print("Predicted : FAILED to extract features")
        except Exception as e:
            print(f"Error     : {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()

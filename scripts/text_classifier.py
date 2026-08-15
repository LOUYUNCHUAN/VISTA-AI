import os
import json
import argparse
import sys
from tqdm import tqdm

try:
    from transformers import pipeline
except ImportError:
    print("Error: transformers is not installed. Please run: pip install transformers torch")
    sys.exit(1)

def classify_text(classifier, transcript):
    if not transcript or len(transcript.strip()) < 2:
        return {"prediction": 0, "probability": 0.0}
        
    labels = ["aggressive bullying and targeted harassment", "friends joking around and casual conversation"]
    try:
        # Zero-shot classification automatically assigns probabilities to the candidate labels
        result = classifier(transcript, candidate_labels=labels)
        
        # Extract the probability for the "bullying" label
        bully_idx = result["labels"].index("aggressive bullying and targeted harassment")
        prob = result["scores"][bully_idx]
        
        prediction = 1 if prob > 0.5 else 0
        return {"prediction": prediction, "probability": float(prob)}
    except Exception as e:
        print(f"Error classifying: {e}")
        return {"prediction": 0, "probability": 0.0}

def main():
    print("Loading Local Transformer Model (BART Zero-Shot)... This may take a moment.")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcripts_path = os.path.join(project_root, 'data', 'transcripts.json')
    
    if not os.path.exists(transcripts_path):
        print(f"Error: {transcripts_path} not found. Run transcribe_audio.py first.")
        sys.exit(1)
        
    with open(transcripts_path, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
        
    print(f"Loaded {len(transcripts)} transcripts.")
    print("Running Semantic Analysis locally...")
    
    results = {}
    
    for filename, transcript in tqdm(transcripts.items()):
        prediction = classify_text(classifier, transcript)
        results[filename] = prediction

    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    output_path = os.path.join(models_dir, 'nlp_predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Semantic Classification complete! Saved to {output_path}")

if __name__ == "__main__":
    main()

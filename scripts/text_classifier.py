import os
import json
import argparse
import time
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai is not installed. Please run: pip install google-generativeai python-dotenv")
    import sys
    sys.exit(1)

# Load environment variables
load_dotenv()

def classify_text(model, transcript):
    """
    Sends the transcript to Gemini and forces it to return JSON
    containing a binary decision and a probability score.
    """
    if not transcript or len(transcript.strip()) < 2:
        return {"prediction": 0, "probability": 0.0} # Not enough text to classify

    prompt = f"""
    You are an expert toxicity and conflict analyzer. 
    Analyze the following transcribed speech and determine if it indicates bullying, harassment, violence, fighting, or aggressive targeted insults.
    
    Transcription: "{transcript}"
    
    Respond ONLY in valid JSON format with exactly two fields:
    - "prediction": 1 if it is bullying/conflict, 0 if it is normal/safe.
    - "probability": A float between 0.0 and 1.0 indicating your confidence that it is bullying/conflict.
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up markdown if Gemini wrapped it in ```json ... ```
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        result = json.loads(text.strip())
        
        # Validate schema
        if "prediction" in result and "probability" in result:
            return result
        else:
            print(f"Unexpected JSON schema from Gemini: {result}")
            return {"prediction": 0, "probability": 0.0}
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"prediction": 0, "probability": 0.0}

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    
    # We use Flash because it is insanely fast and cheap, perfectly suited for this
    model = genai.GenerativeModel('gemini-1.5-flash')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcripts_path = os.path.join(project_root, 'data', 'transcripts.json')
    
    if not os.path.exists(transcripts_path):
        print(f"Error: {transcripts_path} not found. Run transcribe_audio.py first.")
        sys.exit(1)
        
    with open(transcripts_path, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
        
    print(f"Loaded {len(transcripts)} transcripts.")
    print("Connecting to Gemini API for Semantic Analysis...")
    
    results = {}
    
    from tqdm import tqdm
    for filename, transcript in tqdm(transcripts.items()):
        # Rate limiting protection (Flash is fast, but we should be polite)
        time.sleep(0.1) 
        
        prediction = classify_text(model, transcript)
        results[filename] = prediction

    # Save to JSON
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    output_path = os.path.join(models_dir, 'nlp_predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Semantic Classification complete! Saved to {output_path}")

if __name__ == "__main__":
    main()

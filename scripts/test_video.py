import os
import sys
import argparse
import joblib
import json

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Error: moviepy is not installed. Please run: pip install moviepy")
    sys.exit(1)
    
try:
    import whisper
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError:
    print("Error: openai-whisper or google-generativeai is not installed.")
    sys.exit(1)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.inference import run_asymmetric_inference

def extract_audio_from_video(video_path, audio_out_path):
    print(f"Extracting audio from {os.path.basename(video_path)}...")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_out_path, logger=None)
        return True
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        return False

def load_models():
    print("Loading Models (this may take a few seconds)...")
    load_dotenv()
    
    models_dir = os.path.join(project_root, 'models')
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    fusion_path = os.path.join(models_dir, 'fusion_model.joblib')
    
    if not os.path.exists(svm_path) or not os.path.exists(fusion_path):
        print("Error: SVM or Fusion models not found. Please train them first.")
        sys.exit(1)
        
    svm_pipeline = joblib.load(svm_path)
    fusion_model = joblib.load(fusion_path)
    
    whisper_model = whisper.load_model("base")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-3.6-flash')
    
    return svm_pipeline, fusion_model, whisper_model, gemini_model

def main():
    parser = argparse.ArgumentParser(description="Test a full video using Asymmetric Multimodal Inference.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video or audio file to test.")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: File '{args.video}' not found.")
        sys.exit(1)
        
    ext = os.path.splitext(args.video)[1].lower()
    is_video = ext in ['.mp4', '.mov', '.mkv', '.avi']
    
    audio_path = args.video
    if is_video:
        audio_path = os.path.join(project_root, "temp_test_audio.wav")
        if not extract_audio_from_video(args.video, audio_path):
            sys.exit(1)
            
    svm_pipeline, fusion_model, whisper_model, gemini_model = load_models()
    
    print("\nStarting Asymmetric Inference Pipeline...")
    results = run_asymmetric_inference(
        audio_path=audio_path,
        svm_pipeline=svm_pipeline,
        fusion_model=fusion_model,
        whisper_model=whisper_model,
        gemini_model=gemini_model,
        chunk_duration=15,
        overlap=5
    )
    
    print("\n" + "="*50)
    print("                TEST RESULTS                ")
    print("="*50)
    
    print("\n📝 FULL CONVERSATION TRANSCRIPT:")
    print(f'"{results["transcript"]}"')
    
    print(f"\n🧠 SEMANTIC (GLOBAL) PROBABILITY: {results['global_text_prob']*100:.1f}%")
    
    print("\n⏱️ TIMELINE ANALYSIS (SLIDING WINDOW):")
    detected = False
    for segment in results["timeline"]:
        start_m = int(segment["start"] // 60)
        start_s = int(segment["start"] % 60)
        end_m = int(segment["end"] // 60)
        end_s = int(segment["end"] % 60)
        
        timestamp = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
        f_prob = segment['fusion_prob'] * 100
        a_prob = segment['audio_prob'] * 100
        
        status_icon = "🔴 WARNING" if segment["is_bullying"] else "🟢 Safe   "
        if segment["is_bullying"]:
            detected = True
            
        print(f"[{timestamp}] {status_icon} | Final Prob: {f_prob:5.1f}% | (Acoustic: {a_prob:5.1f}%)")
        
    print("="*50)
    if detected:
        print("🚨 BULLYING DETECTED IN THIS VIDEO!")
    else:
        print("✅ No bullying detected in this video.")
        
    # Cleanup temp audio if we created it
    if is_video and os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    main()

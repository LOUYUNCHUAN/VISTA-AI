import os
import json
import argparse
import warnings

try:
    import whisper
except ImportError:
    print("Error: openai-whisper is not installed. Please run: pip install openai-whisper")
    import sys
    sys.exit(1)

# Suppress FP16 warnings on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def transcribe_directory(model, dir_path, transcripts):
    """
    Scans a directory for .wav files, transcribes them using Whisper,
    and adds the results to the transcripts dictionary.
    """
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return

    files_to_process = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.wav'):
                files_to_process.append((root, filename))
                
    if not files_to_process:
        print(f"No .wav files found in {dir_path}")
        return

    print(f"\nTranscribing {len(files_to_process)} files in {dir_path}...")
    
    from tqdm import tqdm
    for root, filename in tqdm(files_to_process):
        file_path = os.path.join(root, filename)
        
        # Whisper transcription
        try:
            result = model.transcribe(file_path, fp16=False)
            transcripts[filename] = result["text"].strip()
        except Exception as e:
            print(f"Error transcribing {filename}: {e}")
            transcripts[filename] = ""

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio chunks using local Whisper model.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bull_dir = os.path.join(project_root, 'data', 'audio', 'bull')
    notbully_dir = os.path.join(project_root, 'data', 'audio', 'notbully')
    
    print(f"Loading Whisper model '{args.model}' (this may take a moment to download on first run)...")
    model = whisper.load_model(args.model)
    print("Model loaded successfully.")

    transcripts = {}
    
    transcribe_directory(model, bull_dir, transcripts)
    transcribe_directory(model, notbully_dir, transcripts)

    # Save to JSON
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'transcripts.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Transcription complete! Saved {len(transcripts)} transcripts to {output_path}")

if __name__ == "__main__":
    main()

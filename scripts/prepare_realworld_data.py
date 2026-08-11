import os
import librosa
import soundfile as sf
import numpy as np
import shutil
from tqdm import tqdm

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, 'data', 'test')
    raw_dir = os.path.join(project_root, 'data', 'raw')
    
    # Map filenames to the 4 categories
    def get_category(filename):
        lower_name = filename.lower()
        if "quiet" in lower_name:
            return "quiet_conversation"
        elif "normal" in lower_name:
            return "normal_talking"
        elif "fight" in lower_name:
            return "fighting"
        elif "argument" in lower_name or "bully" in lower_name:
            return "argument"
        return None

    # We want 2-second chunks
    sr = 22050
    chunk_length = 2 * sr
    hop_length = 1 * sr  # 1-second overlap
    
    counters = {
        "argument": 0,
        "fighting": 0,
        "normal_talking": 0,
        "quiet_conversation": 0
    }
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return
        
    files = sorted([f for f in os.listdir(test_dir) if not f.startswith('.')])
    
    print(f"Segmenting {len(files)} test files into 2-second chunks for training...")
    
    for file in tqdm(files):
        category = get_category(file)
        if category is None:
            print(f"Skipping {file}: Could not map to a category.")
            continue
            
        file_path = os.path.join(test_dir, file)
        
        try:
            # Load audio
            y, _ = librosa.load(file_path, sr=sr)
            
            # Create overlapping chunks
            for i in range(0, len(y) - chunk_length + 1, hop_length):
                chunk = y[i:i + chunk_length]
                
                # Save chunk
                counters[category] += 1
                out_name = f"{category}_{counters[category]:04d}.wav"
                out_path = os.path.join(raw_dir, out_name)
                sf.write(out_path, chunk, sr)
                
            # Handle remainder if there is any substantial audio left (at least 1 sec)
            remainder = len(y) % hop_length
            last_start = len(y) - chunk_length
            if remainder > sr and last_start + hop_length < len(y):
                chunk = y[-chunk_length:] # take the last 2 seconds
                counters[category] += 1
                out_name = f"{category}_{counters[category]:04d}.wav"
                out_path = os.path.join(raw_dir, out_name)
                sf.write(out_path, chunk, sr)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    print("\nExtraction complete! Dataset distribution:")
    for cat, count in counters.items():
        print(f"- {cat}: {count} chunks")

if __name__ == "__main__":
    main()

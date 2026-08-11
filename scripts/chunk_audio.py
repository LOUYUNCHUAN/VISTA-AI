import os
import argparse
import librosa
import soundfile as sf
import random
from tqdm import tqdm

def chunk_audio_file(file_path, output_dir, prefix, chunk_duration=15, overlap_duration=5):
    try:
        y, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
        
    chunk_samples = int(chunk_duration * sr)
    step_samples = int((chunk_duration - overlap_duration) * sr)
    
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    
    if len(y) < chunk_samples:
        output_path = os.path.join(output_dir, f"{prefix}_{name}_chunk000{ext}")
        sf.write(output_path, y, sr)
        return
        
    chunk_idx = 0
    for start in range(0, len(y) - chunk_samples + 1, step_samples):
        end = start + chunk_samples
        chunk_y = y[start:end]
        output_path = os.path.join(output_dir, f"{prefix}_{name}_chunk{chunk_idx:03d}{ext}")
        sf.write(output_path, chunk_y, sr)
        chunk_idx += 1
        
    if end < len(y) and (len(y) - end) > (3 * sr):
        chunk_y = y[end:]
        output_path = os.path.join(output_dir, f"{prefix}_{name}_chunk{chunk_idx:03d}{ext}")
        sf.write(output_path, chunk_y, sr)

def process_category(category_dir, train_out, test_out, prefix, test_size=0.2, chunk_duration=15, overlap_duration=5):
    if not os.path.exists(category_dir):
        print(f"Missing {category_dir}")
        return
        
    files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
    if not files:
        print(f"No files in {category_dir}")
        return
        
    # Set seed for reproducible splits
    random.seed(42)
    files.sort() # Ensure consistent order before shuffle
    random.shuffle(files)
    
    split_idx = max(1, int(len(files) * (1 - test_size))) # At least 1 training file if possible
    
    # If there's only 1 file, put it in test so we can actually evaluate, wait, if only 1 file, put in train
    if len(files) == 1:
        train_files = files
        test_files = []
    else:
        train_files = files[:split_idx]
        test_files = files[split_idx:]
    
    print(f"  {len(train_files)} files for training, {len(test_files)} files for testing.")
    
    print("  Processing Training files...")
    for f in tqdm(train_files):
        chunk_audio_file(os.path.join(category_dir, f), train_out, prefix, chunk_duration, overlap_duration)
        
    print("  Processing Testing files...")
    for f in tqdm(test_files):
        chunk_audio_file(os.path.join(category_dir, f), test_out, prefix, chunk_duration, overlap_duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_duration", type=int, default=15)
    parser.add_argument("--overlap_duration", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_out = os.path.join(project_root, "data", "chunks", "train")
    test_out = os.path.join(project_root, "data", "chunks", "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    
    print(f"=== Chunking & Splitting Pipeline ===")
    
    print("\n[Category: Bullying]")
    bull_dir = os.path.join(project_root, "data", "audio", "bull")
    process_category(bull_dir, train_out, test_out, "bullying", args.test_size, args.chunk_duration, args.overlap_duration)
    
    print("\n[Category: Not Bullying]")
    notbully_dir = os.path.join(project_root, "data", "audio", "notbully")
    process_category(notbully_dir, train_out, test_out, "not_bullying", args.test_size, args.chunk_duration, args.overlap_duration)
    
    print("\n✅ All chunking and splitting complete!")

import os
import subprocess
import glob

def extract_audio(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    files = [
        "4 Aug 2026 at 12:22 PM.mp4",
        "4 Aug 2026 at 12:25 PM.mp4",
        "4 Aug 2026 at 12:27 PM.mp4",
        "4 Aug 2026 at 12:28 PM.mp4",
        "4 Aug 2026 at 12:29 PM.mp4",
        "4 Aug 2026 at 12:32 PM.mp4",
        "4 Aug 2026 at 12:33 PM.mp4",
        "4 Aug 2026 at 12:34 PM.mp4",
        "4 Aug 2026 at 12:43 PM.mp4",
        "4 Aug 2026 at 12:47 PM.mp4",
        "4 Aug 2026 at 12:48 PM.mp4",
        "4 Aug 2026 at 12:50 PM.mp4",
        "not bully 4 Aug 2026 at 12:46 PM.mp4",
        "not bully 4 Aug 2026 at 12:52 PM.mp4",
        "not bully 4 Aug 2026 at 12:55 PM.mp4"
    ]
    video_files = [os.path.join(video_dir, f) for f in files]
    
    print(f"DEBUG: Using hardcoded list of {len(files)} files to bypass TCC permissions.")

    print(f"Found {len(video_files)} videos. Extracting audio...")
    
    for video_file in video_files:
        filename = os.path.basename(video_file)
        name, _ = os.path.splitext(filename)
        
        # Check if it starts with "not bully" or "no bully" (case insensitive)
        is_not_bully = filename.lower().startswith("not bully") or filename.lower().startswith("no bully")
        
        # Determine strict prefix for train.py
        prefix = "not_bullying_" if is_not_bully else "bullying_"
        
        # Clean up the rest of the name for a nice filename
        clean_name = name.lower().replace("not bully ", "").replace("no bully ", "").replace(" ", "_")
        
        final_output_file = os.path.join(output_dir, f"{prefix}{clean_name}.wav")
        
        print(f"  {filename} -> {os.path.basename(final_output_file)}")
        
        # Run ffmpeg to extract audio as wav
        # -vn: no video
        # -acodec pcm_s16le: standard 16-bit wav
        # -ar 22050: 22.05kHz sample rate (matches librosa)
        # -ac 1: mono channel
        command = [
            "ffmpeg",
            "-i", video_file,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            final_output_file,
            "-y"
        ]
        
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Failed to process {filename}. Is FFmpeg installed? Try: brew install ffmpeg")
            
    print("\n✅ Extraction complete! Files saved to data/raw/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract audio from videos and format for VISTA-AI.")
    parser.add_argument("--video_dir", default="data/test/video", help="Directory containing raw videos")
    parser.add_argument("--output_dir", default="data/raw", help="Directory to save extracted .wav files")
    
    args = parser.parse_args()
    extract_audio(args.video_dir, args.output_dir)

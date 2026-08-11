import os
import subprocess
import glob

def extract_audio():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(project_root, "data", "videos")
    audio_dir = os.path.join(project_root, "data", "audio")
    
    categories = ["bull", "notbully"]
    
    for category in categories:
        input_cat_dir = os.path.join(video_dir, category)
        output_cat_dir = os.path.join(audio_dir, category)
        os.makedirs(output_cat_dir, exist_ok=True)
        
        if not os.path.exists(input_cat_dir):
            print(f"Directory not found: {input_cat_dir}")
            continue
            
        videos = glob.glob(os.path.join(input_cat_dir, "*.mp4")) + glob.glob(os.path.join(input_cat_dir, "*.mpeg"))
        
        print(f"Found {len(videos)} videos in {category}. Extracting...")
        
        for i, video_file in enumerate(videos):
            filename = os.path.basename(video_file)
            name, _ = os.path.splitext(filename)
            clean_name = name.lower().replace(" ", "_").replace("-", "_")
            
            prefix = "bullying" if category == "bull" else "not_bullying"
            final_output_file = os.path.join(output_cat_dir, f"{prefix}_{clean_name}.wav")
            
            print(f"  {filename} -> {os.path.basename(final_output_file)}")
            
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
                print(f"❌ Failed to process {filename}.")
                
    print("\n✅ Extraction complete! Files saved to data/audio/bull and data/audio/notbully")

if __name__ == "__main__":
    extract_audio()

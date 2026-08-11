import os
import subprocess

def extract_new_notbully():
    input_dir = "data/test/video/notbully"
    output_dir = "data/raw/train"
    os.makedirs(output_dir, exist_ok=True)
    
    # We only process the 7 new files (the other 3 were already processed previously)
    new_files = [
        "Create_a_10-second_video_showing_202608060852.mp4",
        "People_joking_inside_restroom_202608060825.mp4",
        "People_shouting_in_restroom_202608060829.mp4",
        "Students_joking_in_bathroom_202608060909.mp4",
        "Students_teasing_each_other_in_202608060915.mp4",
        "一大群人欢呼庆祝_202608051853.mp4",
        "三四个人大声互相开玩笑_202608051856.mp4"
    ]
    
    print(f"Extracting {len(new_files)} new Not Bullying videos...")
    
    for i, filename in enumerate(new_files):
        video_file = os.path.join(input_dir, filename)
        final_output_file = os.path.join(output_dir, f"not_bullying_new_{i}.wav")
        
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
            
    print("\n✅ New extraction complete! Files added to data/raw/train/")

if __name__ == "__main__":
    extract_new_notbully()

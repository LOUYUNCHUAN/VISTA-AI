import os
import shutil
import zipfile
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'raw')
    temp_dir = os.path.join(project_root, 'data', 'temp_ravdess')
    
    # Clean previous raw data
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # URL for RAVDESS Audio Speech Actors 01-24 (approx 400MB)
    zip_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = os.path.join(temp_dir, "ravdess.zip")
    
    print("Downloading RAVDESS dataset from Zenodo (~400MB)... This may take a few minutes.")
    download_file(zip_url, zip_path)
    
    print("Extracting RAVDESS dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    print("Structuring files into categories...")
    count_playful = 0
    count_bullying = 0
    count_ambient = 0
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".wav"):
                # RAVDESS filename format: 03-01-05-01-01-01-01.wav
                # The 3rd identifier is Emotion: 
                # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
                parts = file.split("-")
                if len(parts) < 3:
                    continue
                    
                emotion_code = parts[2]
                
                category = None
                if emotion_code in ["03", "08"]: # Happy, Surprised
                    category = "playful_banter"
                    count_playful += 1
                    dest_name = f"playful_banter_{count_playful:04d}.wav"
                elif emotion_code in ["05", "06", "07"]: # Angry, Fearful, Disgust
                    category = "bullying_conflict"
                    count_bullying += 1
                    dest_name = f"bullying_conflict_{count_bullying:04d}.wav"
                elif emotion_code in ["01", "02"]: # Neutral, Calm
                    category = "ambient_noise"
                    count_ambient += 1
                    dest_name = f"ambient_noise_{count_ambient:04d}.wav"
                    
                if category:
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(data_dir, dest_name)
                    shutil.move(src_path, dest_path)
                    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    print(f"Successfully processed RAVDESS dataset!")
    print(f"Playful Banter: {count_playful} files")
    print(f"Bullying Conflict: {count_bullying} files")
    print(f"Ambient Noise: {count_ambient} files")

if __name__ == "__main__":
    main()

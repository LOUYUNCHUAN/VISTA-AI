import os
import argparse
import glob
from huggingface_hub import HfApi, snapshot_download, login
from dotenv import load_dotenv

# Load environment variables from .env (such as HF_TOKEN)
load_dotenv()

def print_dataset_summary():
    """
    Prints a local inventory of dataset components ready for sync.
    """
    print("\n" + "=" * 70)
    print("📦 LOCAL DATASET INVENTORY")
    print("=" * 70)
    
    syn_wavs = glob.glob("data/audio/out/*.wav")
    yt_wavs = glob.glob("data/audio/youtube/*.wav")
    yt_captions = glob.glob("data/audio/youtube/captions/*.vtt")
    features = glob.glob("data/features/*.pt")
    models = glob.glob("models/*.pt")
    
    print(f"• Synthesized Audio (data/audio/out/):          {len(syn_wavs)} WAV files")
    print(f"• Real-World YouTube Audio (data/audio/youtube/): {len(yt_wavs)} WAV files")
    print(f"• YouTube Captions (data/audio/youtube/captions/): {len(yt_captions)} VTT files")
    print(f"• Multimodal Tensors (data/features/):            {len(features)} PT files (WavLM + MPNet)")
    print(f"• Trained Checkpoints (models/):                  {len(models)} PT weight files")
    print(f"• Metadata Manifests:                             "
          f"{'✅ dialogues.jsonl ' if os.path.exists('data/raw/dialogues.jsonl') else '❌ '}"
          f"{'✅ train_test_split.json ' if os.path.exists('data/train_test_split.json') else '❌ '}"
          f"{'✅ youtube_metadata.jsonl' if os.path.exists('data/youtube_metadata.jsonl') else '❌ '}")
    print("=" * 70 + "\n")

def upload_dataset(repo_id, local_dirs, commit_message=None):
    """
    Uploads the local data and model directories to a Hugging Face Dataset repository.
    """
    print_dataset_summary()
    print(f"🚀 Preparing to upload local folders {local_dirs} to Hugging Face dataset: {repo_id}")
    api = HfApi()

    # Attempt to create the repository if it doesn't already exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        print(f"✅ Verified repository '{repo_id}' exists on Hugging Face.")
    except Exception as e:
        print(f"⚠️ Warning during repo creation/verification: {e}")
        print("Make sure your HF_TOKEN has 'Write' permissions for the organization.")

    commit_desc = commit_message or "Sync multimodal CSAT dataset (synthetic + YouTube + features + models)"

    for local_dir in local_dirs:
        if not os.path.exists(local_dir):
            print(f"⚠️ Directory '{local_dir}' does not exist, skipping.")
            continue
            
        print(f"⏳ Uploading files from '{local_dir}'... This may take a few minutes.")
        
        # Ignore temporary cache and OS clutter while keeping youtube and out audio
        ignore_patterns = ["audio/tmp/*", "*/.DS_Store", "__pycache__/*", "*.tmp"] if local_dir == "data" else ["*/.DS_Store"]
        
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=local_dir,  # Mirrors local directory structure on the Hub
            ignore_patterns=ignore_patterns,
            commit_message=f"{commit_desc} ({local_dir})"
        )
        print(f"✅ Uploaded '{local_dir}' successfully.")
        
    print("\n🎉 All folders uploaded! The full dataset and models are synchronized on Hugging Face.")

def download_dataset(repo_id, local_dirs):
    """
    Downloads the dataset and models from Hugging Face back to the local machine.
    """
    print(f"📥 Downloading dataset and checkpoints from Hugging Face: {repo_id}...")
    
    allow_patterns = [f"{d}/*" for d in local_dirs]
    ignore_patterns = ["data/audio/tmp/*", "*.DS_Store"]
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns
    )
    print(f"✅ Download complete! The dataset and models are ready in your workspace.")
    print_dataset_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Dataset & Model Sync Tool for VISTA-AI")
    parser.add_argument("action", choices=["upload", "download", "status"], help="Choose 'upload', 'download', or 'status'.")
    parser.add_argument("--repo-id", type=str, default="Vista-AI/CustomerServiceAudio", help="Hugging Face Dataset Repo ID")
    parser.add_argument("--dirs", nargs="+", default=["data", "models"], help="Local directories to sync (default: data models)")
    parser.add_argument("--commit-message", type=str, default=None, help="Custom commit message for upload")
    
    args = parser.parse_args()
    
    if args.action == "status":
        print_dataset_summary()
        exit(0)
        
    # Authenticate seamlessly if HF_TOKEN is in the .env file
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("🔐 Authenticated with Hugging Face via HF_TOKEN.")
    else:
        print("⚠️ Warning: No HF_TOKEN found in .env.")
        print("If the dataset is Private, or if you are Uploading, you must either:")
        print("1. Add HF_TOKEN=your_token to your .env file.")
        print("2. Run 'huggingface-cli login' in your terminal.")
        print("-" * 50)

    if args.action == "upload":
        upload_dataset(args.repo_id, args.dirs, commit_message=args.commit_message)
    elif args.action == "download":
        download_dataset(args.repo_id, args.dirs)


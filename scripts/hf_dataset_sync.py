import os
import argparse
from huggingface_hub import HfApi, snapshot_download, login
from dotenv import load_dotenv

# Load environment variables from .env (such as HF_TOKEN)
load_dotenv()

def upload_dataset(repo_id, local_dir):
    """
    Uploads the local data directory to a Hugging Face Dataset repository.
    """
    print(f"🚀 Preparing to upload local '{local_dir}' folder to Hugging Face dataset: {repo_id}")
    api = HfApi()
    

    # Attempt to create the repository if it doesn't already exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        print(f"✅ Verified repository '{repo_id}' exists on Hugging Face.")
    except Exception as e:
        print(f"⚠️ Warning during repo creation/verification: {e}")
        print("Make sure your HF_TOKEN has 'Write' permissions for the Vista-AI organization.")

    # Upload the entire folder directly to the dataset repository
    print(f"⏳ Uploading files from {local_dir}... This may take a few minutes depending on audio file size.")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data"  # Mirrors the local directory structure on the hub
    )
    print("✅ Upload complete! Your partner can now pull the dataset.")

def download_dataset(repo_id, local_dir):
    """
    Downloads the dataset from Hugging Face back to the local machine.
    """
    print(f"📥 Downloading dataset from Hugging Face: {repo_id}...")
    
    # Download the 'data/' folder from the hub into the local current directory
    # (which will safely place it inside the local 'data/' folder)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=".",
        allow_patterns="data/*"
    )
    print(f"✅ Download complete! The dataset is ready in your '{local_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Dataset Sync Tool for Vista-AI")
    parser.add_argument("action", choices=["upload", "download"], help="Choose whether to 'upload' to HF or 'download' from HF.")
    parser.add_argument("--repo-id", type=str, default="Vista-AI/CustomerServiceAudio", help="The Hugging Face Dataset Repository ID (e.g., Vista-AI/dataset-name)")
    parser.add_argument("--dir", type=str, default="data", help="The local directory containing the dataset.")
    
    args = parser.parse_args()
    
    # Authenticate seamlessly if HF_TOKEN is in the .env file
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("🔐 Authenticated with Hugging Face via HF_TOKEN.")
    else:
        print("⚠️ Warning: No HF_TOKEN found in .env.")
        print("If the Vista-AI dataset is Private, or if you are Uploading, you must either:")
        print("1. Add HF_TOKEN=your_token to your .env file.")
        print("2. Run 'huggingface-cli login' in your terminal.")
        print("-" * 50)

    if args.action == "upload":
        upload_dataset(args.repo_id, args.dir)
    elif args.action == "download":
        download_dataset(args.repo_id, args.dir)

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

# Load environment variables from .env
load_dotenv()

def get_api_and_repo():
    token = os.getenv("hugging_face_api_token")
    if not token:
        print("Error: hugging_face_api_token not found in .env file.")
        sys.exit(1)
        
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/vista-ai-dataset"
    return api, repo_id

def upload_dataset():
    api, repo_id = get_api_and_repo()
    
    print(f"Ensuring repository {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    
    # Create branch name based on today's date (YYYY-MM-DD)
    branch_name = datetime.now().strftime("%Y-%m-%d")
    print(f"Creating/using branch: {branch_name}")
    
    try:
        api.create_branch(repo_id=repo_id, branch=branch_name, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Note on branch creation: {e}")
        
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    if not os.path.exists(data_dir):
        print(f"Error: Local data directory '{data_dir}' does not exist. Nothing to upload.")
        sys.exit(1)
        
    print(f"Uploading {data_dir} to {repo_id} on branch {branch_name}...")
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch_name,
        commit_message=f"Dataset backup for {branch_name}"
    )
    print("Upload complete! ✅")

def download_dataset():
    api, repo_id = get_api_and_repo()
    
    print(f"Fetching branches for repository {repo_id}...")
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error accessing repository. Does it exist? {e}")
        sys.exit(1)
        
    branches = [branch.name for branch in refs.branches]
    
    # Filter for date branches
    date_branches = []
    for b in branches:
        try:
            # Validate format YYYY-MM-DD
            datetime.strptime(b, "%Y-%m-%d")
            date_branches.append(b)
        except ValueError:
            continue
            
    if not date_branches:
        print("No date-formatted branches found (e.g., 2026-08-02). Cannot determine latest dataset.")
        sys.exit(1)
        
    # Sort dates and pick the latest
    date_branches.sort(reverse=True)
    latest_branch = date_branches[0]
    
    print(f"Latest branch detected: {latest_branch}")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading dataset from branch '{latest_branch}' into {data_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=latest_branch,
        local_dir=data_dir,
        token=os.getenv("hugging_face_api_token")
    )
    print("Download complete! ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync VISTA-AI dataset with Hugging Face")
    parser.add_argument("--upload", action="store_true", help="Upload local data/ directory to Hugging Face")
    parser.add_argument("--download", action="store_true", help="Download latest dataset from Hugging Face to local data/ directory")
    
    args = parser.parse_args()
    
    if args.upload:
        upload_dataset()
    elif args.download:
        download_dataset()
    else:
        print("Please specify --upload or --download")
        parser.print_help()

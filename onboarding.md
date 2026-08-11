# VISTA-AI Onboarding Guide

Welcome to the VISTA-AI Multimodal Bullying Detection project! This guide will help you set up your local environment, download the dataset, and run the training pipeline for testing purposes.

## Prerequisites
- Python 3.8+
- A Hugging Face account with an Access Token (needed for dataset downloads)

## Step 1: Install Dependencies
Navigate to the project root directory and install the required Python packages:

```bash
# Navigate to the project root
cd /path/to/VISTA-AI

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set up Environment Variables
We use Hugging Face to host and version-control our real-world audio dataset. You will need your API token to download it.

1. Create a file named `.env` in the root directory (`VISTA-AI/.env`).
2. Add your Hugging Face API token to the file like this:
```env
hugging_face_api_token=hf_your_actual_token_here
```

## Step 3: Download the Dataset
We have a custom synchronization script that automatically queries our private Hugging Face repository and pulls the most recent dataset branch.

Run the following command from the project root:
```bash
python scripts/hf_dataset_sync.py --download
```
*Note: This will automatically create a `data/` directory and populate it with the latest real-world audio chunks.*

## Step 4: Run the Training Pipeline
Once the data is downloaded, you can run the audio classification training script. This script will extract the 199-dimensional acoustic features, train our baseline SVM model, and output the validation metrics.

```bash
python src/train.py
```
*Note: Model artifacts (like the trained SVM and feature scaler) will be saved to the `models/` directory, and metrics will be logged.*

## Step 5: Test the User Interface
If you want to test the model on new audio files visually, you can launch our Streamlit web application:

```bash
streamlit run app.py
```

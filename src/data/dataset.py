import os
import glob
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class RealCSATDataset(Dataset):
    """
    Loads the extracted text and acoustic PyTorch tensors from the features directory.
    """
    def __init__(self, features_dir="data/features"):
        self.files = glob.glob(os.path.join(features_dir, "*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No feature files found in {features_dir}! Did you run scripts/04_extract_features.py?")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the dictionary of tensors
        data = torch.load(self.files[idx], map_location="cpu")
        mel_spec = data.get("mel_spec", None) # Shape: (S, 1, 128, 312)
        chunked_text = data.get("chunked_text", None) # Shape: (S, 768)
        label = data["label"]
        return mel_spec, chunked_text, label

def pad_collate(batch):
    """
    Custom collate_fn to pad the varying lengths of acoustic frames.
    """
    mels = [item[0] for item in batch]
    chunked_texts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # mels is list of 4D tensors (S, 1, 128, 312) -> pad to (batch, max_S, 1, 128, 312)
    mels_stacked = None
    chunked_texts_padded = None
    if all(m is not None for m in mels) and all(c is not None for c in chunked_texts):
        mels_stacked = pad_sequence(mels, batch_first=True, padding_value=0.0)
        chunked_texts_padded = pad_sequence(chunked_texts, batch_first=True, padding_value=0.0)
    
    labels_stacked = torch.stack(labels)
    
    return mels_stacked, chunked_texts_padded, labels_stacked

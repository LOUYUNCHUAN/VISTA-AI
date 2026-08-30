import os
import glob
import json
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPLIT_MANIFEST_PATH = "data/train_test_split.json"

def get_or_create_splits(features_dir="data/features", test_ratio=0.15, seed=42):
    """
    Creates or loads a deterministic conversation-level train/test split.
    Guarantees:
    1. Zero data leakage: All chunks from the same dialogue/video belong strictly to train OR test.
    2. YouTube preservation: At least 1 YouTube video per category is held out exclusively for test.
    """
    if os.path.exists(SPLIT_MANIFEST_PATH):
        try:
            with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    all_files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
    if not all_files:
        return {"train": [], "test": []}
        
    random.seed(seed)
    
    # Separate synthetic dialogues vs YouTube videos
    yt_by_class = defaultdict(list)
    syn_by_class = defaultdict(list)
    
    for f in all_files:
        basename = os.path.basename(f)
        data = torch.load(f, map_location="cpu")
        label = data["label"].item() if isinstance(data["label"], torch.Tensor) else int(data["label"])
        
        if basename.startswith("yt_"):
            yt_by_class[label].append(f)
        else:
            syn_by_class[label].append(f)
            
    train_files = []
    test_files = []
    
    # 1. Split YouTube videos: hold out at least 1 video per class for test
    for label, files in yt_by_class.items():
        random.shuffle(files)
        # Hold out 1 video for smaller classes, or ~20% for larger classes
        n_test = max(1, int(len(files) * 0.25))
        test_files.extend(files[:n_test])
        train_files.extend(files[n_test:])
        
    # 2. Split Synthetic dialogues: hold out 15% stratified per class
    for label, files in syn_by_class.items():
        random.shuffle(files)
        n_test = max(1, int(len(files) * test_ratio))
        test_files.extend(files[:n_test])
        train_files.extend(files[n_test:])
        
    split_manifest = {
        "train": sorted(train_files),
        "test": sorted(test_files)
    }
    
    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)
    with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
        
    return split_manifest


class RealCSATDataset(Dataset):
    """
    Loads extracted multimodal PyTorch tensors with strict conversation-level isolation.
    """
    def __init__(self, features_dir="data/features", split="all", test_ratio=0.15, seed=42):
        self.split = split
        splits = get_or_create_splits(features_dir=features_dir, test_ratio=test_ratio, seed=seed)
        
        if split == "train":
            self.files = splits["train"]
        elif split == "test":
            self.files = splits["test"]
        elif split == "all":
            self.files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
        else:
            raise ValueError(f"Unknown split: '{split}'. Must be 'train', 'test', or 'all'.")
            
        if len(self.files) == 0:
            raise RuntimeError(f"No feature files found for split='{split}' in {features_dir}!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        
        # Dual Transformer format
        audio_embeds = data.get("audio_embeds", None)
        text_embeds = data.get("text_embeds", None)
        
        # Legacy Hybrid CNN format fallback
        if audio_embeds is None and "mel_spec" in data:
            audio_embeds = data.get("mel_spec")
        if text_embeds is None and "chunked_text" in data:
            text_embeds = data.get("chunked_text")
            
        label = data["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return text_embeds, audio_embeds, label


def pad_collate(batch):
    """
    Custom collate_fn to pad varying sequence lengths across dialogues and generate padding masks.
    """
    text_embeds_list = [item[0] for item in batch]
    audio_embeds_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    batch_size = len(batch)
    lengths = [t.size(0) for t in text_embeds_list]
    max_len = max(lengths)
    
    # Pad sequences
    text_padded = pad_sequence(text_embeds_list, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audio_embeds_list, batch_first=True, padding_value=0.0)
    
    # Construct boolean padding mask (True where index >= sequence length)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            padding_mask[i, l:] = True
            
    labels_stacked = torch.stack(labels)
    
    return text_padded, audio_padded, padding_mask, labels_stacked



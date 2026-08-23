import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.architecture import YShapedHybridCNN
from data.dataset import RealCSATDataset, pad_collate

def train_model():
    print(f"Initializing Multimodal Architecture: HYBRID_CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the unified model
    model = YShapedHybridCNN(num_classes=4, text_dim=768).to(device)
    
    # 2. Define Loss Functions
    classification_criterion = nn.CrossEntropyLoss()
    
    # 3. Define Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 4. Load Data
    try:
        dataset = RealCSATDataset(features_dir="data/features")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    except Exception as e:
        print(e)
        return
    
    num_epochs = 10
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        # Dataset returns mels (Spectrogram), chunked_texts, labels
        for batch_idx, (mels, chunked_texts, labels) in enumerate(dataloader):
            labels = labels.to(device)
            if mels is not None:
                mels = mels.to(device)
            if chunked_texts is not None:
                chunked_texts = chunked_texts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass & Loss
            if mels is None or chunked_texts is None:
                raise ValueError("mel_spec or chunked_text is None. Run feature extraction again to generate chunked features.")
            logits, _, _ = model(chunked_texts, mels)
            loss = classification_criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            
        epoch_acc = 100. * correct / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")
    print("Training Complete for HYBRID_CNN architecture!")
    
    # Save the model weights
    import os
    os.makedirs("models", exist_ok=True)
    save_path = "models/hybrid_cnn_weights.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    train_model()

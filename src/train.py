import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.architecture import MultimodalCSATClassifier, InfoNCELoss
from data.dataset import RealCSATDataset, pad_collate



def train_model():
    print("Initializing Multimodal Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the unified model
    # OpenSMILE eGeMAPSv02 LLDs have 25 dimensions, not 88
    model = MultimodalCSATClassifier(num_classes=4, text_dim=768, audio_input_dim=25).to(device)
    
    # 2. Define Loss Functions
    classification_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = InfoNCELoss(temperature=0.1).to(device)
    
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
    alpha = 0.5 # Weighting factor for contrastive loss
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        for batch_idx, (text, audio, labels) in enumerate(dataloader):
            text, audio, labels = text.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, proj_audio, proj_text = model(text, audio)
            
            # Calculate Losses
            # Task 1: Classification Loss (Cross Entropy)
            cls_loss = classification_criterion(logits, labels)
            
            # Task 2: Contrastive Alignment Loss (InfoNCE)
            # This forces the network to learn a shared latent space BEFORE the GMU fusion
            align_loss = contrastive_criterion(proj_audio, proj_text)
            
            # Combined Loss
            loss = cls_loss + (alpha * align_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            
        epoch_acc = 100. * correct / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")
        
    print("Training Complete! The GMU and Temporal Prosody network have been trained from scratch.")

if __name__ == "__main__":
    train_model()

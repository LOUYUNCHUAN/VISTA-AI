import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.architecture import DualTransformerClassifier, EnhancedDualTransformerClassifier
from data.dataset import RealCSATDataset, pad_collate

def train_single_model(model_class, model_name, weights_filename, epochs=25, batch_size=16, lr=2e-4):
    print("\n" + "=" * 75)
    print(f"🚀 Training {model_name} on Apple Silicon GPU (MPS)")
    print("=" * 75)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # 1. Instantiate Model
    model = model_class(
        num_classes=4,
        audio_dim=768,
        text_dim=768,
        d_model=512,
        nhead=8,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_dataset = RealCSATDataset(features_dir="data/features", split="train")
    test_dataset = RealCSATDataset(features_dir="data/features", split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for text_embeds, audio_embeds, padding_mask, labels in train_loader:
            text_embeds, audio_embeds = text_embeds.to(device), audio_embeds.to(device)
            padding_mask, labels = padding_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(text_embeds, audio_embeds, padding_mask=padding_mask)
            loss = classification_criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # Test Evaluation
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for text_embeds, audio_embeds, padding_mask, labels in test_loader:
                text_embeds, audio_embeds = text_embeds.to(device), audio_embeds.to(device)
                padding_mask, labels = padding_mask.to(device), labels.to(device)
                
                logits, _, _ = model(text_embeds, audio_embeds, padding_mask=padding_mask)
                loss = classification_criterion(logits, labels)
                
                test_loss += loss.item() * labels.size(0)
                test_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                test_total += labels.size(0)
                
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100.0 * test_correct / test_total
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.1f}% | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:5.1f}% | LR: {current_lr:.6f}")
        
    # Evaluate held-out YouTube videos
    print(f"\n🔍 {model_name} Held-Out YouTube Evaluation:")
    classes = ["Very Unsatisfied", "Unsatisfied", "Satisfied", "Very Satisfied"]
    model.eval()
    yt_correct, yt_total = 0, 0
    with torch.no_grad():
        for f in test_dataset.files:
            if os.path.basename(f).startswith("yt_"):
                data = torch.load(f, map_location="cpu")
                t_emb = data["text_embeds"].unsqueeze(0).to(device)
                a_emb = data["audio_embeds"].unsqueeze(0).to(device)
                lbl = data["label"].item() if isinstance(data["label"], torch.Tensor) else int(data["label"])
                logits, _, _ = model(t_emb, a_emb)
                pred = torch.argmax(logits, dim=1).item()
                is_correct = (pred == lbl)
                if is_correct: yt_correct += 1
                yt_total += 1
                status = "✅" if is_correct else "❌"
                print(f"  {status} [{os.path.basename(f)}] True: {classes[lbl]:<16} | Pred: {classes[pred]:<16}")
                
    yt_acc = (100.0 * yt_correct / yt_total) if yt_total > 0 else 0.0
    print(f"🎯 Held-Out YouTube Accuracy: {yt_correct}/{yt_total} ({yt_acc:.1f}%)")
    
    # Save Model Weights
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", weights_filename)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved weights to {save_path}")
    
    return {
        "name": model_name,
        "train_acc": epoch_train_acc,
        "test_acc": epoch_test_acc,
        "yt_acc": yt_acc
    }

def main():
    parser = argparse.ArgumentParser(description="Train VISTA-AI CSAT Models")
    parser.add_argument("--model_version", choices=["v1", "v2", "both"], default="both", help="Model version to train")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    results = []
    
    if args.model_version in ["v1", "both"]:
        r1 = train_single_model(
            model_class=DualTransformerClassifier,
            model_name="V1 Baseline (Linear + Mean Pooling)",
            weights_filename="dual_transformer_v1_weights.pt",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        results.append(r1)
        
    if args.model_version in ["v2", "both"]:
        r2 = train_single_model(
            model_class=EnhancedDualTransformerClassifier,
            model_name="V2 Upgraded (MLP ResBlock + [CLS] Token)",
            weights_filename="dual_transformer_v2_weights.pt",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        # Also copy as standard default weights
        torch.save(torch.load("models/dual_transformer_v2_weights.pt"), "models/dual_transformer_weights.pt")
        torch.save(torch.load("models/dual_transformer_v2_weights.pt"), "models/hybrid_cnn_weights.pt")
        results.append(r2)
        
    if len(results) > 1:
        print("\n" + "=" * 75)
        print("🏆 SIDE-BY-SIDE ARCHITECTURAL BENCHMARK COMPARISON")
        print("=" * 75)
        print(f"{'Model Architecture':<45} | {'Train Acc':<10} | {'Test Acc':<10} | {'Held-Out YouTube'}")
        print("-" * 75)
        for r in results:
            print(f"{r['name']:<45} | {r['train_acc']:8.1f}% | {r['test_acc']:8.1f}% | {r['yt_acc']:8.1f}%")
        print("=" * 75)

if __name__ == "__main__":
    main()


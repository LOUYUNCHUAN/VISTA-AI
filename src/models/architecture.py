import torch
import torch.nn as nn
import torch.nn.functional as F



class YShapedHybridCNN(nn.Module):
    """
    Hybrid Architecture:
    - Audio: 2D CNN over Log-Mel Spectrograms
    - Text: Pre-trained Transformer embeddings (768-dim) directly inputted
    """
    def __init__(self, num_classes=4, text_dim=768):
        super().__init__()
        
        # Audio 2D CNN Branch (Expects input shape: B, 1, 128, T)
        self.audio_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # Output: (B, 128)
        )
        
        # Shared Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 + text_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_embeds, mel_spec):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - mel_spec: (Batch, Segments, 1, 128, 312)
        """
        B, S, C, H, W = mel_spec.size()
        
        # Flatten Batch and Segments for parallel processing
        mel_spec_flat = mel_spec.view(B * S, C, H, W)
        text_embeds_flat = text_embeds.view(B * S, -1)
        
        h_audio_flat = self.audio_branch(mel_spec_flat) # (B*S, 128)
        
        # 1D Vector Concatenation per segment
        fused_flat = torch.cat([h_audio_flat, text_embeds_flat], dim=1) # (B*S, 896)
        
        # Shared Classification Head per segment
        logits_flat = self.classifier(fused_flat) # (B*S, num_classes)
        
        # Reshape back to sequence
        logits_seq = logits_flat.view(B, S, -1) # (B, S, num_classes)
        
        # Late Aggregation: Mean Pooling across segments
        logits = logits_seq.mean(dim=1) # (B, num_classes)
        
        return logits, None, None # Return None for contrastive projections to match API

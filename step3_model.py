"""
============================================================
STEP 3: EFFICIENTNET-B4 MODEL SETUP
============================================================
- Loads pretrained EfficientNet-B4 from timm
- Replaces classifier head for 5-class DR detection
- Supports Phase 1 (frozen) and Phase 2 (unfrozen)
============================================================
"""

import torch
import torch.nn as nn
import timm
from config import MODEL_NAME, NUM_CLASSES, PRETRAINED, DEVICE


# ─────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────
class DRClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super(DRClassifier, self).__init__()

        # Load EfficientNet-B4 pretrained on ImageNet
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,          # Remove default classifier head
            global_pool='avg'       # Global average pooling
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        # Custom classification head for DR
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)     # 5 output classes (Grade 0-4)
        )

    def forward(self, x):
        features = self.backbone(x)         # Extract features
        output   = self.classifier(features) # Classify
        return output

    def freeze_backbone(self):
        """Phase 1: Freeze backbone, train only classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("  Backbone FROZEN — Training classifier head only")

    def unfreeze_backbone(self):
        """Phase 2: Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone UNFROZEN — Fine-tuning entire model")

    def count_parameters(self):
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen   = total - trainable
        return total, trainable, frozen


# ─────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────
def get_model(phase=1):
    """
    Returns model ready for training.
    phase=1 → backbone frozen (train head only)
    phase=2 → all layers unfrozen (full fine-tune)
    """
    model = DRClassifier()
    model = model.to(DEVICE)

    if phase == 1:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    total, trainable, frozen = model.count_parameters()
    print(f"\n  Model          : EfficientNet-B4")
    print(f"  Total params   : {total:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Frozen         : {frozen:,}")
    print(f"  Device         : {DEVICE}")

    return model


# ─────────────────────────────────────────────────────────
# MAIN — Verify model output shape
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   STEP 3: MODEL SETUP")
    print("=" * 55)

    # Test Phase 1
    print("\n  [PHASE 1 — Frozen Backbone]")
    model = get_model(phase=1)

    # Forward pass with dummy input
    dummy = torch.randn(2, 3, 380, 380).to(DEVICE)
    out   = model(dummy)
    print(f"\n  Input shape    : {dummy.shape}")
    print(f"  Output shape   : {out.shape}  <- should be [2, 5]")
    print(f"  Output classes : {out.shape[1]}  <- should be 5 (Grade 0-4)")

    # Show class outputs
    probs = torch.softmax(out, dim=1)
    print(f"\n  Sample probabilities (random weights):")
    for i, (cls, name) in enumerate({0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}.items()):
        print(f"    Grade {cls} ({name:14s}): {probs[0][i].item():.4f}")

    print("\n  STEP 3 COMPLETE! Next: python step4_train.py")

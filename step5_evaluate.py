"""
============================================================
STEP 5: EVALUATION + CLASSIFICATION REPORT
============================================================
- Loads best trained model
- Runs on test set
- Outputs: Accuracy, Kappa Score, Per-class metrics
- Confusion matrix
- Full classification report (Grade 0-4)
============================================================
"""

import numpy as np
import torch
from torch.cuda.amp import autocast
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_DIR,
    NUM_CLASSES, CLASS_NAMES, MIXED_PRECISION
)
from step2_dataset import get_dataloaders
from step3_model import get_model


# ─────────────────────────────────────────────────────────
# LOAD BEST MODEL
# ─────────────────────────────────────────────────────────
def load_best_model():
    ckpt_path = Path(CHECKPOINT_DIR) / "best_model_phase2.pth"
    if not ckpt_path.exists():
        ckpt_path = Path(CHECKPOINT_DIR) / "best_model_phase1.pth"
        print(f"  Phase 2 checkpoint not found, using Phase 1")

    model     = get_model(phase=2)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Best Val Acc     : {checkpoint['val_acc']:.2f}%")
    return model


# ─────────────────────────────────────────────────────────
# RUN INFERENCE ON TEST SET
# ─────────────────────────────────────────────────────────
def get_predictions(model, loader):
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)

            if MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            probs  = torch.softmax(outputs, dim=1)
            preds  = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


# ─────────────────────────────────────────────────────────
# PRINT CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────
def print_classification_report(preds, labels, probs):
    print("\n" + "=" * 65)
    print("   CLASSIFICATION REPORT — DIABETIC RETINOPATHY")
    print("=" * 65)

    # Overall metrics
    acc   = accuracy_score(labels, preds) * 100
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')

    print(f"\n  Overall Accuracy     : {acc:.2f}%")
    print(f"  Quadratic Kappa      : {kappa:.4f}  (competition metric)")
    print(f"  Kappa Interpretation : {interpret_kappa(kappa)}")

    # Per-class accuracy
    print(f"\n  Per-Class Results:")
    print(f"  {'Grade':<8} {'Class Name':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*68}")

    report = classification_report(
        labels, preds,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )

    for i in range(NUM_CLASSES):
        name = CLASS_NAMES[i]
        r    = report[name]
        print(f"  Grade {i}  {name:<20} "
              f"{r['precision']:>10.3f} "
              f"{r['recall']:>10.3f} "
              f"{r['f1-score']:>10.3f} "
              f"{int(r['support']):>10}")

    print(f"\n  Macro Avg F1   : {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted F1    : {report['weighted avg']['f1-score']:.3f}")

    # Try AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            np.eye(NUM_CLASSES)[labels], probs,
            multi_class='ovr', average='macro'
        )
        print(f"  ROC-AUC (macro): {auc:.4f}")
    except Exception:
        pass

    return acc, kappa


def interpret_kappa(kappa):
    if kappa >= 0.81: return "Almost Perfect"
    if kappa >= 0.61: return "Substantial"
    if kappa >= 0.41: return "Moderate"
    if kappa >= 0.21: return "Fair"
    return "Slight"


# ─────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(preds, labels):
    import os
    os.makedirs(LOG_DIR, exist_ok=True)

    cm       = confusion_matrix(labels, preds)
    cm_norm  = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    class_labels = [f"Grade {i}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.suptitle('DR Classification — Confusion Matrix', fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(LOG_DIR) / "confusion_matrix.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"\n  Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   STEP 5: EVALUATION")
    print("=" * 55)

    _, val_loader, test_loader = get_dataloaders()

    model = load_best_model()

    print("\n  Running inference on test set...")
    preds, labels, probs = get_predictions(model, test_loader)

    acc, kappa = print_classification_report(preds, labels, probs)
    plot_confusion_matrix(preds, labels)

    print("\n" + "=" * 55)
    print("   STEP 5 COMPLETE!")
    print("=" * 55)
    print(f"""
  Final Results:
  Accuracy : {acc:.2f}%
  Kappa    : {kappa:.4f}

  Next Step: python step6_gradcam.py
    """)

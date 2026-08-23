import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score,
    accuracy_score, roc_curve, auc
)
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import sys

# ── PATHS ────────────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\checkpoints\best_model_phase2.pth"
TEST_CSV     = r"C:\Users\deves\OneDrive\Desktop\archive\test.csv"
TEST_IMG_DIR = r"C:\Users\deves\OneDrive\Desktop\archive\test_images\test_images"
LOG_DIR      = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\logs"
MODEL_DIR    = r"C:\Users\deves\OneDrive\Desktop\Model"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 380
BATCH_SIZE  = 16
NUM_WORKERS = 0


# ── Dataset ──────────────────────────────────────────────────
class APTOSTestDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df      = pd.read_csv(csv_path)
        self.img_dir = img_dir

        if "id_code" in self.df.columns:
            self.df = self.df.rename(
                columns={"id_code": "image_id", "diagnosis": "label"})
        elif "image" in self.df.columns:
            self.df = self.df.rename(
                columns={"image": "image_id", "level": "label"})

        self.transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),   # resize all images to same size
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = str(row["image_id"])
        label  = int(row["label"])

        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))

        img = self.transform(image=img)["image"]
        return img, label


# ── Load model ────────────────────────────────────────────────
print("=" * 60)
print("  BINARY CLASSIFICATION — APTOS MODEL")
print("  DR vs No DR")
print("=" * 60)

sys.path.append(MODEL_DIR)
from step3_model import DRClassifier

model = DRClassifier().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  Loaded      : {MODEL_PATH}")
print(f"  Val Acc     : {ckpt['val_acc']:.2f}% (5-class)")
print(f"  Device      : {DEVICE}")

# ── Load test data ────────────────────────────────────────────
dataset = APTOSTestDataset(TEST_CSV, TEST_IMG_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=NUM_WORKERS,
                     pin_memory=True)
print(f"  Test images : {len(dataset):,}")
print(f"  Batches     : {len(loader):,}")
print(f"\n  Running inference...")

# ── Inference ─────────────────────────────────────────────────
all_probs_5class  = []
all_labels_5class = []

with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        with autocast('cuda', enabled=True):
            outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs_5class.append(probs)
        all_labels_5class.extend(labels.numpy())
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} done...")

all_probs_5class  = np.vstack(all_probs_5class)
all_labels_5class = np.array(all_labels_5class)
print(f"  Inference complete!")

# ── Convert to binary ─────────────────────────────────────────
prob_no_dr = all_probs_5class[:, 0]
prob_dr    = all_probs_5class[:, 1:].sum(axis=1)

binary_preds  = (prob_dr >= 0.5).astype(int)
binary_labels = (all_labels_5class > 0).astype(int)

# ── Metrics ───────────────────────────────────────────────────
accuracy    = accuracy_score(binary_labels, binary_preds) * 100
roc_auc     = roc_auc_score(binary_labels, prob_dr)
kappa       = cohen_kappa_score(binary_labels, binary_preds)
cm          = confusion_matrix(binary_labels, binary_preds)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100

print(f"\n  ── BINARY RESULTS ──────────────────────────")
print(f"  Accuracy    : {accuracy:.2f}%")
print(f"  ROC-AUC     : {roc_auc:.4f}")
print(f"  Kappa       : {kappa:.4f}")
print(f"  Sensitivity : {sensitivity:.2f}%  (DR detection rate)")
print(f"  Specificity : {specificity:.2f}%  (No DR detection rate)")

print(f"\n  ── CONFUSION MATRIX ────────────────────────")
print(f"  True Negative  (No DR → No DR)  : {tn:>4}")
print(f"  False Positive (No DR → DR)     : {fp:>4}")
print(f"  False Negative (DR    → No DR)  : {fn:>4}")
print(f"  True Positive  (DR    → DR)     : {tp:>4}")

print(f"\n  ── PER CLASS REPORT ────────────────────────")
print(classification_report(binary_labels, binary_preds,
                             target_names=["No DR", "DR Present"],
                             digits=3))

# ── Confusion matrix plot ─────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No DR", "DR Present"],
            yticklabels=["No DR", "DR Present"])
plt.title("Binary Confusion Matrix — APTOS Model")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(LOG_DIR, "confusion_matrix_binary.png")
plt.savefig(cm_path, dpi=150)
plt.close()

# ── ROC curve ────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(binary_labels, prob_dr)
roc_val     = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2,
         label=f"ROC curve (AUC = {roc_val:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Binary DR Classification")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(LOG_DIR, "roc_curve_binary.png")
plt.savefig(roc_path, dpi=150)
plt.close()

print(f"  Confusion matrix : {cm_path}")
print(f"  ROC curve        : {roc_path}")
print(f"\n  BINARY EVALUATION COMPLETE!")
print("=" * 60)
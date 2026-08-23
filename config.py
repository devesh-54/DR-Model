"""
============================================================
CONFIG.PY — APTOS 2019 Model (archive folder)
============================================================
"""

import torch

# ─────────────────────────────────────────────────────────
# DATASET 1 — APTOS 2019 (folder name: archive)
# ─────────────────────────────────────────────────────────
RAW_TRAIN_DIR    = r"C:\Users\deves\OneDrive\Desktop\archive\train_images\train_images"
RAW_VAL_DIR      = r"C:\Users\deves\OneDrive\Desktop\archive\val_images\val_images"
RAW_TEST_DIR     = r"C:\Users\deves\OneDrive\Desktop\archive\test_images\test_images"

PROC_TRAIN_DIR   = r"C:\Users\deves\OneDrive\Desktop\archive\processed\train_images"
PROC_VAL_DIR     = r"C:\Users\deves\OneDrive\Desktop\archive\processed\val_images"
PROC_TEST_DIR    = r"C:\Users\deves\OneDrive\Desktop\archive\processed\test_images"

TRAIN_CSV        = r"C:\Users\deves\OneDrive\Desktop\archive\train_1.csv"
VAL_CSV          = r"C:\Users\deves\OneDrive\Desktop\archive\valid.csv"
TEST_CSV         = r"C:\Users\deves\OneDrive\Desktop\archive\test.csv"
MERGED_CSV       = r"C:\Users\deves\OneDrive\Desktop\archive\merged_train.csv"

# ─────────────────────────────────────────────────────────
# DATASET 2 — EYEPACS
# ─────────────────────────────────────────────────────────
DS2_TRAIN1_DIR   = r"C:\Users\deves\OneDrive\Desktop\DATASET2\resized_train\resized_train"
DS2_TRAIN2_DIR   = r"C:\Users\deves\OneDrive\Desktop\DATASET2\resized_train_cropped\resized_train_cropped"
DS2_CSV1         = r"C:\Users\deves\OneDrive\Desktop\DATASET2\trainLabels.csv"
DS2_CSV2         = r"C:\Users\deves\OneDrive\Desktop\DATASET2\trainLabels_cropped.csv"
MERGED_TRAIN_DIR = r"C:\Users\deves\OneDrive\Desktop\archive\processed\merged_train"

# ─────────────────────────────────────────────────────────
# PROJECT OUTPUT PATHS
# ─────────────────────────────────────────────────────────
CHECKPOINT_DIR   = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\checkpoints"
GRADCAM_DIR      = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\gradcam_outputs"
REPORT_DIR       = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\reports"
LOG_DIR          = r"C:\Users\deves\OneDrive\Desktop\DiabeticRetinopathy\logs"

# ─────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────
MODEL_NAME       = "efficientnet_b4"
IMAGE_SIZE       = 380
NUM_CLASSES      = 5
PRETRAINED       = True

# ─────────────────────────────────────────────────────────
# CLASSIFICATION LABELS
# ─────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild NPDR: Microaneurysms only.",
    2: "Moderate NPDR: More than microaneurysms but less than severe NPDR.",
    3: "Severe NPDR: Extensive hemorrhages, venous beading, IRMA.",
    4: "Proliferative DR: Neovascularization or vitreous/pre-retinal hemorrhage."
}

CLASS_RECOMMENDATIONS = {
    0: "Routine annual screening recommended.",
    1: "Follow-up in 9-12 months recommended.",
    2: "Follow-up in 6 months. Consider referral to ophthalmologist.",
    3: "Urgent referral to ophthalmologist within 1 month.",
    4: "Immediate referral to ophthalmologist. High risk of vision loss."
}

CLASS_COLORS = {
    0: (0, 200, 0),
    1: (255, 255, 0),
    2: (255, 165, 0),
    3: (255, 69, 0),
    4: (255, 0, 0)
}

# ─────────────────────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────────────────────
PHASE1_EPOCHS    = 10
PHASE1_LR        = 1e-3
PHASE2_EPOCHS    = 20
PHASE2_LR        = 1e-4
BATCH_SIZE       = 16
NUM_WORKERS      = 2
WEIGHT_DECAY     = 1e-4
PATIENCE         = 5
MIXED_PRECISION  = True

# ─────────────────────────────────────────────────────────
# CLAHE SETTINGS
# ─────────────────────────────────────────────────────────
CLAHE_CLIP       = 2.0
CLAHE_TILE       = (8, 8)

# ─────────────────────────────────────────────────────────
# DEVICE & SEED
# ─────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED             = 42


def print_config():
    print("=" * 55)
    print("   PROJECT CONFIGURATION — APTOS 2019")
    print("=" * 55)
    print(f"  Device         : {DEVICE}")
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Image Size     : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"  Num Classes    : {NUM_CLASSES}")
    print(f"  Batch Size     : {BATCH_SIZE}")
    print(f"  Mixed Prec     : {MIXED_PRECISION}")
    print(f"\n  APTOS 2019 (archive):")
    print(f"    Train CSV    : {TRAIN_CSV}")
    print(f"    Val CSV      : {VAL_CSV}")
    print(f"    Test CSV     : {TEST_CSV}")
    print("\n  Classes:")
    for k, v in CLASS_NAMES.items():
        print(f"    Grade {k} → {v}")
    print("=" * 55)


if __name__ == "__main__":
    print_config()

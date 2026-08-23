"""
============================================================
STEP 2: DATASET + AUGMENTATION PIPELINE
============================================================
- Loads CLAHE processed images
- Applies heavy augmentation for training
- Light transforms for validation/test
- PyTorch Dataset class ready for DataLoader
============================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────
# YOUR EXACT PATHS
# ─────────────────────────────────────────────────────────
PROC_TRAIN_DIR  = r"C:\Users\deves\OneDrive\Desktop\archive\processed\train_images"
PROC_VAL_DIR    = r"C:\Users\deves\OneDrive\Desktop\archive\processed\val_images"
PROC_TEST_DIR   = r"C:\Users\deves\OneDrive\Desktop\archive\processed\test_images"

TRAIN_CSV       = r"C:\Users\deves\OneDrive\Desktop\archive\train_1.csv"
VAL_CSV         = r"C:\Users\deves\OneDrive\Desktop\archive\valid.csv"
TEST_CSV        = r"C:\Users\deves\OneDrive\Desktop\archive\test.csv"

IMAGE_SIZE      = 380
BATCH_SIZE      = 16    # Safe for RTX 4050 6.44GB VRAM
NUM_WORKERS     = 2     # Safe for laptop GPU
NUM_CLASSES     = 5     # Grade 0, 1, 2, 3, 4


# ─────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────

# TRAINING — Heavy augmentation to prevent overfitting
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    # Geometric augmentations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5),

    # Color augmentations (important for retinal images)
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    ),

    # Noise & blur
    A.GaussNoise(p=0.2),
    A.MotionBlur(blur_limit=3, p=0.2),

    # Cutout — forces model to learn robust features
    A.CoarseDropout(p=0.3),

    # Normalize with ImageNet stats (required for pretrained EfficientNet-B4)
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# VALIDATION & TEST — No augmentation, only resize + normalize
val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


# ─────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────
class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        """
        Args:
            csv_path  : Path to CSV file (train_1.csv / valid.csv / test.csv)
            img_dir   : Path to processed image folder
            transform : Albumentations transform pipeline
        """
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = Path(img_dir)
        self.transform = transform

        # Auto-detect column names
        self.id_col    = self._find_column(['id_code', 'image_id', 'filename', 'image'])
        self.label_col = self._find_column(['diagnosis', 'label', 'target', 'class'])

        print(f"  Dataset loaded: {len(self.df)} images")
        print(f"  ID column     : '{self.id_col}'")
        if self.label_col:
            print(f"  Label column  : '{self.label_col}'")
            print(f"  Class dist    : {self.df[self.label_col].value_counts().sort_index().to_dict()}")
        else:
            print(f"  Label column  : Not found (test set)")

    def _find_column(self, candidates):
        """Find column name from list of possible names."""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path
        img_id   = self.df.iloc[idx][self.id_col]
        img_name = f"{img_id}.png"
        img_path = self.img_dir / img_name

        # Fallback to .jpg if .png not found
        if not img_path.exists():
            img_path = self.img_dir / f"{img_id}.jpg"

        # Read image with OpenCV (BGR) → convert to RGB
        image = cv2.imread(str(img_path))
        if image is None:
            # Return black image if file is missing
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image     = augmented['image']

        # Get label (if available)
        if self.label_col and self.label_col in self.df.columns:
            label = int(self.df.iloc[idx][self.label_col])
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, torch.tensor(-1, dtype=torch.long)


# ─────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────
def get_dataloaders():
    """
    Creates and returns train, val, test DataLoaders.
    """
    print("\n" + "=" * 55)
    print("   LOADING DATASETS")
    print("=" * 55)

    print("\n  [TRAIN]")
    train_dataset = APTOSDataset(TRAIN_CSV, PROC_TRAIN_DIR, train_transforms)

    print("\n  [VALIDATION]")
    val_dataset   = APTOSDataset(VAL_CSV,   PROC_VAL_DIR,   val_transforms)

    print("\n  [TEST]")
    test_dataset  = APTOSDataset(TEST_CSV,  PROC_TEST_DIR,  val_transforms)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           # Shuffle training data every epoch
        num_workers=NUM_WORKERS,
        pin_memory=True,        # Faster GPU transfer
        drop_last=True          # Drop incomplete last batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("\n" + "=" * 55)
    print("   DATALOADER SUMMARY")
    print("=" * 55)
    print(f"  Train batches : {len(train_loader)}  ({len(train_dataset)} images)")
    print(f"  Val batches   : {len(val_loader)}  ({len(val_dataset)} images)")
    print(f"  Test batches  : {len(test_loader)}  ({len(test_dataset)} images)")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────
# VERIFY — Test one batch loads correctly
# ─────────────────────────────────────────────────────────
def verify_batch(train_loader):
    print("\n" + "=" * 55)
    print("   VERIFYING BATCH")
    print("=" * 55)

    images, labels = next(iter(train_loader))

    print(f"  Batch image shape : {images.shape}")
    print(f"  Expected          : torch.Size([{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}])")
    print(f"  Batch label shape : {labels.shape}")
    print(f"  Labels in batch   : {labels.tolist()}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Image min/max     : {images.min():.3f} / {images.max():.3f}")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device            : {device}")
    print(f"  GPU               : {torch.cuda.get_device_name(0)}")

    print("\n  BATCH VERIFICATION PASSED!")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    verify_batch(train_loader)

    print("\n" + "=" * 55)
    print("   STEP 2 COMPLETE!")
    print("=" * 55)
    print("""
  Next Step: Run step3_model.py
    """)

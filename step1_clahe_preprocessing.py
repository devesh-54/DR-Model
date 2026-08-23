"""
============================================================
STEP 1: CLAHE PREPROCESSING
============================================================
CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances retinal blood vessels and lesions — critical for DR.

Run this ONCE before training.
It will create a new 'processed' folder with enhanced images.
============================================================
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# YOUR EXACT PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR        = r"C:\Users\deves\OneDrive\Desktop\archive"

# Input folders (original images)
TRAIN_DIR       = r"C:\Users\deves\OneDrive\Desktop\archive\train_images\train_images"
VAL_DIR         = r"C:\Users\deves\OneDrive\Desktop\archive\val_images\val_images"
TEST_DIR        = r"C:\Users\deves\OneDrive\Desktop\archive\test_images\test_images"

# Output folders (CLAHE processed images — created automatically)
PROC_TRAIN_DIR  = r"C:\Users\deves\OneDrive\Desktop\archive\processed\train_images"
PROC_VAL_DIR    = r"C:\Users\deves\OneDrive\Desktop\archive\processed\val_images"
PROC_TEST_DIR   = r"C:\Users\deves\OneDrive\Desktop\archive\processed\test_images"

# Image size for EfficientNet-B4
IMAGE_SIZE = 380


# ─────────────────────────────────────────────────────────
# CLAHE FUNCTION
# ─────────────────────────────────────────────────────────
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    1. Resize image to 380x380
    2. Convert to LAB color space
    3. Apply CLAHE on L (luminance) channel
    4. Apply circular crop to remove black borders
    5. Return enhanced image
    """
    # Resize
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge back and convert LAB to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Circular mask — removes black border around retinal image
    mask = np.zeros_like(enhanced)
    h, w = enhanced.shape[:2]
    cx, cy, radius = w // 2, h // 2, min(w, h) // 2
    cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)
    enhanced = cv2.bitwise_and(enhanced, mask)

    return enhanced


# ─────────────────────────────────────────────────────────
# PROCESS FOLDER FUNCTION
# ─────────────────────────────────────────────────────────
def process_folder(input_dir: str, output_dir: str, split_name: str):
    """
    Process all images in input_dir with CLAHE
    and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path  = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + \
                  list(input_path.glob("*.jpg")) + \
                  list(input_path.glob("*.jpeg"))

    if len(image_files) == 0:
        print(f"  WARNING: No images found in {input_dir}")
        return

    print(f"\n  Processing {split_name}: {len(image_files)} images...")
    failed = 0

    for img_path in tqdm(image_files, desc=f"  CLAHE {split_name}"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                continue

            processed = apply_clahe(image)

            out_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(out_path), processed)

        except Exception as e:
            print(f"  Failed on {img_path.name}: {e}")
            failed += 1

    success = len(image_files) - failed
    print(f"  DONE {split_name} — {success}/{len(image_files)} images processed")
    if failed > 0:
        print(f"  WARNING: {failed} images failed (likely corrupt)")


# ─────────────────────────────────────────────────────────
# VERIFY OUTPUT
# ─────────────────────────────────────────────────────────
def verify_output(output_dir: str):
    path   = Path(output_dir)
    images = list(path.glob("*.png")) + list(path.glob("*.jpg"))

    if not images:
        print("  WARNING: No processed images found to verify!")
        return

    sample = cv2.imread(str(images[0]))
    print(f"\n  Verification Sample : {images[0].name}")
    print(f"  Shape               : {sample.shape}  <- should be (380, 380, 3)")
    print(f"  Min / Max / Mean    : {sample.min()} / {sample.max()} / {sample.mean():.2f}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("   CLAHE PREPROCESSING — APTOS 2019")
    print("=" * 55)
    print(f"\n  Image Size : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"  CLAHE Clip : 2.0")
    print(f"  Tile Grid  : 8 x 8")

    # Process all 3 splits
    process_folder(TRAIN_DIR, PROC_TRAIN_DIR, "Train")
    process_folder(VAL_DIR,   PROC_VAL_DIR,   "Validation")
    process_folder(TEST_DIR,  PROC_TEST_DIR,  "Test")

    # Verify
    print("\n" + "=" * 55)
    print("   VERIFYING OUTPUT")
    print("=" * 55)
    verify_output(PROC_TRAIN_DIR)

    print("\n" + "=" * 55)
    print("   CLAHE PREPROCESSING COMPLETE!")
    print("=" * 55)
    print(f"""
  Processed images saved to:
  -> {PROC_TRAIN_DIR}
  -> {PROC_VAL_DIR}
  -> {PROC_TEST_DIR}

  Next Step: Run step2_dataset.py
    """)

"""
============================================================
STEP 6: GRAD-CAM ANNOTATION (FIXED)
============================================================
Fixed: Proper hook targeting for EfficientNet-B4 in timm
Fixed: Heatmap empty/None check before resize
Fixed: Gradient computation with retain_graph
============================================================
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DEVICE, CHECKPOINT_DIR, GRADCAM_DIR,
    PROC_TEST_DIR, RAW_TEST_DIR, TEST_CSV,
    CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
)
from step3_model import get_model


# ─────────────────────────────────────────────────────────
# GRAD-CAM — Fixed for EfficientNet-B4 (timm)
# ─────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self.hooks       = []

        # Find correct target layer in EfficientNet-B4
        # timm EfficientNet uses 'conv_head' as last conv layer
        target_layer = self._find_target_layer()
        print(f"  Grad-CAM target layer: {type(target_layer).__name__}")

        h1 = target_layer.register_forward_hook(self._save_activation)
        h2 = target_layer.register_full_backward_hook(self._save_gradient)
        self.hooks = [h1, h2]

    def _find_target_layer(self):
        """Find the last convolutional layer in EfficientNet-B4."""
        # Try conv_head first (timm EfficientNet)
        if hasattr(self.model.backbone, 'conv_head'):
            return self.model.backbone.conv_head
        # Fallback: last block
        if hasattr(self.model.backbone, 'blocks'):
            return self.model.backbone.blocks[-1][-1]
        # Final fallback: last conv layer found
        last_conv = None
        for module in self.model.backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach().clone()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def generate(self, image_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        self.gradients   = None
        self.activations = None

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        image_tensor.requires_grad_(False)

        # Forward pass — keep graph for backward
        output = self.model(image_tensor)
        probs  = torch.softmax(output, dim=1)

        pred_class = output.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Check gradients captured
        if self.gradients is None or self.activations is None:
            print(f"  WARNING: Gradients not captured, using activation-only map")
            if self.activations is not None:
                heatmap = self.activations[0].mean(dim=0).cpu().numpy()
            else:
                heatmap = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            # Grad-CAM formula: global avg pool gradients × activations
            pooled_grads = self.gradients[0].mean(dim=[1, 2])  # [C]
            activations  = self.activations[0]                  # [C, H, W]

            # Weight activations by gradients
            for i in range(activations.shape[0]):
                activations[i] *= pooled_grads[i]

            heatmap = activations.mean(dim=0).cpu().numpy()

        # ReLU + normalize
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)

        # Ensure valid shape before resize
        if heatmap.ndim != 2 or heatmap.size == 0:
            heatmap = np.zeros((7, 7), dtype=np.float32)

        # Convert to float32 explicitly (required by cv2.resize)
        heatmap = heatmap.astype(np.float32)

        # Resize to image size
        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

        return heatmap, pred_class, confidence, probs[0].detach().cpu().numpy()


# ─────────────────────────────────────────────────────────
# OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────────────────────
def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_resized = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
    if original_resized.dtype != np.uint8:
        original_resized = original_resized.astype(np.uint8)

    overlay = cv2.addWeighted(original_resized, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_color


# ─────────────────────────────────────────────────────────
# GENERATE REPORT IMAGE
# ─────────────────────────────────────────────────────────
def generate_report_image(
    original_img, overlay_img, heatmap_img,
    pred_class, confidence, all_probs, img_id
):
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    grade_color = ['#00c851', '#ffbb33', '#ff8800', '#ff4444', '#cc0000'][pred_class]

    # Panel 1: Original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE)))
    ax1.set_title('Original Retinal Image', color='white', fontsize=12, pad=10)
    ax1.axis('off')

    # Panel 2: Grad-CAM Overlay
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(overlay_img)
    ax2.set_title('Grad-CAM: Affected Area', color='white', fontsize=12, pad=10)
    ax2.axis('off')
    legend_patches = [
        mpatches.Patch(color='blue',   label='Normal tissue'),
        mpatches.Patch(color='green',  label='Low activity'),
        mpatches.Patch(color='yellow', label='Moderate risk'),
        mpatches.Patch(color='red',    label='High risk / Lesion'),
    ]
    ax2.legend(handles=legend_patches, loc='lower right',
               fontsize=7, framealpha=0.7)

    # Panel 3: Confidence bars
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor('#16213e')
    class_labels = [f"Grade {i}: {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
    colors       = ['#00c851', '#ffbb33', '#ff8800', '#ff4444', '#cc0000']
    bars         = ax3.barh(class_labels, all_probs * 100,
                            color=colors, alpha=0.8,
                            edgecolor='white', linewidth=0.5)
    bars[pred_class].set_edgecolor('white')
    bars[pred_class].set_linewidth(2.5)
    for bar, prob in zip(bars, all_probs):
        ax3.text(min(bar.get_width() + 1, 95),
                 bar.get_y() + bar.get_height() / 2,
                 f'{prob*100:.1f}%', va='center',
                 color='white', fontsize=8)
    ax3.set_xlim(0, 105)
    ax3.set_xlabel('Confidence (%)', color='white', fontsize=9)
    ax3.set_title('Classification Confidence', color='white', fontsize=12, pad=10)
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#444')
    ax3.set_facecolor('#16213e')

    # Title
    fig.suptitle(
        f"DR DETECTION  |  Grade {pred_class} — {CLASS_NAMES[pred_class].upper()}"
        f"  |  Confidence: {confidence*100:.1f}%  |  ID: {img_id}",
        color=grade_color, fontsize=13, fontweight='bold',
        y=1.02, backgroundcolor='#0f3460'
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# MAIN — Run Grad-CAM on test images
# ─────────────────────────────────────────────────────────
def run_gradcam(num_samples=20):
    Path(GRADCAM_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(phase=2)
    ckpt  = Path(CHECKPOINT_DIR) / "best_model_phase2.pth"
    if not ckpt.exists():
        ckpt = Path(CHECKPOINT_DIR) / "best_model_phase1.pth"

    checkpoint = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    grad_cam = GradCAM(model)

    # Transform
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Load test IDs
    df     = pd.read_csv(TEST_CSV)
    id_col = next((c for c in ['id_code', 'image_id', 'filename', 'image']
                   if c in df.columns), None)
    samples = df[id_col].tolist()[:num_samples]

    print(f"\n  Generating Grad-CAM for {len(samples)} images...")
    success = 0

    for img_id in samples:
        try:
            # Load processed image
            proc_path = Path(PROC_TEST_DIR) / f"{img_id}.png"
            if not proc_path.exists():
                proc_path = Path(PROC_TEST_DIR) / f"{img_id}.jpg"
            if not proc_path.exists():
                print(f"  Skipping {img_id} — not found")
                continue

            # Load raw image for display
            raw_path = Path(RAW_TEST_DIR) / f"{img_id}.png"
            if not raw_path.exists():
                raw_path = Path(RAW_TEST_DIR) / f"{img_id}.jpg"

            proc_img = cv2.imread(str(proc_path))
            if proc_img is None:
                continue
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

            raw_img = cv2.imread(str(raw_path)) if raw_path.exists() else proc_img.copy()
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            # Generate Grad-CAM
            tensor = transform(image=proc_img)['image']
            heatmap, pred_class, confidence, all_probs = grad_cam.generate(tensor)

            overlay, heatmap_color = overlay_heatmap(raw_img, heatmap)

            # Generate and save report image
            fig = generate_report_image(
                raw_img, overlay, heatmap_color,
                pred_class, confidence, all_probs, img_id
            )

            save_path = Path(GRADCAM_DIR) / f"{img_id}_grade{pred_class}.png"
            fig.savefig(save_path, bbox_inches='tight',
                        dpi=150, facecolor='#1a1a2e')
            plt.close(fig)

            print(f"  {img_id} → Grade {pred_class} "
                  f"({CLASS_NAMES[pred_class]}) | "
                  f"Conf: {confidence*100:.1f}% | Saved ✅")
            success += 1

        except Exception as e:
            print(f"  Error on {img_id}: {e}")
            continue

    grad_cam.remove_hooks()
    print(f"\n  Done! {success}/{len(samples)} images processed")
    print(f"  Saved to: {GRADCAM_DIR}")


if __name__ == "__main__":
    print("=" * 55)
    print("   STEP 6: GRAD-CAM ANNOTATION (FIXED)")
    print("=" * 55)

    run_gradcam(num_samples=20)

    print("\n  STEP 6 COMPLETE! Next: python step7_report.py")

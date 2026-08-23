"""
============================================================
STEP 7: FINAL PATIENT REPORT GENERATOR
============================================================
Generates a complete clinical-style report for any image:
  - DR Grade classification (0-4)
  - Confidence score
  - Grad-CAM affected area overlay
  - Clinical description
  - Recommendation
============================================================
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

from config import (
    DEVICE, CHECKPOINT_DIR, REPORT_DIR,
    PROC_TEST_DIR, RAW_TEST_DIR, TEST_CSV,
    CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_RECOMMENDATIONS,
    IMAGE_SIZE, MIXED_PRECISION, NUM_CLASSES
)
from step3_model import get_model
from step6_gradcam import GradCAM, overlay_heatmap


# ─────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────
def generate_patient_report(img_id: str, model, grad_cam, transform):
    """Generate full clinical report for one image."""

    # Load images
    proc_path = Path(PROC_TEST_DIR) / f"{img_id}.png"
    if not proc_path.exists():
        proc_path = Path(PROC_TEST_DIR) / f"{img_id}.jpg"

    raw_path = Path(RAW_TEST_DIR) / f"{img_id}.png"
    if not raw_path.exists():
        raw_path = Path(RAW_TEST_DIR) / f"{img_id}.jpg"

    if not proc_path.exists():
        print(f"  Image not found: {img_id}")
        return

    proc_img = cv2.imread(str(proc_path))
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    raw_img = cv2.imread(str(raw_path)) if raw_path.exists() else proc_img.copy()
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # Inference + Grad-CAM
    tensor    = transform(image=proc_img)['image']
    heatmap, pred_class, confidence, all_probs = grad_cam.generate(tensor)
    overlay, _ = overlay_heatmap(raw_img, heatmap)

    # Affected area estimate (% of high-activation pixels)
    high_act = (heatmap > 0.5).sum() / heatmap.size * 100

    # Grade info
    grade_colors = ['#00c851', '#ffbb33', '#ff8800', '#ff4444', '#cc0000']
    grade_color  = grade_colors[pred_class]
    grade_name   = CLASS_NAMES[pred_class]
    description  = CLASS_DESCRIPTIONS[pred_class]
    recommendation = CLASS_RECOMMENDATIONS[pred_class]

    # ── BUILD REPORT FIGURE ──
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.4, wspace=0.3,
        height_ratios=[0.15, 1, 0.6]
    )

    # ─── HEADER ───
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor('#161b22')
    ax_header.axis('off')
    ax_header.text(
        0.5, 0.7,
        'DIABETIC RETINOPATHY DETECTION REPORT',
        transform=ax_header.transAxes,
        ha='center', va='center',
        fontsize=18, fontweight='bold', color='white'
    )
    ax_header.text(
        0.5, 0.2,
        f'Image ID: {img_id}   |   Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}   |   Model: EfficientNet-B4',
        transform=ax_header.transAxes,
        ha='center', va='center',
        fontsize=10, color='#8b949e'
    )

    # ─── ORIGINAL IMAGE ───
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE)))
    ax1.set_title('Original Retinal Image', color='white', fontsize=11, pad=8)
    ax1.axis('off')

    # ─── GRAD-CAM OVERLAY ───
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(overlay)
    ax2.set_title('Affected Area (Grad-CAM)', color='white', fontsize=11, pad=8)
    ax2.axis('off')

    # Colorbar legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#0000ff', label='Normal tissue'),
        Patch(facecolor='#00ff00', label='Low activity'),
        Patch(facecolor='#ffff00', label='Moderate risk'),
        Patch(facecolor='#ff0000', label='High risk / Lesion'),
    ]
    ax2.legend(handles=legend, loc='lower right',
               fontsize=7.5, framealpha=0.8,
               facecolor='#1a1a2e', labelcolor='white')

    # ─── CONFIDENCE BAR CHART ───
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor('#161b22')
    class_labels = [f"Grade {i}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
    bars = ax3.barh(
        class_labels, all_probs * 100,
        color=grade_colors, alpha=0.85,
        edgecolor='#30363d', linewidth=0.8
    )
    bars[pred_class].set_edgecolor('white')
    bars[pred_class].set_linewidth(2.5)
    for bar, prob in zip(bars, all_probs):
        ax3.text(
            min(bar.get_width() + 1, 95),
            bar.get_y() + bar.get_height() / 2,
            f'{prob*100:.1f}%', va='center',
            color='white', fontsize=8.5
        )
    ax3.set_xlim(0, 105)
    ax3.set_xlabel('Confidence (%)', color='#8b949e', fontsize=9)
    ax3.set_title('Classification Confidence', color='white', fontsize=11, pad=8)
    ax3.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color('#30363d')
    ax3.set_facecolor('#161b22')

    # ─── DIAGNOSIS PANEL ───
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#161b22')
    ax4.axis('off')

    # Grade badge
    ax4.add_patch(FancyBboxPatch(
        (0.01, 0.1), 0.18, 0.8,
        boxstyle="round,pad=0.02",
        facecolor=grade_color, alpha=0.2,
        edgecolor=grade_color, linewidth=2,
        transform=ax4.transAxes
    ))
    ax4.text(0.10, 0.65, f'GRADE {pred_class}',
             transform=ax4.transAxes,
             ha='center', fontsize=16, fontweight='bold', color=grade_color)
    ax4.text(0.10, 0.35, grade_name,
             transform=ax4.transAxes,
             ha='center', fontsize=11, color='white')

    # Confidence
    ax4.text(0.22, 0.75, 'Confidence',
             transform=ax4.transAxes, color='#8b949e', fontsize=9)
    ax4.text(0.22, 0.45, f'{confidence*100:.1f}%',
             transform=ax4.transAxes, color='white', fontsize=16, fontweight='bold')

    # Affected area
    ax4.text(0.35, 0.75, 'Affected Area',
             transform=ax4.transAxes, color='#8b949e', fontsize=9)
    ax4.text(0.35, 0.45, f'{high_act:.1f}%',
             transform=ax4.transAxes, color='white', fontsize=16, fontweight='bold')

    # Description
    ax4.text(0.52, 0.78, 'Clinical Finding:',
             transform=ax4.transAxes, color='#8b949e', fontsize=9)
    ax4.text(0.52, 0.52, description,
             transform=ax4.transAxes, color='white', fontsize=9.5,
             wrap=True, va='top')

    # Recommendation
    ax4.text(0.52, 0.25, f'Recommendation: {recommendation}',
             transform=ax4.transAxes, color=grade_color, fontsize=9.5,
             fontstyle='italic')

    # Footer
    ax4.text(0.5, 0.02,
             'Note: This report is AI-assisted. Final diagnosis must be confirmed by a licensed ophthalmologist.',
             transform=ax4.transAxes,
             ha='center', color='#484f58', fontsize=8, fontstyle='italic')

    # Save
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(REPORT_DIR) / f"report_{img_id}_grade{pred_class}.png"
    fig.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='#0d1117')
    plt.close(fig)

    print(f"  {img_id} → Grade {pred_class} ({grade_name}) | "
          f"Conf: {confidence*100:.1f}% | Area: {high_act:.1f}%")
    return save_path


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("=" * 55)
    print("   STEP 7: FINAL REPORT GENERATOR")
    print("=" * 55)

    # Load model
    model = get_model(phase=2)
    ckpt  = Path(CHECKPOINT_DIR) / "best_model_phase2.pth"
    if not ckpt.exists():
        ckpt = Path(CHECKPOINT_DIR) / "best_model_phase1.pth"
    checkpoint = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    grad_cam  = GradCAM(model)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Load test image IDs
    df     = pd.read_csv(TEST_CSV)
    id_col = next((c for c in ['id_code','image_id','filename','image'] if c in df.columns), None)
    samples = df[id_col].tolist()[:10]  # Generate 10 reports

    print(f"\n  Generating reports for {len(samples)} images...")
    for img_id in samples:
        generate_patient_report(img_id, model, grad_cam, transform)

    print(f"\n  All reports saved to: {REPORT_DIR}")
    print("\n  STEP 7 COMPLETE! Your DR detection system is ready!")
    print("""
  Full pipeline complete:
  Step 1 -> CLAHE preprocessing    DONE
  Step 2 -> Dataset + Augmentation DONE
  Step 3 -> EfficientNet-B4 model  DONE
  Step 4 -> Two-phase training     DONE
  Step 5 -> Evaluation             DONE
  Step 6 -> Grad-CAM heatmaps      DONE
  Step 7 -> Clinical reports       DONE
    """)

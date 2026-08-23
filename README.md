# Diabetic Retinopathy Detection
## EfficientNet-B4 + Grad-CAM | PyTorch | APTOS 2019

---

## Project Structure

```
DiabeticRetinopathy/
├── config.py                  ← All settings (edit this only)
├── step1_clahe_preprocessing.py
├── step2_dataset.py
├── step3_model.py
├── step4_train.py
├── step5_evaluate.py
├── step6_gradcam.py
├── step7_report.py
└── run_all.py                 ← Run everything at once
```

---

## Setup

### 1. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations opencv-python pandas numpy matplotlib scikit-learn tqdm Pillow seaborn
```

### 2. Verify GPU
```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 4050
```

---

## Run Step by Step

```bash
python step1_clahe_preprocessing.py   # Preprocess images (run once)
python step2_dataset.py               # Verify dataset loads
python step3_model.py                 # Verify model output
python step4_train.py                 # Train (3-5 hours)
python step5_evaluate.py              # Evaluate on test set
python step6_gradcam.py               # Generate heatmaps
python step7_report.py                # Generate reports
```

## OR Run All At Once

```bash
python run_all.py
```

---

## Classification Output

| Grade | Class | Description |
|-------|-------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild DR | Microaneurysms only |
| 2 | Moderate DR | More lesions, less than severe |
| 3 | Severe DR | Extensive hemorrhages |
| 4 | Proliferative DR | Neovascularization — high blindness risk |

---

## Expected Results

- Accuracy: ~85-90%
- Quadratic Kappa: ~0.85-0.90
- Training Time: ~3-5 hours on RTX 4050

---

## Output Files

```
checkpoints/
  best_model_phase1.pth     ← Phase 1 best weights
  best_model_phase2.pth     ← Phase 2 best weights (use this)

logs/
  confusion_matrix.png      ← Confusion matrix plot

gradcam_outputs/
  {image_id}_grade{n}.png   ← Heatmap overlay per image

reports/
  report_{image_id}_grade{n}.png  ← Full clinical report per image
```

"""
============================================================
RUN_ALL.PY — Run Entire Pipeline With One Command
============================================================
Runs all 7 steps in sequence:
  python run_all.py
============================================================
"""

import subprocess
import sys
import time

steps = [
    ("STEP 1", "step1_clahe_preprocessing.py", "CLAHE Preprocessing"),
    ("STEP 2", "step2_dataset.py",              "Dataset Verification"),
    ("STEP 3", "step3_model.py",                "Model Setup"),
    ("STEP 4", "step4_train.py",                "Two-Phase Training"),
    ("STEP 5", "step5_evaluate.py",             "Evaluation"),
    ("STEP 6", "step6_gradcam.py",              "Grad-CAM Annotation"),
    ("STEP 7", "step7_report.py",               "Report Generation"),
]

print("=" * 55)
print("   DIABETIC RETINOPATHY — FULL PIPELINE")
print("=" * 55)

for step_id, script, name in steps:
    print(f"\n{'='*55}")
    print(f"  Running {step_id}: {name}")
    print(f"{'='*55}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR in {step_id}! Stopping pipeline.")
        sys.exit(1)

    print(f"\n  {step_id} DONE in {elapsed:.1f}s")

print("\n" + "=" * 55)
print("   ALL STEPS COMPLETE!")
print("=" * 55)
print("""
  Outputs:
  -> Processed images   : archive/processed/
  -> Model checkpoints  : DiabeticRetinopathy/checkpoints/
  -> Evaluation plots   : DiabeticRetinopathy/logs/
  -> Grad-CAM images    : DiabeticRetinopathy/gradcam_outputs/
  -> Patient reports    : DiabeticRetinopathy/reports/
""")

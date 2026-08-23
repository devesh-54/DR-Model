"""
============================================================
STEP 4: TWO-PHASE TRAINING
============================================================
Phase 1 — Frozen backbone, train classifier head only
           Fast convergence, avoids destroying pretrained weights
Phase 2 — Unfreeze all layers, fine-tune with low LR
           Adapts entire model to retinal DR features
============================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_DIR,
    PHASE1_EPOCHS, PHASE1_LR,
    PHASE2_EPOCHS, PHASE2_LR,
    WEIGHT_DECAY, PATIENCE, MIXED_PRECISION,
    NUM_CLASSES, CLASS_NAMES
)
from step2_dataset import get_dataloaders
from step3_model import get_model


# ─────────────────────────────────────────────────────────
# LOSS FUNCTION — Label Smoothing for better generalization
# ─────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=NUM_CLASSES, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls       = classes

    def forward(self, pred, target):
        confidence  = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot     = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val
        log_prob    = torch.log_softmax(pred, dim=1)
        loss        = -(smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss


# ─────────────────────────────────────────────────────────
# TRAINING LOOP — One epoch
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        if MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds      = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {correct/total*100:.2f}%")

    return total_loss / len(loader), correct / total * 100


# ─────────────────────────────────────────────────────────
# VALIDATION LOOP
# ─────────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds      = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        correct / total * 100,
        np.array(all_preds),
        np.array(all_labels)
    )


# ─────────────────────────────────────────────────────────
# TRAIN PHASE
# ─────────────────────────────────────────────────────────
def train_phase(model, train_loader, val_loader, phase):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    lr          = PHASE1_LR if phase == 1 else PHASE2_LR
    epochs      = PHASE1_EPOCHS if phase == 1 else PHASE2_EPOCHS
    ckpt_path   = Path(CHECKPOINT_DIR) / f"best_model_phase{phase}.pth"

    criterion   = LabelSmoothingLoss()
    optimizer   = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler      = GradScaler() if MIXED_PRECISION else None

    best_val_acc  = 0.0
    patience_cnt  = 0
    history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n{'='*55}")
    print(f"   PHASE {phase} TRAINING")
    print(f"{'='*55}")
    print(f"  Epochs     : {epochs}")
    print(f"  LR         : {lr}")
    print(f"  Optimizer  : AdamW  |  Scheduler: CosineAnnealing")
    print(f"  Loss       : LabelSmoothing(0.1)")
    print(f"  Checkpoint : {ckpt_path}")

    for epoch in range(1, epochs + 1):
        start = time.time()
        print(f"\n  Epoch [{epoch}/{epochs}]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        # Validate
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)

        scheduler.step()

        elapsed = time.time() - start
        print(f"\n  Train  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val    Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")
        print(f"  LR        : {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Time      : {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_acc':    val_acc,
                'val_loss':   val_loss,
            }, ckpt_path)
            print(f"  SAVED best model — Val Acc: {val_acc:.2f}%")
            patience_cnt = 0
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{PATIENCE})")

        # Early stopping
        if patience_cnt >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    print(f"\n  Phase {phase} Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Checkpoint saved: {ckpt_path}")

    # Load best weights back
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    return model, history


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── PHASE 1: Frozen backbone ──
    print("\n  Loading model for Phase 1...")
    model = get_model(phase=1)
    model, history1 = train_phase(model, train_loader, val_loader, phase=1)

    # ── PHASE 2: Unfreeze all ──
    print("\n  Switching to Phase 2 — Unfreezing backbone...")
    model.unfreeze_backbone()
    model, history2 = train_phase(model, train_loader, val_loader, phase=2)

    print("\n" + "=" * 55)
    print("   TRAINING COMPLETE!")
    print("=" * 55)
    print(f"""
  Checkpoints saved:
  -> {CHECKPOINT_DIR}\\best_model_phase1.pth
  -> {CHECKPOINT_DIR}\\best_model_phase2.pth

  Next Step: python step5_evaluate.py
    """)

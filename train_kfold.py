"""
train_kfold.py

5-Fold Cross Validation for SEM -> Gini Regression + 4-class classification
Fusion model: image + metadata.

Reports per fold:
- MSE / MAE (original Gini scale if normalized)
- Accuracy
- Confusion matrix (rows=true, cols=pred)

Added:
- Per-class Precision / Recall / IoU (Jaccard)
- Macro Precision / Macro Recall
- mIoU

Aggregates mean ± std across folds.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from config import Config
from dataset import ArestySEMDataset, RobustTargetNormalizer
from model import ArestyRegClsModel


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# Device Selection
# ---------------------------
def pick_device():
    """Automatically pick best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------
# Stratified K-Fold Split
# ---------------------------
def make_stratified_folds(labels, k, seed):
    """
    Create stratified folds to maintain class distribution across folds.
    """
    labels = np.asarray(labels)
    n = len(labels)
    idx = np.arange(n)

    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)

    # If only one class exists, fallback to random split
    if uniq.size < 2:
        rng.shuffle(idx)
        return np.array_split(idx, k)

    # Group indices by class
    by_class = {c: idx[labels == c].copy() for c in uniq}
    for c in uniq:
        rng.shuffle(by_class[c])

    # Distribute samples into folds evenly per class
    folds = [[] for _ in range(k)]
    for c in uniq:
        for j, i in enumerate(by_class[c]):
            folds[j % k].append(int(i))

    folds = [np.array(f, dtype=int) for f in folds]

    # Shuffle each fold
    for f in folds:
        rng.shuffle(f)

    return folds


# ---------------------------
# Metrics from Confusion Matrix
# ---------------------------
def metrics_from_confusion(cm: np.ndarray, eps: float = 1e-9):
    """
    Compute classification metrics from confusion matrix.

    cm: rows=true labels, cols=predicted labels
    """
    cm = cm.astype(np.float64)

    # True positives per class
    tp = np.diag(cm)

    # False positives and false negatives
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # Metrics
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "iou_per_class": iou,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "miou": float(np.mean(iou)),
    }


# ---------------------------
# Evaluation Loop
# ---------------------------
@torch.no_grad()
def eval_epoch(model, loader, device, target_normalizer=None, num_classes=4):
    """
    Evaluate model on validation set.
    Computes regression + classification metrics.
    """
    model.eval()

    mse_sum, mae_sum, n = 0.0, 0.0, 0
    correct = 0

    # Confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for x, score, label, meta in loader:
        x = x.to(device)
        score = score.to(device)
        label = label.to(device)
        meta = meta.to(device)

        pred, logits = model(x, meta)

        # Classification predictions
        pred_cls = logits.argmax(dim=1)

        # Accuracy
        correct += (pred_cls == label).sum().item()

        # Build confusion matrix
        for t, p in zip(label.view(-1), pred_cls.view(-1)):
            cm[int(t), int(p)] += 1

        # Undo normalization if applied
        if target_normalizer is not None:
            pred_np = target_normalizer.inverse(pred.detach().cpu().numpy())
            score_np = target_normalizer.inverse(score.detach().cpu().numpy())

            pred = torch.tensor(pred_np, dtype=torch.float32, device=device)
            score = torch.tensor(score_np, dtype=torch.float32, device=device)

        # Regression metrics
        mse_sum += torch.sum((pred - score) ** 2).item()
        mae_sum += torch.sum(torch.abs(pred - score)).item()
        n += x.size(0)

    cm_np = cm.cpu().numpy()
    extra = metrics_from_confusion(cm_np)

    return {
        "mse": mse_sum / max(1, n),
        "mae": mae_sum / max(1, n),
        "acc": correct / max(1, n),
        "cm": cm_np,
        **extra,
    }


# ---------------------------
# Training Loop (One Epoch)
# ---------------------------
def train_one_epoch(model, loader, opt, device, scaler, cls_loss_fn, cls_lambda: float):
    """
    Train model for one epoch using combined loss:
    - Regression loss (MSE)
    - Classification loss (CrossEntropy)
    """
    model.train()

    total_loss, n = 0.0, 0

    for x, score, label, meta in loader:
        x = x.to(device)
        score = score.to(device)
        label = label.to(device)
        meta = meta.to(device)

        opt.zero_grad(set_to_none=True)

        # Mixed precision training (if enabled)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred, logits = model(x, meta)

                loss_reg = nn.functional.mse_loss(pred, score)
                loss_cls = cls_loss_fn(logits, label)

                loss = loss_reg + cls_lambda * loss_cls

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        else:
            pred, logits = model(x, meta)

            loss_reg = nn.functional.mse_loss(pred, score)
            loss_cls = cls_loss_fn(logits, label)

            loss = loss_reg + cls_lambda * loss_cls

            loss.backward()
            opt.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(1, n)


# --- EVERYTHING ABOVE THIS STAYS THE SAME ---


def main():
    """
    Main training pipeline:
    - Load dataset
    - Perform stratified 5-fold CV
    - Train model per fold
    - Aggregate results
    """
    cfg = Config()
    set_seed(cfg.SEED)

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    csv_path = "master_regularity_full_clean.csv"
    img_dir = "images"

    device = pick_device()
    print("Device:", device)

    # ---------------------------
    # Data Augmentation
    # ---------------------------
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # ---------------------------
    # Load Dataset
    # ---------------------------
    base_ds = ArestySEMDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        img_size=cfg.IMG_SIZE,
        target_col=cfg.TARGET_COL,
        transform=None,
        target_normalizer=None
    )

    print("Dataset size:", len(base_ds))

    # Show class distribution
    class_counts = base_ds.df["label"].value_counts().sort_index()

    print("\nFull dataset class counts:")
    for c, n in class_counts.items():
        print(f"Class {c+1}: {n} images")

    labels = base_ds.df["label"].to_numpy()

    all_results = []

    num_classes = 4
    cls_lambda = getattr(cfg, "CLS_LAMBDA", 0.7)

    # ---------------------------
    # Cross Validation Loop
    # ---------------------------
    for repeat in range(1):
        print(f"\n===== REPEAT {repeat+1} =====")

        folds = make_stratified_folds(labels, k=5, seed=cfg.SEED + repeat)

        fold_results = []

        for fold_id in range(5):
            # Split indices
            val_idx = folds[fold_id]
            train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_id])

            # Show training distribution
            train_labels = base_ds.df.iloc[train_idx]["label"]
            counts = train_labels.value_counts().sort_index()

            print(f"\nFold {fold_id+1} TRAINING class distribution:")
            for c, n in counts.items():
                print(f"Class {c+1}: {n} images")

            # ---------------------------
            # Target Normalization
            # ---------------------------
            normalizer = None
            if cfg.NORM_TARGET:
                normalizer = RobustTargetNormalizer()
                train_scores_raw = base_ds.df.iloc[train_idx]["score_raw"].to_numpy()
                normalizer.fit(train_scores_raw)

            # ---------------------------
            # Datasets + Loaders
            # ---------------------------
            ds_train = ArestySEMDataset(
                csv_path, img_dir, cfg.IMG_SIZE,
                cfg.TARGET_COL, train_tfm, normalizer
            )

            ds_val = ArestySEMDataset(
                csv_path, img_dir, cfg.IMG_SIZE,
                cfg.TARGET_COL, val_tfm, normalizer
            )

            train_set = Subset(ds_train, train_idx.tolist())
            val_set = Subset(ds_val, val_idx.tolist())

            # ---------------------------
            # Class Balancing (Sampler)
            # ---------------------------
            from torch.utils.data import WeightedRandomSampler

            train_labels_np = base_ds.df.iloc[train_idx]["label"].to_numpy()
            class_counts = np.bincount(train_labels_np, minlength=num_classes)

            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[train_labels_np]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            train_loader = DataLoader(
                train_set,
                batch_size=4,
                sampler=sampler,
                num_workers=0
            )

            val_loader = DataLoader(
                val_set,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )

            # ---------------------------
            # Model + Optimizer
            # ---------------------------
            model = ArestyRegClsModel(
                in_channels=cfg.IN_CHANNELS,
                num_classes=num_classes,
                use_pretrained=cfg.USE_PRETRAINED
            ).to(device)

            cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.LR,
                weight_decay=cfg.WEIGHT_DECAY
            )

            scaler = torch.amp.GradScaler("cuda") if (cfg.USE_AMP and device.type == "cuda") else None

            # ---------------------------
            # Training Loop
            # ---------------------------
            best_val = float("inf")
            patience = cfg.EARLY_STOP_PATIENCE
            best_metrics = None

            print(f"\n--- Fold {fold_id+1}/5 ---")

            for epoch in range(1, cfg.EPOCHS + 1):
                train_one_epoch(
                    model, train_loader, opt,
                    device, scaler, cls_loss_fn, cls_lambda
                )

                metrics = eval_epoch(
                    model,
                    val_loader,
                    device,
                    target_normalizer=normalizer if cfg.NORM_TARGET else None,
                    num_classes=num_classes,
                )

                # Early stopping check
                improved = (best_val - metrics["mse"]) > cfg.MIN_DELTA

                if improved:
                    best_val = metrics["mse"]
                    best_metrics = metrics
                    patience = cfg.EARLY_STOP_PATIENCE
                else:
                    patience -= 1
                    if patience <= 0:
                        break

            # ---------------------------
            # Print Fold Results
            # ---------------------------
            print(
                f"Fold {fold_id+1} BEST | MSE={best_metrics['mse']:.4f} "
                f"MAE={best_metrics['mae']:.4f} ACC={best_metrics['acc']:.3f} "
                f"mIoU={best_metrics['miou']:.3f} "
                f"MacroP={best_metrics['macro_precision']:.3f} "
                f"MacroR={best_metrics['macro_recall']:.3f}"
            )

            print("Confusion (rows=true, cols=pred):\n", best_metrics["cm"])
            print("IoU per class:", np.round(best_metrics["iou_per_class"], 3))
            print("P  per class :", np.round(best_metrics["precision_per_class"], 3))
            print("R  per class :", np.round(best_metrics["recall_per_class"], 3))

            fold_results.append(best_metrics)

        # Store all folds
        all_results.extend(fold_results)

    # ---------------------------
    # Final Aggregated Metrics
    # ---------------------------
    mses = np.array([r["mse"] for r in all_results], dtype=float)
    maes = np.array([r["mae"] for r in all_results], dtype=float)
    accs = np.array([r["acc"] for r in all_results], dtype=float)

    mious = np.array([r["miou"] for r in all_results], dtype=float)
    mPs = np.array([r["macro_precision"] for r in all_results], dtype=float)
    mRs = np.array([r["macro_recall"] for r in all_results], dtype=float)

    print("\n===== 5-Fold Cross-Validation Results =====")
    print(f"MSE mean={mses.mean():.4f} std={mses.std(ddof=1):.4f}")
    print(f"MAE mean={maes.mean():.4f} std={maes.std(ddof=1):.4f}")
    print(f"RMSE mean={np.sqrt(mses).mean():.4f}")
    print(f"ACC mean={accs.mean():.3f} std={accs.std(ddof=1):.3f}")
    print(f"ACC median={np.median(accs):.3f}")
    print(f"mIoU mean={mious.mean():.3f} std={mious.std(ddof=1):.3f}")
    print(f"MacroP mean={mPs.mean():.3f} std={mPs.std(ddof=1):.3f}")
    print(f"MacroR mean={mRs.mean():.3f} std={mRs.std(ddof=1):.3f}")

    # Aggregate confusion matrix
    cm_total = np.zeros((4,4), dtype=int)
    for r in all_results:
        cm_total += r["cm"]

    cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True)

    print("\n===== FINAL AGGREGATED CONFUSION MATRIX =====")
    print(cm_total)

    print("\n===== NORMALIZED =====")
    print(np.round(cm_norm, 3))


if __name__ == "__main__":
    main()
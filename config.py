"""
config.py

Purpose:
Central location for experiment settings used across the entire pipeline.
Updating values here automatically affects training, dataset loading,
and model behavior for consistency and reproducibility.
"""

from dataclasses import dataclass

@dataclass
class Config:
    # -------------------------
    # Reproducibility Settings
    # -------------------------
    SEED: int = 42
    # Ensures consistent results across runs (data splits, weight initialization, etc.)

    # -------------------------
    # Image / Input Settings
    # -------------------------
    IMG_SIZE: int = 224
    # All SEM images are resized to a fixed resolution for compatibility with CNN backbone

    # -------------------------
    # Training Hyperparameters
    # -------------------------
    BATCH_SIZE: int = 4
    # Small batch size improves generalization for small datasets (83 samples)

    EPOCHS: int = 50
    # Maximum training epochs; early stopping usually terminates earlier

    LR: float = 5e-5
    # Low learning rate for stable fine-tuning of pretrained ResNet

    WEIGHT_DECAY: float = 1e-4
    # L2 regularization to reduce overfitting

    # Mixed precision (primarily useful for CUDA GPUs; disabled for MPS/CPU)
    USE_AMP: bool = False

    # -------------------------
    # Target / Label Settings
    # -------------------------
    TARGET_COL: str = "Gini"
    # Continuous regression target (regularity score)

    NORM_TARGET: bool = True
    # Applies robust normalization (median + IQR) for more stable regression training

    CLS_LAMBDA = 0.7
    # Weight balancing classification vs regression loss:
    # total_loss = regression_loss + CLS_LAMBDA * classification_loss
    # Higher value prioritizes classification accuracy

    # -------------------------
    # Early Stopping
    # -------------------------
    EARLY_STOP_PATIENCE: int = 15
    # Number of epochs to wait without improvement before stopping

    MIN_DELTA: float = 1e-4
    # Minimum improvement in validation MSE required to reset patience

    # -------------------------
    # Model Configuration
    # -------------------------
    USE_PRETRAINED: bool = True
    # Uses ImageNet pretrained ResNet18 weights (critical for small datasets)

    IN_CHANNELS: int = 1
    # Number of input channels:
    # 1 = grayscale SEM images
    # 2 = potential future extension (image + mask)

    # -------------------------
    # Output / Logging
    # -------------------------
    OUT_DIR: str = "runs/aresty"
    # Directory to save trained models, logs, and outputs
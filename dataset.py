"""
dataset.py

Defines the PyTorch Dataset used to load SEM images and regularity scores.

Returns per sample:
    x      : image tensor (1,H,W)
    score  : regression target (normalized if normalizer provided)
    label  : class label from Gini bins (0..3)
    meta   : metadata tensor [fluence, delay, pulses, fluence_mask, delay_mask, pulses_mask]
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def build_filename(image_id: str) -> str:
    """
    Converts dataset image_id into actual filename.

    Example:
        input  -> "sample_1_3"
        output -> "sample_1_LIPSS03.jpg"

    Ensures consistent formatting for loading SEM images.
    """
    parts = image_id.split("_")
    prefix = "_".join(parts[:2])
    num = int(parts[2])
    return f"{prefix}_LIPSS{num:02d}.jpg"


class RobustTargetNormalizer:
    """
    Robust normalization using median and IQR (Interquartile Range).

    Why:
    - More stable than mean/std when data contains outliers
    - Improves regression training stability

    Formula:
        y_scaled = (y - median) / IQR
    """

    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.float32)
        self.median = float(np.median(y))
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        self.iqr = float(q3 - q1)

        # Avoid division by zero if all values identical
        if self.iqr == 0:
            self.iqr = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        return (y - self.median) / self.iqr

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled, dtype=np.float32)
        return y_scaled * self.iqr + self.median


def _load_master_csv(csv_path: str) -> pd.DataFrame:
    """
    Handles two CSV formats:
    1. Standard CSV with header row
    2. Custom format where actual header appears after a few rows

    Ensures flexibility for messy/raw dataset exports.
    """
    try:
        df = pd.read_csv(csv_path)
        if "image_id" in df.columns:
            return df
    except Exception:
        pass

    # fallback for messy CSV format
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    return df


def _to_float_or_nan(v) -> float:
    """
    Safely converts value to float.
    Returns NaN if conversion fails or value is missing.
    """
    try:
        if v is None:
            return float("nan")
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _val_and_mask(v: float) -> tuple[float, float]:
    """
    Returns:
        value: actual value (0 if missing)
        mask : 1 if valid, 0 if missing

    Allows model to distinguish between:
    - true zero
    - missing value
    """
    if np.isnan(v):
        return 0.0, 0.0
    return float(v), 1.0


class ArestySEMDataset(Dataset):
    """
    Custom dataset for SEM regularity prediction.

    Expected CSV columns:
        - image_id
        - target_col (default: "Gini")

    Optional metadata columns:
        - fluence_j_cm2
        - delay_ps
        - double_pulses / pulse

    Outputs:
        image tensor + regression target + classification label + metadata
    """

    def __init__(
        self,
        csv_path,
        img_dir,
        img_size=224,
        target_col="Gini",
        transform=None,
        target_normalizer: RobustTargetNormalizer | None = None,
        print_class_counts=False,
    ):
        # Load dataset
        self.df = _load_master_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.target_col = target_col
        self.transform = transform

        # -------------------------
        # Validation checks
        # -------------------------
        if "image_id" not in self.df.columns:
            raise ValueError(f"CSV must contain 'image_id'. Found columns: {list(self.df.columns)}")

        if self.target_col not in self.df.columns:
            raise ValueError(f"CSV missing target column '{self.target_col}'. Found columns: {list(self.df.columns)}")

        # -------------------------
        # Process regression target
        # -------------------------
        self.df["score_raw"] = pd.to_numeric(self.df[self.target_col], errors="coerce")
        self.df = self.df.dropna(subset=["score_raw"]).reset_index(drop=True)

        # -------------------------
        # Convert Gini → 4 classes
        # -------------------------
        g = self.df["score_raw"]

        self.df["label"] = 0
        self.df.loc[(g >= 0.3) & (g < 0.5), "label"] = 1
        self.df.loc[(g >= 0.5) & (g < 0.65), "label"] = 2
        self.df.loc[g >= 0.65, "label"] = 3

        if print_class_counts:
            print("Gini class counts:", self.df["label"].value_counts().sort_index().to_dict())

        # -------------------------
        # Normalize regression target
        # -------------------------
        self.target_normalizer = target_normalizer
        if self.target_normalizer is not None and self.target_normalizer.median is not None:
            self.df["score"] = self.target_normalizer.transform(self.df["score_raw"].to_numpy())
        else:
            self.df["score"] = self.df["score_raw"].astype(np.float32)

        # -------------------------
        # Quick sanity check for images
        # -------------------------
        for img_id in self.df["image_id"].head(min(10, len(self.df))):
            img_name = build_filename(str(img_id))
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")

    def __len__(self):
        return len(self.df)

    def _load_gray(self, path):
        """
        Loads grayscale SEM image and applies transform if provided.
        """
        img = Image.open(path).convert("L")

        if self.transform is not None:
            return self.transform(img)

        # fallback if no transform
        img = img.resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # Load image
        # -------------------------
        img_name = build_filename(str(row["image_id"]))
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        x = self._load_gray(img_path)

        # Skip degenerate images (all pixels identical)
        if float(x.max()) == float(x.min()):
            new_idx = (idx + 1) % len(self.df)
            return self.__getitem__(new_idx)

        # -------------------------
        # Targets
        # -------------------------
        score = torch.tensor(float(row["score"]), dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        # -------------------------
        # Metadata (with masks)
        # -------------------------
        fluence = _to_float_or_nan(row.get("fluence_j_cm2", np.nan))
        delay = _to_float_or_nan(row.get("delay_ps", np.nan))

        # Prefer double_pulses, fallback to pulse
        pulses_raw = row.get("double_pulses", row.get("pulse", np.nan))
        pulses = _to_float_or_nan(pulses_raw)

        fluence_v, fluence_m = _val_and_mask(fluence)
        delay_v, delay_m = _val_and_mask(delay)
        pulses_v, pulses_m = _val_and_mask(pulses)

        meta = torch.tensor(
            [fluence_v, delay_v, pulses_v, fluence_m, delay_m, pulses_m],
            dtype=torch.float32
        )

        return x, score, label, meta
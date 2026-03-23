# SEM Regularity Prediction (Fusion CNN + Metadata)

A deep learning fusion model combining CNN-based image features and metadata for predicting surface regularity from SEM images.

---

## Overview

This project predicts surface regularity from Scanning Electron Microscope (SEM) images using a deep learning approach. The model integrates image-based features with metadata to perform:

* **Regression** → Predict continuous regularity score (Gini-based)
* **Classification** → Predict discrete regularity class (4 classes)

This work is motivated by materials science applications, specifically analyzing Laser-Induced Periodic Surface Structures (LIPSS).

---

## Model Architecture

The model is a **fusion neural network** consisting of two branches:

### Image Branch

* ResNet18 backbone
* Modified for grayscale SEM images
* Outputs a 512-dimensional feature vector

### Metadata Branch

* Lightweight Multi-Layer Perceptron (MLP)
* Outputs a 32-dimensional feature vector

### Fusion Layer

* Concatenates image and metadata features
* Feeds into:

  * Regression head (predicts Gini score)
  * Classification head (predicts 4 classes)

---

## Features

* 5-Fold Stratified Cross Validation
* Joint Regression + Classification training
* Class imbalance handling via `WeightedRandomSampler`
* Data augmentation for improved generalization
* Early stopping
* Mixed precision training (CUDA support)

### Evaluation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Accuracy
* Confusion Matrix
* Precision / Recall (per class)
* Intersection over Union (IoU / Jaccard Index)
* Macro Precision / Recall
* Mean IoU (mIoU)

---

## Dataset

The dataset includes:

* SEM images
* Metadata features
* Continuous regularity scores
* Discrete labels (4 classes)

Project structure:

```
project_root/
│── images/
│   ├── img_1.png
│   ├── img_2.png
│   └── ...
│
│── master_regularity_full_clean.csv
```

The full dataset is included in this repository for reproducibility and experimentation.

---

## Installation

```bash
git clone https://github.com/KaranVyas7/sem-regularity-prediction.git
cd sem-regularity-prediction

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

Run training with 5-fold cross-validation:

```bash
python train_kfold.py
```

This will output:

* Per-fold metrics
* Confusion matrices
* Aggregated performance statistics

---

## Results

Example output:

```
Fold 1 BEST | MSE=0.0052 MAE=0.0562 ACC=0.667 mIoU=0.479

Confusion Matrix:
[[4 0 0 0]
 [2 2 0 0]
 [0 2 1 1]
 [0 0 1 5]]
```

### Summary

* Accuracy: ~0.65–0.70
* mIoU: ~0.45–0.50

The model performs well overall but shows reduced performance on minority classes due to dataset imbalance.

---

## Reproducibility

To reproduce results:

1. Clone the repository
2. Install dependencies
3. Run:

   ```bash
   python train_kfold.py
   ```

All experiments are fully reproducible using the provided dataset and configuration.

---

## Training Details

* Loss Function:

  ```
  loss = MSE + λ * CrossEntropy
  ```
* Optimizer: AdamW
* Data Augmentation:

  * Random crop
  * Horizontal & vertical flips
  * Rotation
  * Color jitter
  * Gaussian blur

---

## Future Work

* Increase dataset size
* Incorporate segmentation masks
* Experiment with deeper architectures (ResNet50, EfficientNet)
* Hyperparameter tuning
* Ensemble methods
* Integrate FFT-based features

## Author

**Karan Vyas**
Rutgers University – New Brunswick
Computer Science & Data Science

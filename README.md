# SEM Regularity Prediction 

## Overview

This project focuses on predicting surface regularity from Scanning Electron Microscope (SEM) images using a deep learning approach. The model combines image-based features with metadata to perform both:

* **Regression** → Predict continuous regularity score (Gini-based)
* **Classification** → Predict discrete regularity class (4 classes)

The system is designed for materials science applications, specifically analyzing Laser-Induced Periodic Surface Structures (LIPSS).

---

## Model Architecture

The model is a **fusion neural network** consisting of:

* **Image Branch**

  * ResNet18 backbone
  * Modified for grayscale SEM images
  * Outputs 512-dimensional feature vector

* **Metadata Branch**

  * Small Multi-Layer Perceptron (MLP)
  * Outputs 32-dimensional feature vector

* **Fusion Layer**

  * Concatenates image + metadata features
  * Feeds into:

    * Regression head (Gini score)
    * Classification head (4 classes)

---

## Features

* 5-Fold Stratified Cross Validation
* Joint Regression + Classification training
* Class imbalance handling via **WeightedRandomSampler**
* Data augmentation for robustness
* Early stopping for stability
* Mixed precision training (if CUDA available)
* Detailed evaluation metrics:

  * MSE / MAE
  * Accuracy
  * Confusion Matrix
  * Precision / Recall (per class)
  * IoU (Jaccard Index)
  * Macro Precision / Recall
  * Mean IoU (mIoU)

---

## Dataset

The dataset consists of:

* SEM images
* Corresponding metadata features
* Continuous regularity scores
* Discrete labels (4 classes)

Expected structure:

```
project_root/
│── images/
│   ├── img_1.png
│   ├── img_2.png
│   └── ...
│
│── master_regularity_full_clean.csv
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/sem-regularity.git
cd sem-regularity
```

### 2. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Usage

Run training with 5-fold cross-validation:

```
python train_kfold.py
```

Output includes:

* Per-fold metrics
* Confusion matrices
* Aggregated performance statistics

---

## Training Details

* Loss Function:

  * Regression: Mean Squared Error (MSE)
  * Classification: CrossEntropyLoss (with label smoothing)
  * Combined Loss:

    ```
    loss = MSE + λ * CrossEntropy
    ```

* Optimizer:

  * AdamW

* Data Augmentation:

  * Random crop
  * Flips (horizontal + vertical)
  * Rotation
  * Color jitter
  * Gaussian blur

---

## Evaluation Metrics

### Regression

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Classification

* Accuracy
* Confusion Matrix
* Precision / Recall per class
* Intersection over Union (IoU)
* Macro Precision / Recall
* Mean IoU (mIoU)

---

## Example Output

```
Fold 1 BEST | MSE=0.0052 MAE=0.0562 ACC=0.667 mIoU=0.479

Confusion Matrix:
[[4 0 0 0]
 [2 2 0 0]
 [0 2 1 1]
 [0 0 1 5]]
```

---

## Key Observations

* The model performs well on dominant classes but may struggle with minority classes due to dataset imbalance.
* Weighted sampling improves class balance during training.
* Joint regression + classification helps stabilize learning.

---

## Future Improvements

* Increase dataset size (current dataset is small)
* Add segmentation masks for spatial learning
* Use deeper architectures (ResNet50 / EfficientNet)
* Hyperparameter tuning (learning rate, batch size)
* Ensemble models
* Incorporate FFT-based features from SEM images

---

## Research Context

This work is part of the **Aresty Research Program at Rutgers University**, focused on quantifying LIPSS regularity using computer vision and machine learning techniques.

---

## Author

Karan Vyas
Rutgers University – New Brunswick
Computer Science & Data Science
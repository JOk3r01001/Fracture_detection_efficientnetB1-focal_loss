# Fracture_detection_efficientnetB1-focal_loss

Binary classification of bone fractures from X-ray images using EfficientNetB1 and Focal Loss.

## Overview

This project trains a deep learning model to classify X-ray images as **Fractured** or **Non-fractured**. It uses a pretrained EfficientNetB1 as a backbone and a custom classification head. The model is optimized for high recall to minimize missed fractures in a medical context.

## Dataset structure

- **Total images:** 4,083
- **Fractured:** 717
- **Non-fractured:** 3,366
- **Split:** 80 / 10 / 10 (train / val / test), stratified

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.9046 |
| Precision | 0.7143 |
| Recall | 0.7639 |
| AUC | 0.9225 |
| F1 (Fracture) | 0.74 |

### Threshold Analysis

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| 0.30 | 0.2834 | 0.9722 | 0.4389 | 0.5623 |
| 0.40 | 0.4599 | 0.8750 | 0.6029 | 0.7971 |
| **0.50** | **0.7143** | **0.7639** | **0.7383** | **0.9046** |
| 0.60 | 0.8246 | 0.6528 | 0.7287 | 0.9144 |
| 0.70 | 0.9535 | 0.5694 | 0.7130 | 0.9193 |

> For medical use, threshold **0.40** is recommended — it achieves Recall of **0.875**, capturing the majority of fractures at the cost of more false positives.

## Model Architecture

- **Backbone:** EfficientNetB1 pretrained on ImageNet (fully trainable)
- **Head:** GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Sigmoid
- **Input size:** 640×640
- **Loss:** Focal Loss (gamma=2.0, alpha=0.25)
- **Optimizer:** AdamW (lr=0.0005)

## Training

- **Epochs:** up to 50 (EarlyStopping on `val_recall`, patience=10)
- **Batch size:** 8
- **Augmentation:** horizontal flip, rotation ±15°, zoom 10%, brightness ±20%
- **ReduceLROnPlateau:** factor=0.5, patience=5, min_lr=1e-7


## Requirements

```
tensorflow
scikit-learn
pandas
numpy
Pillow
matplotlib
```

## Dataset credit

> Iftekharul Abedeen et al.
> *FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation.*
> Scientific Data, 10(1), 2023.
> https://doi.org/10.1038/s41597-023-02432-4


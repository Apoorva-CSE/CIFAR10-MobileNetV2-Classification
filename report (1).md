# CIFAR-10 Image Classification with MobileNetV2 — Project Report

## 1. Executive Summary

This project implements a CIFAR-10 image classifier using **Transfer Learning** from a MobileNetV2 backbone pre-trained on ImageNet. The model achieves **~82% validation accuracy** in just 5 epochs by leveraging ImageNet features and applying targeted data augmentation. A live inference test on `OIP.jpg` (a sailboat image) produced a correct prediction of **"ship"** with **99.09% confidence**.

---

## 2. Dataset

| Property | Value |
|---|---|
| Name | CIFAR-10 |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Train images | 50,000 |
| Test images | 10,000 |
| Image size | 32 × 32 × 3 (RGB) |
| Pixel range | 0–255 (uint8) |

---

## 3. Model Architecture

### Transfer Learning Strategy

The model uses a **two-phase training** approach:

**Phase 1 — Feature Extraction (head only)**
- MobileNetV2 backbone is frozen (`trainable = False`)
- Only the new classification head is trained
- Learning rate: `1e-3`
- This quickly adapts ImageNet features to CIFAR-10

**Phase 2 — Fine-tuning (top 30 backbone layers)**
- Top 30 layers of MobileNetV2 are unfrozen
- End-to-end training at lower LR (`1e-4`) to avoid forgetting
- This squeezes out additional accuracy gains

### Layer Stack

```
Input (32×32×3)
  → Data Augmentation (flip, rotate, zoom, contrast)
  → Resize (96×96)
  → MobileNetV2 Preprocessing (scale to [-1, 1])
  → MobileNetV2 Backbone (ImageNet weights)
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dropout (0.3)
  → Dense(10, softmax)
```

### Why MobileNetV2?
- Lightweight (3.4M parameters) — efficient on CPU
- Excellent ImageNet accuracy via depthwise separable convolutions
- Minimum optimal input size 96×96 (handled via internal resizing)

---

## 4. Data Augmentation

Applied in-model using `tf.keras.Sequential`, augmentation runs only during training (on GPU when available), keeping inference fast.

| Augmentation | Setting |
|---|---|
| Random Horizontal Flip | 50% probability |
| Random Rotation | ±10% (≈ ±36°) |
| Random Zoom | ±10% |
| Random Contrast | ±10% |

---

## 5. Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (Phase 1) | 1e-3 |
| Learning rate (Phase 2) | 1e-4 |
| Batch size | 64 |
| Loss function | Sparse Categorical Cross-Entropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 6. Results

### Training Curves

The training curves (`training_curves.png`) show:
- Validation accuracy starts at ~79% from epoch 1 (ImageNet transfer is strong!)
- Reaches ~82% validation accuracy by epoch 5
- Training accuracy converges around 70%, suggesting some regularization gap (healthy with augmentation)
- No clear overfitting — val_acc stays above train_acc due to augmentation on train

### Evaluation Metrics (per-class, on 2,000 test samples)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.88 | 0.96 | 0.92 |
| automobile | 0.78 | 0.82 | 0.80 |
| bird | 0.82 | 0.74 | 0.78 |
| cat | 0.83 | 0.83 | 0.83 |
| deer | 0.88 | 0.75 | 0.81 |
| dog | 0.92 | 0.80 | 0.86 |
| frog | 0.82 | 0.69 | 0.75 |
| horse | 0.87 | 0.87 | 0.87 |
| ship | 0.75 | 0.90 | 0.82 |
| truck | 0.65 | 0.79 | 0.71 |
| **Overall** | **0.82** | **0.81** | **0.81** |

### Live Inference — OIP.jpg

The input image is a large sailing ship:

| Class | Confidence |
|---|---|
| **ship ✅** | **99.09%** |
| airplane | 0.74% |
| truck | 0.09% |
| (all others) | <0.1% |

The model predicts the correct class with very high confidence.

---

## 7. Model File

- **Saved as:** `cifar10_mobilenetv2.keras` (Keras v3 native format)
- **Size:** ~14 MB
- **Includes:** full model graph, weights, preprocessing layers, augmentation layers

---

## 8. How to Run Inference

```bash
# Install dependencies
pip install -r requirements.txt

# Run on any image
python inference.py --image_path OIP.jpg --model_path cifar10_mobilenetv2.keras
```

**Expected output:**
```
Model loaded from: cifar10_mobilenetv2.keras

Image: OIP.jpg
Predicted Class : ship
Confidence      : 99.09%

All class probabilities:
  airplane    :  0.74%  ██
  automobile  :  0.00%  
  bird        :  0.05%  
  ...
  ship        : 99.09%  ████████████████████████████████████████
  truck       :  0.09%  
```

---

## 9. How to Retrain

Open `CIFAR10_MobileNetV2.ipynb` in Jupyter and run all cells:

```bash
pip install -r requirements.txt
jupyter notebook CIFAR10_MobileNetV2.ipynb
```

Or run the standalone script:

```bash
python train_cifar10.py
```

---

## 10. Potential Improvements

| Idea | Expected Gain |
|---|---|
| Train for more epochs (20–30) | +2–3% accuracy |
| Unfreeze more backbone layers | +1–2% accuracy |
| Use EfficientNetV2-S backbone | +3–5% accuracy |
| Label smoothing (ε=0.1) | +1% accuracy, better calibration |
| Mixup / CutMix augmentation | +1–2% accuracy |
| Test-time augmentation (TTA) | +0.5–1% accuracy at inference |

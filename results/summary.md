# Results Summary

## Dataset Overview

| Split | Images | Makes | Models |
|-------|--------|-------|--------|
| Train | 16,016 | 75 | 431 |
| Test | 14,939 | 75 | 431 |
| Total | 30,955 | 75 | 431 |

## Model Comparison (Make Classification - 75 classes)

### Validation Set Results (During Training)

| Model | Pretrained | Loss | Best Val Acc | Train Acc | Params | Epochs | Time |
|-------|------------|------|--------------|-----------|--------|--------|------|
| SimpleCNN | No | Label Smooth | 74.78% | 85.80% | 1.60M | 100 | 0:22:40 |
| EfficientNet-B0 | Yes | Label Smooth | 97.75% | 99.93% | 4.10M | 100 | 0:52:42 |
| ResNet50 | Yes | CE | 97.75% | 99.97% | 24.60M | 100 | 1:21:39 |
| ResNet50 | Yes | Focal | 96.69% | 99.94% | 24.60M | 100 | 1:21:48 |
| ResNet50 | Yes | Label Smooth | 97.57% | 99.95% | 24.60M | 100 | 1:23:22 |
| ResNet50-SE | Yes | Label Smooth | 98.00% | 99.99% | 24.36M | 100 | 1:25:33 |
| **Hierarchical** | Yes | Hierarchical | **98.94%** | 100.00% | 25.87M | 100 | 1:24:28 |

### Test Set Results (Final Evaluation)

| Model | Test Top-1 | Test Top-5 | Precision | Recall | F1 |
|-------|------------|------------|-----------|--------|-----|
| SimpleCNN | 53.08% | 77.12% | 0.5721 | 0.4260 | 0.4592 |
| EfficientNet-B0 | 86.95% | 94.50% | 0.8974 | 0.8327 | 0.8528 |
| ResNet50 (CE) | 82.28% | 91.88% | 0.8386 | 0.7687 | 0.7893 |
| ResNet50 (Focal) | 80.39% | 91.12% | 0.8299 | 0.7419 | 0.7634 |
| ResNet50 (LS) | 82.69% | 88.47% | 0.7941 | 0.8142 | 0.7699 |
| ResNet50-SE | 84.48% | 93.78% | 0.9107 | 0.8208 | 0.8532 |
| **Hierarchical** | **87.55%** | **94.77%** | **0.9096** | **0.8486** | **0.8702** |

## Loss Function Comparison (ResNet50)

| Loss Function | Val Acc | Test Acc | Notes |
|---------------|---------|----------|-------|
| Cross-Entropy | 97.75% | 82.28% | baseline |
| Focal (gamma=2.0) | 96.69% | 80.39% | -1.89% test |
| Label Smoothing (0.1) | 97.57% | 82.69% | +0.41% test |

## Hierarchical Model Results

| Task | Val Accuracy | Train Accuracy |
|------|--------------|----------------|
| Make (75 classes) | 98.94% | 100.00% |
| Model (431 classes) | 95.51% | 99.99% |

## Baseline Comparison

| Method | Source | Make Acc |
|--------|--------|----------|
| OverFeat All-View | Yang et al., 2015 | 82.9% |
| **Our Best (Hierarchical)** | This work | **87.55%** |
| Improvement | - | **+4.65%** |

## Key Findings

1. **Transfer learning is essential**: Pretrained models (82-88% test) vastly outperform training from scratch (53% test)
2. **Hierarchical learning achieves best results**: Joint make+model training achieves best test accuracy (87.55%)
3. **EfficientNet offers best efficiency**: 86.95% test accuracy with only 4.1M parameters (vs 24-26M for ResNet variants)
4. **SE attention provides improvement**: +1.79% test accuracy over standard ResNet50 with label smoothing
5. **Loss function has limited impact on test accuracy**: All ResNet50 variants within 2% on test set

---
*Results from 100-epoch training run (2026-01-09)*

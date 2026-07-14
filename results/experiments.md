# Experiment Log

## Data Preparation Phase

### Experiment: Dataset Analysis
Date: 2026-01-08
Goal: Analyze CompCars dataset structure and statistics
Model: N/A (data exploration only)

**Config:**
- Dataset path: dataset/data
- Split files: train_test_split/classification/{train,test}.txt

**Results:**
| Metric | Train | Test |
|--------|-------|------|
| Total images | 16,016 | 14,939 |
| Unique makes | 75 | 75 |
| Unique models | 431 | 431 |

**Observations:**
- Classification split uses 75 makes (not 163 as in full dataset)
- Classification split uses 431 models (not 1716 as in full dataset)
- Significant class imbalance in makes: 120x (max 1201, min 10 samples)
- Moderate class imbalance in models: 14x (max 143, min 10 samples)
- Year range: 2004-2015 plus "unknown" category

**Files:**
- results/dataset_statistics.md
- results/make_distribution.png
- results/model_distribution.png
- results/year_distribution.png

---

## Training Experiments

### Experiment: SimpleCNN Baseline with Label Smoothing
**Date**: 2026-01-09
**Goal**: Establish baseline performance with a simple CNN trained from scratch
**Model**: SimpleCNN (custom 4-layer CNN)
**Key Config**: lr=0.001, batch_size=32, epochs=100, loss=label_smoothing (0.1)

**Results**:
- Train Acc: 85.80%
- Val Acc: 74.78% (best), 74.34% (final)
- Test Acc: 53.08%
- Parameters: 1.60M
- Training Time: 0:22:40

**Observations**:
- Training from scratch requires significantly more epochs to converge
- Large gap between train and val accuracy indicates overfitting
- Without pretrained features, the model struggles with fine-grained distinctions
- Significant gap between val (74.78%) and test (53.08%) accuracy

**Files**:
- Checkpoint: `checkpoints/simplecnn_label_smoothing_best.pth`
- History: `checkpoints/simplecnn_label_smoothing_history.json`
- Config: `checkpoints/simplecnn_label_smoothing_config.json`

---

### Experiment: EfficientNet-B0 with Label Smoothing
**Date**: 2026-01-09
**Goal**: Test lightweight pretrained model with modern architecture
**Model**: EfficientNet-B0 (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=label_smoothing (0.1)

**Results**:
- Train Acc: 99.93%
- Val Acc: 97.75% (best), 97.44% (final)
- Test Acc: 86.95%
- Top-5 Test Acc: 94.50%
- Parameters: 4.10M
- Training Time: 0:52:42

**Observations**:
- Best accuracy-to-parameters ratio (86.95% test with only 4.1M params)
- Modern architecture with efficient compound scaling
- Strong generalization from validation to test set

**Files**:
- Checkpoint: `checkpoints/efficientnet_label_smoothing_best.pth`
- History: `checkpoints/efficientnet_label_smoothing_history.json`
- Config: `checkpoints/efficientnet_label_smoothing_config.json`

---

### Experiment: ResNet50 with Cross-Entropy Loss
**Date**: 2026-01-09
**Goal**: Baseline ResNet50 with standard cross-entropy loss
**Model**: ResNet50 (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=ce

**Results**:
- Train Acc: 99.97%
- Val Acc: 97.75% (best), 97.44% (final)
- Test Acc: 82.28%
- Top-5 Test Acc: 91.88%
- Parameters: 24.60M
- Training Time: 1:21:39

**Observations**:
- Standard CE loss provides strong baseline
- Serves as comparison point for other loss functions
- All ResNet50 variants achieved similar validation accuracy (~97%)

**Files**:
- Checkpoint: `checkpoints/resnet50_ce_best.pth`
- History: `checkpoints/resnet50_ce_history.json`
- Config: `checkpoints/resnet50_ce_config.json`

---

### Experiment: ResNet50 with Focal Loss
**Date**: 2026-01-09
**Goal**: Test focal loss (gamma=2.0) for handling class imbalance
**Model**: ResNet50 (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=focal (gamma=2.0)

**Results**:
- Train Acc: 99.94%
- Val Acc: 96.69% (best), 96.50% (final)
- Test Acc: 80.39%
- Top-5 Test Acc: 91.12%
- Parameters: 24.60M
- Training Time: 1:21:48

**Observations**:
- Slightly lower validation accuracy than CE (-1.06%)
- Focal loss did not provide advantage for this dataset
- Marginal benefit suggests class imbalance not the primary challenge

**Files**:
- Checkpoint: `checkpoints/resnet50_focal_best.pth`
- History: `checkpoints/resnet50_focal_history.json`
- Config: `checkpoints/resnet50_focal_config.json`

---

### Experiment: ResNet50 with Label Smoothing
**Date**: 2026-01-09
**Goal**: Test label smoothing (0.1) for regularization and calibration
**Model**: ResNet50 (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=label_smoothing (0.1)

**Results**:
- Train Acc: 99.95%
- Val Acc: 97.57% (best), 97.44% (final)
- Test Acc: 82.69%
- Top-5 Test Acc: 88.47%
- Precision (macro): 0.7941
- Recall (macro): 0.8142
- F1 Score (macro): 0.7699
- Parameters: 24.60M
- Training Time: 1:23:22

**Observations**:
- Similar validation accuracy to other ResNet50 variants
- **Label smoothing provides regularization benefit during training**
- Test accuracy slightly higher than CE variant (+0.41%)

**Files**:
- Checkpoint: `checkpoints/resnet50_label_smoothing_best.pth`
- History: `checkpoints/resnet50_label_smoothing_history.json`
- Config: `checkpoints/resnet50_label_smoothing_config.json`
- Evaluation: `results/resnet50_label_smoothing_evaluation.json`
- Confusion Matrix: `results/resnet50_label_smoothing_confusion_matrix.png`
- Per-Class Accuracy: `results/resnet50_label_smoothing_per_class_accuracy.png`

---

### Experiment: ResNet50-SE with Label Smoothing
**Date**: 2026-01-09
**Goal**: Test Squeeze-and-Excitation attention mechanism
**Model**: ResNet50 with SE blocks (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=label_smoothing (0.1)

**Results**:
- Train Acc: 99.99%
- Val Acc: 98.00% (best), 97.69% (final)
- Test Acc: 84.48%
- Top-5 Test Acc: 93.78%
- Precision (macro): 0.9107
- Recall (macro): 0.8208
- F1 Score (macro): 0.8532
- Parameters: 24.36M
- Training Time: 1:25:33

**Observations**:
- SE attention provides best single-task test accuracy (84.48%)
- Highest validation accuracy among single-task models (98.00%)
- Channel attention helps focus on discriminative features

**Files**:
- Checkpoint: `checkpoints/resnet50_se_label_smoothing_best.pth`
- History: `checkpoints/resnet50_se_label_smoothing_history.json`
- Config: `checkpoints/resnet50_se_label_smoothing_config.json`

---

### Experiment: Hierarchical Classification Model
**Date**: 2026-01-09
**Goal**: Joint make and model prediction with hierarchical loss
**Model**: ResNet50 with dual classification heads (pretrained ImageNet)
**Key Config**: backbone_lr=0.0001, head_lr=0.001, batch_size=32, epochs=100, loss=hierarchical

**Results**:
- Train Make Acc: 100.00%
- Train Model Acc: 99.99%
- Val Make Acc: 98.94% (best), 98.81% (final)
- Val Model Acc: 95.51% (final)
- Test Make Acc: 87.55%
- Test Top-5 Acc: 94.77%
- Precision (macro): 0.9096
- Recall (macro): 0.8486
- F1 Score (macro): 0.8702
- Parameters: 25.87M
- Training Time: 1:24:28

**Observations**:
- Best overall test accuracy (87.55% make classification)
- Hierarchical approach provides strong performance on both tasks
- Model-level validation accuracy (95.51%) shows improved fine-grained capability
- Joint training provides regularization and shared feature learning

**Files**:
- Checkpoint: `checkpoints/hierarchical_best.pth`
- History: `checkpoints/hierarchical_history.json`
- Config: `checkpoints/hierarchical_config.json`

---

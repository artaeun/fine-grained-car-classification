# Observations Log

## 2026-01-08: Data Preparation

### Dataset Discovery
- The initial plan mentioned 163 makes and 1716 models, but the official classification split only uses:
  - 75 unique makes
  - 431 unique models
- This is consistent with the original paper (Yang et al., 2015) which mentions 431 car models for classification

### Class Imbalance
- Make classification has severe imbalance (120x ratio between max and min class)
- Model classification has moderate imbalance (14x ratio)
- Class weights implemented in dataset.py for weighted loss training

### Data Quality
- Some images have "unknown" year field instead of numeric year
- All images load correctly as RGB
- Image sizes vary widely (need pre-resize for efficiency)

### Implementation Notes
- DataLoader prefetch_factor and persistent_workers only work with num_workers > 0
- Added conditional handling in get_dataloader() function
- Transforms follow ImageNet normalization (required for pretrained models)

---

## 2026-01-08: Model Architecture (SimpleCNN)

### Design Decisions
- chose vgg-style architecture for simplicity and educational value
- 4 conv blocks with increasing channels: 64 → 128 → 256 → 512
- used batch normalization after each conv layer for training stability
- global average pooling reduces parameters vs full flatten
- dropout 0.5 before classifier prevents overfitting on small dataset

### Parameter Count
- total parameters: ~1.6M (much smaller than ResNet50's 25.6M)
- fast to train: estimated ~40 min on RTX 3060

### Architecture Notes
- first conv uses 7x7 kernel with stride 2 (similar to ResNet stem)
- subsequent convs use 3x3 kernels (standard choice)
- final adaptive avg pool outputs 1x1 spatial size regardless of input

### Expected Behavior
- should underperform pretrained models significantly (40-50% vs 75-85%)
- will demonstrate the value of transfer learning in report

---

## 2026-01-08: Model Architecture (ResNet50)

### Design Decisions
- chose ResNet50 as main pretrained model for best accuracy/time ratio
- uses ImageNet pretrained weights (IMAGENET1K_V1) for rich visual features
- replaced final FC layer with custom two-layer classifier head
- added dropout (0.5 and 0.3) before both FC layers to prevent overfitting
- intermediate FC layer (512 units) provides gradual dimensionality reduction

### Why ResNet50 over ResNet18/101?
- ResNet18: faster but less capacity (11.7M params vs 25.6M)
- ResNet101/152: diminishing returns, exceeds 2-hour training target
- ResNet50: sweet spot with 50 layers and bottleneck blocks

### Transfer Learning Strategy
- lower learning rate (0.0001) for backbone to preserve ImageNet features
- backbone already knows edges, textures, shapes from 1000-class pretraining
- classifier head learns car-specific features with standard LR

### Parameter Count
- total parameters: ~24.6M
- trainable parameters: ~24.6M (all layers trainable)
- bottleneck blocks (1x1→3x3→1x1) more efficient than basic blocks

### Expected Behavior
- should achieve 75-82% top-1 accuracy
- significant improvement over SimpleCNN (40-50%)
- competitive with original OverFeat results (82.9%)

---

## 2026-01-08: Model Architecture (EfficientNet-B0)

### Design Decisions
- chose EfficientNet-B0 to compare modern efficient architecture against ResNet50
- uses ImageNet pretrained weights (IMAGENET1K_V1)
- replaced classifier with single FC layer (simpler than ResNet50's two-layer head)
- dropout 0.3 matches original EfficientNet design

### Why EfficientNet-B0?
- compound scaling: neural architecture search found optimal depth/width/resolution balance
- MBConv blocks with depthwise separable convolutions are parameter efficient
- squeeze-and-excitation attention improves feature selection
- 6x fewer parameters than ResNet50 (4.1M vs 24.6M) with similar accuracy

### Key Innovations
- Swish activation (x * sigmoid(x)) outperforms ReLU
- depthwise separable convolutions reduce compute cost
- SE attention recalibrates channel-wise features
- inverted residuals (expand→depthwise→project) from MobileNetV2

### Transfer Learning Strategy
- lower learning rate (0.0001) for backbone to preserve ImageNet features
- same strategy as ResNet50 for fair comparison
- simpler classifier head since EfficientNet already has rich feature representation

### Parameter Count
- total parameters: ~4.1M
- trainable parameters: ~4.1M (all layers trainable)
- much more efficient than ResNet50 (24.6M)

### Expected Behavior
- should achieve 75-82% top-1 accuracy (similar to ResNet50)
- faster training due to fewer parameters
- demonstrates that newer architectures achieve efficiency gains

---

## 2026-01-08: Model Architecture (ResNet50-SE)

### Design Decisions
- chose to add SE blocks after ResNet stages for originality element
- uses ImageNet pretrained weights as backbone (IMAGENET1K_V1)
- SE blocks placed after each of the 4 ResNet layers
- simpler single-FC classifier head (vs two-layer in base ResNet50)

### Why SE Attention for Fine-Grained Classification?
- channel attention helps focus on discriminative features
- car makes differ in grille shapes, headlights, body lines - SE learns which channels capture these
- position-invariant attention is robust to different viewpoints
- proven effective in original SE-Net paper for ImageNet classification

### SE Block Implementation Details
- reduction ratio of 16: squeeze to C/16 dimensions, then expand back
- two FC layers: Linear(C→C/16) + ReLU + Linear(C/16→C) + Sigmoid
- output weights in [0,1] range multiply original features
- placed after each ResNet stage, not inside bottleneck blocks

### Transfer Learning Strategy
- lower learning rate (0.0001) for backbone to preserve ImageNet features
- SE blocks initialized randomly but small (only ~0.7M params total)
- SE weights learn to recalibrate pretrained features for car classification

### Parameter Count
- total parameters: ~24.4M (slightly more than ResNet50 due to SE blocks)
- SE blocks add: 8K + 32K + 131K + 524K = ~696K parameters
- trainable parameters: ~24.4M (all layers trainable)

### Expected Behavior
- should achieve 80-85% top-1 accuracy (best model)
- slight training time increase due to SE forward pass overhead
- demonstrates attention mechanism understanding for originality points

---

## 2026-01-08: Model Architecture (HierarchicalClassifier)

### Design Decisions
- chose coarse-to-fine multi-task learning approach for originality
- single shared ResNet50 backbone (practical vs 75 separate model classifiers)
- two parallel classification heads for make (75) and model (431) classes
- hierarchical loss combines both tasks with tunable alpha weight

### Why Multi-Task Hierarchical Learning?
- exploits natural car hierarchy: make → model
- shared backbone learns features useful for both tasks
- more practical than training 75+ separate model classifiers (would take 150+ hours)
- enables error correlation analysis: "do make errors cause model errors?"

### HierarchicalLoss Design
- alpha=0.3 by default: 30% make loss + 70% model loss
- focuses more on harder fine-grained model classification
- label smoothing (0.1) prevents overconfidence
- total_loss, make_loss, model_loss all returned for monitoring

### Architecture Choices
- ResNet50 backbone with fc replaced by Identity
- two-layer classifier heads (2048→512→num_classes)
- dropout before both FC layers (0.5 and 0.3)
- parallel heads share same 2048-dim feature vector

### Training Modifications
- separate train_hierarchical_epoch and validate_hierarchical functions
- dataset returns (image, make_label, model_label) triplets
- scheduler and early stopping track make accuracy (coarse task)
- history tracks both make and model accuracy separately

### Parameter Count
- total parameters: ~25.9M
- backbone: ~23.5M
- make head: ~1.09M
- model head: ~1.27M (larger due to 431 output classes)

### Expected Behavior
- make accuracy: 78-83% (comparable to single-task)
- model accuracy: 65-70% (harder fine-grained task)
- demonstrates multi-task learning and hierarchical reasoning

---

## 2026-01-09: Training Framework (Phase 3 Extended)

### Loss Functions Implemented
- **CrossEntropyLoss (ce)**: baseline loss function
- **Weighted CrossEntropyLoss (weighted_ce)**: uses class weights for imbalance
- **FocalLoss (focal)**: focuses on hard examples, gamma=2.0 by default
- **Label Smoothing (label_smoothing)**: built-in PyTorch CE with smoothing=0.1
- **WeightedFocalLoss (weighted_focal)**: combines focal loss with class weights

### Why Multiple Loss Functions?
- class imbalance in CompCars dataset (120x ratio for makes, 14x for models)
- focal loss down-weights easy examples, focuses on hard-to-classify samples
- label smoothing prevents overconfidence and improves generalization
- allows ablation study comparing loss function effects

### Differential Learning Rates Design
- pretrained backbone: 0.0001 (10x lower than head)
- classifier head: 0.001 (standard learning rate)
- preserves rich ImageNet features in backbone
- allows head to learn car-specific patterns faster

### Why Differential LR?
- pretrained features are already good for visual recognition
- aggressive updates would destroy useful representations
- classifier head is randomly initialized, needs faster learning
- proven effective in fine-tuning transfer learning literature

### Gradient Accumulation Implementation
- ACCUMULATION_STEPS config (default=1, no accumulation)
- effective batch size = batch_size * accumulation_steps
- useful when running out of VRAM on large models
- gradients accumulated before optimizer step

### When to Use Gradient Accumulation
- if batch_size=32 causes OOM, use batch_size=16 with accumulation_steps=2
- maintains effective batch statistics for BatchNorm
- small training time overhead from extra forward passes

### Hierarchical Model Plotting
- added plot_hierarchical_curves function
- 2x2 subplot layout: loss, make accuracy, model accuracy, learning rate
- separate curves for training and validation
- automatically saved to results/ directory

### Implementation Files
- `src/losses.py`: FocalLoss, LabelSmoothingCrossEntropy, WeightedFocalLoss, get_loss_function
- `src/training.py`: get_optimizer_with_differential_lr, train_one_epoch_with_accumulation, plot_hierarchical_curves
- `train.py`: added LOSS_TYPE, BACKBONE_LR, HEAD_LR, ACCUMULATION_STEPS config

---

## 2026-01-09: Full Training Run (100 Epochs)

### Training Session Overview
- Ran 7 experiments sequentially using 02_train.ipynb
- Total training time: ~8 hours
- All models trained with seed=2128831 for reproducibility
- Mixed precision training (AMP) enabled throughout

### Final Results Summary
| Experiment | Best Val Acc | Test Acc | Training Time |
|------------|--------------|----------|---------------|
| simplecnn_label_smoothing | 74.78% | 53.08% | 0:22:40 |
| efficientnet_label_smoothing | 97.75% | 86.95% | 0:52:42 |
| resnet50_ce | 97.75% | 82.28% | 1:21:39 |
| resnet50_focal | 96.69% | 80.39% | 1:21:48 |
| resnet50_label_smoothing | 97.57% | 82.69% | 1:23:22 |
| resnet50_se_label_smoothing | 98.00% | 84.48% | 1:25:33 |
| **hierarchical** | **98.94%** | **87.55%** | 1:24:28 |

### Key Observations

#### Transfer Learning is Essential
- SimpleCNN (from scratch): 74.78% validation, 53.08% test accuracy
- Pretrained models: 96-99% validation, 80-88% test accuracy
- Gap of 22-35% on test set demonstrates critical importance of ImageNet pretraining

#### All Pretrained Models Achieve High Validation Accuracy
| Loss Function | ResNet50 Val Acc | Test Acc |
|---------------|------------------|----------|
| Cross-Entropy | 97.75% | 82.28% |
| Focal (gamma=2.0) | 96.69% | 80.39% |
| Label Smoothing | 97.57% | 82.69% |

- All variants achieve similar validation accuracy (~97%)
- Test accuracy varies within 2% range
- Loss function has limited impact on final generalization

#### Hierarchical Model Achieves Best Results
- Make accuracy: 98.94% validation, 87.55% test (best overall)
- Model accuracy: 95.51% validation (fine-grained task)
- Joint training provides regularization for coarse-grained task
- Exceeds original paper baseline (82.9%) by +4.65%

#### Validation vs Test Accuracy Gap
All models show significant gap between validation and test accuracy:

| Model | Val Acc | Test Acc | Gap |
|-------|---------|----------|-----|
| SimpleCNN | 74.78% | 53.08% | 21.70% |
| EfficientNet | 97.75% | 86.95% | 10.80% |
| ResNet50 (CE) | 97.75% | 82.28% | 15.47% |
| ResNet50-SE | 98.00% | 84.48% | 13.52% |
| Hierarchical | 98.94% | 87.55% | 11.39% |

- All pretrained models show 10-15% gap
- SimpleCNN has largest gap (21.70%)
- EfficientNet and Hierarchical have best generalization

#### Training Efficiency
| Model | Params | Time | Test Acc |
|-------|--------|------|----------|
| EfficientNet | 4.1M | 0:52:42 | 86.95% |
| SimpleCNN | 1.6M | 0:22:40 | 53.08% |
| ResNet50 (LS) | 24.6M | 1:23:22 | 82.69% |
| ResNet50-SE | 24.4M | 1:25:33 | 84.48% |
| Hierarchical | 25.9M | 1:24:28 | 87.55% |

- EfficientNet best accuracy-to-parameters ratio (86.95% with 4.1M params)
- Hierarchical model achieves best test accuracy (87.55%)

### Unexpected Findings
1. Focal loss did not outperform cross-entropy despite 120x class imbalance
2. SE attention provided +1.79% test improvement over base ResNet50
3. EfficientNet (4.1M params) reaches 86.95% vs ResNet50's 82.69% (24.6M params)
4. Hierarchical approach exceeds baseline by larger margin on test set (+4.65%)

### What Would Be Done Differently
1. More aggressive data augmentation for better generalization
2. Test-time augmentation for final evaluation
3. Ensemble methods combining best models

---

## 2026-01-09: Results vs Original Paper

### Comparison with Yang et al., 2015
| Method | Make Acc (Test) |
|--------|-----------------|
| OverFeat All-View (paper) | 82.9% |
| Our Hierarchical | 87.55% |
| **Improvement** | **+4.65%** |

### Why We Outperform 2015 Results
1. Modern architectures (ResNet50 vs OverFeat)
2. Better pretrained weights (ImageNet-1K V1 vs 2015 era)
3. Improved training techniques (AMP, label smoothing, differential LR)
4. Better optimizers (AdamW vs SGD)

### Notes
- Comparison based on test set evaluation (87.55%)
- Official classification split used (75 makes, 431 models)
- Results evaluated on same test split as original paper

---

## 2026-01-10: Generalization Gap Analysis

### What is Generalization Gap?
- Defined as: Training Accuracy - Validation Accuracy
- Measures degree of overfitting during training
- Lower values indicate better generalization to unseen data
- High gap suggests model memorizes training data rather than learning generalizable features

### Analysis Implementation
Added `compute_generalization_gap_stats()` function to 04_analysis.ipynb:
- Computes full generalization gap vector per epoch for each model
- Calculates average gap after epoch 20 (post-convergence phase)
- Calculates average gap for last 10 epochs (final training phase)
- Reports final epoch gap value

### Output Files
- `results/comparison/generalization_gap_stats.csv` - raw statistics
- `results/comparison/generalization_gap_stats.md` - formatted documentation

### Key Metrics
| Metric | Description |
|--------|-------------|
| Avg Gap (Epoch 21+) | Average gap after initial convergence |
| Avg Gap (Last 10) | Average gap in final training phase |
| Final Gap | Gap at the last epoch of training |

### Interpretation Guidelines
- Gap < 5%: Excellent generalization
- Gap 5-10%: Good generalization, moderate overfitting
- Gap 10-20%: Significant overfitting
- Gap > 20%: Severe overfitting, model may be memorizing

### Why This Analysis Matters
- Helps identify which models overfit most
- Shows effectiveness of regularization techniques (label smoothing, dropout)
- Provides insight into training dynamics
- Useful for comparing different loss functions and architectures

---

## 2026-01-09: Comprehensive Evaluation (Phase 5)

### Evaluation Notebook Design Decisions

#### Models Evaluated
Selected 7 primary experiments (best variant per architecture):
1. `simplecnn_label_smoothing` - baseline without transfer learning
2. `efficientnet_label_smoothing` - lightweight modern architecture
3. `resnet50_ce` - standard loss baseline
4. `resnet50_focal` - focal loss for class imbalance
5. `resnet50_label_smoothing` - best single-task model
6. `resnet50_se_label_smoothing` - attention mechanism (originality)
7. `hierarchical` - joint make/model prediction

#### Per-Model Outputs (results/{model_name}/)
Each model generates the following artifacts:
- `confusion_matrix.png` - 75x75 class confusion heatmap
- `per_class_accuracy.png` - top/bottom 15 classes bar chart
- `sample_correct.png` - 4x4 grid of correct predictions with class labels
- `sample_incorrect.png` - 4x4 grid of misclassifications with true/pred labels
- `evaluation.json` - all metrics in JSON format
- `classification_report.txt` - sklearn per-class precision/recall/f1

#### Cross-Model Outputs (results/)
Comparative visualizations for the report:
- `comparison_training_curves.png` - loss/accuracy vs epochs for all models
- `comparison_accuracy_bar.png` - top-1 and top-5 accuracy grouped bars
- `comparison_acc_vs_params.png` - accuracy vs model size scatter plot
- `comparison_summary.csv` - tabular data for report tables
- `comparison_summary.md` - markdown format summary

### Visualization Design Choices

#### Confusion Matrix
- Used seaborn heatmap with Blues colormap
- Disabled annotations for 75-class matrix (unreadable)
- Full matrix shown without filtering to reveal error patterns
- Diagonal represents correct classifications

#### Per-Class Accuracy
- Shows bottom 15 and top 15 classes side-by-side
- Highlights which makes are hardest/easiest to classify
- Uses class indices (not names) since make names not in dataset

#### Sample Predictions
- 4x4 grid layout (16 samples)
- Random sampling with fixed seed for reproducibility
- Original images loaded (without transforms) for display clarity
- Green titles for correct, red titles for incorrect predictions
- Misclassification grids show both true and predicted labels

#### Training Curves
- 2x2 subplot layout: train loss, val loss, train acc, val acc
- All models plotted on same axes for direct comparison
- Used tab10 colormap for distinguishable colors
- Legend with model display names

#### Accuracy Comparison
- Grouped bar chart with top-1 and top-5 side by side
- Value annotations on bars for precise reading
- X-axis shows display names (e.g., "ResNet50 (LS)" not "resnet50_label_smoothing")

### Metrics Computed

#### Primary Metrics
- Top-1 Accuracy: correct / total
- Top-5 Accuracy: true label in top-5 predictions
- Precision (macro): average precision across all classes
- Recall (macro): average recall across all classes
- F1 Score (macro): harmonic mean of precision and recall

#### Additional Analysis for Hierarchical Model
- Make task: 75-class coarse classification
- Model task: 431-class fine-grained classification
- Error correlation: % of model errors when make is also wrong

### Code Style Compliance
- Lowercase comments only
- No list comprehensions (all converted to for loops)
- No ternary operators (if/else blocks used)
- No argparse (configuration variables at top)
- Expanded dictionary literals (step-by-step assignment)

### Output Directory Structure
```
results/
├── comparison_training_curves.png
├── comparison_accuracy_bar.png
├── comparison_acc_vs_params.png
├── comparison_summary.csv
├── comparison_summary.md
├── simplecnn_label_smoothing/
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── sample_correct.png
│   ├── sample_incorrect.png
│   ├── evaluation.json
│   └── classification_report.txt
├── efficientnet_label_smoothing/
│   └── [same structure]
├── resnet50_ce/
│   └── [same structure]
├── resnet50_focal/
│   └── [same structure]
├── resnet50_label_smoothing/
│   └── [same structure]
├── resnet50_se_label_smoothing/
│   └── [same structure]
└── hierarchical/
    └── [same structure]
```

### Report-Ready Visualizations

#### Required (per REPORT_GUIDE.md)
1. Training curves: `results/comparison_training_curves.png`
2. Confusion matrix: `results/{best_model}/confusion_matrix.png`
3. Architecture comparison: `results/comparison_accuracy_bar.png`
4. Per-class accuracy: `results/{best_model}/per_class_accuracy.png`

#### Recommended
5. Sample predictions: `results/{model}/sample_correct.png`, `sample_incorrect.png`
6. Accuracy vs parameters: `results/comparison_acc_vs_params.png`

#### Data for Tables
- CSV import: `results/comparison_summary.csv`
- Markdown copy: `results/comparison_summary.md`

### Hierarchical Model Dual-Head Visualization

Added dedicated `plot_hierarchical_samples()` function to showcase the hierarchical model's dual-head classification capability.

#### Display Format
Each sample shows both predictions:
```
make: 42 (ok)        <- green if correct
model: 187 (X)       <- red if incorrect
  true: 203          <- shows true label when wrong
```

#### Color Coding
- Green title: both make AND model correct
- Orange title: one correct, one wrong
- Red title: both wrong

#### Why Separate Function
- Standard `plot_sample_predictions()` only shows single-task output
- Hierarchical model outputs two predictions per image
- Need to display both heads to demonstrate the multi-task architecture
- Shows correlation between make/model errors (can the model get the model right if make is wrong?)

#### Output Files
- `results/hierarchical/sample_correct.png` - samples where make prediction is correct
- `results/hierarchical/sample_incorrect.png` - samples where make prediction is wrong
- Both show the model prediction status alongside make prediction

---

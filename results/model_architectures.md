# Model Architectures

## SimpleCNN (Baseline)

### Purpose
- baseline model trained from scratch (no transfer learning)
- establishes lower-bound performance for comparison
- demonstrates the value of pretrained models

### Architecture

```
Input: [batch, 3, 224, 224]
    |
Conv Block 1: Conv2d(3→64, k=7, s=2, p=3) + BN + ReLU + MaxPool(k=3, s=2, p=1)
    | Output: [batch, 64, 56, 56]
    |
Conv Block 2: Conv2d(64→128, k=3, p=1) + BN + ReLU + MaxPool(k=2)
    | Output: [batch, 128, 28, 28]
    |
Conv Block 3: Conv2d(128→256, k=3, p=1) + BN + ReLU + MaxPool(k=2)
    | Output: [batch, 256, 14, 14]
    |
Conv Block 4: Conv2d(256→512, k=3, p=1) + BN + ReLU + AdaptiveAvgPool(1,1)
    | Output: [batch, 512, 1, 1]
    |
Flatten: [batch, 512]
    |
Classifier: Dropout(0.5) + Linear(512→num_classes)
    | Output: [batch, num_classes]
```

### Parameters
| Component | Parameters |
|-----------|------------|
| Conv Block 1 | ~9,536 |
| Conv Block 2 | ~73,984 |
| Conv Block 3 | ~295,680 |
| Conv Block 4 | ~1,180,672 |
| Classifier (75 classes) | ~38,475 |
| **Total** | **~1,598,347 (~1.6M)** |

### Design Choices
- VGG-style architecture with increasing channel depth (64→128→256→512)
- batch normalization after each conv for stable training
- dropout (0.5) before classifier to prevent overfitting
- global average pooling instead of flatten to reduce parameters
- relatively small for fast training (~40 min estimated)

### Expected Performance
- top-1 accuracy: 40-50%
- intentionally low to show transfer learning benefit

### Usage
```python
from src.models import SimpleCNN, get_model_summary

model = SimpleCNN(num_classes=75)
get_model_summary(model)
```

### File
- `src/models.py`

---

## ResNet50Classifier (Pretrained)

### Purpose
- main model leveraging transfer learning from ImageNet
- demonstrates significant improvement over training from scratch
- uses pretrained weights to extract rich visual features

### Architecture

```
Input: [batch, 3, 224, 224]
    |
ResNet50 Backbone (pretrained on ImageNet):
    | Conv1: Conv2d(3→64, k=7, s=2, p=3) + BN + ReLU + MaxPool
    | Layer1: 3x Bottleneck blocks (64→256 channels)
    | Layer2: 4x Bottleneck blocks (256→512 channels)
    | Layer3: 6x Bottleneck blocks (512→1024 channels)
    | Layer4: 3x Bottleneck blocks (1024→2048 channels)
    | AdaptiveAvgPool(1,1)
    | Output: [batch, 2048]
    |
Custom Classifier Head:
    | Dropout(0.5)
    | Linear(2048→512) + ReLU
    | Dropout(0.3)
    | Linear(512→num_classes)
    | Output: [batch, num_classes]
```

### Parameters
| Component | Parameters |
|-----------|------------|
| ResNet50 Backbone | ~23.5M |
| Custom Classifier (75 classes) | ~1.1M |
| **Total** | **~24.6M** |
| **Trainable** | **~24.6M** |

### Design Choices
- ResNet50 backbone with ImageNet pretrained weights
- bottleneck blocks (1x1→3x3→1x1) for efficiency
- residual connections solve vanishing gradient problem
- custom classifier head with two FC layers
- dropout before both FC layers to prevent overfitting
- lower learning rate (0.0001) to preserve pretrained features

### Expected Performance
- top-1 accuracy: 75-82%
- significant improvement over SimpleCNN baseline
- competitive with original OverFeat results (82.9%)

### Training Configuration
- learning rate: 0.0001 (lower than default to preserve pretrained features)
- all other settings: same as default

### Usage
```python
from src.models import ResNet50Classifier, get_model_summary

model = ResNet50Classifier(num_classes=75, pretrained=True)
get_model_summary(model)
```

### File
- `src/models.py`

---

## EfficientNetClassifier (Pretrained)

### Purpose
- compare modern efficient architecture against ResNet50
- similar or better accuracy with significantly fewer parameters
- uses compound scaling to balance network depth, width, and resolution

### Architecture

```
Input: [batch, 3, 224, 224]
    |
EfficientNet-B0 Backbone (pretrained on ImageNet):
    | Stem: Conv2d(3→32, k=3, s=2) + BN + Swish
    | MBConv1 (k=3): 32→16 channels, 1 block
    | MBConv6 (k=3): 16→24 channels, 2 blocks
    | MBConv6 (k=5): 24→40 channels, 2 blocks
    | MBConv6 (k=3): 40→80 channels, 3 blocks
    | MBConv6 (k=5): 80→112 channels, 3 blocks
    | MBConv6 (k=5): 112→192 channels, 4 blocks
    | MBConv6 (k=3): 192→320 channels, 1 block
    | Head: Conv2d(320→1280, k=1) + BN + Swish + AdaptiveAvgPool
    | Output: [batch, 1280]
    |
Custom Classifier Head:
    | Dropout(0.3)
    | Linear(1280→num_classes)
    | Output: [batch, num_classes]
```

### Parameters
| Component | Parameters |
|-----------|------------|
| EfficientNet-B0 Backbone | ~4.0M |
| Custom Classifier (75 classes) | ~96K |
| **Total** | **~4.1M** |
| **Trainable** | **~4.1M** |

### Design Choices
- EfficientNet-B0 backbone with ImageNet pretrained weights
- MBConv (Mobile Inverted Bottleneck) blocks with squeeze-and-excitation
- Swish activation (x * sigmoid(x)) instead of ReLU
- compound scaling balances depth, width, resolution efficiently
- single FC layer classifier (simpler than ResNet50 head)
- dropout 0.3 matches original EfficientNet design

### Key Differences from ResNet50
| Aspect | ResNet50 | EfficientNet-B0 |
|--------|----------|-----------------|
| Parameters | ~24.6M | ~4.1M (6x smaller) |
| Block Type | Bottleneck (conv) | MBConv (depthwise separable) |
| Activation | ReLU | Swish |
| Attention | None | Squeeze-and-Excitation |
| Feature Dim | 2048 | 1280 |

### Expected Performance
- top-1 accuracy: 75-82% (similar to ResNet50)
- faster training due to fewer parameters
- more memory efficient

### Training Configuration
- learning rate: 0.0001 (lower than default to preserve pretrained features)
- all other settings: same as default

### Usage
```python
from src.models import EfficientNetClassifier, get_model_summary

model = EfficientNetClassifier(num_classes=75, pretrained=True)
get_model_summary(model)
```

### File
- `src/models.py`

---

## ResNet50_SE (Originality Element)

### Purpose
- demonstrate understanding of attention mechanisms (originality points)
- improve fine-grained classification by focusing on discriminative parts
- learn to weight channel importance adaptively

### SEBlock Architecture

```
Input: [batch, C, H, W]
    |
Squeeze: AdaptiveAvgPool2d(1)
    | Output: [batch, C, 1, 1] → [batch, C]
    |
Excitation:
    | Linear(C → C/16) + ReLU
    | Linear(C/16 → C) + Sigmoid
    | Output: [batch, C] → [batch, C, 1, 1]
    |
Scale: Input * Weights
    | Output: [batch, C, H, W]
```

### Full Model Architecture

```
Input: [batch, 3, 224, 224]
    |
ResNet50 Backbone (pretrained on ImageNet):
    | Stem: Conv2d(3→64, k=7, s=2, p=3) + BN + ReLU + MaxPool
    |
    | Layer1: 3x Bottleneck blocks (64→256 channels)
    | SE Block (256 channels, reduction=16)
    |
    | Layer2: 4x Bottleneck blocks (256→512 channels)
    | SE Block (512 channels, reduction=16)
    |
    | Layer3: 6x Bottleneck blocks (512→1024 channels)
    | SE Block (1024 channels, reduction=16)
    |
    | Layer4: 3x Bottleneck blocks (1024→2048 channels)
    | SE Block (2048 channels, reduction=16)
    |
    | AdaptiveAvgPool(1,1)
    | Output: [batch, 2048]
    |
Custom Classifier Head:
    | Dropout(0.5)
    | Linear(2048→num_classes)
    | Output: [batch, num_classes]
```

### Parameters
| Component | Parameters |
|-----------|------------|
| ResNet50 Backbone | ~23.5M |
| SE Block 1 (256 ch) | 256*(256/16) + (256/16)*256 = 8,192 |
| SE Block 2 (512 ch) | 512*(512/16) + (512/16)*512 = 32,768 |
| SE Block 3 (1024 ch) | 1024*(1024/16) + (1024/16)*1024 = 131,072 |
| SE Block 4 (2048 ch) | 2048*(2048/16) + (2048/16)*2048 = 524,288 |
| Custom Classifier (75 classes) | ~153K |
| **Total** | **~24.4M** |
| **Trainable** | **~24.4M** |

### Design Choices
- SE blocks added after each ResNet stage (not inside blocks) for simplicity
- reduction ratio of 16 balances expressiveness vs parameters
- single FC layer classifier (simpler than ResNet50's two-layer head)
- dropout 0.5 before classifier to prevent overfitting

### Why SE Attention for Cars?
| Aspect | Benefit |
|--------|---------|
| Channel attention | Focus on discriminative features (grilles, headlights, body shape) |
| Adaptive weighting | Different channels matter for different car makes |
| Position invariant | Attention applies globally, robust to viewpoint |
| Minimal overhead | Only ~0.7M extra parameters over ResNet50 |

### Expected Performance
- top-1 accuracy: 80-85% (should be best model)
- slight increase in training time due to SE blocks
- justifies added complexity with improved accuracy

### Training Configuration
- learning rate: 0.0001 (lower than default to preserve pretrained features)
- all other settings: same as default

### Usage
```python
from src.models import ResNet50_SE, get_model_summary

model = ResNet50_SE(num_classes=75, pretrained=True)
get_model_summary(model)
```

### File
- `src/models.py`

---

## HierarchicalCarClassifier (Coarse-to-Fine)

### Purpose
- exploit the natural hierarchy: make (75 classes) -> model (431 classes)
- shared backbone learns features useful for both tasks
- multi-task learning demonstrates understanding of hierarchical classification
- enables rich error analysis (correlation between make and model errors)

### Architecture

```
Input: [batch, 3, 224, 224]
    |
ResNet50 Backbone (pretrained on ImageNet):
    | Stem: Conv2d(3→64, k=7, s=2, p=3) + BN + ReLU + MaxPool
    | Layer1-4: Standard ResNet50 bottleneck blocks
    | AdaptiveAvgPool(1,1)
    | Output: [batch, 2048]
    |
    +------------------+------------------+
    |                                     |
Make Head (Coarse):                  Model Head (Fine):
    | Dropout(0.5)                       | Dropout(0.5)
    | Linear(2048→512) + ReLU            | Linear(2048→512) + ReLU
    | Dropout(0.3)                       | Dropout(0.3)
    | Linear(512→75)                     | Linear(512→431)
    | Output: [batch, 75]                | Output: [batch, 431]
```

### HierarchicalLoss

```
Loss = α * CrossEntropy(make) + (1-α) * CrossEntropy(model)

Where:
- α = 0.3 (default): Focus more on model classification
- α = 0.5: Equal weight to both tasks
- α = 0.7: Focus more on make classification
- Label smoothing = 0.1 for regularization
```

### Parameters
| Component | Parameters |
|-----------|------------|
| ResNet50 Backbone | ~23.5M |
| Make Head (75 classes) | 2048*512 + 512 + 512*75 + 75 = ~1.09M |
| Model Head (431 classes) | 2048*512 + 512 + 512*431 + 431 = ~1.27M |
| **Total** | **~25.9M** |
| **Trainable** | **~25.9M** |

### Design Choices
- single shared backbone (more practical than separate networks)
- two-layer classifier heads with intermediate 512-dim layer
- dropout before both FC layers to prevent overfitting
- alpha=0.3 focuses more on fine-grained model classification
- label smoothing for regularization

### Multi-Task Learning Benefits
| Aspect | Benefit |
|--------|---------|
| Shared representations | Backbone learns features useful for both tasks |
| Regularization | Multi-task loss acts as implicit regularization |
| Error analysis | Can measure make-model error correlation |
| Interpretability | Two-level predictions more explainable |

### Expected Performance
- make top-1 accuracy: 78-83%
- model top-1 accuracy: 65-70%
- demonstrates that shared features help both tasks

### Training Configuration
- learning rate: 0.0001 (lower than default to preserve pretrained features)
- hierarchical alpha: 0.3 (focus more on model classification)
- scheduler tracks make accuracy for LR reduction
- all other settings: same as default

### Usage
```python
from src.models import HierarchicalCarClassifier, HierarchicalLoss

model = HierarchicalCarClassifier(num_makes=75, num_models=431, pretrained=True)
criterion = HierarchicalLoss(alpha=0.3)

# forward pass returns two outputs
make_logits, model_logits = model(images)

# loss returns total, make_loss, model_loss
loss, make_loss, model_loss = criterion(
    make_logits, model_logits, make_labels, model_labels
)
```

### File
- `src/models.py`

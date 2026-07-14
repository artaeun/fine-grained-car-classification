# Generalization Gap Analysis

Generated: 2026-01-11 01:15

## Summary Statistics

The generalization gap measures overfitting as (Train Accuracy - Validation Accuracy).
Lower values indicate better generalization.

| Model            | Avg Gap (Last 10)   |
|:-----------------|:--------------------|
| SimpleCNN        | 11.48%              |
| EfficientNet-B0  | 2.47%               |
| ResNet50 (CE)    | 2.38%               |
| ResNet50 (Focal) | 3.52%               |
| ResNet50 (LS)    | 2.62%               |
| ResNet50-SE      | 2.22%               |
| Hierarchical     | 1.18%               |

## Interpretation

- **Avg Gap (Last 10)**: Average gap in final training phase (last 10 epochs)

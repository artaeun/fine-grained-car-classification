# What is in `results/`

Everything the analysis in the [main README](../README.md) rests on. The dataset and the checkpoints are not in this repository, so this directory is the reproducible artefact: notebooks `03_evaluation.ipynb` and `04_analysis.ipynb` read from here and regenerate every table and figure without retraining.

Seven runs are recorded, one directory each: `simplecnn_label_smoothing`, `efficientnet_label_smoothing`, `resnet50_ce`, `resnet50_focal`, `resnet50_label_smoothing`, `resnet50_se_label_smoothing`, `hierarchical`.

## Cross-model comparison, `comparison/`

| File | Contains | What it proves |
|---|---|---|
| `summary.csv`, `summary.md` | One row per run: parameters, top-1, top-5, macro precision, recall, F1, best validation accuracy | The headline table. Hierarchical 87.55% top-1, EfficientNet-B0 86.95% at a sixth of the parameters, SimpleCNN 53.08% from scratch |
| `generalization_gap_stats.csv`, `.md` | Mean train-minus-validation accuracy gap over the last ten epochs of each run | Hierarchical generalises best (1.18 points), SimpleCNN worst (11.48). The auxiliary head acts as a regulariser |
| `top_confusions.md` | The ten worst true-to-predicted pairs for the three strongest models, with counts and percentage of the true class | Volkswagen is the predicted label in 7 of the hierarchical model's 10 worst pairs. Brabus is called Benz 66.7% of the time by EfficientNet-B0 |
| `inference_timing.json` | Per-run wall time, per-batch and per-sample latency, throughput, measured on the full 14,939-image test set, RTX 3060, batch size 32 | The cost axis. EfficientNet-B0 844 img/s, ResNet50 411 img/s, Hierarchical 420 img/s, so the second head is free |
| `accuracy_comparison.png` | Grouped bars, top-1 and top-5 per model | Accuracy ranking at a glance |
| `acc_vs_params.png` | Test accuracy against parameter count | The accuracy-cost front. EfficientNet-B0 is the left-hand corner |
| `val_accuracy_overlay.png`, `val_loss_overlay.png` | All seven validation curves on one axis | Pretrained models converge by roughly epoch 20; SimpleCNN plateaus low |
| `generalization_gap_overlay.png` | Train-minus-validation gap per epoch, all runs | Only SimpleCNN's overfitting is visible in the train-to-validation signal. The real gap is validation to test, and it is invisible here |

## Per-run directories

| File | Contains | What it proves |
|---|---|---|
| `classification_report.txt` | scikit-learn per-class precision, recall, F1 and support for all 75 makes, plus macro and weighted averages | Where the macro F1 is lost. 9 of 75 make classes score below 0.80 F1 on the hierarchical model; one hits precision 1.00 with recall 0.39 |
| `classification_report_model.txt` (hierarchical only) | The same for all 431 model classes | The fine-grained task is harder: 144 of 431 classes fall below 0.80 F1 |
| `evaluation_data.json` | `indices`, `make_predictions`, `make_labels`, `make_metrics`, and for the hierarchical run `model_predictions`, `model_labels`, `model_metrics`. Metrics are top-1, top-5, macro precision, recall, F1 | The raw evidence behind every reported number, per test image |
| `figures/confusion_matrix.png` | 75x75 make confusion matrix | The confusion structure, Volkswagen column included |
| `figures/per_class_accuracy_make.png`, `per_class_accuracy_model.png` | Per-class accuracy bars, sorted | Head classes are near-saturated, tail classes carry the error |
| `figures/sample_correct.png`, `sample_incorrect.png` (and `_model` variants) | Image grids of hits and misses with predicted and true labels | Qualitative check on what the residual errors actually look like |

### On `evaluation_data.json`

It stores indices, predictions, labels and the summary metrics. The raw probability matrices are deliberately dropped, because a 14,939 by 431 float array per run pushes the repository past GitHub's file size limit and buys nothing that the argmax predictions do not already carry. Any confusion matrix and any per-class precision, recall or F1 can be recomputed from the prediction and label vectors that are here. What cannot be recomputed is anything needing calibrated confidences: top-k for k above 5, reliability diagrams, entropy-based rejection.

## Top-level files

| File | Contains | What it proves |
|---|---|---|
| `dataset_statistics.md` | Split sizes, class counts, imbalance ratios, source and citation | 16,016 train and 14,939 test images, 75 makes, 431 models, make imbalance 120:1 |
| `model_architectures.md` | Layer-by-layer description of all five architectures with parameter counts | What was actually built, down to the SE reduction ratio and the head widths |
| `experiments.md` | Per-run configuration and outcome log: loss, hyperparameters, epochs, final metrics | The audit trail from configuration to number |
| `observations.md` | Running notes taken during the work, including the findings that changed the conclusions | Where the focal-loss failure and the validation-to-test gap were first noticed |
| `summary.md` | Human-readable version of the comparison table plus the OverFeat reference point | The one-page result |
| `visualization_metadata.json` | The index-to-name maps for makes and models | Turns integer labels in `evaluation_data.json` back into "Volkswagen" and "Skoda" |
| `make_distribution.png`, `model_distribution.png`, `year_distribution.png` | Class frequency histograms | The long tail, and why macro recall is the honest metric |
| `<run>_training_curves.png` | Per-run loss and accuracy curves, train and validation | Convergence behaviour of each individual run |

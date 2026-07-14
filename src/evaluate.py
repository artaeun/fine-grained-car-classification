# evaluation functions for compcars classification

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from tqdm import tqdm
from pathlib import Path

from src.visualization import plot_confusion_matrix, plot_per_class_accuracy
from src.utils import top_k_accuracy


# Configuration contants

# dataset paths
DATASET_PATH = 'dataset/data'
TRAIN_SPLIT = 'dataset/data/train_test_split/classification/train.txt'
TEST_SPLIT = 'dataset/data/train_test_split/classification/test.txt'

# model configuration
MODEL_NAME = 'resnet50_label_smoothing'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'

# dataset configuration
TASK = 'make'
NUM_CLASSES = 75
NUM_MODELS = 431
IMAGE_SIZE = 224

# dataloader configuration
BATCH_SIZE = 32
NUM_WORKERS = 4

# device
USE_CUDA = True


# evaluation functions

def evaluate(model, dataloader, device, num_classes):
    # evaluate model on a dataset and compute all metrics
    # returns dictionary with top1, top5, precision, recall, f1

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='evaluating'):
            images = images.to(device)

            # forward pass
            outputs = model(images)
            # get probabilities
            probs = F.softmax(outputs, dim=1)
            # get predictions
            _, preds = outputs.max(1)

            # store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # calculate top-1 accuracy
    top1_acc = accuracy_score(all_labels, all_preds)
    # calculate top-5 accuracy
    top5_acc = top_k_accuracy(all_probs, all_labels, k=5)

    # calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    # build results dictionary
    results = {}
    results['top1_accuracy'] = top1_acc
    results['top5_accuracy'] = top5_acc
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['predictions'] = all_preds
    results['labels'] = all_labels
    results['probabilities'] = all_probs
    return results


def evaluate_hierarchical(model, dataloader, device):
    # evaluate hierarchical model on both make and model tasks
    # returns dictionary with metrics for both tasks

    model.eval()

    # storage for predictions and labels
    make_preds = []
    make_labels = []
    make_probs = []
    model_preds = []
    model_labels = []
    model_probs = []

    with torch.no_grad():
        for images, make_label, model_label in tqdm(dataloader, desc='evaluating'):
            images = images.to(device)

            # forward pass
            make_logits, model_logits = model(images)

            # get probabilities
            make_prob = F.softmax(make_logits, dim=1)
            model_prob = F.softmax(model_logits, dim=1)

            # get predictions
            _, make_pred = make_logits.max(1)
            _, model_pred = model_logits.max(1)

            # store make results
            make_preds.extend(make_pred.cpu().numpy())
            make_labels.extend(make_label.numpy())
            make_probs.extend(make_prob.cpu().numpy())

            # store model results
            model_preds.extend(model_pred.cpu().numpy())
            model_labels.extend(model_label.numpy())
            model_probs.extend(model_prob.cpu().numpy())

    # convert to numpy arrays
    make_preds = np.array(make_preds)
    make_labels = np.array(make_labels)
    make_probs = np.array(make_probs)

    model_preds = np.array(model_preds)
    model_labels = np.array(model_labels)
    model_probs = np.array(model_probs)

    # calculate make metrics
    make_top1 = accuracy_score(make_labels, make_preds)
    make_top5 = top_k_accuracy(make_probs, make_labels, k=5)
    make_precision, make_recall, make_f1, _ = precision_recall_fscore_support(
        make_labels,
        make_preds,
        average='macro',
        zero_division=0
    )

    # calculate model metrics
    model_top1 = accuracy_score(model_labels, model_preds)
    model_top5 = top_k_accuracy(model_probs, model_labels, k=5)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        model_labels,
        model_preds,
        average='macro',
        zero_division=0
    )

    # build results dictionary
    results = {}

    results['make_top1_accuracy'] = make_top1
    results['make_top5_accuracy'] = make_top5
    results['make_precision'] = make_precision
    results['make_recall'] = make_recall
    results['make_f1_score'] = make_f1
    results['make_predictions'] = make_preds
    results['make_labels'] = make_labels

    results['model_top1_accuracy'] = model_top1
    results['model_top5_accuracy'] = model_top5
    results['model_precision'] = model_precision
    results['model_recall'] = model_recall
    results['model_f1_score'] = model_f1
    results['model_predictions'] = model_preds
    results['model_labels'] = model_labels

    return results


def get_classification_report(labels, predictions, class_names=None, output_dict=False):
    # generate classification report with per-class metrics
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )
    return report


def analyze_hierarchical_errors(make_labels, make_preds, model_labels, model_preds):
    # Analyze correlation between make and model errors
    # returns dictionary with error analysis

    num_samples = len(make_labels)

    make_correct_model_correct = 0
    make_correct_model_wrong = 0
    make_wrong_model_correct = 0
    make_wrong_model_wrong = 0

    for i in range(num_samples):
        make_is_correct = make_preds[i] == make_labels[i]
        model_is_correct = model_preds[i] == model_labels[i]

        if make_is_correct:
            if model_is_correct:
                make_correct_model_correct = make_correct_model_correct + 1
            else:
                make_correct_model_wrong = make_correct_model_wrong + 1
        else:
            if model_is_correct:
                make_wrong_model_correct = make_wrong_model_correct + 1
            else:
                make_wrong_model_wrong = make_wrong_model_wrong + 1

    # Calculate percentages
    total = num_samples
    pct_both_correct = 100.0 * make_correct_model_correct / total
    pct_make_only = 100.0 * make_correct_model_wrong / total
    pct_model_only = 100.0 * make_wrong_model_correct / total
    pct_both_wrong = 100.0 * make_wrong_model_wrong / total

    # calculate error correlation
    model_errors = make_correct_model_wrong + make_wrong_model_wrong
    if model_errors > 0:
        error_correlation = 100.0 * make_wrong_model_wrong / model_errors
    else:
        error_correlation = 0.0

    # Build results
    analysis = {}
    analysis['make_correct_model_correct'] = make_correct_model_correct
    analysis['make_correct_model_wrong'] = make_correct_model_wrong
    analysis['make_wrong_model_correct'] = make_wrong_model_correct
    analysis['make_wrong_model_wrong'] = make_wrong_model_wrong
    analysis['pct_both_correct'] = pct_both_correct
    analysis['pct_make_correct_model_wrong'] = pct_make_only
    analysis['pct_make_wrong_model_correct'] = pct_model_only
    analysis['pct_both_wrong'] = pct_both_wrong
    analysis['error_correlation'] = error_correlation

    return analysis


def print_evaluation_results(results, model_name='model'):
    #Print evaluation results in formatted table

    print("\n" + "=" * 60)
    print(f"evaluation results: {model_name}")
    print("=" * 60)

    print(f"\ntop-1 accuracy: {results['top1_accuracy'] * 100:.2f}%")
    print(f"top-5 accuracy: {results['top5_accuracy'] * 100:.2f}%")
    print(f"precision (macro): {results['precision']:.4f}")
    print(f"recall (macro): {results['recall']:.4f}")
    print(f"f1 score (macro): {results['f1_score']:.4f}")

    print("\n" + "=" * 60)


def print_hierarchical_results(results, model_name='hierarchical'):
    # print hierarchical evaluation results

    print("\n" + "=" * 60)
    print(f"evaluation results: {model_name}")
    print("=" * 60)

    print("\n--- make classification (75 classes) ---")
    print(f"top-1 accuracy: {results['make_top1_accuracy'] * 100:.2f}%")
    print(f"top-5 accuracy: {results['make_top5_accuracy'] * 100:.2f}%")
    print(f"precision (macro): {results['make_precision']:.4f}")
    print(f"recall (macro): {results['make_recall']:.4f}")
    print(f"f1 score (macro): {results['make_f1_score']:.4f}")

    print("\n--- model classification (431 classes) ---")
    print(f"top-1 accuracy: {results['model_top1_accuracy'] * 100:.2f}%")
    print(f"top-5 accuracy: {results['model_top5_accuracy'] * 100:.2f}%")
    print(f"precision (macro): {results['model_precision']:.4f}")
    print(f"recall (macro): {results['model_recall']:.4f}")
    print(f"f1 score (macro): {results['model_f1_score']:.4f}")

    print("\n" + "=" * 60)


# main evaluation function
def evaluate_model(model_name, checkpoint_dir, results_dir, device):
    # main function to evaluate a trained model

    from src.dataset import CompCarsDataset, get_val_transforms, get_dataloader
    from src.models import count_parameters
    from src.utils import get_model_for_inference, load_checkpoint, save_evaluation_results
    print("\n" + "=" * 60)
    print(f"evaluating model: {model_name}")
    print("=" * 60)

    # determine task from model name
    if 'hierarchical' in model_name:
        task = 'hierarchical'
    else:
        task = TASK
    # paths
    checkpoint_path = Path(checkpoint_dir) / f'{model_name}_best.pth'

    if not checkpoint_path.exists():
        print(f"checkpoint not found: {checkpoint_path}")
        return None

    print(f"loading checkpoint: {checkpoint_path}")

    # create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # create model
    print("creating model...")
    if task == 'hierarchical':
        model = get_model_for_inference(model_name, NUM_CLASSES, NUM_MODELS)
    else:
        model = get_model_for_inference(model_name, NUM_CLASSES)

    # load checkpoint
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # print model info
    params = count_parameters(model)
    print(f"model parameters: {params['total']:,}")

    # load test dataset
    print("loading test dataset...")
    test_transforms = get_val_transforms(IMAGE_SIZE)

    test_dataset = CompCarsDataset(
        root_dir=DATASET_PATH,
        split_file=TEST_SPLIT,
        transform=test_transforms,
        task=task
    )

    print(f"test samples: {len(test_dataset)}")

    # create dataloader
    test_loader = get_dataloader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # run evaluation
    print("running evaluation...")

    if task == 'hierarchical':
        results = evaluate_hierarchical(model, test_loader, device)
        print_hierarchical_results(results, model_name)

        # analyze hierarchical errors
        error_analysis = analyze_hierarchical_errors(
            results['make_labels'],
            results['make_predictions'],
            results['model_labels'],
            results['model_predictions']
        )

        print("\n--- hierarchical error analysis ---")
        print(f"make correct, model correct: {error_analysis['make_correct_model_correct']} ({error_analysis['pct_both_correct']:.1f}%)")
        print(f"make correct, model wrong: {error_analysis['make_correct_model_wrong']} ({error_analysis['pct_make_correct_model_wrong']:.1f}%)")
        print(f"make wrong, model correct: {error_analysis['make_wrong_model_correct']} ({error_analysis['pct_make_wrong_model_correct']:.1f}%)")
        print(f"make wrong, model wrong: {error_analysis['make_wrong_model_wrong']} ({error_analysis['pct_both_wrong']:.1f}%)")
        print(f"\n{error_analysis['error_correlation']:.1f}% of model errors occur when make is also wrong")

        results['error_analysis'] = error_analysis

        # plot confusion matrix for make task
        cm_path = results_path / f'{model_name}_make_confusion_matrix.png'
        plot_confusion_matrix(
            results['make_labels'],
            results['make_predictions'],
            save_path=cm_path,
            title=f'{model_name} - Make Confusion Matrix'
        )

        # plot per-class accuracy for make task
        pca_path = results_path / f'{model_name}_make_per_class_accuracy.png'
        plot_per_class_accuracy(
            results['make_labels'],
            results['make_predictions'],
            save_path=pca_path,
            title=f'{model_name} - Make Per-Class Accuracy'
        )

    else:
        results = evaluate(model, test_loader, device, NUM_CLASSES)
        print_evaluation_results(results, model_name)

        # plot confusion matrix
        cm_path = results_path / f'{model_name}_confusion_matrix.png'
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            save_path=cm_path,
            title=f'{model_name} Confusion Matrix'
        )

        # plot per-class accuracy
        pca_path = results_path / f'{model_name}_per_class_accuracy.png'
        plot_per_class_accuracy(
            results['labels'],
            results['predictions'],
            save_path=pca_path,
            title=f'{model_name} Per-Class Accuracy'
        )

    # save results
    save_evaluation_results(results, model_name, results_dir)

    return results


# main

if __name__ == '__main__':
    # setup device
    if USE_CUDA:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"using device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("using device: cpu")
    else:
        device = torch.device('cpu')
        print("using device: cpu")

    # run evaluation
    results = evaluate_model(
        MODEL_NAME,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        device
    )

    print("\nevaluation complete!")

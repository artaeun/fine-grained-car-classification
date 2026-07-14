import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

from src.models import (
    SimpleCNN,
    ResNet50Classifier,
    EfficientNetClassifier,
    ResNet50_SE,
    HierarchicalCarClassifier
)

# some functions that i didn't know where else to put

# loss type suffixes that can be appended to model names
LOSS_SUFFIXES = []
LOSS_SUFFIXES.append('_label_smoothing')
LOSS_SUFFIXES.append('_focal')
LOSS_SUFFIXES.append('_ce')
LOSS_SUFFIXES.append('_weighted_ce')
LOSS_SUFFIXES.append('_weighted_focal')

# models that use pretrained backbones
PRETRAINED_MODELS = []
PRETRAINED_MODELS.append('resnet50')
PRETRAINED_MODELS.append('efficientnet')
PRETRAINED_MODELS.append('resnet50_se')
PRETRAINED_MODELS.append('hierarchical')


# helper functions

def _strip_loss_suffix(model_name: str) -> str:
    # extract base model name by removing loss type suffixes
    base_name = model_name

    for suffix in LOSS_SUFFIXES:
        if suffix in base_name:
            base_name = base_name.replace(suffix, '')

    return base_name


def get_model(
    model_name: str,
    num_classes: int,
    num_models: Optional[int] = None,
    pretrained: Optional[bool] = None
):
    # unified model factory function
    # handles model names with or without loss type suffixes

    base_name = _strip_loss_suffix(model_name)

    # determine pretrained default based on model type
    if pretrained is None:
        pretrained = base_name in PRETRAINED_MODELS

    # create model based on base name
    if base_name == 'simplecnn':
        model = SimpleCNN(num_classes=num_classes)

    elif base_name == 'resnet50':
        model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)

    elif base_name == 'efficientnet':
        model = EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)

    elif base_name == 'resnet50_se':
        model = ResNet50_SE(num_classes=num_classes, pretrained=pretrained)

    elif base_name == 'hierarchical':
        if num_models is None:
            raise ValueError("num_models required for hierarchical model")
        model = HierarchicalCarClassifier(
            num_makes=num_classes,
            num_models=num_models,
            pretrained=pretrained
        )

    else:
        raise ValueError(f"unknown model: {model_name} (base: {base_name})")

    return model


def get_model_for_inference(model_name: str, num_classes: int, num_models: Optional[int] = None):
    # get model for inference (pretrained=False since weights will be loaded)
    # use this when loading from checkpoint to avoid downloading pretrained weights
    model = get_model(model_name, num_classes, num_models, pretrained=False)
    return model


def load_checkpoint(model, checkpoint_path, device):
    # load model weights from checkpoint file
    # handles both full checkpoint dict and direct state dict formats

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


def save_config(config: Dict[str, Any], path) -> None:
    # save configuration dictionary to json file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def get_config_for_model(
    model_name: str,
    loss_type: str,
    num_classes: int,
    num_models: Optional[int] = None,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 5,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    use_differential_lr: bool = True,
    backbone_lr: float = 0.0001,
    head_lr: float = 0.001,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    seed: int = 2128831,
    task: str = 'make',
    model_configs: Optional[Dict] = None
) -> Dict[str, Any]:
    # build configuration dictionary for a model and loss type

    config = {}
    config['model_name'] = model_name
    config['task'] = task
    config['num_classes'] = num_classes
    config['num_models'] = num_models
    config['batch_size'] = batch_size
    config['num_epochs'] = num_epochs
    config['learning_rate'] = learning_rate
    config['weight_decay'] = weight_decay
    config['scheduler_patience'] = scheduler_patience
    config['scheduler_factor'] = scheduler_factor
    config['early_stopping_patience'] = early_stopping_patience
    config['use_amp'] = use_amp
    config['seed'] = seed

    # loss function config
    config['loss_type'] = loss_type
    config['label_smoothing'] = label_smoothing
    config['focal_gamma'] = focal_gamma

    # differential learning rates config
    config['use_differential_lr'] = use_differential_lr
    config['backbone_lr'] = backbone_lr
    config['head_lr'] = head_lr

    # gradient accumulation config
    config['accumulation_steps'] = accumulation_steps

    # apply model-specific overrides if provided
    if model_configs is not None:
        if model_name in model_configs:
            overrides = model_configs[model_name]
            for key, value in overrides.items():
                config[key] = value

    return config


def top_k_accuracy(probs, labels, k: int = 5) -> float:
    # calculate top-k accuracy from probability predictions

    num_samples = len(labels)
    correct = 0

    for i in range(num_samples):
        sample_probs = probs[i]
        top_k_indices = np.argsort(sample_probs)[-k:]
        true_label = labels[i]

        if true_label in top_k_indices:
            correct = correct + 1

    accuracy = correct / num_samples
    return accuracy


def save_evaluation_results(results: Dict, model_name: str, save_dir) -> Path:
    # save evaluation results to json file
    # handles numpy array conversion and skips large arrays

    save_path = Path(save_dir) / f'{model_name}_evaluation.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # convert numpy arrays to lists for json serialization
    results_to_save = {}

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            # skip large arrays (predictions, labels, etc.)
            if len(value) > 1000:
                continue
            results_to_save[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            results_to_save[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            results_to_save[key] = int(value)
        else:
            results_to_save[key] = value

    with open(save_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"evaluation results saved to {save_path}")

    return save_path


def is_pretrained_model(model_name: str) -> bool:
    # check if a model uses pretrained weights

    base_name = _strip_loss_suffix(model_name)
    is_pretrained = base_name in PRETRAINED_MODELS
    return is_pretrained


def is_hierarchical_model(model_name: str) -> bool:
    # check if a model is hierarchical (dual-head)

    is_hierarchical = 'hierarchical' in model_name
    return is_hierarchical

import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# main library used by notebookx for training

# experiments to run: list of (model_name, loss_type) tuples
# available models: 'simplecnn', 'resnet50', 'efficientnet', 'resnet50_se', 'hierarchical'
# available losses: 'ce', 'label_smoothing', 'focal', 'weighted_ce', 'weighted_focal'

EXPERIMENTS = []
EXPERIMENTS.append(('resnet50', 'label_smoothing'))
EXPERIMENTS.append(('resnet50', 'ce'))
EXPERIMENTS.append(('resnet50', 'focal'))
EXPERIMENTS.append(('simplecnn', 'label_smoothing'))
EXPERIMENTS.append(('efficientnet', 'label_smoothing'))
EXPERIMENTS.append(('resnet50_se', 'label_smoothing'))
EXPERIMENTS.append(('hierarchical', 'hierarchical'))

# dataset configuration
DATA_ROOT = 'dataset/data'
TRAIN_SPLIT = 'dataset/data/train_test_split/classification/train.txt'
TASK = 'make'
NUM_CLASSES = 75
NUM_MODELS = 431

# hierarchical training configuration
HIERARCHICAL_ALPHA = 0.3

# training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1

# loss function configuration
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0

# differential learning rates
USE_DIFFERENTIAL_LR = True
BACKBONE_LR = 0.0001
HEAD_LR = 0.001

# gradient accumulation
ACCUMULATION_STEPS = 1

# scheduler configuration
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# early stopping
EARLY_STOPPING_PATIENCE = 5

# mixed precision
USE_AMP = True

# paths
CHECKPOINT_DIR = Path('checkpoints')
RESULTS_DIR = Path('results')

# random seed for reproducibility - my student id
SEED = 2128831

# per-model config overrides
MODEL_CONFIGS = {}
MODEL_CONFIGS['resnet50'] = {}
MODEL_CONFIGS['resnet50']['learning_rate'] = 0.0001
MODEL_CONFIGS['efficientnet'] = {}
MODEL_CONFIGS['efficientnet']['learning_rate'] = 0.0001
MODEL_CONFIGS['resnet50_se'] = {}
MODEL_CONFIGS['resnet50_se']['learning_rate'] = 0.0001
MODEL_CONFIGS['hierarchical'] = {}
MODEL_CONFIGS['hierarchical']['learning_rate'] = 0.0001

###################################################################

# training history class
class TrainingHistory:
    # tracks training metrics across epochs

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.lr = []

    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.lr.append(lr)

    def save(self, path):
        # save history to json file
        data = {}
        data['train_loss'] = self.train_loss
        data['train_acc'] = self.train_acc
        data['val_loss'] = self.val_loss
        data['val_acc'] = self.val_acc
        data['lr'] = self.lr

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        # load history from json file
        with open(path, 'r') as f:
            data = json.load(f)

        self.train_loss = data['train_loss']
        self.train_acc = data['train_acc']
        self.val_loss = data['val_loss']
        self.val_acc = data['val_acc']
        self.lr = data['lr']


# early stopping class

class EarlyStopping:
    # stops training when validation metric stops improving
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter = self.counter + 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp=True):
    # train model for one epoch

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='training')

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss = running_loss + loss.item()

        _, predicted = outputs.max(1)
        total = total + labels.size(0)
        correct = correct + predicted.eq(labels).sum().item()

        current_acc = 100.0 * correct / total
        progress_bar.set_postfix(loss=loss.item(), acc=f'{current_acc:.2f}%')

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_one_epoch_with_accumulation(model, dataloader, optimizer, criterion, device, scaler,
                                       use_amp=True, accumulation_steps=2):
    # train model for one epoch with gradient accumulation
    # effective batch size = actual batch size * accumulation_steps

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc='training')

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # mixed precision forward pass
        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # step optimizer every accumulation_steps batches
            step_needed = (batch_idx + 1) % accumulation_steps == 0
            if step_needed:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            step_needed = (batch_idx + 1) % accumulation_steps == 0
            if step_needed:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        running_loss = running_loss + loss.item() * accumulation_steps

        # calculate accuracy
        _, predicted = outputs.max(1)
        total = total + labels.size(0)
        correct = correct + predicted.eq(labels).sum().item()

        # update progress bar
        current_acc = 100.0 * correct / total
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps, acc=f'{current_acc:.2f}%')

    # handle remaining gradients if batches not divisible by accumulation_steps
    remainder = len(dataloader) % accumulation_steps
    if remainder != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    # validate model on validation set

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='validating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss = running_loss + loss.item()

            _, predicted = outputs.max(1)
            total = total + labels.size(0)
            correct = correct + predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_hierarchical_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp=True):
    # train hierarchical model for one epoch
    # returns: (loss, make_accuracy, model_accuracy)

    model.train()
    running_loss = 0.0
    make_correct = 0
    model_correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='training')

    for images, make_labels, model_labels in progress_bar:
        images = images.to(device)
        make_labels = make_labels.to(device)
        model_labels = model_labels.to(device)

        optimizer.zero_grad()

        # mixed precision forward pass
        if use_amp:
            with autocast(device_type='cuda'):
                make_logits, model_logits = model(images)
                loss, make_loss, model_loss = criterion(
                    make_logits, model_logits, make_labels, model_labels
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            make_logits, model_logits = model(images)
            loss, make_loss, model_loss = criterion(
                make_logits, model_logits, make_labels, model_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss = running_loss + loss.item()

        # calculate accuracies
        _, make_pred = make_logits.max(1)
        _, model_pred = model_logits.max(1)
        total = total + make_labels.size(0)
        make_correct = make_correct + make_pred.eq(make_labels).sum().item()
        model_correct = model_correct + model_pred.eq(model_labels).sum().item()

        # update progress bar
        make_acc = 100.0 * make_correct / total
        model_acc = 100.0 * model_correct / total
        progress_bar.set_postfix(
            loss=loss.item(),
            make_acc=f'{make_acc:.1f}%',
            model_acc=f'{model_acc:.1f}%'
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_make_acc = 100.0 * make_correct / total
    epoch_model_acc = 100.0 * model_correct / total

    return epoch_loss, epoch_make_acc, epoch_model_acc


def validate_hierarchical(model, dataloader, criterion, device):
    # validate hierarchical model
    # returns: (loss, make_accuracy, model_accuracy)

    model.eval()
    running_loss = 0.0
    make_correct = 0
    model_correct = 0
    total = 0

    with torch.no_grad():
        for images, make_labels, model_labels in tqdm(dataloader, desc='validating'):
            images = images.to(device)
            make_labels = make_labels.to(device)
            model_labels = model_labels.to(device)

            make_logits, model_logits = model(images)
            loss, make_loss, model_loss = criterion(
                make_logits, model_logits, make_labels, model_labels
            )

            running_loss = running_loss + loss.item()

            _, make_pred = make_logits.max(1)
            _, model_pred = model_logits.max(1)
            total = total + make_labels.size(0)
            make_correct = make_correct + make_pred.eq(make_labels).sum().item()
            model_correct = model_correct + model_pred.eq(model_labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_make_acc = 100.0 * make_correct / total
    epoch_model_acc = 100.0 * model_correct / total

    return epoch_loss, epoch_make_acc, epoch_model_acc


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    # save model checkpoint with training state
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['val_acc'] = val_acc

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    # load model checkpoint and restore training state
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']

    return epoch, val_acc


def get_optimizer_with_differential_lr(model, backbone_lr, head_lr, weight_decay):
    # create optimizer with different learning rates for backbone and classifier
    # backbone (pretrained): lower lr to preserve features
    # classifier head (new): higher lr for faster learning

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # check if parameter belongs to classifier head
        is_head = False
        if 'fc' in name:
            is_head = True
        if 'classifier' in name:
            is_head = True
        if 'head' in name:
            is_head = True

        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # create parameter groups with different learning rates
    param_groups = []

    if len(backbone_params) > 0:
        backbone_group = {}
        backbone_group['params'] = backbone_params
        backbone_group['lr'] = backbone_lr
        param_groups.append(backbone_group)

    if len(head_params) > 0:
        head_group = {}
        head_group['params'] = head_params
        head_group['lr'] = head_lr
        param_groups.append(head_group)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    return optimizer


def plot_training_curves(history, save_path=None):
    # plot training and validation curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = []
    for i in range(len(history.train_loss)):
        epochs.append(i + 1)

    # loss plot
    axes[0].plot(epochs, history.train_loss, 'b-', label='train')
    axes[0].plot(epochs, history.val_loss, 'r-', label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('loss curves')
    axes[0].legend()
    axes[0].grid(True)

    # accuracy plot
    axes[1].plot(epochs, history.train_acc, 'b-', label='train')
    axes[1].plot(epochs, history.val_acc, 'r-', label='val')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy (%)')
    axes[1].set_title('accuracy curves')
    axes[1].legend()
    axes[1].grid(True)

    # learning rate plot
    axes[2].plot(epochs, history.lr, 'g-')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('learning rate')
    axes[2].set_title('learning rate schedule')
    axes[2].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")

    plt.close()


def plot_hierarchical_curves(history_data, save_path=None):
    # plot training curves for hierarchical model
    # includes make accuracy, model accuracy, and loss

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = []
    for i in range(len(history_data['train_loss'])):
        epochs.append(i + 1)

    # loss plot
    axes[0, 0].plot(epochs, history_data['train_loss'], 'b-', label='train')
    axes[0, 0].plot(epochs, history_data['val_loss'], 'r-', label='val')
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('loss curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # make accuracy plot
    axes[0, 1].plot(epochs, history_data['train_make_acc'], 'b-', label='train')
    axes[0, 1].plot(epochs, history_data['val_make_acc'], 'r-', label='val')
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('accuracy (%)')
    axes[0, 1].set_title('make accuracy curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # model accuracy plot
    axes[1, 0].plot(epochs, history_data['train_model_acc'], 'b-', label='train')
    axes[1, 0].plot(epochs, history_data['val_model_acc'], 'r-', label='val')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('accuracy (%)')
    axes[1, 0].set_title('model accuracy curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # learning rate plot
    axes[1, 1].plot(epochs, history_data['lr'], 'g-')
    axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylabel('learning rate')
    axes[1, 1].set_title('learning rate schedule')
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")

    plt.close()

def get_config_for_model_local(model_name, loss_type):
    # get configuration for a specific model and loss type

    from src.utils import get_config_for_model #need to import here 
    # to avoid circular imports with src.utils which also imports from src.training
    # this would be cleaner if we refactor config management into a separate module, but for now this works
    # i ain't got time to make production level code man
    config = get_config_for_model(
        model_name=model_name,
        loss_type=loss_type,
        num_classes=NUM_CLASSES,
        num_models=NUM_MODELS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        scheduler_patience=SCHEDULER_PATIENCE,
        scheduler_factor=SCHEDULER_FACTOR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        label_smoothing=LABEL_SMOOTHING,
        focal_gamma=FOCAL_GAMMA,
        use_differential_lr=USE_DIFFERENTIAL_LR,
        backbone_lr=BACKBONE_LR,
        head_lr=HEAD_LR,
        accumulation_steps=ACCUMULATION_STEPS,
        use_amp=USE_AMP,
        seed=SEED,
        task=TASK,
        model_configs=MODEL_CONFIGS
    )
    # add val_ratio
    config['val_ratio'] = VAL_RATIO
    return config

def train_model(model_name, loss_type, train_loader, val_loader, device):
    # train a single model with specified loss function
    from src.models import count_parameters, HierarchicalLoss
    from src.losses import get_loss_function
    from src.utils import get_model, save_config, is_pretrained_model

    # get config for this model
    config = get_config_for_model_local(model_name, loss_type)

    # create experiment name
    experiment_name = f"{model_name}_{loss_type}"

    print("\n" + "=" * 60)
    print(f"TRAINING: {experiment_name.upper()}")
    print("=" * 60)

    # create model
    print(f"\ncreating model: {model_name}")
    model = get_model(model_name, config['num_classes'])
    model = model.to(device)

    # print model info
    params = count_parameters(model)
    print(f"total parameters: {params['total']:,}")
    print(f"trainable parameters: {params['trainable']:,}")

    # loss function
    criterion = get_loss_function(
        loss_type=config['loss_type'],
        num_classes=config['num_classes'],
        label_smoothing=config['label_smoothing'],
        focal_gamma=config['focal_gamma']
    )
    print(f"loss function: {config['loss_type']}")

    # optimizer
    use_diff_lr = config['use_differential_lr'] and is_pretrained_model(model_name)

    if use_diff_lr:
        optimizer = get_optimizer_with_differential_lr(
            model,
            backbone_lr=config['backbone_lr'],
            head_lr=config['head_lr'],
            weight_decay=config['weight_decay']
        )
        print(f"using differential lr: backbone={config['backbone_lr']}, head={config['head_lr']}")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience']
    )

    # mixed precision scaler
    scaler = GradScaler('cuda')

    # early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    # training history
    history = TrainingHistory()

    # best model tracking
    best_val_acc = 0.0

    # save config
    config_path = CHECKPOINT_DIR / f'{experiment_name}_config.json'
    save_config(config, config_path)
    print(f"saved config to: {config_path}")

    # training loop
    print(f"\nstarting training for {config['num_epochs']} epochs...")
    print(f"batch size: {config['batch_size']}")
    if use_diff_lr:
        print(f"backbone lr: {config['backbone_lr']}, head lr: {config['head_lr']}")
    else:
        print(f"learning rate: {config['learning_rate']}")
    if config['accumulation_steps'] > 1:
        effective_batch = config['batch_size'] * config['accumulation_steps']
        print(f"gradient accumulation: {config['accumulation_steps']} steps (effective batch: {effective_batch})")
    print("-" * 50)

    start_time = datetime.now()

    for epoch in range(config['num_epochs']):
        print(f"\nepoch {epoch + 1}/{config['num_epochs']}")

        # get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # train
        if config['accumulation_steps'] > 1:
            train_loss, train_acc = train_one_epoch_with_accumulation(
                model, train_loader, optimizer, criterion, device, scaler,
                config['use_amp'], config['accumulation_steps']
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, config['use_amp']
            )

        # validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # update history
        history.update(train_loss, train_acc, val_loss, val_acc, current_lr)

        # print epoch summary
        print(f"train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%")
        print(f"val loss: {val_loss:.4f} | val acc: {val_acc:.2f}%")
        print(f"learning rate: {current_lr:.6f}")

        # update scheduler
        scheduler.step(val_acc)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINT_DIR / f'{experiment_name}_best.pth'
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"saved best model (val_acc: {val_acc:.2f}%)")

        # check early stopping
        if early_stopping(val_acc):
            print(f"\nearly stopping triggered at epoch {epoch + 1}")
            break

    end_time = datetime.now()
    training_time = end_time - start_time

    # save training history
    history_path = CHECKPOINT_DIR / f'{experiment_name}_history.json'
    history.save(history_path)
    print(f"\nsaved history to: {history_path}")

    # plot training curves
    plot_path = RESULTS_DIR / f'{experiment_name}_training_curves.png'
    plot_training_curves(history, save_path=plot_path)

    # save summary
    summary = {}
    summary['experiment_name'] = experiment_name
    summary['model_name'] = model_name
    summary['loss_type'] = config['loss_type']
    summary['total_parameters'] = params['total']
    summary['trainable_parameters'] = params['trainable']
    summary['epochs_trained'] = len(history.train_loss)
    summary['best_val_accuracy'] = best_val_acc
    summary['final_train_accuracy'] = history.train_acc[-1]
    summary['final_val_accuracy'] = history.val_acc[-1]
    summary['training_time'] = str(training_time)
    summary['checkpoint_path'] = str(CHECKPOINT_DIR / f'{experiment_name}_best.pth')

    summary_path = CHECKPOINT_DIR / f'{experiment_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary to: {summary_path}")

    # print final summary
    print("\n" + "-" * 50)
    print(f"{experiment_name.upper()} COMPLETE")
    print(f"best val accuracy: {best_val_acc:.2f}%")
    print(f"training time: {training_time}")
    print("-" * 50)

    return best_val_acc, training_time


def train_hierarchical_model(model_name, loss_type, train_loader, val_loader, device, num_models):
    # train hierarchical model with dual outputs
    from src.models import count_parameters, HierarchicalLoss
    from src.utils import get_model, save_config

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 60)

    # get config for this model
    config = get_config_for_model_local(model_name, loss_type)

    # create model
    print(f"\ncreating model: {model_name}")
    model = get_model(model_name, config['num_classes'], num_models=num_models)
    model = model.to(device)

    # print model info
    params = count_parameters(model)
    print(f"total parameters: {params['total']:,}")
    print(f"trainable parameters: {params['trainable']:,}")

    # hierarchical loss function
    criterion = HierarchicalLoss(alpha=HIERARCHICAL_ALPHA)

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience']
    )

    # mixed precision scaler
    scaler = GradScaler('cuda')

    # early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    # training history
    history_data = {}
    history_data['train_loss'] = []
    history_data['train_make_acc'] = []
    history_data['train_model_acc'] = []
    history_data['val_loss'] = []
    history_data['val_make_acc'] = []
    history_data['val_model_acc'] = []
    history_data['lr'] = []

    # best model tracking
    best_val_make_acc = 0.0

    # save config
    config_path = CHECKPOINT_DIR / f'{model_name}_config.json'
    save_config(config, config_path)
    print(f"saved config to: {config_path}")

    # training loop
    print(f"\nstarting training for {config['num_epochs']} epochs...")
    print(f"batch size: {config['batch_size']}")
    print(f"learning rate: {config['learning_rate']}")
    print(f"hierarchical alpha: {HIERARCHICAL_ALPHA}")
    print("-" * 50)

    start_time = datetime.now()

    for epoch in range(config['num_epochs']):
        print(f"\nepoch {epoch + 1}/{config['num_epochs']}")

        # get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # train
        train_loss, train_make_acc, train_model_acc = train_hierarchical_epoch(
            model, train_loader, optimizer, criterion, device, scaler, config['use_amp']
        )

        # validate
        val_loss, val_make_acc, val_model_acc = validate_hierarchical(
            model, val_loader, criterion, device
        )

        # update history
        history_data['train_loss'].append(train_loss)
        history_data['train_make_acc'].append(train_make_acc)
        history_data['train_model_acc'].append(train_model_acc)
        history_data['val_loss'].append(val_loss)
        history_data['val_make_acc'].append(val_make_acc)
        history_data['val_model_acc'].append(val_model_acc)
        history_data['lr'].append(current_lr)

        # print epoch summary
        print(f"train loss: {train_loss:.4f}")
        print(f"train make acc: {train_make_acc:.2f}% | train model acc: {train_model_acc:.2f}%")
        print(f"val make acc: {val_make_acc:.2f}% | val model acc: {val_model_acc:.2f}%")
        print(f"learning rate: {current_lr:.6f}")

        # update scheduler
        scheduler.step(val_make_acc)

        # save best model
        if val_make_acc > best_val_make_acc:
            best_val_make_acc = val_make_acc
            checkpoint_path = CHECKPOINT_DIR / f'{model_name}_best.pth'
            save_checkpoint(model, optimizer, epoch, val_make_acc, checkpoint_path)
            print(f"saved best model (val_make_acc: {val_make_acc:.2f}%)")

        # check early stopping
        if early_stopping(val_make_acc):
            print(f"\nearly stopping triggered at epoch {epoch + 1}")
            break

    end_time = datetime.now()
    training_time = end_time - start_time

    # save training history
    history_path = CHECKPOINT_DIR / f'{model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"\nsaved history to: {history_path}")

    # plot hierarchical training curves
    plot_path = RESULTS_DIR / f'{model_name}_training_curves.png'
    plot_hierarchical_curves(history_data, save_path=plot_path)

    # save summary
    summary = {}
    summary['model_name'] = model_name
    summary['total_parameters'] = params['total']
    summary['trainable_parameters'] = params['trainable']
    summary['epochs_trained'] = len(history_data['train_loss'])
    summary['best_val_make_accuracy'] = best_val_make_acc
    summary['final_train_make_accuracy'] = history_data['train_make_acc'][-1]
    summary['final_train_model_accuracy'] = history_data['train_model_acc'][-1]
    summary['final_val_make_accuracy'] = history_data['val_make_acc'][-1]
    summary['final_val_model_accuracy'] = history_data['val_model_acc'][-1]
    summary['training_time'] = str(training_time)
    summary['checkpoint_path'] = str(CHECKPOINT_DIR / f'{model_name}_best.pth')

    summary_path = CHECKPOINT_DIR / f'{model_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary to: {summary_path}")

    # print final summary
    print("\n" + "-" * 50)
    print(f"{model_name.upper()} COMPLETE")
    print(f"best val make accuracy: {best_val_make_acc:.2f}%")
    print(f"final val model accuracy: {history_data['val_model_acc'][-1]:.2f}%")
    print(f"training time: {training_time}")
    print("-" * 50)

    return best_val_make_acc, training_time

# fml why did i spent so much time on this project when i could have just done something else smh
# i wrote so much damn code i wanna freaking die and no one's gonna read it anyway.- fucking waste of my life

def main():
    from src.dataset import (
        CompCarsDataset,
        get_train_transforms,
        get_val_transforms,
        get_dataloader,
        create_train_val_split
    )
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"using device: cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"gpu: {gpu_name}")
    else:
        device = torch.device('cpu')
        print(f"using device: cpu")

    # load data
    print("\nloading datasets...")

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    full_train_dataset = CompCarsDataset(
        root_dir=DATA_ROOT,
        split_file=TRAIN_SPLIT,
        transform=train_transform,
        task=TASK
    )

    print(f"full training set: {len(full_train_dataset)} images")
    print(f"number of classes: {full_train_dataset.num_makes}")

    # create train/val split
    print("\ncreating train/val split...")

    train_indices, val_indices = create_train_val_split(
        full_train_dataset,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    print(f"training samples: {len(train_indices)}")
    print(f"validation samples: {len(val_indices)}")

    # create datasets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)

    val_full_dataset = CompCarsDataset(
        root_dir=DATA_ROOT,
        split_file=TRAIN_SPLIT,
        transform=val_transform,
        task=TASK
    )
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

    # create dataloaders
    print("\ncreating dataloaders...")

    train_loader = get_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    print(f"training batches: {len(train_loader)}")
    print(f"validation batches: {len(val_loader)}")

    # check if hierarchical model needs separate data loading
    experiment_models = []
    for exp in EXPERIMENTS:
        experiment_models.append(exp[0])

    has_hierarchical = 'hierarchical' in experiment_models

    # create hierarchical dataloaders if needed
    hierarchical_train_loader = None
    hierarchical_val_loader = None
    num_models_dataset = NUM_MODELS

    if has_hierarchical:
        print("\ncreating hierarchical dataloaders...")

        hierarchical_train_dataset = CompCarsDataset(
            root_dir=DATA_ROOT,
            split_file=TRAIN_SPLIT,
            transform=train_transform,
            task='hierarchical'
        )
        hierarchical_train_subset = torch.utils.data.Subset(
            hierarchical_train_dataset, train_indices
        )

        hierarchical_val_dataset = CompCarsDataset(
            root_dir=DATA_ROOT,
            split_file=TRAIN_SPLIT,
            transform=val_transform,
            task='hierarchical'
        )
        hierarchical_val_subset = torch.utils.data.Subset(
            hierarchical_val_dataset, val_indices
        )

        hierarchical_train_loader = get_dataloader(
            hierarchical_train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        hierarchical_val_loader = get_dataloader(
            hierarchical_val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

        num_models_dataset = hierarchical_train_dataset.num_models
        print(f"hierarchical: {num_models_dataset} model classes")

    # train each experiment
    results = []
    total_experiments = len(EXPERIMENTS)

    for exp_idx, (model_name, loss_type) in enumerate(EXPERIMENTS):
        print(f"\n{'#' * 70}")
        print(f"# EXPERIMENT {exp_idx + 1}/{total_experiments}: {model_name} + {loss_type}")
        print(f"{'#' * 70}")

        if model_name == 'hierarchical':
            # use hierarchical training
            best_acc, train_time = train_hierarchical_model(
                model_name,
                loss_type,
                hierarchical_train_loader,
                hierarchical_val_loader,
                device,
                num_models=num_models_dataset
            )
            experiment_name = model_name
        else:
            # use standard training
            best_acc, train_time = train_model(model_name, loss_type, train_loader, val_loader, device)
            experiment_name = f"{model_name}_{loss_type}"

        result = {}
        result['experiment'] = experiment_name
        result['model'] = model_name
        result['loss_type'] = loss_type
        result['best_val_accuracy'] = best_acc
        result['training_time'] = str(train_time)
        results.append(result)

    # print final summary
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n{'experiment':<30} {'best val acc':>15} {'time':>20}")
    print("-" * 65)

    for result in results:
        experiment = result['experiment']
        acc = result['best_val_accuracy']
        time = result['training_time']
        print(f"{experiment:<30} {acc:>14.2f}% {time:>20}")

    print("-" * 55)


if __name__ == '__main__':
    main()

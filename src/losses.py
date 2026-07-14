import torch
import torch.nn as nn
import torch.nn.functional as F

# loss functions for compcars classification

class FocalLoss(nn.Module):
    # focal loss for handling class imbalance
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # compute cross entroy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # compute focal weight: (1 - pt)^gamma
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * ce_loss

        # apply reduction
        if self.reduction == 'mean':
            result = focal_loss.mean()
        elif self.reduction == 'sum':
            result = focal_loss.sum()
        else:
            result = focal_loss

        return result


class LabelSmoothingCrossEntropy(nn.Module):
    # cross entropy loss with label smoothing
    # smoothing=0.1 => 10% of probability is spread across all classes

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)

        # create smoothed labels
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (num_classes - 1)

        # compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)

        # create one-hot targets with smoothing
        one_hot = torch.zeros_like(log_probs)
        one_hot.fill_(smooth_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)

        # compute loss
        loss = -torch.sum(one_hot * log_probs, dim=1)

        # apply reduction
        if self.reduction == 'mean':
            result = loss.mean()
        elif self.reduction == 'sum':
            result = loss.sum()
        else:
            result = loss

        return result


class WeightedFocalLoss(nn.Module):
    # focal loss with class weights for severe imbalance
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        # compute cross entropy with class weights
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # apply focal weight
        focal_loss = focal_weight * ce_loss

        # apply reduction
        if self.reduction == 'mean':
            result = focal_loss.mean()
        elif self.reduction == 'sum':
            result = focal_loss.sum()
        else:
            result = focal_loss

        return result


def get_loss_function(loss_type, num_classes=None, class_weights=None, label_smoothing=0.1, focal_gamma=2.0):
    # factory function to get loss function by name
    # loss_type options: 'ce', 'weighted_ce', 'focal', 'label_smoothing', 'weighted_focal'

    if loss_type == 'ce':
        # standard cross entropy
        criterion = nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        # weighted cross entropy for class imbalance
        if class_weights is None:
            raise ValueError("class_weights required for weighted_ce loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == 'focal':
        # focal loss for hard example mining
        criterion = FocalLoss(gamma=focal_gamma)

    elif loss_type == 'label_smoothing':
        # cross entropy with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type == 'weighted_focal':
        # focal loss with class weights
        criterion = WeightedFocalLoss(class_weights=class_weights, gamma=focal_gamma)

    else:
        raise ValueError(f"unknown loss type: {loss_type}")

    return criterion

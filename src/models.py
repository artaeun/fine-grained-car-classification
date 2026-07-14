import torch
import torch.nn as nn
from torchvision import models

# model architectures for compcars classification

class SimpleCNN(nn.Module):
    # baseline cnn trained from scratch
    # vgg-style architecture with 4 conv blocks amd increasing channels
    # used as baseline and to prove difference in performance without transfer learning

    def __init__(self, num_classes=75):
        super().__init__()

        # conv block 1: 224 -> 112 -> 56
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv block 2: 56 -> 28
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # conv block 3: 28 -> 14
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # conv block 4: 14 -> 7 -> 1 (global pool)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier head
        self.dropout = nn.Dropout(0.5) #to prevent overfitting, dropout at 50%
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.global_pool(x)

        # flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # classifier
        x = self.dropout(x)
        output = self.fc(x)

        return output


class ResNet50Classifier(nn.Module):
    # resnet50 classifier with pretrained imagenet weights (using IMAGENET1K_V1)
    # levereges transfer learning
    # (!) i personalized it by replacing the final fc with custom classifier head

    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()

        # load pretrained resnet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features  # 2048 features

        # replace final fc layer with custom classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output


class SEBlock(nn.Module):
    # squeeze-and-excitation block for channel attention
    # main steps:
    # 1. squeeze step: global average pooling reduces HxW to 1x1
    # 2. excitation step: fc layers learn channel importance's weights
    # 3. scale step: multiply original features by learned weights to recalibrate channel importance

    def __init__(self, channels, reduction=16): #shuld be a good compromize between expresiveness vs parameters
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Calculate reduced channels
        reduced_channels = channels // reduction

        # excitation: two fc layers with relu and sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_channels = x.size(1)

        # 1. Squeeze step
        squeezed = self.squeeze(x)
        squeezed = squeezed.view(batch_size, num_channels)

        # 2. Excitation step
        weights = self.excitation(squeezed)
        weights = weights.view(batch_size, num_channels, 1, 1)

        # 3. Scale
        scaled = x * weights

        return scaled


class ResNet50_SE(nn.Module):
    # resnet50 with SE attention blocks after each stage
    # my attempt at improving resnet by adding attention, inspired by the SE-net paper
    # should use SE blocks to learn to weight the channel importance in a variable way

    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()

        # load pretrained resnet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # add SE blocks after each layer (for simplicity)
        self.se1 = SEBlock(256) #layer1 with 256 channels
        self.se2 = SEBlock(512) # layer 2 and so on
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

        # get number of features from original fc layer
        in_features = self.backbone.fc.in_features  # 2048 feats.

        # custom classifier head as final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # conv1 + bn1 + Relu + maxpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) # batch norm after conv1
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # layer1 + SE attention
        x = self.backbone.layer1(x)
        x = self.se1(x)

        # layer2 + SE attention
        x = self.backbone.layer2(x)
        x = self.se2(x)

        # layer3 + SE attention
        x = self.backbone.layer3(x)
        x = self.se3(x)

        # layer4 + SE attention
        x = self.backbone.layer4(x)
        x = self.se4(x)

        # global average pooling
        x = self.backbone.avgpool(x)

        # flatten
        x = torch.flatten(x, 1)

        # classifier
        output = self.backbone.fc(x)

        return output


class EfficientNetClassifier(nn.Module):
    # efficientnet-b0 with pretrained imagenet weights (IMAGENET1K_V1)
    # should be a more efficient architecture against ResNet and have a similar
    #  accuracy with fewer parameters

    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()

        # load pretrained efficientnet-b0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280 features from original classifier

        # custom head instead of the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output


class HierarchicalCarClassifier(nn.Module):
    # single backbone with two classification heads for hierarchical classification
    # using the natural hierarchy in the data to try and improve performance
    # shared backbone learns features useful that for both tasks

    def __init__(self, num_makes=75, num_models=431, pretrained=True):
        super().__init__()

        # load pretrained resnet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # get number of features from original fc layer
        features_dim = self.backbone.fc.in_features  # 2048 feats.

        self.backbone.fc = nn.Identity()# remove original classifier

        # coarse classification head
        self.make_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(features_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_makes)
        )

        # fine classification head
        self.model_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(features_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_models)
        )

        # store num classes for reference
        self.num_makes = num_makes
        self.num_models = num_models

    def forward(self, x):
        # shared feature extraction
        features = self.backbone(x)

        # parallel classification
        make_logits = self.make_head(features)
        model_logits = self.model_head(features)

        return make_logits, model_logits


class HierarchicalLoss(nn.Module):
    # combined loss for hierarchical classification.
    # loss = alpha * cross_entropy(make) + (1-alpha) * cross_entropy(model)
    # alpha is the trade-off between tasks

    def __init__(self, alpha=0.3, label_smoothing=0.1):
        super().__init__()

        self.alpha = alpha
        self.make_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.model_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, make_logits, model_logits, make_labels, model_labels):
        make_loss = self.make_criterion(make_logits, make_labels)
        model_loss = self.model_criterion(model_logits, model_labels)
        total_loss = self.alpha * make_loss + (1 - self.alpha) * model_loss

        return total_loss, make_loss, model_loss


def count_parameters(model):
    # counts total/trainable parameters in a model
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params = total_params + num_params

        if param.requires_grad:
            trainable_params = trainable_params + num_params

    result = {}
    result['total'] = total_params
    result['trainable'] = trainable_params

    return result


def get_model_summary(model, input_size=(1, 3, 224, 224)):
    # model summary (including layer shapes)

    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)

    print("model architecture:")
    print("-" * 50)

    # forward pass to get output shape
    model.eval()
    with torch.no_grad():
        output = model(x)
    output_shape = output.shape
    print(f"input shape: {input_size}")
    print(f"output shape: {tuple(output_shape)}")
    print("-" * 50)

    # count parameters
    params = count_parameters(model)
    total_millions = params['total'] / 1_000_000
    trainable_millions = params['trainable'] / 1_000_000
    print(f"total parameters: {params['total']:,} ({total_millions:.2f}M)")
    print(f"trainable parameters: {params['trainable']:,} ({trainable_millions:.2f}M)")

    return params


if __name__ == '__main__':    # test model creation and forward pass

    # test simplecnn
    print("\n=== SimpleCNN Model ===\n")
    model = SimpleCNN(num_classes=75)
    params = get_model_summary(model)

    print("\ntesting forward pass...")
    test_input = torch.randn(4, 3, 224, 224)
    output = model(test_input)
    print(f"batch size: 4")
    print(f"output shape: {output.shape}")

    # test Resnet50
    print("\n\n=== ResNet50Classifier Model ===\n")
    model = ResNet50Classifier(num_classes=75, pretrained=False)
    params = get_model_summary(model)

    print("\ntesting forward pass...")
    test_input = torch.randn(4, 3, 224, 224)
    output = model(test_input)
    print(f"batch size: 4")
    print(f"output shape: {output.shape}")

    # test EfficientMet
    print("\n\n=== EfficientNetClassifier Model ===\n")
    model = EfficientNetClassifier(num_classes=75, pretrained=False)
    params = get_model_summary(model)

    print("\ntesting forward pass...")
    test_input = torch.randn(4, 3, 224, 224)
    output = model(test_input)
    print(f"batch size: 4")
    print(f"output shape: {output.shape}")

    print("\nmodel test complete!")

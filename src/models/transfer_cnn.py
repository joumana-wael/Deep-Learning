"""Pretrained (ImageNet) CNN classifiers for transfer learning on CUB.

The model keeps the same layout as ScratchCNN:

    model.features   -> convolutional encoder (pretrained backbone)
    model.classifier -> pooling + dropout + linear head (new, 200 classes)

so the training engine, Grad-CAM (target model.features[-1]) and later tasks
(colorization / captioning encoders) can treat both models the same way.

Training strategies:
    "frozen"   - feature extraction: the whole backbone is frozen and only the
                 new head is trained.
    "partial"  - the last backbone stage and the head are trained.
    "finetune" - every layer is trained; the backbone uses a smaller learning
                 rate than the head (discriminative learning rates).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# Pretrained torchvision weights expect ImageNet normalization, not the
# 0.5/0.5 scaling used by the scratch model.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STRATEGIES = ("frozen", "partial", "finetune")


def _resnet_parts(builder, weights):
    """Split a torchvision ResNet into (features, feature_dim)."""
    backbone = builder(weights=weights)

    features = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )

    return features, backbone.fc.in_features


def _efficientnet_parts(builder, weights):
    """Split a torchvision EfficientNet into (features, feature_dim)."""
    backbone = builder(weights=weights)

    return backbone.features, backbone.classifier[-1].in_features


# Registry of supported backbones: name -> (part builder, model builder, weights)
BACKBONES = {
    "resnet18": (
        _resnet_parts,
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        _resnet_parts,
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V2,
    ),
    "efficientnet_b0": (
        _efficientnet_parts,
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}


class TransferCNN(nn.Module):
    def __init__(
        self,
        num_classes=200,
        backbone="resnet50",
        pretrained=True,
        strategy="finetune",
        dropout=0.4,
    ):
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {', '.join(BACKBONES)}"
            )

        build_parts, builder, weights = BACKBONES[backbone]

        self.backbone_name = backbone
        self.pretrained = pretrained

        # Pretrained encoder (ImageNet weights unless pretrained=False)
        self.features, self.feature_dim = build_parts(
            builder,
            weights if pretrained else None,
        )

        # New classification head, same shape as the scratch model's head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        self.strategy = None
        self.set_strategy(strategy)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Return the pooled encoder embedding (before dropout/linear)."""
        x = self.features(x)
        return self.classifier[1](self.classifier[0](x))

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def set_strategy(self, strategy):
        """Freeze or unfreeze backbone layers for the chosen strategy."""
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {', '.join(STRATEGIES)}"
            )

        self.strategy = strategy

        # Start with everything frozen in the backbone
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        if strategy == "finetune":
            trainable = [self.features]
        elif strategy == "partial":
            # The last stage holds the most task-specific features
            trainable = [self.features[-1]]
        else:
            trainable = []

        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad = True

        # The new head is always trained
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        """Keep BatchNorm layers of frozen backbone parts in eval mode.

        Without this, model.train() would keep updating the ImageNet running
        mean/variance of frozen layers, so "frozen" would not really be frozen.
        """
        super().train(mode)

        if mode:
            for module in self.features.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    is_frozen = not any(
                        parameter.requires_grad
                        for parameter in module.parameters()
                    )
                    if is_frozen:
                        module.eval()

        return self

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------
    def parameter_groups(self, head_lr, backbone_lr):
        """Return optimizer parameter groups.

        The head is always group 0 so optimizer.param_groups[0]["lr"]
        (logged by the training engine) is the head learning rate.
        """
        groups = [{
            "params": list(self.classifier.parameters()),
            "lr": head_lr,
            "name": "head",
        }]

        backbone_params = [
            parameter
            for parameter in self.features.parameters()
            if parameter.requires_grad
        ]

        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone",
            })

        return groups

    def count_parameters(self):
        """Return (trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return trainable, total


def save_model_config(path, model, class_names, extra=None):
    """Save what is needed to rebuild a TransferCNN for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "TransferCNN",
        "backbone": model.backbone_name,
        "strategy": model.strategy,
        "num_classes": len(class_names),
        "dropout": model.classifier[2].p,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_names": list(class_names),
    }

    if extra:
        config.update(extra)

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def load_transfer_model(run_dir, device="cpu"):
    """Rebuild a trained TransferCNN from a training output directory.

    Expects run_dir/model_config.json and run_dir/best_model.pth.

    Returns:
        (model in eval mode, config dict)
    """
    run_dir = Path(run_dir)

    with (run_dir / "model_config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    model = TransferCNN(
        num_classes=config["num_classes"],
        backbone=config["backbone"],
        pretrained=False,
        strategy=config["strategy"],
        dropout=config["dropout"],
    )

    checkpoint = torch.load(
        run_dir / "best_model.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device).eval(), config

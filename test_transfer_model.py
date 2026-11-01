
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.transfer_cnn import BACKBONES, TransferCNN


def check_shapes_and_freezing():
    for backbone in BACKBONES:
        for strategy in ("frozen", "partial", "finetune"):
            # pretrained=False keeps the test offline and fast
            model = TransferCNN(
                num_classes=200,
                backbone=backbone,
                pretrained=False,
                strategy=strategy,
            )
            model.eval()

            outputs = model(torch.randn(2, 3, 224, 224))
            assert outputs.shape == (2, 200)

            embedding = model.extract_features(torch.randn(2, 3, 224, 224))
            assert embedding.shape == (2, model.feature_dim)

            trainable, total = model.count_parameters()
            head = sum(p.numel() for p in model.classifier.parameters())

            if strategy == "frozen":
                assert trainable == head
            elif strategy == "finetune":
                assert trainable == total
            else:
                assert head < trainable < total

            print(
                f"{backbone:16s} {strategy:9s} "
                f"trainable={trainable:>11,} / {total:,}"
            )


def check_frozen_backbone_does_not_change():
    torch.manual_seed(42)

    model = TransferCNN(
        num_classes=10,
        backbone="resnet18",
        pretrained=False,
        strategy="frozen",
    )
    before = {
        name: tensor.clone()
        for name, tensor in model.features.state_dict().items()
    }

    optimizer = AdamW(model.parameter_groups(head_lr=0.01, backbone_lr=0.0))
    assert len(optimizer.param_groups) == 1

    model.train()
    # Frozen BatchNorm layers must stay in eval mode during training
    assert all(
        not module.training
        for module in model.features.modules()
        if isinstance(module, nn.BatchNorm2d)
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 10, (8,))
    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        optimizer.step()

    # Weights and BatchNorm running statistics are unchanged
    for name, tensor in model.features.state_dict().items():
        assert torch.equal(before[name], tensor), name

    print("Frozen backbone stays unchanged during training.")


def check_model_can_learn():
    torch.manual_seed(42)

    model = TransferCNN(
        num_classes=4,
        backbone="resnet18",
        pretrained=False,
        strategy="finetune",
        dropout=0.0,
    )
    optimizer = AdamW(model.parameter_groups(head_lr=0.001, backbone_lr=0.001))
    criterion = nn.CrossEntropyLoss()

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    model.train()
    for step in range(30):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        accuracy = (model(images).argmax(dim=1) == labels).float().mean()

    print(f"Overfit check: loss={loss.item():.4f}, accuracy={accuracy:.0%}")
    assert accuracy == 1.0


def main():
    check_shapes_and_freezing()
    check_frozen_backbone_does_not_change()
    check_model_can_learn()
    print("Transfer model checks passed.")


if __name__ == "__main__":
    main()

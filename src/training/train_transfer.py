"""Train a pretrained CNN (transfer learning) on the shared CUB split.

Examples:
    python -m src.training.train_transfer --config configs/transfer_frozen.json
    python -m src.training.train_transfer --config configs/transfer_finetune.json
    python -m src.training.train_transfer --backbone resnet18 --strategy partial

Values given on the command line override values from the config file.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.loaders import create_classification_loaders
from src.models.transfer_cnn import (
    BACKBONES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STRATEGIES,
    TransferCNN,
    save_model_config,
)
from src.training.engine import fit
from src.training.logger import save_training_history
from src.training.tracker import (
    setup_mlflow,
    start_run,
    end_run,
    log_parameters,
    log_metrics,
    log_artifact,
)


DEFAULTS = {
    "backbone": "resnet50",
    "strategy": "finetune",
    "pretrained": True,
    "epochs": 15,
    "batch_size": 32,
    "head_lr": 0.001,
    "backbone_lr": 0.0001,
    "weight_decay": 0.0001,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "seed": 42,
    "num_workers": 0,
    "output_dir": None,
    "run_name": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transfer learning classifier."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file with run settings (see configs/).",
    )
    parser.add_argument("--backbone", choices=list(BACKBONES))
    parser.add_argument("--strategy", choices=list(STRATEGIES))
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_const",
        const=False,
        help="Use random weights (ablation only).",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-name", type=str)

    args = parser.parse_args()

    # Defaults < config file < command line
    settings = dict(DEFAULTS)

    if args.config is not None:
        with open(args.config, encoding="utf-8") as file:
            file_settings = json.load(file)

        unknown = set(file_settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        settings.update(file_settings)

    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value

    if settings["run_name"] is None:
        settings["run_name"] = (
            f"transfer_{settings['backbone']}_{settings['strategy']}"
        )

    if settings["output_dir"] is None:
        settings["output_dir"] = f"outputs/{settings['run_name']}"

    return settings


def main():
    settings = parse_args()

    # Reproducibility
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("TRANSFER LEARNING")
    print("=" * 50)
    print("Device:", device)
    for key, value in settings.items():
        print(f"{key}: {value}")

    # Data: same split and augmentation as the scratch model, but with the
    # ImageNet normalization the pretrained weights were trained with.
    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=settings["batch_size"],
            seed=seed,
            num_workers=settings["num_workers"],
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    # Model
    model = TransferCNN(
        num_classes=len(class_names),
        backbone=settings["backbone"],
        pretrained=settings["pretrained"],
        strategy=settings["strategy"],
        dropout=settings["dropout"],
    ).to(device)

    trainable_params, total_params = model.count_parameters()
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"/ {total_params:,}"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=settings["label_smoothing"],
    )

    optimizer = AdamW(
        model.parameter_groups(
            head_lr=settings["head_lr"],
            backbone_lr=settings["backbone_lr"],
        ),
        weight_decay=settings["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
    )

    # Output paths
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "training_history.csv"
    config_path = output_dir / "model_config.json"
    summary_path = output_dir / "run_summary.json"

    # Experiment tracking
    setup_mlflow(
        experiment_name="bird_classification"
    )

    start_run(
        run_name=settings["run_name"]
    )

    try:
        # fit() logs epochs, batch size, head learning rate and optimizer
        log_parameters({
            "model": "TransferCNN",
            "backbone": settings["backbone"],
            "strategy": settings["strategy"],
            "pretrained": settings["pretrained"],
            "head_lr": settings["head_lr"],
            "backbone_lr": settings["backbone_lr"],
            "weight_decay": settings["weight_decay"],
            "dropout": settings["dropout"],
            "label_smoothing": settings["label_smoothing"],
            "seed": seed,
            "normalization": "imagenet",
            "trainable_params": trainable_params,
            "total_params": total_params,
        })

        start_time = time.perf_counter()

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=settings["epochs"],
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
        )

        elapsed_seconds = time.perf_counter() - start_time

        save_training_history(
            history=history,
            output_path=history_path,
        )

        best = max(
            history,
            key=lambda result: result["val_accuracy"],
        )

        summary = {
            "run_name": settings["run_name"],
            "backbone": settings["backbone"],
            "strategy": settings["strategy"],
            "pretrained": settings["pretrained"],
            "trainable_params": trainable_params,
            "total_params": total_params,
            "best_epoch": best["epoch"],
            "best_val_accuracy": best["val_accuracy"],
            "best_val_loss": best["val_loss"],
            "elapsed_seconds": elapsed_seconds,
            "settings": settings,
        }

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        save_model_config(
            path=config_path,
            model=model,
            class_names=class_names,
        )

        log_metrics(
            {
                "best_val_accuracy": best["val_accuracy"],
                "best_val_loss": best["val_loss"],
                "elapsed_seconds": elapsed_seconds,
            },
            step=best["epoch"],
        )

        for path in (history_path, config_path, summary_path):
            log_artifact(path)

    finally:
        end_run()

    print("\nTransfer learning completed!")
    print(f"Best validation accuracy: {best['val_accuracy']:.2f}% "
          f"(epoch {best['epoch']})")
    print(f"Time: {elapsed_seconds / 60:.1f} minutes")
    print("Best checkpoint:", checkpoint_path)
    print("Training history:", history_path)
    print("Model config:", config_path)


if __name__ == "__main__":
    main()

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.training.engine import fit
from src.training.logger import save_training_history
from src.training.tracker import (
    setup_mlflow,
    start_run,
    end_run,
    log_artifact,
)


def load_config(config_path):
    """Load training settings from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train a Scratch CNN classifier."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a latest training checkpoint.",
    )

    # Load YAML defaults first; explicit CLI arguments override them.
    config_args, _ = parser.parse_known_args()
    config = load_config(config_args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Resume from:", args.resume_from)

    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=0,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    model = ScratchCNN(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_checkpoint.pth"
    history_path = output_dir / "training_history.csv"

    setup_mlflow(experiment_name="bird_classification")
    start_run(run_name="classification_training")

    try:
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            checkpoint_path=checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            scheduler=scheduler,
            resume_from=args.resume_from,
        )

        save_training_history(
            history=history,
            output_path=history_path,
        )

        log_artifact(history_path)

    finally:
        end_run()

    print("\nTraining completed!")
    print("Best checkpoint:", checkpoint_path)
    print("Latest checkpoint:", last_checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.training.engine import fit
from src.training.logger import save_training_history


def main():
    parser = argparse.ArgumentParser(
        description="Train a Scratch CNN classifier."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training_run",
        help="Directory for checkpoints and logs.",
    )

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
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

    # Data
    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=0,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    # Model
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

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "training_history.csv"

    # Train
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
    )

    # Save experiment history
    save_training_history(
        history=history,
        output_path=history_path,
    )

    print("\nTraining completed!")
    print("Best checkpoint:", checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.scratch_cnn import ScratchCNN
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def main():
    # Use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Create a model
    model = ScratchCNN(
        num_classes=200
    ).to(device)

    # Create an optimizer
    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    # Where the checkpoint will be saved
    checkpoint_path = Path(
        "outputs/test_checkpoint.pth"
    )

    # Example metrics
    metrics = {
        "train_loss": 5.02,
        "train_accuracy": 1.71,
        "val_loss": 4.95,
        "val_accuracy": 2.75,
    }

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    print("\nSaving checkpoint...")

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics=metrics,
    )

    print("Checkpoint saved:", checkpoint_path)

    # -----------------------------
    # Create a NEW model
    # -----------------------------
    new_model = ScratchCNN(
        num_classes=200
    ).to(device)

    new_optimizer = Adam(
        new_model.parameters(),
        lr=0.001,
    )

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    print("\nLoading checkpoint...")

    epoch, loaded_metrics = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device=device,
    )

    # -----------------------------
    # Check the results
    # -----------------------------
    print("\nLoaded epoch:", epoch)
    print("Loaded metrics:", loaded_metrics)

    # Verify that the model weights match
    for original, loaded in zip(
        model.parameters(),
        new_model.parameters(),
    ):
        assert torch.equal(
            original,
            loaded,
        )

    print("\nCheckpoint test passed! ✅")


if __name__ == "__main__":
    main()
    
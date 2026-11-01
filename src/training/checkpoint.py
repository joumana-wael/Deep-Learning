from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
):
    """Save the current training state."""

    # Make sure the output folder exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect everything we want to save
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }

    # Save everything into one file
    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    device,
):
    """Load a saved training checkpoint.

    Returns:
        The saved epoch and metrics.
    """

    # Load the checkpoint file
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # Restore the model's learned weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore the optimizer's state
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Return information about where training stopped
    return checkpoint["epoch"], checkpoint["metrics"]
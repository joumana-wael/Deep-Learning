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
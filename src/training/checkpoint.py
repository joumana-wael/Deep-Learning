from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    scheduler=None,
    best_val_accuracy=-1.0,
    history=None,
):
    """Save the complete training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "metrics": metrics,
        "best_val_accuracy": best_val_accuracy,
        "history": history or [],
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    device,
    scheduler=None,
    return_state=False,
):
    """Restore model, optimizer, and optional scheduler state."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    if return_state:
        return checkpoint

    # Preserve compatibility with the original function.
    return checkpoint["epoch"], checkpoint["metrics"] 
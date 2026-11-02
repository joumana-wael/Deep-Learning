
import torch

from src.training.tracker import (
    log_metrics,
    log_parameters,
    log_artifact,
)
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    if train_total == 0:
        raise ValueError("The training loader is empty.")

    return (
        train_loss_sum / train_total,
        100.0 * train_correct / train_total,
    )


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    if val_total == 0:
        raise ValueError("The validation loader is empty.")

    return (
        val_loss_sum / val_total,
        100.0 * val_correct / val_total,
    )


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path=None,
    scheduler=None,
    resume_from=None,
    last_checkpoint_path=None,
):
    history = []
    best_val_accuracy = -1.0
    start_epoch = 0

    if resume_from is not None:
        print(f"Resuming from {resume_from}")

        state = load_checkpoint(
            resume_from,
            model,
            optimizer,
            device,
            scheduler=scheduler,
            return_state=True,
        )

        start_epoch = state["epoch"]
        best_val_accuracy = state.get("best_val_accuracy", -1.0)
        history = state.get("history", [])

        print(f"Resuming at epoch {start_epoch + 1}")

    if start_epoch >= epochs:
        print("Requested epoch count already reached.")
        return history

    log_parameters({
        "epochs": epochs,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": train_loader.batch_size,
        "optimizer": optimizer.__class__.__name__,
        "loss_function": criterion.__class__.__name__,
        "resumed_from_epoch": start_epoch,
    })

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_results = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        history.append(epoch_results)

        log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        }, step=epoch + 1)

        is_best = val_accuracy > best_val_accuracy

        if is_best:
            best_val_accuracy = val_accuracy

        if last_checkpoint_path is not None:
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                epoch_results,
                scheduler=scheduler,
                best_val_accuracy=best_val_accuracy,
                history=history,
            )

        if checkpoint_path is not None and is_best:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                epoch_results,
                scheduler=scheduler,
                best_val_accuracy=best_val_accuracy,
                history=history,
            )
            print(f"Best checkpoint saved: {val_accuracy:.2f}%")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    if checkpoint_path is not None:
        log_artifact(checkpoint_path)

    if last_checkpoint_path is not None:
        log_artifact(last_checkpoint_path)

    return history

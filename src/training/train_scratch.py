
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN


def main():
    # ==========================================
    # 1. Reproducibility
    # ==========================================
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ==========================================
    # 2. Configure GPU
    # ==========================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("SCRATCH CNN - EXPERIMENT 2")
    print("=" * 50)

    print("Training device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )
        torch.cuda.empty_cache()

    # ==========================================
    # 3. Load Dataset
    # ==========================================
    batch_size = 8

    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=batch_size,
            seed=seed,
            num_workers=0,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    print("\nDataset information:")
    print("Number of classes:", len(class_names))
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Batch size:", batch_size)

    # ==========================================
    # 4. Create Model
    # ==========================================
    model = ScratchCNN(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    # Reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
    )

    # ==========================================
    # 5. Experiment Output Directory
    # ==========================================
    output_dir = Path("outputs/scratch_cnn_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    csv_path = output_dir / "training_history.csv"

    epochs = 20
    best_val_accuracy = -1.0

    # ==========================================
    # 6. Training History CSV
    # ==========================================
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # ======================================
        # 7. Training Loop
        # ======================================
        for epoch in range(epochs):

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # ----------------------------------
            # Training phase
            # ----------------------------------
            model.train()

            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch_index, (images, labels) in enumerate(
                train_loader
            ):

                images = images.to(
                    device,
                    non_blocking=(device.type == "cuda")
                )

                labels = labels.to(
                    device,
                    non_blocking=(device.type == "cuda")
                )

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss_sum += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                train_correct += (
                    predictions == labels
                ).sum().item()

                train_total += labels.size(0)

            train_loss = train_loss_sum / train_total

            train_accuracy = (
                100.0 * train_correct / train_total
            )

            # ----------------------------------
            # Validation phase
            # ----------------------------------
            model.eval()

            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(
                        device,
                        non_blocking=(device.type == "cuda")
                    )

                    labels = labels.to(
                        device,
                        non_blocking=(device.type == "cuda")
                    )

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    val_loss_sum += (
                        loss.item() * images.size(0)
                    )

                    predictions = outputs.argmax(dim=1)

                    val_correct += (
                        predictions == labels
                    ).sum().item()

                    val_total += labels.size(0)

            val_loss = val_loss_sum / val_total

            val_accuracy = (
                100.0 * val_correct / val_total
            )

            # ----------------------------------
            # Learning rate scheduling
            # ----------------------------------
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            # ----------------------------------
            # Print results
            # ----------------------------------
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}%"
            )

            print(
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

            print(
                f"Learning Rate: {current_lr:.6f}"
            )

            # ----------------------------------
            # Save CSV metrics
            # ----------------------------------
            writer.writerow({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })

            file.flush()

            # ----------------------------------
            # Save best model
            # ----------------------------------
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "epoch": epoch + 1,
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss,
                        "seed": seed,
                    },
                    checkpoint_path,
                )

                print(
                    "Best model saved!",
                    f"Validation accuracy: {best_val_accuracy:.2f}%"
                )

            # ----------------------------------
            # GPU memory information
            # ----------------------------------
            if device.type == "cuda":

                allocated = (
                    torch.cuda.memory_allocated(0)
                    / (1024 ** 3)
                )

                print(
                    f"GPU memory allocated: {allocated:.2f} GB"
                )

    # ==========================================
    # 8. Final Results
    # ==========================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 2 COMPLETED")
    print("=" * 50)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print("Saved model:", checkpoint_path)
    print("Training history:", csv_path)


if __name__ == "__main__":
    main()

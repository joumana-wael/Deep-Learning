import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.training.engine import fit
def main():
    # Choose GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the data
    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=8,
            seed=42,
            num_workers=0,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    # Create the model
    model = ScratchCNN(
        num_classes=len(class_names)
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    # Train and validate for multiple epochs
history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=3,
)
print("\nTraining history:")

for epoch_results in history:
    print(epoch_results)

    # Validate
    print("Validating...")

    val_loss, val_accuracy = validate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # Print results
    print("\nResults:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.2f}%")


if __name__ == "__main__":
    main()
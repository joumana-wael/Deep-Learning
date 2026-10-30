
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    train_loader, _, class_names = create_classification_loaders(
        batch_size=16,
        seed=42,
        augment_training=False,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    device = torch.device("cpu")
    model = ScratchCNN(num_classes=len(class_names)).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    print("Testing whether the CNN can learn 16 images...")

    for step in range(101):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(images).argmax(dim=1)
                accuracy = (
                    (predictions == labels).float().mean().item() * 100
                )
            model.train()

            print(
                f"Step {step:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    print("Diagnostic complete.")


if __name__ == "__main__":
    main()

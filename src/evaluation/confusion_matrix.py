import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN


def generate_confusion_matrix(
    checkpoint_path="outputs/scratch_cnn_v2/best_model.pth",
    output_dir="outputs/evaluation/scratch_cnn",
    batch_size=32,
):
    """Generate and save the ScratchCNN confusion matrix."""

    # 1. Select CPU or GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the trained model checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 3. Rebuild the model and load its trained weights.
    model = ScratchCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Load the validation dataset.
    _, val_loader, loader_class_names = create_classification_loaders(
        batch_size=batch_size,
        seed=42,
        num_workers=0,
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment_training=False,
        pin_memory=False,
    )

    if list(loader_class_names) != list(class_names):
        raise ValueError("Checkpoint class names do not match dataset classes.")

    # 5. Generate predictions.
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Generating predictions"):
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    # 6. Calculate the confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
    )

    # 7. Create the output folder.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 8. Save the complete 200 x 200 matrix as CSV.
    csv_path = output_path / "confusion_matrix.csv"
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%d")

    # 9. Select the 20 most represented actual classes for visualization.
    class_totals = matrix.sum(axis=1)
    selected_classes = np.argsort(class_totals)[-20:]
    selected_matrix = matrix[np.ix_(selected_classes, selected_classes)]

    # 10. Create the confusion matrix image.
    fig, ax = plt.subplots(figsize=(16, 14))

    image = ax.imshow(selected_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    selected_names = [class_names[i] for i in selected_classes]

    ax.set_xticks(range(len(selected_names)))
    ax.set_yticks(range(len(selected_names)))

    ax.set_xticklabels(selected_names, rotation=90, fontsize=8)
    ax.set_yticklabels(selected_names, fontsize=8)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("ScratchCNN Confusion Matrix (20 Most Represented Classes)")

    fig.tight_layout()

    # 11. Save the image.
    image_path = output_path / "confusion_matrix.png"
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    print("\nConfusion matrix generated successfully!")
    print(f"Full matrix CSV: {csv_path}")
    print(f"Visualization: {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix for ScratchCNN."
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/scratch_cnn_v2/best_model.pth",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/scratch_cnn",
    )

    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    generate_confusion_matrix(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
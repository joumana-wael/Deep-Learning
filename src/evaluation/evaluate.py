
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.evaluation.metrics import calculate_metrics


def evaluate_scratch_model(
    checkpoint_path="outputs/scratch_cnn_v2/best_model.pth",
    output_dir="outputs/evaluation/scratch_cnn",
    batch_size=32,
):
    """
    Evaluate the saved Scratch CNN on the validation dataset.
    """

    # 1. Select the device (CPU or GPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the saved checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

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
        raise ValueError(
            "Checkpoint class names do not match dataset class names."
        )

    # 5. Make predictions.
    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)

            # Convert model scores into probabilities.
            probabilities = torch.softmax(outputs, dim=1)

            # Choose the class with the highest probability.
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    # 6. Convert predictions into NumPy arrays.
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # 7. Calculate all five evaluation metrics.
    results = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    # 8. Save the evaluation results.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results["model"] = "ScratchCNN"
    results["checkpoint_epoch"] = int(checkpoint["epoch"])
    results["num_classes"] = num_classes
    results["num_validation_samples"] = len(y_true)

    results_path = output_path / "metrics.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # 9. Display the results.
    print("\nEvaluation Results")
    print("------------------")

    for metric_name, value in results.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Scratch CNN model."
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/scratch_cnn_v2/best_model.pth",
        help="Path to the saved model checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/scratch_cnn",
        help="Directory where evaluation results will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images processed at a time.",
    )

    args = parser.parse_args()

    evaluate_scratch_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


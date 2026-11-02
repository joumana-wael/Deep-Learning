import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN


def analyze_errors(
    checkpoint_path="outputs/scratch_cnn_v2/best_model.pth",
    output_dir="outputs/evaluation/scratch_cnn",
    batch_size=32,
):
    """Analyze correct and incorrect ScratchCNN predictions."""

    # 1. Select the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 3. Rebuild the trained model.
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

    # 5. Collect predictions and confidence scores.
    records = []
    image_index = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing errors"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, predictions = torch.max(probabilities, dim=1)

            for actual, predicted, score in zip(
                labels.tolist(),
                predictions.cpu().tolist(),
                confidence.cpu().tolist(),
            ):
                records.append({
                    "image_index": image_index,
                    "actual_class": class_names[actual],
                    "predicted_class": class_names[predicted],
                    "confidence": float(score),
                    "correct": actual == predicted,
                })

                image_index += 1

    # 6. Convert the results into a DataFrame.
    results = pd.DataFrame(records)

    # 7. Separate incorrect predictions.
    errors = results[results["correct"] == False].copy()

    # 8. Create the output folder.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 9. Save all predictions.
    all_predictions_path = output_path / "all_predictions.csv"
    results.to_csv(all_predictions_path, index=False)

    # 10. Save only incorrect predictions.
    errors_path = output_path / "misclassified_samples.csv"
    errors.to_csv(errors_path, index=False)

    # 11. Print a summary.
    total = len(results)
    correct = int(results["correct"].sum())
    incorrect = len(errors)

    print("\nError Analysis Results")
    print("----------------------")
    print(f"Total validation images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {correct / total:.4f}")

    if incorrect > 0:
        print(
            "Average confidence on incorrect predictions: "
            f"{errors['confidence'].mean():.4f}"
        )

    print(f"\nAll predictions saved to: {all_predictions_path}")
    print(f"Misclassified samples saved to: {errors_path}")

    return results, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze ScratchCNN classification errors."
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

    analyze_errors(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
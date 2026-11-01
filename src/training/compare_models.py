"""Compare the scratch CNN against the transfer learning runs.

Every run is evaluated on the same validation split, each with its own input
normalization (0.5/0.5 for scratch, ImageNet for pretrained models), and the
results are written as a table plus training-curve plots.

Example:
    python -m src.training.compare_models \
        --run scratch=outputs/scratch_cnn_v2 \
        --run frozen=outputs/transfer_resnet50_frozen \
        --run finetune=outputs/transfer_resnet50_finetune

A run directory needs training_history.csv. If it also has best_model.pth the
checkpoint is evaluated; model_config.json marks it as a transfer run.
Use --skip-eval to build the table from the training histories only.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.models.transfer_cnn import load_transfer_model


SCRATCH_MEAN = (0.5, 0.5, 0.5)
SCRATCH_STD = (0.5, 0.5, 0.5)


def read_history(path):
    """Read a training_history.csv file into a list of float dicts."""
    with Path(path).open(encoding="utf-8") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def load_run_model(run_dir, device):
    """Return (model, mean, std, kind) for a run directory."""
    run_dir = Path(run_dir)

    if (run_dir / "model_config.json").is_file():
        model, config = load_transfer_model(run_dir, device=device)
        return model, tuple(config["mean"]), tuple(config["std"]), "transfer"

    checkpoint = torch.load(
        run_dir / "best_model.pth",
        map_location=device,
    )
    state_dict = checkpoint["model_state_dict"]
    num_classes = state_dict["classifier.3.weight"].shape[0]

    model = ScratchCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)

    return model.to(device).eval(), SCRATCH_MEAN, SCRATCH_STD, "scratch"


def evaluate(model, loader, num_classes, device):
    """Return top-1 accuracy, top-5 accuracy and macro F1 (all in %)."""
    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).cpu()

            predictions = outputs.argmax(dim=1)
            top5 = outputs.topk(5, dim=1).indices

            top5_correct += (
                top5 == labels.unsqueeze(1)
            ).any(dim=1).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):
                confusion[label, prediction] += 1

    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp(min=1)
    recall = true_positive / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "val_top1": 100.0 * true_positive.sum().item() / total,
        "val_top5": 100.0 * top5_correct / total,
        "val_macro_f1": 100.0 * f1.mean().item(),
    }


def summarize_history(history):
    best = max(history, key=lambda row: row["val_accuracy"])
    return {
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": min(row["val_loss"] for row in history),
        "train_accuracy_at_best": best["train_accuracy"],
        # Large gap = the model memorizes the training set
        "generalization_gap": best["train_accuracy"] - best["val_accuracy"],
    }


def plot_curves(histories, metric, ylabel, output_path):
    figure, axis = plt.subplots(figsize=(8, 5))

    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axis.plot(
            epochs,
            [row[f"val_{metric}"] for row in history],
            label=f"{name} (val)",
        )
        axis.plot(
            epochs,
            [row[f"train_{metric}"] for row in history],
            linestyle="--",
            alpha=0.5,
            color=axis.lines[-1].get_color(),
            label=f"{name} (train)",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Scratch vs transfer learning: {ylabel.lower()}")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_bars(rows, output_path):
    names = [row["run"] for row in rows]
    evaluated = all(isinstance(row.get("val_top1"), float) for row in rows)
    key = "val_top1" if evaluated else "best_val_accuracy"
    values = [row[key] for row in rows]

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(names, values, color="#4C72B0")
    axis.bar_label(bars, fmt="%.1f%%")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Best validation accuracy per model")
    axis.set_ylim(0, 100)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_markdown(rows, output_path):
    columns = list(rows[0].keys())

    def fmt(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        if isinstance(value, int) and value >= 10000:
            return f"{value:,}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(fmt(row[column]) for column in columns) + " |"
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_runs(values):
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Use NAME=DIR for --run, got: {value}")
        name, run_dir = value.split("=", 1)
        runs[name] = Path(run_dir)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Compare scratch and transfer learning runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=DIR of a training output directory (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not evaluate checkpoints; use training histories only.",
    )
    args = parser.parse_args()

    runs = parse_runs(args.run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    histories = {}
    rows = []

    for name, run_dir in runs.items():
        print(f"\n=== {name}: {run_dir}")

        history = read_history(run_dir / "training_history.csv")
        histories[name] = history

        row = {"run": name, **summarize_history(history)}

        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as file:
                summary = json.load(file)
            row["model"] = f"{summary['backbone']} ({summary['strategy']})"
            row["trainable_params"] = summary["trainable_params"]
            row["minutes"] = summary["elapsed_seconds"] / 60
        else:
            row["model"] = "ScratchCNN"
            row["trainable_params"] = sum(
                p.numel() for p in ScratchCNN().parameters()
            )
            row["minutes"] = float("nan")

        if not args.skip_eval and (run_dir / "best_model.pth").is_file():
            model, mean, std, _ = load_run_model(run_dir, device)

            _, val_loader, class_names = create_classification_loaders(
                batch_size=args.batch_size,
                mean=mean,
                std=std,
                augment_training=False,
                pin_memory=(device.type == "cuda"),
            )

            row.update(evaluate(model, val_loader, len(class_names), device))

        for key, value in row.items():
            print(f"{key}: {value}")

        rows.append(row)

    # Same columns for every row so the table lines up
    columns = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    rows = [{key: row.get(key, "") for key in columns} for row in rows]

    table_csv = output_dir / "comparison_table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, output_dir / "comparison_table.md")

    plot_curves(
        histories, "accuracy", "Accuracy (%)",
        output_dir / "accuracy_curves.png",
    )
    plot_curves(
        histories, "loss", "Loss",
        output_dir / "loss_curves.png",
    )
    plot_bars(rows, output_dir / "best_accuracy.png")

    print("\nComparison saved to:", output_dir)


if __name__ == "__main__":
    main()

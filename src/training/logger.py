from pathlib import Path
import csv


def save_training_history(history, output_path):
    """Save training history to a CSV file.

    Args:
        history: List of dictionaries returned by fit().
        output_path: Path where the CSV file will be saved.
    """

    if not history:
        raise ValueError("Training history is empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = history[0].keys()

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"Training history saved to: {output_path}")
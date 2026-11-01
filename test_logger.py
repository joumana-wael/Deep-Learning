from pathlib import Path

from src.training.logger import save_training_history


def main():

    history = [
        {
            "epoch": 1,
            "learning_rate": 0.001,
            "train_loss": 5.36,
            "train_accuracy": 0.88,
            "val_loss": 5.15,
            "val_accuracy": 1.42,
        },
        {
            "epoch": 2,
            "learning_rate": 0.001,
            "train_loss": 5.13,
            "train_accuracy": 1.11,
            "val_loss": 5.05,
            "val_accuracy": 1.67,
        },
        {
            "epoch": 3,
            "learning_rate": 0.0005,
            "train_loss": 5.03,
            "train_accuracy": 1.65,
            "val_loss": 4.95,
            "val_accuracy": 3.08,
        },
    ]

    output_path = Path(
        "outputs/test_experiment/training_history.csv"
    )

    save_training_history(
        history=history,
        output_path=output_path,
    )

    assert output_path.exists()

    print("\nLogger test passed! ✅")
    print("Saved file:", output_path)


if __name__ == "__main__":
    main()
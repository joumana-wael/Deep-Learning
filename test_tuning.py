from pathlib import Path

from src.tuning.runner import run_tuning


def main():

    # Small test: only two configurations
    configs = [
        {
            "learning_rate": 0.001,
            "batch_size": 8,
        },
        {
            "learning_rate": 0.0005,
            "batch_size": 8,
        },
    ]

    results, best_result = run_tuning(
        configs=configs,
        epochs=2,
        output_dir=Path("outputs/test_tuning"),
        seed=42,
    )

    assert len(results) == 2
    assert "best_val_accuracy" in best_result

    print("\nTuning runner test passed! ✅")
    print("Best result:", best_result)


if __name__ == "__main__":
    main()
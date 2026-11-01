from src.training.tracker import setup_mlflow


def main():
    setup_mlflow(
        experiment_name="test_experiment"
    )

    print("MLflow tracker imported successfully! ✅")


if __name__ == "__main__":
    main()
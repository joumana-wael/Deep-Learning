from pathlib import Path

import mlflow


def setup_mlflow(
    experiment_name="bird_classification",
    tracking_uri="sqlite:///mlflow.db",
):
    """Configure MLflow tracking for the project."""

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_parameters(params):
    """Log hyperparameters for the current MLflow run."""

    mlflow.log_params(params)


def log_metrics(metrics, step):
    """Log metrics for one epoch."""

    mlflow.log_metrics(metrics, step=step)


def log_artifact(path):
    """Log a file as an MLflow artifact."""

    path = Path(path)

    if path.exists():
        mlflow.log_artifact(str(path))


def start_run(run_name=None):
    """Start an MLflow experiment run."""

    return mlflow.start_run(run_name=run_name)


def end_run():
    """End the currently active MLflow run."""

    if mlflow.active_run() is not None:
        mlflow.end_run()
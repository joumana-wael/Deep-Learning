
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification evaluation metrics.

    Args:
        y_true: Actual class labels.
        y_pred: Predicted class labels.
        y_prob: Optional prediction scores for each class.
                Shape: (number of samples, number of classes).

    Returns:
        A dictionary containing accuracy, precision, recall,
        F1-score, and optionally top-5 accuracy.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    # Calculate Top-5 Accuracy if prediction scores are provided.
    if y_prob is not None:
        top5_predictions = np.argsort(y_prob, axis=1)[:, -5:]

        correct_top5 = [
            true_label in predicted_labels
            for true_label, predicted_labels
            in zip(y_true, top5_predictions)
        ]

        metrics["top5_accuracy"] = float(np.mean(correct_top5))

    return metrics

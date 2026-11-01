from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN
from src.training.engine import fit
from src.training.logger import save_training_history


def run_trial(
    config,
    trial_number,
    epochs,
    device,
    output_dir,
    seed=42,
):
    """Run one hyperparameter tuning trial."""

    trial_seed = seed + trial_number

    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    trial_dir = (
        Path(output_dir)
        / f"trial_{trial_number:03d}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"TRIAL {trial_number}")
    print("=" * 50)
    print("Configuration:", config)

    start_time = time.perf_counter()

    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=config["batch_size"],
            seed=trial_seed,
            num_workers=0,
            augment_training=True,
            pin_memory=(device.type == "cuda"),
        )
    )

    model = ScratchCNN(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
    )

    checkpoint_path = trial_dir / "best_model.pth"

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
    )

    save_training_history(
        history=history,
        output_path=trial_dir / "training_history.csv",
    )

    elapsed_time = time.perf_counter() - start_time

    best_epoch_result = max(
        history,
        key=lambda result: result["val_accuracy"],
    )

    result = {
        "trial": trial_number,
        **config,
        "best_epoch": best_epoch_result["epoch"],
        "best_val_accuracy": (
            best_epoch_result["val_accuracy"]
        ),
        "best_val_loss": best_epoch_result["val_loss"],
        "elapsed_seconds": elapsed_time,
    }

    print(
        f"\nTrial {trial_number} completed!"
    )
    print(
        f"Best validation accuracy: "
        f"{result['best_val_accuracy']:.2f}%"
    )
    print(
        f"Time: {elapsed_time:.2f} seconds"
    )

    return result


def save_tuning_results(results, output_path):
    """Save the summary of all tuning trials to CSV."""

    if not results:
        raise ValueError("No tuning results to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nTuning results saved to: {output_path}"
    )


def run_tuning(
    configs,
    epochs,
    output_dir,
    seed=42,
):
    """Run multiple hyperparameter tuning trials."""

    if not configs:
        raise ValueError("No configurations provided.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(output_dir)

    print("=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    print("Device:", device)
    print("Trials:", len(configs))
    print("Epochs per trial:", epochs)

    results = []

    for trial_number, config in enumerate(
        configs,
        start=1,
    ):
        result = run_trial(
            config=config,
            trial_number=trial_number,
            epochs=epochs,
            device=device,
            output_dir=output_dir,
            seed=seed,
        )

        results.append(result)

    save_tuning_results(
        results=results,
        output_path=output_dir / "tuning_results.csv",
    )

    best_result = max(
        results,
        key=lambda result: result["best_val_accuracy"],
    )

    print("\n" + "=" * 50)
    print("TUNING COMPLETED")
    print("=" * 50)
    print("Best configuration:", {
        key: best_result[key]
        for key in best_result
        if key in configs[0]
    })
    print(
        f"Best validation accuracy: "
        f"{best_result['best_val_accuracy']:.2f}%"
    )

    return results, best_result
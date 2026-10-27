"""Build shared training and validation DataLoaders for CUB classification.

Uses the committed split CSV; never regenerates splits or reads the official test
set. Import create_classification_loaders from this module in training programs.
"""

from pathlib import Path
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.dataset import CUBClassificationDataset
from src.data.transforms import build_classification_transform


def _seed_worker(worker_id):
    """Seed augmentation RNGs in each loader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_classification_loaders(
    batch_size=32,
    seed=42,
    num_workers=0,
    size=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    augment_training=True,
    split_csv=None,
    pin_memory=False,
):
    """Return (train_loader, val_loader, class_names).

    Call once during experiment setup. This seeds Python's random generator
    (used by horizontal flipping with num_workers=0). Loader generators seed
    sample ordering and worker startup independently from the model's RNG.
    Seed model initialization separately in the training program.

    Scratch defaults: RGB, resize/pad to 224, optional training flip, tensor
    conversion, and fixed 0.5 mean/std scaling. For transfer learning, review
    the selected weights' complete input requirements before changing settings.

    Keep num_workers=0 for initial debugging. Windows programs using workers
    must call this function under an if __name__ == "__main__": guard.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Use three channel means and three positive std values.")

    project_root = Path(__file__).resolve().parents[2]
    if split_csv is None:
        split_path = project_root / "data/splits/cub_classification_split.csv"
    else:
        split_path = Path(split_csv)
        if not split_path.is_absolute():
            split_path = project_root / split_path

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Shared split CSV not found: {split_path}. "
            "Get the committed CSV from the team; do not make a new split."
        )

    random.seed(seed)

    source = load_dataset("Multimodal-Fatima/CUB_train", split="train")
    class_names = list(source.features["label"].names)

    train_dataset = CUBClassificationDataset(
        source_dataset=source,
        split_csv=split_path,
        split="train",
        transform=build_classification_transform(
            training=augment_training, size=size, mean=mean, std=std,
        ),
    )
    val_dataset = CUBClassificationDataset(
        source_dataset=source,
        split_csv=split_path,
        split="validation",
        transform=build_classification_transform(
            training=False, size=size, mean=mean, std=std,
        ),
    )

    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    if train_indices & val_indices:
        raise ValueError("Training and validation contain overlapping rows.")
    if train_indices | val_indices != set(range(len(source))):
        raise ValueError("The partitions do not cover the original training set.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return train_loader, val_loader, class_names


def main():
    print("Loading shared classification data...", flush=True)
    train_loader, val_loader, class_names = create_classification_loaders()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Classes:", len(class_names))

    for name, loader in [("Training", train_loader), ("Validation", val_loader)]:
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert torch.isfinite(images).all().item()
        assert images.min().item() >= -1.00001
        assert images.max().item() <= 1.00001
        assert labels.shape == (images.shape[0],)
        assert labels.dtype == torch.int64
        assert ((labels >= 0) & (labels < len(class_names))).all().item()

        print(f"\n{name} batch:")
        print("Image shape:", tuple(images.shape))
        print("Label shape:", tuple(labels.shape))
        print("Label type:", labels.dtype)

    print("\nShared loaders checks passed.")


if __name__ == "__main__":
    main()


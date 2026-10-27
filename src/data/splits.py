
"""Create reproducible classification splits without exact-image leakage."""

import hashlib
from collections import defaultdict
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


def create_split_indices(dataset, validation_fraction=0.20, seed=42):
    """Return train/validation row indices, keeping exact duplicates together.

    Args:
        dataset: Original dataset with 'image' and integer 'label' columns.
        validation_fraction: Fraction of unique-image groups for validation.
        seed: Random seed controlling the group assignment.

    Returns:
        A tuple: (sorted training indices, sorted validation indices).
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    labels = list(dataset["label"])
    fingerprint_groups = defaultdict(list)

    # Build groups from image content, rather than filenames.
    for index in range(len(dataset)):
        try:
            image = dataset[index]["image"].convert("RGB")
            image.load()

            fingerprint = hashlib.sha256()
            fingerprint.update(
                f"{image.width}x{image.height}|RGB|".encode("ascii")
            )
            fingerprint.update(image.tobytes())

        except Exception as error:
            # Stop rather than silently leaving a record out of the split.
            raise RuntimeError(
                f"Could not fingerprint image at row {index}."
            ) from error

        fingerprint_groups[fingerprint.hexdigest()].append(index)

    groups = list(fingerprint_groups.values())
    group_labels = []

    for group in groups:
        unique_labels = {labels[index] for index in group}

        if len(unique_labels) != 1:
            raise ValueError(
                f"Identical images have conflicting labels at rows {group}."
            )

        group_labels.append(labels[group[0]])

    # Split groups while approximately preserving class proportions.
    train_groups, validation_groups = train_test_split(
        list(range(len(groups))),
        test_size=validation_fraction,
        random_state=seed,
        stratify=group_labels,
    )

    train_indices = sorted(
        index
        for group_id in train_groups
        for index in groups[group_id]
    )
    validation_indices = sorted(
        index
        for group_id in validation_groups
        for index in groups[group_id]
    )

    # Verify coverage and separation.
    train_set = set(train_indices)
    validation_set = set(validation_indices)

    if train_set & validation_set:
        raise RuntimeError("A record appears in both partitions.")

    if train_set | validation_set != set(range(len(dataset))):
        raise RuntimeError("The split does not cover the complete dataset.")

    for group in groups:
        if not (
            set(group).issubset(train_set)
            or set(group).issubset(validation_set)
        ):
            raise RuntimeError("An identical-image group crosses partitions.")

    all_labels = set(labels)
    if (
        {labels[index] for index in train_indices} != all_labels
        or {labels[index] for index in validation_indices} != all_labels
    ):
        raise RuntimeError("Some classes are missing from a partition.")

    return train_indices, validation_indices

if __name__ == "__main__":
    from datasets import load_dataset

    print("Loading the dataset...", flush=True)

    dataset = load_dataset(
        "Multimodal-Fatima/CUB_train",
        split="train",
    )

    print("Checking images and creating the split...", flush=True)

    train_indices, val_indices = create_split_indices(dataset)

    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Finished.")

    output_path = Path("data/splits/cub_classification_split.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_ids = list(dataset["id"])
    file_names = list(dataset["file_name"])
    labels = list(dataset["label"])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "row_index", "image_id", "file_name", "label", "split"
        ])

        for split_name, indices in [
            ("train", train_indices),
            ("validation", val_indices),
        ]:
            for index in indices:
                writer.writerow([
                    index,
                    image_ids[index],
                    file_names[index],
                    labels[index],
                    split_name,
                ])

    print("Saved split assignments to:", output_path.resolve())
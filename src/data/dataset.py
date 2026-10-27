"""Load classification examples using the saved image split."""

import csv
from pathlib import Path
import random
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.data.transforms import build_classification_transform
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
import textwrap



class CUBClassificationDataset(Dataset):
    def __init__(self, source_dataset, split_csv, split, transform):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        self.source_dataset = source_dataset
        self.transform = transform

        filenames = list(source_dataset["file_name"])
        labels = list(source_dataset["label"])

        if len(set(filenames)) != len(filenames):
            raise ValueError("Source dataset contains repeated filenames.")

        # Match by filename rather than assuming row order is unchanged.
        filename_to_index = {
            filename: index
            for index, filename in enumerate(filenames)
        }

        self.indices = []
        seen_filenames = set()

        with Path(split_csv).open(
            "r", newline="", encoding="utf-8"
        ) as file:
            for row in csv.DictReader(file):
                filename = row["file_name"]

                if filename in seen_filenames:
                    raise ValueError(
                        f"Repeated filename in split CSV: {filename}"
                    )
                seen_filenames.add(filename)

                if filename not in filename_to_index:
                    raise ValueError(
                        f"Split image missing from dataset: {filename}"
                    )

                if row["split"] not in {"train", "validation"}:
                    raise ValueError(
                        f"Unknown split value: {row['split']}"
                    )

                source_index = filename_to_index[filename]

                if int(row["label"]) != labels[source_index]:
                    raise ValueError(
                        f"Label mismatch for image: {filename}"
                    )

                if row["split"] == split:
                    self.indices.append(source_index)

        if seen_filenames != set(filenames):
            raise ValueError("Split CSV does not cover the source dataset.")

        if not self.indices:
            raise ValueError(f"No records found for split: {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        record = self.source_dataset[source_index]

        image = self.transform(record["image"])
        label = int(record["label"])

        return image, label
    
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parents[2]
    split_csv = (
        project_root / "data/splits/cub_classification_split.csv"
    )

    print("Loading source dataset...", flush=True)
    source = load_dataset(
        "Multimodal-Fatima/CUB_train",
        split="train",
    )

    train_dataset = CUBClassificationDataset(
        source_dataset=source,
        split_csv=split_csv,
        split="train",
        transform=build_classification_transform(training=True),
    )

    val_dataset = CUBClassificationDataset(
        source_dataset=source,
        split_csv=split_csv,
        split="validation",
        transform=build_classification_transform(training=False),
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    assert set(train_dataset.indices).isdisjoint(val_dataset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    for name, loader in [
        ("Training", train_loader),
        ("Validation", val_loader),
    ]:
        images, labels = next(iter(loader))

        assert images.shape == (32, 3, 224, 224)
        assert images.dtype == torch.float32
        assert torch.isfinite(images).all().item()
        assert labels.shape == (32,)
        assert labels.dtype == torch.int64
        assert ((labels >= 0) & (labels < 200)).all().item()

        print(f"\n{name} batch:")
        print("Image shape:", tuple(images.shape))
        print("Label shape:", tuple(labels.shape))
        print("Label type:", labels.dtype)
        class_names = source.features["label"].names
        preview_count = min(12, len(images))

        columns = 3
        rows = (preview_count + columns - 1) // columns
        cell_width, cell_height = 240, 275

        grid = Image.new(
            "RGB",
            (columns * cell_width, rows * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(grid)

        for i in range(preview_count):
            # Undo our fixed normalization for display only.
            display_tensor = (
                images[i].detach().cpu() * 0.5 + 0.5
            ).clamp(0, 1)

            picture = to_pil_image(display_tensor)
            species = class_names[labels[i].item()]

            x = (i % columns) * cell_width
            y = (i // columns) * cell_height

            grid.paste(picture, (x + 8, y + 5))
            draw.text(
                (x + 8, y + 233),
                textwrap.fill(species, width=32),
                fill="black",
            )

        preview_path = (
            project_root
            / "outputs/figures"
            / f"{name.lower()}_batch_preview.png"
        )
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(preview_path)

        print("Batch preview saved to:", preview_path)

    print("\nDataset and batch checks passed.")
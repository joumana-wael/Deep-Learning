"""Image transformations for the bird classification pipeline."""

from PIL import Image, ImageOps
import torch
import random
from torchvision import transforms as T
def build_classification_transform(
    training=False,
    size=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    """Build the image-to-tensor pipeline for classification."""

    steps = []

    if training:
        steps.append(RandomHorizontalFlip(p=0.5))

    steps.extend([
        ResizeAndPad(size=size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(steps)
class RandomHorizontalFlip:
    """Flip an image horizontally with probability p."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1.")

        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return ImageOps.mirror(image)

        return image
class ResizeAndPad:
    """Convert to RGB, resize proportionally, and pad to a square."""

    def __init__(self, size=224, fill=(127, 127, 127)):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer.")

        self.size = size
        self.fill = fill

    def __call__(self, image):
        image = image.convert("RGB")

        return ImageOps.pad(
            image,
            size=(self.size, self.size),
            method=Image.Resampling.BILINEAR,
            color=self.fill,
            centering=(0.5, 0.5),
        )
        
if __name__ == "__main__":
    from pathlib import Path

    from datasets import load_dataset
    from PIL import ImageDraw

    print("Loading dataset...", flush=True)
    dataset = load_dataset(
        "Multimodal-Fatima/CUB_train",
        split="train",
    )

    transform = ResizeAndPad(size=224)

    # Ordinary, grayscale, tall, and wide examples from our EDA.
    preview_indices = [0, 726, 3228, 5590]

    cell_width = 340
    row_height = 360

    preview = Image.new(
        "RGB",
        (cell_width * 2, row_height * len(preview_indices)),
        color="white",
    )
    draw = ImageDraw.Draw(preview)

    for row_number, index in enumerate(preview_indices):
        original = dataset[index]["image"]
        prepared = transform(original)

        assert prepared.mode == "RGB"
        assert prepared.size == (224, 224)

        y = row_number * row_height

        draw.text(
            (10, y + 10),
            f"Row {index}: original {original.size}, {original.mode}",
            fill="black",
        )
        draw.text(
            (cell_width + 10, y + 10),
            "Prepared: 224 x 224, RGB",
            fill="black",
        )

        # Resize only the display copy of the original to fit the preview.
        display_original = ImageOps.contain(
            original.convert("RGB"),
            (300, 300),
        )

        preview.paste(
            display_original,
            (
                (cell_width - display_original.width) // 2,
                y + 40 + (300 - display_original.height) // 2,
            ),
        )

        # Display the prepared output at its actual pixel dimensions.
        preview.paste(
            prepared,
            (
                cell_width + (cell_width - 224) // 2,
                y + 40 + (300 - 224) // 2,
            ),
        )

        print(
            f"Row {index}: "
            f"{original.size}, {original.mode} "
            f"-> {prepared.size}, {prepared.mode}"
        )

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "outputs/figures/resize_pad_preview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview.save(output_path)
    print("Preview saved to:", output_path)
        # Force a flip for this preview so we can inspect its effect.
    preview_flip = RandomHorizontalFlip(p=1.0)
    resize_pad = ResizeAndPad(size=224)

    comparison = Image.new(
        "RGB",
        (480, 260 * len(preview_indices)),
        color="white",
    )
    draw = ImageDraw.Draw(comparison)

    for row_number, index in enumerate(preview_indices):
        original = dataset[index]["image"].convert("RGB")

        normal = resize_pad(original)
        flipped = resize_pad(preview_flip(original))

        y = row_number * 260

        draw.text((8, y + 8), f"Row {index}: no flip", fill="black")
        draw.text((248, y + 8), "Horizontal flip", fill="black")

        comparison.paste(normal, (8, y + 28))
        comparison.paste(flipped, (248, y + 28))

    augmentation_path = (
        project_root / "outputs/figures/horizontal_flip_preview.png"
    )
    comparison.save(augmentation_path)

    print("Augmentation preview saved to:", augmentation_path)

    train_transform = build_classification_transform(training=True)
    val_transform = build_classification_transform(training=False)

    # Check ordinary, grayscale, portrait, and wide examples.
    for index in preview_indices:
        original = dataset[index]["image"]

        train_tensor = train_transform(original)
        val_tensor = val_transform(original)

        for name, tensor in [
            ("Training", train_tensor),
            ("Validation", val_tensor),
        ]:
            assert tensor.shape == (3, 224, 224)
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all().item()
            assert tensor.min().item() >= -1.00001
            assert tensor.max().item() <= 1.00001

            print(
                f"Row {index} | {name}: "
                f"shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, "
                f"range=[{tensor.min().item():.3f}, "
                f"{tensor.max().item():.3f}]"
            )

        # Validation processing must give identical results each time.
        assert torch.equal(val_tensor, val_transform(original))

    print("Image-to-tensor pipeline checks passed.") 
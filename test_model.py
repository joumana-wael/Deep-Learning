
import torch
from src.models.scratch_cnn import ScratchCNN


def main():
    model = ScratchCNN(num_classes=200)
    sample_images = torch.randn(32, 3, 224, 224)

    predictions = model(sample_images)

    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()

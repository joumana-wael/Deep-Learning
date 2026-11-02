import argparse
import torch
from PIL import Image

from src.models.scratch_cnn import ScratchCNN
from src.data.transforms import build_classification_transform


def predict_image(image_path, checkpoint_path):
    device = torch.device("cpu")

    # Load the trained model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = ScratchCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    transform = build_classification_transform(
        training=False,
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]

    print("\nPrediction Results")
    print("------------------")
    print(f"Predicted bird: {predicted_class}")
    print(f"Confidence: {confidence.item():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the bird species in an image."
    )

    parser.add_argument(
        "image",
        help="Path to the bird image",
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/scratch_cnn_v2/best_model.pth",
        help="Path to the trained model checkpoint",
    )

    args = parser.parse_args()

    try:
        predict_image(args.image, args.checkpoint)
    except FileNotFoundError:
        print("Error: Image or checkpoint file was not found.")
    except Exception as error:
        print(f"Error: {error}")
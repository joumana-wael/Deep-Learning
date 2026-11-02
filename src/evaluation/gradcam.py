import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.loaders import create_classification_loaders
from src.models.scratch_cnn import ScratchCNN


def main():
    # Use CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the validation data
    _, val_loader, class_names = create_classification_loaders(
        batch_size=32,
        augment_training=False,
    )

    # Load the trained model checkpoint
    checkpoint_path = "outputs/scratch_cnn_v2/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ScratchCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # The final convolutional feature layer
    target_layers = [model.features[14]]

    # Folder where Grad-CAM images will be saved
    output_dir = "outputs/evaluation/scratch_cnn/gradcam"
    os.makedirs(output_dir, exist_ok=True)

    number_of_images = 5
    images_saved = 0

    print("Generating Grad-CAM visualizations...")

    with GradCAM(model=model, target_layers=target_layers) as cam:

        for images, labels in tqdm(val_loader):

            for i in range(images.size(0)):

                if images_saved >= number_of_images:
                    break

                # Prepare one image
                input_tensor = images[i].unsqueeze(0).to(device)
                actual_label = labels[i].item()

                # Get the model prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_label = output.argmax(dim=1).item()
                    confidence = probabilities[0, predicted_label].item()

                # Convert the normalized image back to RGB values between 0 and 1
                rgb_image = images[i].cpu().permute(1, 2, 0).numpy()
                rgb_image = (rgb_image * 0.5) + 0.5
                rgb_image = np.clip(rgb_image, 0, 1).astype(np.float32)

                # Generate the Grad-CAM heatmap
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(predicted_label)],
                )[0]

                # Overlay the heatmap on the original image
                visualization = show_cam_on_image(
                    rgb_image,
                    grayscale_cam,
                    use_rgb=True,
                )

                # Save the visualization
                output_path = os.path.join(
                    output_dir,
                    f"sample_{images_saved + 1}_actual_{actual_label}_predicted_{predicted_label}.png",
                )

                plt.figure(figsize=(8, 6))
                plt.imshow(visualization)
                plt.title(
                    f"Actual: {class_names[actual_label]}\n"
                    f"Predicted: {class_names[predicted_label]} "
                    f"({confidence:.2%} confidence)"
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()

                images_saved += 1

            if images_saved >= number_of_images:
                break

    print("\nGrad-CAM generation completed!")
    print(f"Images saved: {images_saved}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
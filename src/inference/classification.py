"""Single-image classification using the project's models and validation transforms."""

import json
from pathlib import Path


def load_classifier(checkpoint_path, model_kind="auto", device="cpu", class_names_path=None):
    import torch

    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "best_model.pth"
    if not path.is_file():
        raise ValueError(f"Checkpoint not found: {path}")
    config_path = path.parent / "model_config.json"
    if model_kind == "auto":
        model_kind = "transfer" if config_path.is_file() else "scratch"
    if model_kind == "transfer":
        if class_names_path is not None:
            raise ValueError("Transfer class names come from model_config.json; omit --class-names.")
        if not config_path.is_file():
            raise ValueError("Transfer inference requires model_config.json beside the checkpoint.")
        from src.models.transfer_cnn import load_transfer_model
        model, metadata = load_transfer_model(path.parent, device=device, checkpoint_path=path)
        names = metadata["class_names"]
        num_classes = metadata["num_classes"]
        mean, std = metadata["mean"], metadata["std"]
    elif model_kind == "scratch":
        from src.models.scratch_cnn import ScratchCNN
        state = torch.load(path, map_location="cpu", weights_only=True)
        weights = state["model_state_dict"]
        num_classes = weights["classifier.3.weight"].shape[0]
        model = ScratchCNN(num_classes=num_classes)
        model.load_state_dict(weights)
        model = model.to(device).eval()
        names = state.get("class_names")
        if class_names_path is not None:
            with Path(class_names_path).open(encoding="utf-8") as file:
                names = json.load(file)
        mean, std = (0.5,) * 3, (0.5,) * 3
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    if names is not None and (not isinstance(names, list) or len(names) != num_classes or not all(isinstance(n, str) for n in names)):
        raise ValueError("Class names must be a JSON list of strings matching the checkpoint's class count and order.")
    return model, names, mean, std, model_kind


def predict(image_path, checkpoint_path, model_kind="auto", device="auto", top_k=5, class_names_path=None):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Image not found: {image_path}")
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / "best_model.pth"
    if not checkpoint_file.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_file}")
    import torch
    from PIL import Image
    from src.data.transforms import build_classification_transform

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    model, names, mean, std, kind = load_classifier(checkpoint_path, model_kind, device, class_names_path)
    transform = build_classification_transform(training=False, size=224, mean=mean, std=std)
    with Image.open(image_path) as image:
        batch = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model produced non-finite prediction probabilities.")
        scores, indices = probabilities.topk(min(top_k, probabilities.numel()))
    return {
        "task": "classification",
        "model": kind,
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "class_names_available": names is not None,
        "predictions": [
            {"class_id": index, "class_name": names[index] if names is not None else None, "probability": score}
            for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ],
    }

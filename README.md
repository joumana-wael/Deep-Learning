# Bird Species Classification — Milestone 1

CSE 488 Deep Learning project using the caption-enriched CUB-200-2011 dataset. Milestone 1 covers exploratory data analysis, preprocessing, a CNN trained from scratch, ResNet-50 transfer learning, tuning, evaluation and Grad-CAM. Classification uses images and species labels; captions are prepared separately for later tasks.

## 1. Clone and set up the environment

Run in an Anaconda/Miniconda terminal. If the repository is already cloned, open its root folder instead of cloning again.

```bash
git clone https://github.com/joumana-wael/Deep-Learning.git
cd Deep-Learning
conda create -n dl-project python=3.11 -y
conda activate dl-project
python -m pip install --upgrade pip
```

For a local CPU environment:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install datasets pillow numpy pandas scipy scikit-learn matplotlib seaborn tqdm mlflow pyyaml grad-cam jupyter ipykernel
```

For an NVIDIA GPU, use the matching installation command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) instead of the CPU command. In Colab or Kaggle, enable a GPU runtime and check the existing PyTorch installation before replacing it. Internet access is needed for the first dataset and pretrained-weight downloads.

 Check the selected environment:
```bash
python -c "import sys, torch, torchvision, datasets; print(sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

These are dependency installation instructions, not a pinned reproduction of the original GPU environments. After validating your environment, record its packages with `python -m pip freeze > environment-freeze.txt`. Exact results may vary across hardware and library versions.

**Run all commands below from the repository root.** CPU is suitable for checking the data pipeline and inference; full training is much faster on a GPU.

## 2. Repository layout

```text
configs/                 Training and transfer-learning configurations
data/splits/             Shared classification split CSV
data/processed/          Processed captions and review records
notebooks/               EDA and experiment notebooks
src/data/                Splits, captions, transforms, datasets and loaders
src/models/              Original/deeper scratch CNNs and transfer model
src/training/            Training, checkpoints, logging and comparisons
src/tuning/              Configuration generation and trial execution
src/evaluation/          Metrics, predictions, confusion matrix and Grad-CAM
outputs/                 Histories, metrics, figures and available checkpoints
```

## 3. Dataset and shared split

The loader downloads/caches [Multimodal-Fatima/CUB_train](https://huggingface.co/datasets/Multimodal-Fatima/CUB_train). It contains 5,994 images, 200 species and 10 caption entries per image. The [official test partition](https://huggingface.co/datasets/Multimodal-Fatima/CUB_test) contains 5,794 images and is reserved for final evaluation.

Use the committed `data/splits/cub_classification_split.csv`:

| Partition | Images |
|---|---:|
| Training | 4,794 |
| Validation | 1,200 |
| Official test, held separately | 5,794 |

The split uses seed 42, species stratification and image-hash groups. The two exact duplicate-image records stay together to prevent leakage. Do not regenerate a different split for each model. If the split CSV is missing, obtain the committed file from the team before training.

Preprocessing converts every image to RGB, resizes proportionally and pads to 224 × 224. Training uses random horizontal flipping with probability 0.5; validation is deterministic. Augmentation happens when loading an image, so it does not increase the stored training count. Scratch tensors use fixed mean/std `(0.5, 0.5, 0.5)` to map intensities approximately to [-1, 1]; these are not measured dataset statistics. Transfer training uses ImageNet normalization.

Captions are stored separately in `data/processed/captions_official_train.jsonl`. The current policy retains repeated and short captions, normalizes whitespace, and applies reviewed encoding repairs. Of 27 flagged encoding cases, 26 were repaired and one remains unresolved. Captions are not returned by the classification loader.

## 4. Verify and use the shared loaders

```bash
python -m src.data.loaders
```

Expected: 4,794 training images, 1,200 validation images, 200 classes, image tensors of shape `(32, 3, 224, 224)` and `torch.int64` labels of shape `(32,)`.

Teammates can import the loader directly:

```python
from src.data.loaders import create_classification_loaders


def main():
    train_loader, val_loader, class_names = create_classification_loaders(
        batch_size=32,
        seed=42,
        num_workers=0,
        augment_training=True,
    )
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape, len(class_names))
    # Use these loaders in your training and validation loops.


if __name__ == "__main__":
    main()
```

Seed model initialization separately; loader seeds do not initialize model weights. Use the transfer-training entry points below for their matching model preprocessing.

## 5. Train models

Use new output directories for new experiments. Fixed-setting scripts below reuse their listed destinations, so preserve previous results before rerunning them.

### Original scratch baseline

```bash
python -m src.training.train_scratch
```

Trains the 440,776-parameter `ScratchCNN` for 20 epochs with Adam, initial learning rate 0.001 and batch size 8. Outputs include `outputs/scratch_cnn_v2/training_history.csv` and `best_model.pth`. Despite the folder name, this is the original four-block CNN.

### Deeper scratch model

```bash
python -m src.training.train_scratch_v2
```

Trains `ScratchCNNV2`, with 2,430,984 parameters, for 30 epochs. It uses Adam, learning rate 0.001, batch size 8, weight decay 0.0001 and dropout 0.5. Outputs go to `outputs/scratch_cnn_v3/`.

### Configurable scratch training and resumption

```bash
python -m src.training.train --config configs/training.yaml --epochs 20 --batch-size 8 --learning-rate 0.001 --seed 42 --output-dir outputs/my_scratch_run
```

Resume an interrupted run from its latest checkpoint:

```bash
python -m src.training.train --config configs/training.yaml --epochs 20 --batch-size 8 --learning-rate 0.001 --seed 42 --resume-from outputs/my_scratch_run/last_checkpoint.pth --output-dir outputs/my_scratch_run
```

`--epochs 20` means a total target of 20 epochs, not 20 additional epochs. Keep the model and data settings consistent. This engine restores model, optimizer and scheduler state, but does not restore every random-generator state; exact uninterrupted-run equivalence is not guaranteed. Its checkpoint format should not be assumed interchangeable with every evaluation script.

### ResNet-50 transfer learning

```bash
python -m src.training.train_transfer --config configs/transfer_frozen.json --output-dir outputs/my_resnet50_frozen
python -m src.training.train_transfer --config configs/transfer_partial.json --output-dir outputs/my_resnet50_partial
python -m src.training.train_transfer --config configs/transfer_finetune.json --output-dir outputs/my_resnet50_finetune
```

Frozen training updates the classifier; partial fine-tuning updates layer4 and the classifier; full fine-tuning updates the entire network. The configurations use 15 epochs, batch size 32, AdamW and seed 42. Save the checkpoint, `model_config.json`, history and run summary together.

## 6. Hyperparameter search and experiment tracking

```bash
python -m src.tuning.run_experiments
```

This launches training: four grid trials followed by three random-search trials, five epochs each. The space is learning rate `{0.001, 0.0005}` × batch size `{8, 16}`; random search samples without replacement with seed 42. New outputs go to `outputs/grid_search/` and `outputs/random_search/`.

Historical four-trial grid results are in `outputs/phase2_tuning/`. A random-search implementation exists, but completed random-search results were not present in the reviewed artifacts. Running the command above creates new evidence; it does not recover earlier results.

The shared engine/trial runner logs to `sqlite:///mlflow.db`. From the same repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Preserve the database and artifact directories when leaving temporary GPU sessions. Not every standalone training script is guaranteed to use MLflow; retain CSV histories and configuration files as well.

## 7. Evaluate the original scratch checkpoint

First ensure `outputs/scratch_cnn_v2/best_model.pth` is an actual downloaded checkpoint, not a missing file or Git LFS pointer. Train the original baseline or obtain its checkpoint from the team if necessary.

```bash
python -m src.evaluation.evaluate --checkpoint outputs/scratch_cnn_v2/best_model.pth --output-dir outputs/evaluation/scratch_cnn --batch-size 32
```

The evaluation uses the shared validation partition and writes `metrics.json`. Generate the prediction CSVs and confusion matrix separately:

```bash
python -m src.evaluation.error_analysis --checkpoint outputs/scratch_cnn_v2/best_model.pth --output-dir outputs/evaluation/scratch_cnn
python -m src.evaluation.confusion_matrix --checkpoint outputs/scratch_cnn_v2/best_model.pth --output-dir outputs/evaluation/scratch_cnn
```

Inspect `all_predictions.csv`, `misclassified_samples.csv` and the confusion-matrix outputs. Top-1 measures the highest-scoring prediction; Top-5 checks whether the true class appears among the five highest scores; macro F1 weights species equally.

This evaluator reconstructs the original ScratchCNN. Do not pass a deeper CNN or ResNet checkpoint to it.

### Compare saved training histories

```bash
python -m src.training.compare_models --run scratch=outputs/scratch_cnn_v2 --run frozen=outputs/transfer_resnet50_frozen --run partial=outputs/transfer_resnet50_partial --run full=outputs/transfer_resnet50_finetune --skip-eval --output-dir outputs/comparison_histories
```

`--skip-eval` produces history-based comparisons without loading checkpoints. It cannot generate missing Top-5 or macro-F1 scores. Omit it only when the matching checkpoints and model configurations are available for the supported architectures.

## 8. Predict a bird species

```bash
python -m src.evaluation.predict "path/to/bird.jpg" --checkpoint outputs/scratch_cnn_v2/best_model.pth
```

Replace the image path with a real file. This CPU inference command loads the original ScratchCNN, applies deterministic validation transforms and prints a predicted species and softmax score. It expects the original training checkpoint with `model_state_dict` and `class_names`. Scores are not calibrated confidence estimates. A shared inference command for the deeper CNN and ResNet-50 is not implemented by this module.

## 9. Grad-CAM

```bash
python -m src.evaluation.gradcam
```

Requires the `grad-cam` package installed above. The current script uses CPU, the original checkpoint, and `model.features[14]` (the last block's ReLU activation) as its target layer. Maps explain the predicted class logit. Files are written to `outputs/evaluation/scratch_cnn/gradcam/`.

The script processes the first five validation images, all Black footed Albatross examples; the saved five cases are incorrect predictions. This is a limited qualitative subset, not a balanced survey. Additional correct cases in the report were generated separately during report verification and are not produced by this script unchanged. Highlighted background or bird regions are diagnostic observations, not proof of causal reasoning.



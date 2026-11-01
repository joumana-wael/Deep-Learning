# Transfer Learning (Member 3)

Pretrained ImageNet CNNs fine-tuned on the shared CUB-200 split, and compared
against the from-scratch baseline (`ScratchCNN`).

## Files

| File | Purpose |
|---|---|
| `src/models/transfer_cnn.py` | `TransferCNN` model, freezing strategies, save/load helpers |
| `src/training/train_transfer.py` | Training CLI (uses the shared `fit()` engine and MLflow tracker) |
| `src/training/compare_models.py` | Scratch-vs-transfer comparison table and plots |
| `configs/transfer_*.json` | Documented settings for each strategy |
| `test_transfer_model.py` | Shape, freezing, BatchNorm and overfit checks (offline) |

No existing file was changed.

## Model

`TransferCNN` uses the same layout as `ScratchCNN`:

- `model.features`: the pretrained convolutional encoder. For ResNet this is
  `conv1 → bn1 → relu → maxpool → layer1..layer4`.
- `model.classifier`: `AdaptiveAvgPool2d → Flatten → Dropout(0.4) → Linear(…, 200)`.
  This is a new head, initialized randomly.

The shared layout means that:
- the training engine, checkpointing and tuning code work unchanged;
- Grad-CAM can hook `model.features[-1]` on either model;
- `model.extract_features(x)` gives the pooled embedding (2048-d for ResNet-50),
  which the colorization and captioning tasks can reuse as their encoder.

Supported backbones: `resnet18`, `resnet50` (default, `IMAGENET1K_V2`) and
`efficientnet_b0`.

## Freezing and fine-tuning settings

| Strategy | What trains | Learning rates (AdamW) | Trainable params (ResNet-50) |
|---|---|---|---|
| `frozen` (feature extraction) | Head only | head 1e-3 | ~0.41 M of 23.9 M |
| `partial` | `layer4` + head | head 1e-3, backbone 1e-4 | ~15.4 M |
| `finetune` | Everything | head 1e-3, backbone 1e-4 | 23.9 M |

Shared settings: batch size 32, 15 epochs, weight decay 1e-4, dropout 0.4,
`ReduceLROnPlateau` (factor 0.5, patience 2), seed 42. Data uses the same split
and horizontal-flip augmentation as the scratch model.

### Design decisions

- **ImageNet normalization.** The pretrained weights expect inputs normalized
  with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. The shared
  loader already accepts `mean`/`std`, so the transfer runs pass these values.
  The scratch model keeps its 0.5/0.5 normalization.
- **Discriminative learning rates.** The new head starts from random weights
  and needs a large learning rate. The pretrained backbone is already good, so
  it gets a learning rate 10× smaller to avoid destroying its features
  (catastrophic forgetting).
- **Frozen BatchNorm stays frozen.** `model.train()` normally updates BatchNorm
  running statistics even when the weights have `requires_grad=False`.
  `TransferCNN.train()` keeps the BatchNorm layers of frozen parts in eval
  mode, so the frozen backbone really is unchanged. `test_transfer_model.py`
  checks this.
- **Head is param group 0.** The engine logs `optimizer.param_groups[0]["lr"]`
  as `learning_rate`, so in MLflow that value is the head learning rate.
  The backbone learning rate is logged separately as `backbone_lr`.

## Commands

```bash
# Checks (no download, CPU is fine)
python test_transfer_model.py

# Train the three strategies (GPU recommended: Colab/Kaggle T4)
python -m src.training.train_transfer --config configs/transfer_frozen.json
python -m src.training.train_transfer --config configs/transfer_partial.json
python -m src.training.train_transfer --config configs/transfer_finetune.json

# Any setting can be overridden from the command line
python -m src.training.train_transfer --config configs/transfer_finetune.json --backbone efficientnet_b0 --output-dir outputs/transfer_effb0_finetune

# Scratch vs transfer comparison
python -m src.training.compare_models --run scratch=outputs/scratch_cnn_v2 --run frozen=outputs/transfer_resnet50_frozen --run partial=outputs/transfer_resnet50_partial --run finetune=outputs/transfer_resnet50_finetune
```

Each training run writes the following to its `output_dir`:
`best_model.pth`, `training_history.csv`, `model_config.json` (backbone,
strategy, normalization and class names, used to rebuild the model with
`load_transfer_model`) and `run_summary.json` (best score, parameter counts and
training time). The same files are logged to the MLflow
`bird_classification` experiment.

The comparison writes `outputs/comparison/`:
`comparison_table.csv` and `.md`, `accuracy_curves.png`, `loss_curves.png`
and `best_accuracy.png`. Every checkpoint is re-evaluated on the same
validation split with top-1 accuracy, top-5 accuracy and macro F1. Each model
uses its own normalization, so the comparison is fair.

## Results

_Fill in after running on GPU. Paste `outputs/comparison/comparison_table.md` here._

Points for the analysis:
- The gap between frozen and fine-tuned shows how far ImageNet features are
  from fine-grained bird species.
- The gap between scratch and transfer shows what pretraining adds when there
  are only about 24 training images per class.
- Compare the `generalization_gap` (train − val accuracy) column for each model
  to discuss overfitting.

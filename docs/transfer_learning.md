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

Trained on a Google Colab T4 GPU with `notebooks/transfer_learning_colab.ipynb`
(ResNet-50, 15 epochs, seed 42). Every model was evaluated on the same
validation split (1,200 images) by `compare_models`.

| Model | Trainable params | Top-1 acc | Top-5 acc | Macro F1 | Train acc at best epoch | Gap (train − val) | Best epoch | Training time |
|---|---|---|---|---|---|---|---|---|
| ScratchCNN (baseline) | 440,776 | 9.42% | 26.25% | 7.76% | 10.91% | 1.49 | 19 / 20 | — |
| ResNet-50 frozen | 409,800 | 58.25% | 85.75% | 57.12% | 92.70% | 34.45 | 14 / 15 | 8.4 min |
| ResNet-50 partial | 15,374,536 | 72.25% | 92.00% | 71.37% | 100.00% | 27.75 | 15 / 15 | 8.9 min |
| **ResNet-50 finetune** | 23,917,832 | **74.33%** | **92.50%** | **73.98%** | 100.00% | 25.67 | 14 / 15 | 15.7 min |

Plots: `outputs/comparison/best_accuracy.png`, `accuracy_curves.png`,
`loss_curves.png`. The per-epoch history of each run is in
`outputs/transfer_resnet50_*/training_history.csv`. Trained checkpoints
(`best_model.pth`, about 95 MB each) are on the team Google Drive in
`Deep-Learning-transfer/`, not in git.

### Analysis

- **Pretraining is the biggest factor.** With only about 24 training images per
  class, the scratch CNN reaches 9.42%, while even the frozen ResNet-50 reaches
  58.25%. Both have about the same number of *trainable* parameters (0.44 M vs
  0.41 M). The difference comes from the 23.5 M frozen ImageNet weights, which
  already encode edges, textures and object parts that the scratch model cannot
  learn from so little data.
- **Adapting the backbone matters for fine-grained classes.** Training `layer4`
  adds 14 points (58.25% → 72.25%). ImageNet's top-level features are tuned to
  ImageNet's 1,000 classes, and bird species differ in small details (wing bars,
  eye rings, bill shape) that these features do not separate well.
- **Full fine-tuning adds only a little more** (72.25% → 74.33%), but it takes
  about twice as long. Most of the useful adaptation happens in the last stage;
  early layers (edges, colours) transfer almost unchanged.
- **Top-5 accuracy is above 92%** for partial and finetune: when the model is
  wrong, the true species is usually among its top guesses. These errors are
  mostly confusions between visually similar species.
- **Macro F1 is close to top-1 accuracy** for every model. Performance is spread
  evenly across species and is not driven by a few easy classes. CUB-200 has
  about the same number of images in every class, so accuracy and macro F1
  are expected to agree.
- **Overfitting.** Partial and finetune reach 100% training accuracy within a
  few epochs, while validation accuracy plateaus around 72–74% (gap of about
  26–28 points). The frozen model overfits too (34-point gap): its 2,048
  features are enough to memorize about 5,400 images with a linear head.
  For partial and finetune, validation loss stops improving after about epoch 3
  (around 1.15–1.20) while training accuracy goes to 100%. `ReduceLROnPlateau`
  then halved the learning rate two or three times, which added only small
  gains (finetune: 68.7% at epoch 3 → 74.3% at epoch 14). The frozen model's
  validation loss kept falling until the end, so it may gain a little from more
  epochs. Stronger regularization would help (see below).
- **Fine-tuning learns fast.** It reaches 50.8% validation accuracy after one
  epoch, compared with 9.42% for the scratch model after 20 epochs.

### Final choice

The **fine-tuned ResNet-50** (74.33% top-1, 73.98% macro F1) is the best Task 1
classifier and the recommended encoder for the colorization and captioning tasks.

### Limitations and next steps

- Augmentation is only a horizontal flip, the same as for the scratch model so
  that the comparison stays fair. Random resized crops and colour jitter should
  reduce the train/validation gap.
- Training ran for 15 epochs at 224×224. Published CUB results with ResNet-50
  (about 80–85%) usually use 448×448 inputs and longer schedules.
- Learning rates were not tuned for the transfer model; the tuning grid (Member
  4) could be reused with `head_lr`, `backbone_lr` and `strategy`.

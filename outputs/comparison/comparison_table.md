| run | epochs | best_epoch | best_val_accuracy | best_val_loss | train_accuracy_at_best | generalization_gap | model | trainable_params | minutes | val_top1 | val_top5 | val_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scratch | 20 | 19 | 9.42 | 4.50 | 10.91 | 1.49 | ScratchCNN | 440,776 | nan | 9.42 | 26.25 | 7.76 |
| frozen | 15 | 14 | 58.25 | 1.68 | 92.70 | 34.45 | resnet50 (frozen) | 409,800 | 8.40 | 58.25 | 85.75 | 57.12 |
| partial | 15 | 15 | 72.25 | 1.16 | 100.00 | 27.75 | resnet50 (partial) | 15,374,536 | 8.88 | 72.25 | 92.00 | 71.37 |
| finetune | 15 | 14 | 74.33 | 1.13 | 100.00 | 25.67 | resnet50 (finetune) | 23,917,832 | 15.73 | 74.33 | 92.50 | 73.98 |

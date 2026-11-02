
import torch
from src.models.scratch_cnn_v2 import ScratchCNNV2

model = ScratchCNNV2(num_classes=200)

x = torch.randn(8, 3, 224, 224)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

assert output.shape == (8, 200)

print("Model test passed!")

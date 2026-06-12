"""gelu_layernorm: LayerNorm only (MLIR has no GELU kernel, same as test)"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        return self.ln(x)

model = Model().eval()
input_shape = [4, 8]

"""residual_block: x + (x @ W(8x8) + bias(8)) -> (batch,8) — no ReLU for testing"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8) * 0.1)
        self.bias = nn.Parameter(torch.randn(8) * 0.1)

    def forward(self, x):
        return x + (x @ self.weight + self.bias)

model = Model().eval()
input_shape = [4, 8]

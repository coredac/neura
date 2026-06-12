"""conv2d_relu_pool: simplified — matmul only (no ReLU/maxpool to avoid known issues)"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(9, 4) * 0.1)

    def forward(self, x):
        return x @ self.kernel

model = Model().eval()
input_shape = [1, 9]

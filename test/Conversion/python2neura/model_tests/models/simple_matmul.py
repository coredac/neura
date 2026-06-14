"""simple_matmul: input(4x8) @ W(8x16) -> 4x16"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 16) * 0.1)

    def forward(self, x):
        return x @ self.weight

model = Model().eval()
input_shape = [4, 8]

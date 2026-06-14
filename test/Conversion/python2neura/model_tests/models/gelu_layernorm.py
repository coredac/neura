"""gelu_layernorm: LayerNorm + GELU activation"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.ln(x)
        x = torch.nn.functional.gelu(x, approximate='tanh')
        return self.fc(x)

model = Model().eval()
input_shape = [4, 8]

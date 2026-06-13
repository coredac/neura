"""transformer_attention: simplified attention with real softmax"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Parameter(torch.randn(8, 8) * 0.1)
        self.wk = nn.Parameter(torch.randn(8, 8) * 0.1)
        self.wv = nn.Parameter(torch.randn(8, 8) * 0.1)

    def forward(self, x):
        Q = x @ self.wq
        K = x @ self.wk
        V = x @ self.wv
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V

model = Model().eval()
input_shape = [4, 8]

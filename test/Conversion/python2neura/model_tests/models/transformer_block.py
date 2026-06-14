"""transformer_block: single transformer encoder block
Architecture:
  x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
  Attention: Q,K,V projections + scaled dot-product + softmax
  FFN: Linear(8->16) -> GELU -> Linear(16->8)
"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Self-attention projections
        self.wq = nn.Linear(8, 8, bias=False)
        self.wk = nn.Linear(8, 8, bias=False)
        self.wv = nn.Linear(8, 8, bias=False)
        self.attn_ln = nn.LayerNorm(8)
        # Feed-forward network
        self.ffn1 = nn.Linear(8, 16, bias=False)
        self.ffn2 = nn.Linear(16, 8, bias=False)
        self.ffn_ln = nn.LayerNorm(8)

    def forward(self, x):
        # --- Self-attention sub-layer ---
        Q = self.wq(x)   # [4, 8]
        K = self.wk(x)   # [4, 8]
        V = self.wv(x)   # [4, 8]
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn_out = attn @ V                   # [4, 8]
        x = self.attn_ln(x + attn_out)        # residual + layer norm

        # --- FFN sub-layer ---
        ffn = torch.nn.functional.gelu(self.ffn1(x), approximate='tanh')  # [4, 16]
        ffn = self.ffn2(ffn)                  # [4, 8]
        return self.ffn_ln(x + ffn)           # residual + layer norm

model = Model().eval()
input_shape = [4, 8]

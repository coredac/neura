"""two_layer_mlp: W2(16x4) @ relu(W1(8x16) @ x(2x8)) -> (2,4)"""
import torch
import torch.nn as nn

torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 4, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = Model().eval()
input_shape = [2, 8]

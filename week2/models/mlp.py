import torch
from torch import nn

class TrajMLP(nn.Module):
    def __init__(self, history_len=10, future_len=20, hidden_dim=128):
        super().__init__()
        input_dim = history_len * 2
        output_dim = future_len * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.future_len = future_len

    def forward(self, history):
        x = history.reshape(history.shape[0], -1)
        y = self.net(x)
        y = y.reshape(history.shape[0], self.future_len, 2)
        return y

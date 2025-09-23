import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1, output_dim=1 ):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

import torch
import torch.nn as nn
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )   

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        pooled_output = torch.sum(attn_weights * x, dim=1)  # (batch_size, input_dim)
        return pooled_output
import torch
import torch.nn as nn
class CNNHead(nn.Module):
    def __init__(self, input_dim=768, num_filters=128, filter_sizes=[3,4,5], output_dim=1, dropout=0.1):
        super(CNNHead, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, input_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, input_dim)
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len-filter_size+1), ...]*len(filter_sizes)
        pooled = [torch.max(c, dim=2)[0] for c in conved]  # [(batch_size, num_filters), ...]*len(filter_sizes)
        cat = self.dropout(torch.cat(pooled, dim=1))  # (batch_size, num_filters * len(filter_sizes))
        return self.fc(cat)  # (batch_size, output_dim)
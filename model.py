import torch
import torch.nn as nn

# class SentimentModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
#         super(SentimentModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#
#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         lstm_out = lstm_out.mean(dim=1)
#         output = self.fc(lstm_out)
#         return output

import torch
import torch.nn as nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_heads=4, num_layers=2):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(500, embed_dim)  # Vị trí tối đa 500 (max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, output_dim)  # Không cần nhân đôi như LSTM bidirectional

    def forward(self, x):
        # Embedding + Position Embedding
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        embedded = self.embedding(x) + self.position_embedding(position_ids)

        # Transformer Encoder
        transformer_out = self.transformer(embedded.permute(1, 0, 2))  # [seq_len, batch_size, embed_dim]
        transformer_out = transformer_out.mean(dim=0)  # Mean pooling

        # Fully Connected Layer
        output = self.fc(transformer_out)
        return output


import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, n_champs, hidden_dim=512, n_positions=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(n_champs, hidden_dim)
        self.positional_emb = nn.Embedding(n_positions, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x.shape : T, N
        pos = torch.arange(start=0, end=x.shape[0], dtype=torch.long, device=x.device, requires_grad=False)
        pos = pos.unsqueeze(1).expand_as(x)  # (T,) -> (T, N)

        x = self.token_emb(x) + self.positional_emb(pos)
        return self.dropout(x)


class WinnerTransformer(nn.Module):
    def __init__(self, inputblock, transformer, outputblock):
        super().__init__()
        self.inputblock = inputblock
        self.transformer = transformer
        self.outputblock = outputblock

    def forward(self, x_enc, x_dec):
        x_enc = self.inputblock(x_enc)
        x_dec = self.inputblock(x_dec)

        logits = self.transformer(x_enc, x_dec)
        return self.outputblock(logits[0, :, :])


class OutputBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
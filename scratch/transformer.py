from time import perf_counter as now

import numpy as np
import torch
import torch.nn as nn

hidden_dim = 4
n_head = 1
n_layers = 1

transformer = nn.Transformer(d_model=hidden_dim, nhead=n_head, num_decoder_layers=n_layers, num_encoder_layers=n_layers,
                             dim_feedforward=hidden_dim * 4, dropout=0.0, activation='gelu')

embedding_layer = nn.Embedding(20, hidden_dim)

enc = torch.tensor([[1, 2, 3, 4, 5]])
dec = torch.tensor([[0, 10, 11, 12, 13, 14]])

x_enc = embedding_layer.forward(enc.T)
x_dec = embedding_layer.forward(dec.T)

transformer.forward(x_enc, x_dec)


n = 100000
lines = torch.randint(200, size=(n, 5)).numpy()
lines = [list(line) for line in lines]

res = []
for line in lines:
    res.append([0] + line[5:])
res = torch.tensor(res)

res = []
for line in lines:
    line.insert(0, 0)
    res.append(line)
res = torch.tensor(res)

res = np.insert(np.array(lines), 0, np.array([[0]]), axis=1)
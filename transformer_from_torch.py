import sys
import pickle
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Adagrad, SGD
from torch.utils.data import DataLoader, TensorDataset

from winner_transformer_model import InputBlock, WinnerTransformer, OutputBlock

SPECIAL_TOK = 0

X = pd.read_pickle('./X.pkl')
y = pd.read_pickle("./y.pkl")

# X = pd.read_pickle('./X_large.pkl')
# y = pd.read_pickle("./y_large.pkl")
# X = pickle.load('./X_large.pkl')
# y = pickle.load('./y_large.pkl')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)


def enc_dec_split(data):
    enc, dec = [], []
    for line in data:
        enc.append([SPECIAL_TOK] + list(line[:5]))
        dec.append([SPECIAL_TOK] + list(line[:5]))
    return np.array(enc), np.array(dec)


X_train_enc, X_train_dec = enc_dec_split(X_train)
X_test_enc, X_test_dec = enc_dec_split(X_test)

# batch_size = len(X_train_enc) // 150
# batch_size = len(X_train_enc) // 70
batch_size = 5000

X_train_enc = torch.tensor(X_train_enc, device='cpu')
X_train_dec = torch.tensor(X_train_dec, device='cpu')
y_true = torch.tensor(y_train, device='cpu')

X_test_enc = torch.tensor(X_test_enc, device='cpu')
X_test_dec = torch.tensor(X_test_dec, device='cpu')
y_test = torch.tensor(np.array(y_test), device='cpu')

train_ds = TensorDataset(X_train_enc, X_train_dec, y_true)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(X_test_enc, X_test_dec, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)


hidden_dim = 256
n_head = 8
n_layers = 6

inputblock = InputBlock(n_champs=151, hidden_dim=hidden_dim, n_positions=6)
outputblock = OutputBlock(hidden_size=hidden_dim)

transformer = nn.Transformer(d_model=hidden_dim, nhead=n_head, num_decoder_layers=n_layers, num_encoder_layers=n_layers,
                             dim_feedforward=hidden_dim * 4, dropout=0.1, activation='gelu')

model = WinnerTransformer(inputblock, transformer, outputblock)

device = torch.device('cuda:0')
model.to(device)
print(model)

OPTIMIZER = 'AdamW'
optimizers = {'Adam': Adam, "AdamW": AdamW, "Adagrad": Adagrad, "SGD": SGD}
optimizer = optimizers[OPTIMIZER]
optimizer = optimizer(model.parameters())


# train codes from here
total_epochs = 0
n_epochs = 20
print_all = True
verbose = 1
progresses = {int(n_epochs // (100 / i)): i for i in range(1, 101, 1)}
t0, durations = perf_counter(), list()

for epoch in range(n_epochs):
    epoch_loss = 0
    epoch_n_truth = 0
    model.train()
    print('epoch', epoch)
    for iteration, ds in enumerate(train_dl):
        X_encoder, X_decoder, y_true_dec = ds
        X_encoder = X_encoder.T.to(device)
        X_decoder = X_decoder.T.to(device)
        y_true_dec = y_true_dec.to(device)

        log_y_pred = model(X_encoder, X_decoder)

        loss = F.nll_loss(input=log_y_pred, target=y_true_dec.squeeze(), reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        y_pred = log_y_pred.detach().cpu().numpy()
        n_truth = sum(np.argmax(y_pred, axis=-1) == y_true_dec.detach().cpu().numpy())
        epoch_n_truth += n_truth

        if verbose >= 2 or epoch == 0:
            acc = n_truth / len(y_pred)
            loss_s = round(loss.item(), 3)
            print(f"epoch-iter: {total_epochs}-{iteration}, loss: {loss_s}, acc: {acc:.3f}")
            # print('-' * 100)

        del y_true_dec
        del X_encoder
        del X_decoder

    loss_s = round(epoch_loss / (iteration + 1), 3)
    acc = epoch_n_truth / len(X_train)

    model.eval()
    epoch_test_n_truth = 0
    for iteration, ds in enumerate(test_dl):
        X_encoder, X_decoder, y_test_iter = ds
        X_encoder = X_encoder.T.to(device)
        X_decoder = X_decoder.T.to(device)
        y_test_iter = y_test_iter.numpy()

        log_y_pred = model(X_encoder, X_decoder)
        y_pred = log_y_pred.detach().cpu().numpy()
        n_truth = sum(np.argmax(y_pred, axis=-1) == y_test_iter)
        epoch_test_n_truth += n_truth

    test_acc = epoch_test_n_truth / len(y_test)
    print(f"epoch: {total_epochs}, loss: {loss_s}, acc: {acc:.3f}, test_acc: {test_acc:.3f}")
    total_epochs += 1


print('asdf')
print('hasdfla')
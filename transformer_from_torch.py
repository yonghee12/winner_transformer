import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.absolute())
sys.path.insert(0, ROOT)

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
from progress_timer import Timer

from winner_transformer_model import InputBlock, WinnerTransformer, OutputBlock
from functions import *

FROM_PICKLE = True
SPECIAL_TOK = 151

if not FROM_PICKLE:

    X = pd.read_pickle('./data/X_1027.pkl')
    y = pd.read_pickle("./data/y_1027.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, test_size=0.15)

    X_train_enc, X_train_dec = enc_dec_split(X_train, SPECIAL_TOK)
    X_test_enc, X_test_dec = enc_dec_split(X_test, SPECIAL_TOK)

    fnames = ['X_train_enc', 'X_train_dec', 'y_train', 'X_test_enc', 'X_test_dec', 'y_test']
    for fname in fnames:
        print(fname)
        path = os.path.join('data', fname + '.pkl')
        pd.to_pickle(eval(fname), path)

else:
    X_train_enc = pd.read_pickle(os.path.join(ROOT, 'data', 'X_train_enc.pkl'))
    X_train_dec = pd.read_pickle(os.path.join(ROOT, 'data', 'X_train_dec.pkl'))
    y_train = pd.read_pickle(os.path.join(ROOT, 'data', 'y_train.pkl'))
    X_test_enc = pd.read_pickle(os.path.join(ROOT, 'data', 'X_test_enc.pkl'))
    X_test_dec = pd.read_pickle(os.path.join(ROOT, 'data', 'X_test_dec.pkl'))
    y_test = pd.read_pickle(os.path.join(ROOT, 'data', 'y_test.pkl'))

# sample = 1000000
sample = len(X_train_enc)
test_size = 0.05
X_train_enc = X_train_enc[:sample]
X_train_dec = X_train_dec[:sample]
y_train = y_train[:sample]
X_test_enc = X_test_enc[:int(sample * test_size)]
X_test_dec = X_test_dec[:int(sample * test_size)]
y_test = y_test[:int(sample * test_size)]

batch_size = 15000
# batch_size = 1000
batch_size = 2000

X_train_enc = torch.tensor(X_train_enc, device='cpu')
X_train_dec = torch.tensor(X_train_dec, device='cpu')
y_true = torch.tensor(y_train, device='cpu')

X_test_enc = torch.tensor(X_test_enc, device='cpu')
X_test_dec = torch.tensor(X_test_dec, device='cpu')
y_test = torch.tensor(np.array(y_test), device='cpu')

train_ds = TensorDataset(X_train_enc, X_train_dec, y_true)
train_dl_shuffle = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
train_dl_noshuffle = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

test_ds = TensorDataset(X_test_enc, X_test_dec, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)


hidden_dim = 128
n_head = 4
n_layers = 2

print(f"d_model: {hidden_dim}, n_head: {n_head}, n_layers: {n_layers}")

inputblock = InputBlock(n_champs=151 + 1, hidden_dim=hidden_dim, n_positions=6, dropout=0.1)
outputblock = OutputBlock(hidden_size=hidden_dim, dropout=0.1)

transformer = nn.Transformer(d_model=hidden_dim, nhead=n_head, num_decoder_layers=n_layers, num_encoder_layers=n_layers,
                             dim_feedforward=hidden_dim * 4, dropout=0.1, activation='gelu')

model = WinnerTransformer(inputblock, transformer, outputblock, pooling_method='first')

device = torch.device('cuda:0')
model.to(device)
print(model)

OPTIMIZER = 'AdamW'
optimizers = {'Adam': Adam, "AdamW": AdamW, "Adagrad": Adagrad, "SGD": SGD}
optimizer = optimizers[OPTIMIZER]
optimizer = optimizer(model.parameters())

n_iters = len(train_dl_shuffle)
sampling_warmup_size = 0.1
sampling_warmup = int(n_iters * sampling_warmup_size)
sampling_warmup_epochs = 5

# train codes from here
total_epochs = 0
n_epochs = 50
print_all = True
verbose = 1
progresses = {int(n_epochs // (100 / i)): i for i in range(1, 101, 1)}
t0, durations = perf_counter(), list()

for epoch in range(n_epochs):
    epoch_loss = 0
    epoch_n_truth = 0
    model.train()
    print('epoch', epoch)
    if epoch >= sampling_warmup_epochs:
        train_dl = train_dl_shuffle
        do_test = True
    else:
        train_dl = train_dl_noshuffle
        do_test = False

    for iteration, ds in enumerate(train_dl):
        if epoch < sampling_warmup_epochs and iteration == sampling_warmup:
            print(f"sampling warmup steps: {iteration}")
            break

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
    acc = epoch_n_truth / len(X_train_enc)

    if not do_test:
        print(f"epoch: {total_epochs}, loss: {loss_s}")
        continue
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

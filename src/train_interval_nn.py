#!/usr/bin/env python3
"""Train a tiny 1-input, N-hidden-node, 1-output network to classify whether
a scalar x falls inside a target interval, and write out the learned
parameters at chosen snapshot epochs.

This is the "why do we need a hidden layer" example: a single neuron can only
draw one threshold on the number line, so it can never carve out a bounded
interval like (2, 4). Two hidden units can, one learns "x > lo", the other
learns "x < hi", and the output layer combines them.

Architecture:
    x -> Linear(1, hidden) -> activation -> h   (activation is a cli flag)
    h -> Linear(hidden, 1) -> logit -> sigmoid -> P(x in interval)

This script only trains and writes parameters. Use eval_interval_nn.py to
load those parameters back and run inference.

Reads the dataset from a file (see make_interval_dataset.py) rather than
generating it, so the same dataset can be reused across runs.

Requires: torch, numpy (pip install torch numpy --break-system-packages)

Usage:
    python make_interval_dataset.py --out out/interval.data.tsv
    python train_interval_nn.py --data out/interval.data.tsv --out_prefix out/interval
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True,
                    help='dataset file from make_interval_dataset.py (x, y columns)')
    p.add_argument('--hidden', type=int, default=2,
                    help='hidden units (use 1 to show the single-neuron ceiling)')
    p.add_argument('--activation', choices=['sigmoid', 'tanh', 'relu', 'identity'],
                    default='sigmoid', help='hidden layer activation function')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--snapshot_epochs', type=int, nargs='+',
                    default=[1, 5, 15, 30, 60, 120, 200, 400],
                    help='epochs at which to record parameters')
    p.add_argument('--snapshot_every', type=int,
                    help='record every N epochs instead of --snapshot_epochs '
                         '(e.g. --snapshot_every 10)')
    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


ACTIVATIONS = {
    'sigmoid':  torch.sigmoid,
    'tanh':     torch.tanh,
    'relu':     torch.relu,
    'identity': lambda z: z,
}


class TinyIntervalNet(nn.Module):
    def __init__(self, hidden=2, activation='sigmoid'):
        super().__init__()
        self.hidden = nn.Linear(1, hidden)
        self.output = nn.Linear(hidden, 1)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        h = self.activation(self.hidden(x))   # (N, hidden)
        logit = self.output(h).squeeze(-1)    # (N,)
        return logit, h


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    data = np.loadtxt(args.data, skiprows=1, dtype=np.float32)
    x_np, y_np = data[:, 0], data[:, 1]

    x = torch.tensor(x_np).unsqueeze(1)   # (N,1)
    y = torch.tensor(y_np)                # (N,)

    model = TinyIntervalNet(hidden=args.hidden, activation=args.activation)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    if args.snapshot_every:
        snapshot_set = set(range(args.snapshot_every, args.epochs + 1, args.snapshot_every))
    else:
        snapshot_set = set(args.snapshot_epochs)

    header = (['epoch', 'loss', 'acc']
              + [f'h{i+1}_w' for i in range(args.hidden)]
              + [f'h{i+1}_b' for i in range(args.hidden)]
              + [f'out_v{i+1}' for i in range(args.hidden)]
              + ['out_b'])
    rows = []

    def snapshot(epoch, loss_val, acc_val):
        with torch.no_grad():
            w = model.hidden.weight.detach().numpy().ravel()   # (hidden,)
            b = model.hidden.bias.detach().numpy().ravel()     # (hidden,)
            v = model.output.weight.detach().numpy().ravel()   # (hidden,)
            c = float(model.output.bias.detach().numpy()[0])

        row = [epoch, loss_val, acc_val] + list(w) + list(b) + list(v) + [c]
        rows.append(row)

        parts = [f'epoch {epoch:04d}', f'loss={loss_val:.4f}', f'acc={acc_val:.3f}']
        for i in range(args.hidden):
            parts.append(f'h{i+1}_w={w[i]:+.3f}')
            parts.append(f'h{i+1}_b={b[i]:+.3f}')
        for i in range(args.hidden):
            parts.append(f'out_v{i+1}={v[i]:+.3f}')
        parts.append(f'out_b={c:+.3f}')
        print(' '.join(parts))

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if epoch in snapshot_set or epoch == args.epochs:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == y).float().mean().item()
            snapshot(epoch, loss.item(), acc)

    params_file = args.out_prefix + '.params.tsv'
    with open(params_file, 'w') as f:
        f.write(f'# activation={args.activation}\n')
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(f'{v:.6f}' if isinstance(v, float) else str(v)
                               for v in row) + '\n')
    print(f'wrote {params_file}')


if __name__ == '__main__':
    main()

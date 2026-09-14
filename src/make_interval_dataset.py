#!/usr/bin/env python3
"""Generate the interval-classification dataset and save it to a file.

x is sampled uniformly, y is 1 if x falls inside (lo, hi), 0 otherwise. This
is a separate step from training so the same dataset file can be reused
across runs (different hidden sizes, different seeds for the optimizer,
etc.) instead of being silently regenerated inside the training script.

Usage:
    python make_interval_dataset.py --lo 2.0 --hi 4.0 --out out/interval.data.tsv
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lo', type=float, default=2.0, help='interval lower bound')
    p.add_argument('--hi', type=float, default=4.0, help='interval upper bound')
    p.add_argument('--x_min', type=float, default=0.0)
    p.add_argument('--x_max', type=float, default=6.0)
    p.add_argument('--n_points', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def make_dataset(lo, hi, x_min, x_max, n_points, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_points).astype(np.float32)
    y = ((x > lo) & (x < hi)).astype(np.float32)
    return x, y


def main():
    args = get_args()
    x, y = make_dataset(args.lo, args.hi, args.x_min, args.x_max,
                         args.n_points, args.seed)
    np.savetxt(args.out, np.column_stack([x, y]), fmt='%.6f',
               delimiter='\t', header='x\ty', comments='')
    pos_rate = y.mean()
    print(f'wrote {args.out}: {args.n_points} points, '
          f'{pos_rate:.1%} positive, baseline accuracy = {max(pos_rate, 1 - pos_rate):.3f}')


if __name__ == '__main__':
    main()

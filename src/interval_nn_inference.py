#!/usr/bin/env python3
"""Run inference for the tiny interval-classification network, given the
model parameters directly on the command line (as opposed to eval_interval_nn.py,
which loads them from a train_interval_nn.py params file).

Architecture:
    x -> Linear(1, hidden) -> activation -> h
    h -> Linear(hidden, 1) -> logit -> sigmoid -> P(x in interval)

Usage:
    python interval_nn_inference.py \
        --h_w 3.246 -4.748 --h_b -12.963 9.216 \
        --out_v -12.481 -12.286 --out_b 5.840 \
        --x 0.5 1.9 2.1 3.0 3.9 4.1 5.5

    # a different hidden activation, must match what the model was trained with
    python interval_nn_inference.py \
        --h_w 3.246 -4.748 --h_b -12.963 9.216 \
        --out_v -12.481 -12.286 --out_b 5.840 \
        --activation tanh --x 3.0
"""
import argparse
import math


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h_w', type=float, nargs='+', required=True,
                    help='hidden layer weights, one per hidden unit')
    p.add_argument('--h_b', type=float, nargs='+', required=True,
                    help='hidden layer biases, one per hidden unit')
    p.add_argument('--out_v', type=float, nargs='+', required=True,
                    help='output layer weights, one per hidden unit')
    p.add_argument('--out_b', type=float, required=True,
                    help='output layer bias')
    p.add_argument('--activation', choices=['sigmoid', 'tanh', 'relu', 'identity'],
                    default='sigmoid', help='hidden layer activation function')
    p.add_argument('--x', type=float, nargs='+', required=True,
                    help='x values to score')
    return p.parse_args()


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


ACTIVATIONS = {
    'sigmoid':  sigmoid,
    'tanh':     math.tanh,
    'relu':     lambda z: max(0.0, z),
    'identity': lambda z: z,
}


def predict(x, h_w, h_b, out_v, out_b, activation):
    act = ACTIVATIONS[activation]
    h = [act(h_w[i] * x + h_b[i]) for i in range(len(h_w))]
    logit = sum(out_v[i] * h[i] for i in range(len(h_w))) + out_b
    prob = sigmoid(logit)
    return h, prob


def main():
    args = get_args()

    hidden = len(args.h_w)
    if len(args.h_b) != hidden or len(args.out_v) != hidden:
        raise SystemExit(f'--h_w ({len(args.h_w)}), --h_b ({len(args.h_b)}), '
                          f'and --out_v ({len(args.out_v)}) must all have the same length')

    h_cols = [f'h{i+1}' for i in range(hidden)]
    widths = {c: max(6, len(c)) for c in h_cols}
    header = f'{"x":>6}  ' + '  '.join(f'{c:>{widths[c]}}' for c in h_cols) \
             + f'  {"P(x in interval)":>16}  prediction'
    print(f'activation={args.activation}')
    print(header)
    for x in args.x:
        h, prob = predict(x, args.h_w, args.h_b, args.out_v, args.out_b, args.activation)
        pred = 1 if prob >= 0.5 else 0
        row = f'{x:6.2f}  ' + '  '.join(f'{v:{widths[c]}.3f}' for v, c in zip(h, h_cols)) \
              + f'  {prob:16.3f}  {pred}'
        print(row)


if __name__ == '__main__':
    main()

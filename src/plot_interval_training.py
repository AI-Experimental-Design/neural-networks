#!/usr/bin/env python3
"""Plot loss/acc (or any other column) from a train_interval_nn.py params file.

Adapted from AI-Experimental-Design/convolution's src/plot_training_log.py,
which regex-parses "epoch N loss=.. acc=.. meanP(X)=.. meanP(O)=.." log
lines. This version reads the tab-separated params.tsv file
train_interval_nn.py writes instead (it may have a leading
"# activation=..." comment line before the header), and lets you name any
of its columns rather than assuming a fixed set.

Usage:
    python plot_interval_training.py -i out/interval.params.tsv -o out/interval_training.png

    # plot different columns, e.g. the hidden weights instead of loss/acc
    python plot_interval_training.py -i out/interval.params.tsv -o out/interval_weights.png \
        --columns h1_w,h2_w --colors purple,orange
"""
import argparse

import matplotlib
import matplotlib.pyplot
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['legend.numpoints'] = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--params_file', required=True,
                        help='params.tsv file written by train_interval_nn.py')
    parser.add_argument('-o', '--output', required=True,
                        help='Output image path')
    parser.add_argument('--columns', default='loss,acc',
                        help='Comma-separated column names to plot (must exist in the header)')
    parser.add_argument('--colors', default='blue,green,red,magenta,orange,purple',
                        help='Color CSV, one per column')
    parser.add_argument('--plot_width', type=float, default=4)
    parser.add_argument('--plot_height', type=float, default=3)
    parser.add_argument('--title')
    return parser.parse_args()


def parse_params(path):
    with open(path) as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        header = line.strip().split('\t')
        rows = [dict(zip(header, l.strip().split('\t'))) for l in f if l.strip()]
    return header, rows


def main():
    args = get_args()
    header, rows = parse_params(args.params_file)
    if not rows:
        raise SystemExit(f'no data rows found in {args.params_file}')

    epochs = [int(float(r['epoch'])) for r in rows]
    columns = args.columns.split(',')
    for c in columns:
        if c not in header:
            raise SystemExit(f'column {c!r} not in {args.params_file}, have: {", ".join(header)}')

    colors = args.colors.split(',')

    fig = matplotlib.pyplot.figure(figsize=(args.plot_width, args.plot_height), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2)

    plts = []
    for i, col in enumerate(columns):
        values = [float(r[col]) for r in rows]
        p, = ax.plot(epochs, values, '-', color=colors[i % len(colors)], linewidth=1)
        plts.append(p)

    ax.legend(plts, columns, frameon=False, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    ax.set_xlabel('epoch', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    matplotlib.pyplot.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()

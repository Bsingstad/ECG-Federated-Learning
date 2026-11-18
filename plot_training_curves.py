#!/usr/bin/env python

# Supplementary Figure S1: validation AUROC per training centre during training,
# from the training_history.csv files written by run_strategy.py.
#
#   python plot_training_curves.py -r results_revision --seed 1 -o figS1_training_curves.pdf

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("ciil_1epoch", "CIIL 1 epoch", "cycle"), ("ciil", "CIIL", "cycle"),
          ("fedavg_repeated", "Repeated FedAvg", "round"), ("fedsgd", "FedSGD", "round")]
CENTRES = ["st_petersburg", "ptb_diag", "chapman", "ningbo", "georgia", "china"]
LABELS = ["INCART", "PTB Diagnostic", "Chapman–Shaoxing", "Ningbo", "Georgia", "CPSC"]


def get_parser():
    parser = argparse.ArgumentParser(description="Plot validation AUROC per centre.")
    parser.add_argument("-r", "--results_folder", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-o", "--out_file", type=str, default="figS1_training_curves.pdf")
    return parser


def run(args):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (key, title, xcol) in zip(axes.ravel(), PANELS):
        path = os.path.join(args.results_folder, f"{key}_seed{args.seed}", "training_history.csv")
        if not os.path.exists(path):
            ax.set_title(f"{title} (missing)")
            continue
        hist = pd.read_csv(path)
        x = hist[xcol] + (1 if xcol == "cycle" else 0)
        for centre, label in zip(CENTRES, LABELS):
            ax.plot(x, hist[centre], label=label, linewidth=1.2)
        ax.plot(x, hist["mean_val"], color="black", linewidth=2, label="Mean")
        ax.axvline(x.iloc[hist["mean_val"].idxmax()], color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xcol.capitalize())
        ax.set_ylabel("Validation AUROC")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_file)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))

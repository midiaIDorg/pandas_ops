#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas_ops.io import read_df
from pandas_ops.printing import get_to_show

parser = argparse.ArgumentParser(
    description="Show a compact representation of a table."
)
parser.add_argument(
    "data_path",
    help="Data file containing data to be plotted",
    type=Path,
)
parser.add_argument(
    "-c",
    "--column",
    help="Column to use in the plot. May be an expression that'll be eval()'d. Prefix with \"data.\"",
    type=str,
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file path. Will display onscreen if omitted.",
    type=Path,
    default=None,
)
parser.add_argument(
    "--outliers",
    help="Proportion of outliers to eliminate from histogram.",
    type=float,
    default=None,
)

args = parser.parse_args()


if __name__ == "__main__":
    data = read_df(file_path=args.data_path)
    to_plot = eval(args.column)
    if args.outliers:
        import numpy as np

        to_plot = np.sort(to_plot)
        num_to_drop = int(args.outliers * len(to_plot))
        to_plot = to_plot[num_to_drop:-num_to_drop]

    from matplotlib import pyplot as plt

    plt.hist(to_plot, bins=1000)
    plt.title(args.column)

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

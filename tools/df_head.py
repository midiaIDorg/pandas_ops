#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Show a compact representation of a table."
)
parser.add_argument(
    "data_paths",
    help="Feather file(s) containing data to be plotted",
    type=Path,
    nargs="+",
)
parser.add_argument(
    "-c",
    "--columns",
    help="Columns to include in plot",
    nargs="*",
    type=str,
    default=None,
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file path. Will display onscreen if omitted.",
    type=Path,
    default=None,
)
parser.add_argument(
    "-n",
    help="Number of rows to show.",
    type=int,
    default=5,
)
parser.add_argument(
    "--csv",
    help="Run pandas table string representation instead of a csv.",
    action="store_true",
)
parser.add_argument(
    "--columns_only",
    help="Show only columns of submitted tables.",
    action="store_true",
)

args = parser.parse_args()


def get_to_show(data, row_count, pandas_style, columns_only):
    if columns_only:
        return data.columns
    if pandas_style:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", row_count)
        return data
    return data.head(row_count).to_csv(index=data.index.name == "")


if __name__ == "__main__":
    print()
    for data_path in args.data_paths:
        if not args.csv:
            print(data_path)
        data = pd.read_feather(
            path=data_path,
            columns=args.columns,
        )
        to_show = get_to_show(data, args.n, not args.csv, args.columns_only)
        print(to_show, file=sys.stdout)
        if not args.csv:
            print()

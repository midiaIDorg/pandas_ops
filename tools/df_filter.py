#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from pandas_ops.io import empty_df, save_df

parser = argparse.ArgumentParser(description="Open a table, filter rows, save table.")
parser.add_argument(
    "source",
    help="Path to the table.",
    type=Path,
)
parser.add_argument(
    "target",
    help="Path to the table.",
    type=Path,
)
parser.add_argument(
    "--filters",
    nargs="+",
    help="A set of strings defining the filtering conditions, using names of columns.",
)
args = parser.parse_args()


if __name__ == "__main__":
    df = pandas_ops.io.read_df(args.source)
    if len(df) > 0:
        for _filter in args.filters:
            df.query(_filter, inplace=True)
    if len(df) == 0:
        df = empty_df
    else:
        df = df.reset_index(drop=True) 
    save_df(df, args.target)

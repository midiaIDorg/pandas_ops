#!/usr/bin/env python3
import argparse
import sys
from pprint import pprint

import fastparquet
import pandas as pd
from pandas_ops.io import read_df
from tqdm import tqdm

DEBUG = False

parser = argparse.ArgumentParser(
    "Check if all of the tables are the same content-wise."
)
parser.add_argument(
    "tables",
    help="Paths files containing tables.",
    nargs="+",
)

args = parser.parse_args()


def main(args):
    tables = map(read_df, args.tables)
    table_prev = next(tables)
    same = True
    for i, table in enumerate(tables, start=1):
        same = same & table.equals(table_prev)
        if not same:
            print("Not the same")
            print(table.compare(table_prev))
            break
        table_prev = table
    if same:
        print("All the same.")
    sys.exit(not same)


if __name__ == "__main__":
    main(args)

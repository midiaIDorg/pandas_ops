#!/usr/bin/env python3
import argparse
from pprint import pprint

from tqdm import tqdm

import fastparquet
import pandas as pd
from pandas_ops.io import read_df

parser = argparse.ArgumentParser("Save data in a parquet format.")
parser.add_argument(
    "output",
    help="Path to store the output.",
)

parser.add_argument(
    "tables",
    help="Paths files containing tables.",
    nargs="+",
)

parser.add_argument(
    "--partition_on",
    help="Columns to hive-partition data on.",
    nargs="*",
    default=None,
)

parser.add_argument(
    "--verbose",
    help="Be more verbose.",
    action="store_true",
)

args = parser.parse_args()

if args.verbose:
    pprint(args.__dict__)

if __name__ == "__main__":
    table_paths = args.tables
    if args.verbose:
        table_paths = tqdm(table_paths, desc="Rewriting tables into parquet")
    for i, df in enumerate(map(read_df, table_paths)):
        if len(df) > 0:
            fastparquet.write(
                filename=args.output,
                data=df,
                partition_on=args.partition_on,
                append=i > 0,
                file_scheme="simple" if args.partition_on is None else "hive",
            )

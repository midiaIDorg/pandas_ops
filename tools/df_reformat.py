#!/usr/bin/env python3
import argparse

from pandas_ops.io import read_df, save_df

parser = argparse.ArgumentParser(description="Reformat tables.")
parser.add_argument(
    "in_table",
    help="Path with a table file.",
)
parser.add_argument(
    "out_table",
    help="Path to the requested table file.",
)
args = parser.parse_args()


def main(args):
    save_df(read_df(args.in_table), args.out_table)


if __name__ == "__main__":
    main(args)

#!/usr/bin/env python3
import argparse
import pathlib

from pandas_ops.io import read_df, save_df

parser = argparse.ArgumentParser("Cast data into a new format.")
parser.add_argument("input", help="Path to the input table.", type=pathlib.Path)
parser.add_argument("output", help="Path to the output table.", type=pathlib.Path)
parser.add_argument(
    "--verbose",
    help="Be more verbose.",
    action="store_true",
)
args = parser.parse_args()


if __name__ == "__main__":
    assert args.input.exists(), f"File not existings: {args.input}"
    if args.verbose:
        print(f"Translating:\n{args.input}\nto\n{args.output}")
    df = read_df(args.input)
    save_df(df, args.output)

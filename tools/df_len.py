#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import pandas_ops.io
from pandas_ops.printing import get_to_show

parser = argparse.ArgumentParser(description="Count")
parser.add_argument(
    "data_paths",
    help="File(s) containing tables to count lengths of.",
    type=Path,
    nargs="+",
)
parser.add_argument(
    "-o",
    "--output",
    help="Output table location.",
    type=Path,
)
parser.add_argument(
    "--preview",
    help="Run pandas table string representation instead of a csv.",
    action="store_true",
)
parser.add_argument(
    "--csv",
    help="Return a csv.",
    action="store_true",
)
args = parser.parse_args()
if __name__ == "__main__":
    df = pd.DataFrame(
        ((str(path), len(pandas_ops.io.read_df(path))) for path in args.data_paths),
        columns=["path", "count"],
    )
    df["cumulated_count"] = df["count"].cumsum()
    if args.preview or args.csv:
        to_show = get_to_show(df, 10 if args.preview else len(df), not args.csv, False)
        print(to_show, file=sys.stdout)
    else:
        pandas_ops.io.save_df(df, args.output)

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import pandas_ops.io

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
args = parser.parse_args()
if __name__ == "__main__":
    df = pd.DataFrame(
        ((str(path), len(pandas_ops.io.read_df(path))) for path in args.data_paths),
        columns=["path", "count"],
    )
    df["cumulated_count"] = df["count"].cumsum()
    pandas_ops.io.save_df(df, args.output)

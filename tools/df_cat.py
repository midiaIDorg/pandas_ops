#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Concatenate tables.")
parser.add_argument(
    "-o",
    "--output",
    help="Output file path.",
    type=Path,
)
parser.add_argument(
    "path",
    help="Feather file paths containing data to be plotted",
    type=Path,
    nargs="+",
)
args = parser.parse_args()


if __name__ == "__main__":
    ext_2_reader = {
        ".csv": pd.read_csv,
        ".feather": pd.read_feather,
    }

    def stream_of_dfs():
        for p in args.path:
            df = ext_2_reader[p.suffix](p)
            if len(df) > 0:
                yield df

    combined_df = pd.concat(stream_of_dfs(), ignore_index=True)
    combined_df.to_feather(args.output)

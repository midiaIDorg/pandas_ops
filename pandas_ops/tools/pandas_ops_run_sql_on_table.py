#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import tomllib
from pandas_ops.io import read_df

parser = argparse.ArgumentParser(
    description="Summarize a table.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("source", help="Path to the table to summarize.", type=Path)
parser.add_argument("config", help="Path to the table to summarize.", type=Path)
parser.add_argument("target", help="Path to the summary table.", type=Path)
args = parser.parse_args()


def main(config_path: Path, source: Path, target: Path):
    with open(config_path, "rb") as f:
        config: dict = tomllib.load(f)

    duckcon = duckdb.connect()
    if source.suffix == ".startrek":
        df = read_df(source)
        source = Path("df")

    query = config["sql"].format(source=source, target=target)
    duckcon.execute(query)


if __name__ == "__main__":
    main(**args.__dict__)

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
from pandas_ops.io import read_df


class args:  # quick mock
    source = "partial/G8027/MS1@tims@1fd37e91592@default/clusterStats@fast@default/clusterStats.parquet"
    source = "partial/G8027/G8045/MS1@tims@1fd37e91592@default@fast@default@MS2@tims@1fd37e91592@default@fast@default/matcher@prtree@narrow/rough.startrek"
    source = "test.csv"
    target = "/tmp/testing_stats_table.json"
    stats = ["min", "max", "size"]
    full_drop = True


parser = argparse.ArgumentParser(
    description="Summarize a table.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("source", help="Path to the table to summarize.", type=Path)
parser.add_argument(
    "--target", help="A json file with basic statistics.", type=Path, default=None
)
parser.add_argument(
    "--stats",
    nargs="+",
    default=["min", "max", "size"],
    help="Statisctics to calculate.",
)
parser.add_argument(
    "--full_drop", help="Fully drop a table into a json.", action="store_true"
)

args = parser.parse_args()


if __name__ == "__main__":
    res = {}
    df = read_df(args.source)
    if args.full_drop:
        output = df.to_dict(orient="records")
    else:
        output = df.aggregate(args.stats).to_dict()
    if args.target is None:
        print(json.dumps(output, indent=5))
    else:
        with open(args.target, "w") as f:
            json.dump(output, f, indent=5)

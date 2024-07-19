#!/usr/bin/env python3
import argparse
import pathlib
import re
from pprint import pprint

import duckdb
import pandas as pd
import tomllib
from pandas_ops.io import read_df, save_df2

parser = argparse.ArgumentParser(
    "Combine multiple table files into one and add in meta info by regex-based parsing the submitted paths.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "config",
    help="How to parse submitted paths into a table.",
    default="",
    type=str,
)
parser.add_argument("output", help="Path to the output table.", type=pathlib.Path)
parser.add_argument(
    "inputs", help="Path to the input table.", type=pathlib.Path, nargs="+"
)

# parser.add_argument(
#     "--verbose",
#     help="Be more verbose.",
#     action="store_true",
# )
# parser.add_argument(
#     "--paths_regex",
#     help="How to parse submitted paths into a table.",
#     default="",
#     type=str,
# )

# parser.add_argument(
#     "--howtostack",
#     help="How to stack the tables?",
#     choices=["vertical", "horizontal"],
#     default="vertical",
# )
# parser.add_argument(
#     "--partition_cols",
#     nargs="+",
#     help="Which columns should be used to split the parquet file?",
#     type=str,
# )

args = parser.parse_args().__dict__

if __name__ == "__main__":
    dfs = []
    duckcon = duckdb.connect()
    with open(args["config"], "rb") as f:
        config = tomllib.load(f)
        _verbose = config.get("verbose", False)
        if _verbose:
            pprint(config)

    path_pattern = re.compile(config["paths_regex"])

    for path in args["inputs"]:
        if _verbose:
            print(path)
        if not path.exists():
            if _verbose:
                print(f"Missing `{path}`")
        else:
            df = read_df(path)
            if "datawise_filter_sql" in config:
                df = duckcon.query(config["datawise_filter_sql"]).df()
            match = path_pattern.search(str(path))
            if match:
                meta = pd.DataFrame([match.groupdict()] * len(df))
                df = pd.concat([meta, df], axis=1)
            dfs.append(df)

    if _verbose:
        pprint(dfs)
    if len(dfs) > 0:
        out = pd.concat(
            dfs,
            ignore_index=True,
        )
        save_df2(
            out,
            args["output"],
            index=False,
            **config.get("save_kwargs", {}),
        )

# TODO: test it under csvs using different paths.

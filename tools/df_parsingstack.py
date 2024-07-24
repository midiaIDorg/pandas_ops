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
parser.add_argument("output", help="Path to the output table.", type=pathlib.Path)
parser.add_argument(
    "inputs", help="Path to the input table.", type=pathlib.Path, nargs="+"
)

parser.add_argument(
    "--verbose",
    help="Be more verbose.",
    action="store_true",
)
parser.add_argument(
    "--paths_regex",
    help="How to parse submitted paths into a table.",
    default="",
    type=str,
)
parser.add_argument(
    "--howtostack",
    help="How to stack the tables?",
    choices=["vertical", "horizontal"],
    default="vertical",
)
parser.add_argument(
    "--partition_cols",
    nargs="+",
    help="Which columns should be used to split the parquet file?",
    type=str,
    default=None,
)
parser.add_argument(
    "--sql",
    help="How to parse submitted paths into a table.",
    default={},
    type=str,
)
parser.add_argument(
    "--name_of_table_index",
    help="By default, the column name used to distinguish passed in tables in case of their content being the same or difficult to tell apart. Pass in empty string "
    " not to append that column (saving some space for larger datasets, which we do not recommend).",
    default="table_id",
    type=str,
)


args = parser.parse_args().__dict__

if __name__ == "__main__":
    dfs = []
    _verbose = args["verbose"]

    df_filter = lambda df: df
    if args["sql"]:
        duckcon = duckdb.connect()
        with open(args["sql"], "rb") as f:
            config = tomllib.load(f)
            if _verbose:
                pprint(config)
        if "sql" in config:
            df_filter = lambda df: duckdb.query(config["sql"]).df()

    path_pattern = re.compile(args["paths_regex"])

    assert not args["name_of_table_index"] in set(
        path_pattern.groupindex
    ), f"It seems that the provided name of table index, `{args['name_of_table_index']}`, coincides with a name of one of the named patterns in the provided regular expression used to parse the paths of tables,\n`{args['paths_regex']}`\nMake adjustements."

    for table_id, path in enumerate(args["inputs"]):
        if _verbose:
            print(path)
        if not path.exists():
            if _verbose:
                print(f"Missing `{path}`")
        else:
            df = read_df(path)
            df = df_filter(df)
            meta = {}
            if len(args["name_of_table_index"]) > 0:
                meta[args["name_of_table_index"]] = table_id
            match = path_pattern.search(str(path))
            if match:
                meta.update(match.groupdict())
            if len(meta) > 0:
                meta = pd.DataFrame([meta] * len(df))
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
            partition_cols=args["partition_cols"],
        )

# TODO: test it under csvs using different paths.

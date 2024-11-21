import json
from pathlib import Path
from warnings import warn

import click

import duckdb
import numpy as np
import pandas as pd
import tomllib
from pandas_ops.io import read_df, save_df
from pandas_ops.parsers.misc import parse_key_equal_value

output_path = "/tmp/combined_cluster_stats.parquet"
input_paths = (
    Path("tmp/clusters/tims/reformated/49/cluster_stats.parquet"),
    Path("tmp/clusters/tims/reformated/34/additional_cluster_stats.parquet"),
)


@click.command(context_settings={"show_default": True})
@click.argument("output_path", type=Path, nargs=1)
@click.argument("input_paths", type=Path, nargs=-1)
def df_concat(output_path: Path, input_paths: list[Path]) -> None:
    """
    Concatenate tables.

    Arguments:\n
        output_path (pathlib.Path): Where to store outputs.\n
        input_path (list[pathlib.Path]): Which tables to concatenate.
    """
    assert len(input_paths) > 1, "Makes no sense to merge 1 table."
    input_paths = list(input_paths)
    first_path = input_paths.pop()
    out_df = read_df(first_path)
    for in_path in input_paths:
        in_df = read_df(in_path)
        for col in in_df:
            if col not in out_df:
                out_df[col] = in_df[col]
            else:
                assert np.all(
                    out_df[col] == in_df[col]
                ), f"Column `{col}` takes different values in `{first_path}` and `{in_path}`."
    save_df(out_df, output_path)


@click.command(context_settings={"show_default": True})
@click.argument("source_path", type=Path)
@click.argument("config_path_or_sql_str", type=str)
@click.argument("target_path", type=Path)
def apply_sql(source_path: Path, config_path_or_sql_str: str, target_path: Path):
    with open(config_path_or_sql_str, "rb") as f:
        config: dict = tomllib.load(f)

    duckcon = duckdb.connect()
    if source_path.suffix == ".startrek":
        df = read_df(source_path)
        source = Path("df")

    query = config["sql"].format(source=source_path, target=target_path)
    duckcon.execute(query)

import json
from pathlib import Path

import click

import duckdb
import pandas as pd
import tomllib
from pandas_ops.io import read_df, save_df
from pandas_ops.parsers.misc import parse_key_equal_value


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
    in_dfs = (read_df(in_path) for in_path in input_paths)
    out_df = pd.concat(in_dfs, axis=1)
    save_df(out_df, output_path)


@click.command(context_settings={"show_default": True})
@click.argument("source_path", type=Path)
@click.argument("config_path", type=Path)
@click.argument("target_path", type=Path)
def apply_sql(source_path: Path, config_path: Path, target_path: Path):
    with open(config_path, "rb") as f:
        config: dict = tomllib.load(f)

    duckcon = duckdb.connect()
    if source_path.suffix == ".startrek":
        df = read_df(source_path)
        source = Path("df")

    query = config["sql"].format(source=source_path, target=target_path)
    duckcon.execute(query)

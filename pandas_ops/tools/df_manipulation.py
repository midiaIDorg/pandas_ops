import json
import sys

from pathlib import Path
from pprint import pprint
from warnings import warn

import click

import duckdb
import numpy as np
import pandas as pd
import tomllib

from pandas_ops.io import read_df
from pandas_ops.io import save_df
from pandas_ops.parsers.misc import parse_key_equal_value

duckdb_nonnative_formats = (".startrek",)
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
@click.option("--verbose", is_flag=True, help="Flush me with text.")
def apply_sql(
    source_path: Path,
    config_path_or_sql_str: str,
    target_path: Path,
    verbose: bool = False,
):
    with open(config_path_or_sql_str, "rb") as f:
        config: dict = tomllib.load(f)

    duckcon = duckdb.connect()
    if source_path.suffix == ".startrek":
        df = read_df(source_path)
        source_path = "df"

    query = config["sql"].format(source=source_path, target=target_path)
    if verbose:
        pprint(query)
    duckcon.execute(query)


@click.command(context_settings={"show_default": True})
@click.argument("config_path_or_sql_str", type=str)
@click.option(
    "--param",
    "-p",
    type=(str, str),
    multiple=True,
    help="Pass in name of the parameter and value tuples.",
)
@click.option("--source", default=None, help="Source table.")
@click.option("--target", default=None, help="Target table.")
@click.option("--verbose", is_flag=True, help="Be more verbose.")
def run_general_sql(
    config_path_or_sql_str: str,
    param: tuple[tuple[str, str], ...],
    source: str | None = None,
    target: str | None = None,
    verbose: bool = False,
) -> None:
    """Run a general sql.

    Arguments:

        config_path_or_sql_str: A path to the config. If not found, literally call the passed in string as sql.\n
        param: A tuple of tuples of form (<parameter name>,<parameter value>).\n
        source: An (optional) source of the table. Overrides one passed in as `-p source ...`.\n
        target: An (optional) target for the sql result. Overrides one passed in as `-p target ...`.\n
    """
    name_to_param = dict(param)

    if verbose:
        print(name_to_param)

    try:
        with open(config_path_or_sql_str, "rb") as f:
            config: dict = tomllib.load(f)
        sql = config["sql"]
    except FileNotFoundError:
        sql = config_path_or_sql_str

    if source is not None:
        name_to_param["source"] = source
    if target is not None:
        name_to_param["target"] = target

    if "source" in name_to_param:
        source = Path(name_to_param["source"])
        if source.suffix in duckdb_nonnative_formats:
            source_table = read_df(source)
            name_to_param["source"] = "source_table"

    formatted_sql = sql.format(**name_to_param)
    if verbose:
        print(formatted_sql)

    duckcon = duckdb.connect()

    if "target" in name_to_param:  # only to write to .startrek.
        target_path = Path(name_to_param["target"])
        df = duckcon.query(formatted_sql).df()
        save_df(df, target_path)
    else:
        duckcon.query(formatted_sql)

    if verbose:
        print("Done! Go play openrct2 or fheroes2.")

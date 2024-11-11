from pathlib import Path

import click

import pandas as pd
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

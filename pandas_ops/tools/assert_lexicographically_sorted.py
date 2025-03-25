import click

from pandas_ops.io import read_df
from pandas_ops.sortedness import is_sorted_lexicographically
from pathlib import Path


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=Path)
@click.argument("columns", type=Path, nargs=-1)
@click.option("--strictly", is_flag=True)
def assert_lexicographically_sorted(
    input_path: Path, columns, strictly: bool = False
) -> None:
    data = read_df(input_path, columns=columns)
    assert is_sorted_lexicographically(strictly, *[data[col] for col in columns])

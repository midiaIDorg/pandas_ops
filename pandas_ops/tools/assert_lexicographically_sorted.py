import click

from pandas_ops.io import read_df
from pandas_ops.sortedness import is_sorted_lexicographically
from pathlib import Path


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=Path)
@click.argument("columns", nargs=-1)
@click.option("--strictly", is_flag=True)
def assert_lexicographically_sorted(
    input_path: Path,
    columns: list[str],
    strictly: bool = False,
) -> None:
    """Check a table is lexicographically sorted w.r.t. the provided columns.\n

    Arguments:\n
        input_path (Path): Path to a table.
        columns (list[str]): A sequence of strings describing column names.
    """
    if "None.d" in str(input_path):
        print("None.d is trivially lexicographically sorted.")
        return None

    data = read_df(input_path, columns=columns)
    assert is_sorted_lexicographically(
        *[data[col] for col in columns], strictly=strictly
    ), f"`{input_path}` is not strictly lexicographically sorted by {columns}."
    print(f"`{input_path}` is strictly lexicographically sorted by {columns}.")

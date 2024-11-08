import pathlib
from collections import defaultdict

import click

import h5py
import pandas as pd
from mmapped_df import open_new_dataset_dct
from pandas_ops.io import read_df, save_df
from pandas_ops.parsers.misc import parse_key_equal_value


# %load_ext autoreload
# %autoreload 2
# in_hdf = pathlib.Path(
#     "/home/matteo/Projects/midia/pipelines/devel/midia_pipe/tmp/clusters/tims/1fd37e91592/precursor/25/clusters.hdf"
# )
# out_startrek = pathlib.Path("/tmp/clusters.startrek")
# root = "raw/data"
# hdf = h5py.File(in_hdf, mode="r")
def hdf_to_startrek(
    in_hdf: pathlib.Path,
    out_startrek: pathlib.Path,
    root: str = "/",
    verbose: bool = False,
) -> None:
    """
    Translate an HDF-stored table into .startrek format column after column.
    """
    with h5py.File(in_hdf, mode="r") as hdf:
        rootgroup = hdf[root]
        scheme = {}

        prev_size = None
        prev_column_name = ""
        for column_name in rootgroup:
            hdf_array = rootgroup[column_name]
            scheme[column_name] = pd.Series(dtype=hdf_array.dtype)
            size = len(hdf_array)
            if prev_size is not None:
                assert (
                    prev_size == size
                ), f"Differently shaped columns: `{prev_column_name}` has `{prev_size}` entries and `{column_name}` has `{size}`."
            prev_column_name = column_name
            prev_size = size

        df = open_new_dataset_dct(out_startrek, scheme, size)
        for column_name in rootgroup:
            if verbose:
                print(f"cp {root}/{column_name} {out_startrek}/{column_name}")
            df[column_name] = rootgroup[column_name][:]


def trivial_translator(input: pathlib.Path, output: pathlib.Path) -> None:
    """Translate a supported input into output using full RAM copy as intermediary."""
    save_df(read_df(input), output)


_translators: defaultdict = defaultdict(lambda: trivial_translator)
_translators[(".hdf", ".startrek")] = hdf_to_startrek


@click.command(context_settings={"show_default": True})
@click.argument("source", type=pathlib.Path)
@click.argument("target", type=pathlib.Path)
@click.option(
    "--kwarg",
    multiple=True,
    help="Dynamic key-value pairs in key=value format.",
    type=parse_key_equal_value,
)
def reformat_table(source: pathlib.Path, target: pathlib.Path, kwarg=()):
    _translators[(source.suffix, target.suffix)](source, target, **dict(kwarg))

import pathlib
import shutil

from collections import defaultdict

import click

from mmappet import DatasetWriter
from mmappet import open_new_dataset_dct
from tqdm import tqdm

import h5py
import json
import pandas as pd
import toml

from opentimspy import OpenTIMS
from pandas_ops.io import read_df
from pandas_ops.io import save_df
from pandas_ops.parsers.misc import parse_key_equal_value


# %load_ext autoreload
# %autoreload 2
# in_hdf = pathlib.Path(
#     "/home/matteo/Projects/midia/pipelines/devel/midia_pipe/tmp/clusters/tims/1fd37e91592/precursor/25/clusters.hdf"
# )
# out_startrek = pathlib.Path("~/clusters.startrek")
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
            df[column_name][:] = rootgroup[column_name][:]


# in_tdf = pathlib.Path(
#     "/home/matteo/Projects/midia/pipelines/devel/midia_pipe/spectra/G8045.d"
# )
# out_startrek = pathlib.Path("~/G8045.startrek")
def tdf_to_startrek(in_tdf, out_startrek, progressbar: str = ""):
    with OpenTIMS(in_tdf) as OT, DatasetWriter(out_startrek) as DW:
        frames = OT
        if progressbar:
            frames = tqdm(OT, total=len(OT.frames["Id"]), desc=progressbar)
        for frame in frames:
            df = pd.DataFrame(frame, copy=False)
            DW.append_df(df)


def trivial_translator(
    input: pathlib.Path,
    output: pathlib.Path,
    **kwargs,
) -> None:
    """Translate a supported input into output using full RAM copy as intermediary."""
    save_df(read_df(input, **kwargs), output)


_translators: defaultdict = defaultdict(lambda: trivial_translator)
_translators[(".hdf", ".startrek")] = hdf_to_startrek
_translators[(".d", ".startrek")] = tdf_to_startrek


@click.command(context_settings={"show_default": True})
@click.argument("source", type=pathlib.Path)
@click.argument("target", type=pathlib.Path)
@click.option("--force", help="Remove target if it exists.", is_flag=True)
@click.option(
    "--kwarg",
    multiple=True,
    help="Dynamic key-value pairs in key=value format.",
    type=parse_key_equal_value,
)
def reformat_table(
    source: pathlib.Path,
    target: pathlib.Path,
    force: bool = False,
    kwarg=(),
):
    if force:
        shutil.rmtree(target, ignore_errors=True)
    _translators[(source.suffix, target.suffix)](source, target, **dict(kwarg))


@click.command(context_settings={"show_default": True})
@click.argument("source", type=pathlib.Path)
@click.argument("target", type=pathlib.Path)
def json2toml(source: pathlib.Path, target: pathlib.Path):
    """Turn a json file to a toml file."""
    with open(source, "r") as jsonfile, open(target, "w") as tomlfile:
        toml.dump(json.load(jsonfile), tomlfile)


@click.command(context_settings={"show_default": True})
@click.argument("source", type=pathlib.Path)
@click.argument("target", type=pathlib.Path)
def toml2json(source: pathlib.Path, target: pathlib.Path):
    """Turn a toml file to a json file."""
    with open(source, "r") as tomlfile, open(target, "w") as jsonfile:
        json.dump(toml.load(tomlfile), jsonfile, indent=4)

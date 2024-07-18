import functools
import inspect
import warnings
from functools import partial
from pathlib import Path

import mmapped_df
import pandas as pd
import pandas.errors
from pyarrow import ArrowInvalid


def add_kwargs(foo):
    """Make a foo without **kwargs have **kwargs."""
    foo_params = inspect.signature(foo).parameters

    @functools.wraps(foo)
    def wrapper(*args, **kwargs):
        clipped_kwargs = {k: v for k, v in kwargs.items() if k in foo_params}
        return foo(*args, **clipped_kwargs)

    return wrapper


__ext_to_reader = {
    ".csv": add_kwargs(pd.read_csv),
    ".tsv": add_kwargs(partial(pd.read_csv, sep="\t")),
    ".txt": add_kwargs(partial(pd.read_table)),
    ".xlsx": add_kwargs(pd.read_excel),
    ".json": add_kwargs(pd.read_json),
    ".feather": add_kwargs(pd.read_feather),
    ".pandas_hdf": add_kwargs(pd.read_hdf),
    ".parquet": add_kwargs(pd.read_parquet),
    ".startrek": add_kwargs(mmapped_df.open_dataset),
}

__ext_to_methodName = {
    ".csv": "to_csv",
    ".xlsx": "to_excel",
    ".json": "to_json",
    ".feather": "to_feather",
    ".pandas_hdf": "to_hdf",
    ".parquet": "to_parquet",
}

get_extension = lambda file_path: Path(file_path).suffix.lower()
empty_df = pd.DataFrame(columns=["empty"]).reset_index(drop=True)


class MissingColumn(Exception):
    pass


def read_df(file_path: str | Path, *args, **kwargs) -> pd.DataFrame:
    file_extension = get_extension(file_path)

    if "columns" in kwargs and kwargs["columns"] is None:
        del kwargs["columns"]

    if "columns" in kwargs and kwargs["columns"] is not None:
        if "empty" in kwargs["columns"]:
            warnings.warn(
                "Someone uses a column named `empty` in the df. This is a column name reseved for empty dfs."
            )
        if file_extension in (".tsv", ".csv"):
            kwargs["usecols"] = kwargs["columns"]
            del kwargs["columns"]
    try:
        reader = __ext_to_reader[file_extension]
    except KeyError:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    try:
        return reader(file_path, *args, **kwargs)
    except pandas.errors.EmptyDataError:
        return empty_df
    except ArrowInvalid:
        raise MissingColumn("Missing some of the columns.")


def save_df(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    index: bool = False,
    key: str = "data",
    *args,
    **kwargs,
) -> None:
    file_extension = get_extension(file_path)
    match file_extension:
        case ".tsv":
            dataframe.to_csv(file_path, sep="\t", index=index, *args, **kwargs)
        case ".feather":
            dataframe.to_feather(file_path, *args, **kwargs)
        case ".parquet":
            dataframe.to_parquet(file_path, *args, **kwargs)
        case ".pandas_hdf":
            dataframe.to_hdf(file_path, key=key, *args, **kwargs)
        case ".startrek":
            with mmapped_df.DatasetWriter(path=file_path, **kwargs) as data_writer:
                data_writer.append_df(dataframe)
        case other:  # this might obviously not work
            writer = getattr(dataframe, __ext_to_methodName[file_extension])
            try:
                writer(file_path, *args, index=index, **kwargs)
            except TypeError:
                writer(file_path, *args, **kwargs)


__writer_specific_kwargs = {".tsv": dict(sep="\t")}


def save_df2(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *args,
    **kwargs,
) -> None:
    file_extension = get_extension(file_path)
    match file_extension:
        case ".startrek":
            with mmapped_df.DatasetWriter(path=file_path, **kwargs) as data_writer:
                data_writer.append_df(dataframe)
        case other:
            writer_name = f"to_{file_extension[1:]}"
            writer = add_kwargs(getattr(dataframe, writer_name))
            kwargs.update(__writer_specific_kwargs.get(writer_name, {}))
            writer(file_path, *args, **kwargs)

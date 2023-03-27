import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import pandas.errors
from pyarrow import ArrowInvalid

__ext_to_reader = {
    ".csv": pd.read_csv,
    ".tsv": partial(pd.read_csv, sep="\t"),
    ".xlsx": pd.read_excel,
    ".json": pd.read_json,
    ".feather": pd.read_feather,
    ".pandas_hdf": pd.read_hdf,
    ".parquet": pd.read_parquet,
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
    if (
        "columns" in kwargs
        and kwargs["columns"] is not None
        and "empty" in kwargs["columns"]
    ):
        warnings.warn(
            "Someone uses a column named `empty` in the df. This is a column name reseved for empty dfs."
        )
    file_extension = get_extension(file_path)
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
        case ".pandas_hdf":
            dataframe.to_hdf(file_path, key=key, *args, **kwargs)
        case other:  # this might obviously not work
            writer = getattr(dataframe, __ext_to_methodName[file_extension])
            try:
                writer(file_path, *args, index=index, **kwargs)
            except TypeError:
                writer(file_path, *args, **kwargs)

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


def hstack_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, axis=1)


def prepend_zeroth_row(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [df.iloc[[0]], df],
    )


def coordinatewise_minimum(*args):
    return pd.DataFrame(args).min()


def coordinatewise_maximum(*args):
    return pd.DataFrame(args).max()


def get_extents(
    *args: pd.DataFrame, resize_factor: float = 0
) -> tuple[pd.Series, pd.Series]:
    mins = coordinatewise_minimum(*(x.min().to_numpy() for x in args))
    maxs = coordinatewise_maximum(*(x.max().to_numpy() for x in args))
    if resize_factor != 0:
        extents = maxs - mins
        mins -= extents * resize_factor
        maxs += extents * resize_factor
    return mins, maxs


def overwrite_columns(original_df, new_df, prefix=""):
    return pd.concat(
        [
            original_df.drop(columns=new_df.columns),
            new_df.add_prefix(prefix) if prefix else new_df,
        ],
        axis=1,
    )


def add_column_to_pandas_dataframe_without_copying_data(
    df: pd.DataFrame,
    allow_column_overwrites: bool = False,
    **columns: npt.NDArray,
) -> pd.DataFrame:
    """
    Add column without copying space.
    """
    if not allow_column_overwrites:
        for column, values in columns.items():
            assert not column in df.columns, f"Column `{column}` would be overwritten."
            assert len(values) == len(
                df
            ), f"Column `{column}` has size (`{len(values)}`) not conforming to that of `df` ({len(df)})."
    dct = {col: df[col].to_numpy() for col in df.columns}
    dct.update(**columns)
    return pd.DataFrame(dct, copy=False)


def extend_df(df: pd.DataFrame, *other_dfs: pd.DataFrame) -> None:
    _cols = set(df.columns)
    _size = len(df.columns)
    for odf in other_dfs:
        _cols |= set(odf.columns)
        _size += len(odf.columns)
    assert len(_cols) == _size, "Some columns do not have different names."
    for odf in other_dfs:
        for col in odf.columns:
            df[col] = odf[col].to_numpy()


def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


@numba.njit(parallel=True)
def fill_nans(xx, filler):
    for i in numba.prange(len(xx)):
        if np.isnan(xx[i]):
            xx[i] = filler


def cast_to_array_if_possible(arg):
    try:
        return arg.to_numpy()
    except AttributeError:
        return arg


def indexed_long_df_to_tensor(long_df: pd.DataFrame, dtype=np.float64) -> npt.NDArray:
    """Transform a long sparse formated data into a tensor.

    Arguments:
        long_df (pd.DataFrame): A dataframe with row multiindex corresponding to indices of the output tensor.

    Returns:
        npt.NDArray: A dense tensor with with first dims corresponding to the input's index and last dim storing the values.
    """
    maxes = long_df.index.to_frame().apply(max)
    idx_to_values = np.zeros(shape=(*(maxes + 1), len(long_df.columns)), dtype=dtype)
    for idx, params in long_df.iterrows():
        idx_to_values[idx] = params
    return idx_to_values

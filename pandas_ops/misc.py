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

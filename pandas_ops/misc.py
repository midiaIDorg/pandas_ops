import pandas as pd


def hstack_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, axis=1)

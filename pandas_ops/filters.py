import pandas as pd


def quantile_filter_query(
    df: pd.DataFrame,
    min_quantile: float = 0.01,
    max_quantile: float = 0.99,
) -> str:
    """
    Arguments:
        df (pd.DataFrame): A dataframe for which columns we want the quantile filters to be computed.
        min_quantile (float): Lower value of the quantile.
        max_quantile (float): Upper value of the quantile.

    Return:
        str: A string that can be used to query df or any other data frame to get the limits we want.
    """
    assert 0 <= min_quantile
    assert min_quantile < max_quantile
    assert max_quantile <= 1
    assert len(df) > 0

    quantiles = df.quantile([min_quantile, max_quantile]).T.reset_index()
    quantiles.columns = "variable", "lo", "hi"
    return " and ".join(
        f"{r.variable} >= {r.lo} and {r.variable} <= {r.hi}"
        for r in quantiles.itertuples(index=False)
    )

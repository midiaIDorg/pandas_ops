import typing

import pandas as pd
from tqdm import tqdm


def iter_start_end_tuples(size, N) -> typing.Iterator[tuple[int, int]]:
    assert size > 0, "Size must be larger than 0."
    assert N > 0, "N must be larger than 0."
    prev = 0
    while prev < N:
        curr = min(prev + size, N)
        yield prev, curr
        prev = curr


def iter_df_batches(
    df: pd.DataFrame,
    size: int = 10_000_000,
    progressbar_message: str = "",
) -> typing.Iterator[pd.DataFrame]:
    start_end_tuples = iter_start_end_tuples(size, len(df))
    if progressbar_message:
        start_end_tuples = tqdm(
            start_end_tuples,
            total=len(df) // size + 1,
            desc=progressbar_message,
        )
    for start, end in start_end_tuples:
        yield df[start:end]

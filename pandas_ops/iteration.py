import typing

from tqdm import tqdm

import pandas as pd


def iter_start_end_tuples(size, N) -> typing.Iterator[tuple[int, int]]:
    prev = 0
    while prev < N:
        curr = prev + size
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

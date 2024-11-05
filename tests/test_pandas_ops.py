import itertools

import pytest

import numba
import numpy as np
import pandas as pd
from pandas_ops.lex_ops import LexicographicIndex


def test_LexicographicIndex():
    a_min = 1
    a_max = 4
    b_min = 2
    b_max = 5
    c_min = 10
    c_max = 15
    X = pd.DataFrame(
        itertools.product(
            range(a_min, a_max + 1), range(b_min, b_max + 1), range(c_min, c_max + 1)
        ),
        columns=["a", "b", "c"],
    )
    lexidx = LexicographicIndex.from_df(X[["a", "b"]])

    unique_diffs = np.unique(np.diff(lexidx.idx))
    assert len(unique_diffs) == 1, "Number of diffs not 1 on a grid."
    assert unique_diffs[0] == c_max - c_min + 1, "Unexpected diff size."
    assert len(lexidx) == (a_max - a_min + 1) * (
        b_max - b_min + 1
    ), "Wrong number of groups."

    res = np.zeros(len(lexidx), dtype=np.int64)

    @numba.njit
    def no_args_test(a, b):
        return np.sum(a) + np.sum(b)

    with pytest.raises(AssertionError):
        lexidx.simple_map(no_args_test, res, X.a.to_numpy(), X.b.to_numpy())

    @numba.njit
    def test(a, b, *args):
        return np.sum(a) + np.sum(b)

    res1 = np.zeros(len(lexidx), dtype=np.int64)
    lexidx.simple_map(test, res1, X.b.to_numpy(), X.c.to_numpy())

    res2 = np.zeros(len(lexidx), dtype=np.int64)
    lexidx.simple_map(test, res2, X.b, X.c)

    assert np.all(res1 == res2), "Some error in automatic calling of to_numpy()."

    res3 = []
    for group, df in X.groupby(["a", "b"]):
        res3.append(df[["b", "c"]].sum().sum())
    res3 = np.array(res3)

    assert np.all(
        res2 == res3
    ), "Trivial groupby and LexicographicIndex do not return the same results."

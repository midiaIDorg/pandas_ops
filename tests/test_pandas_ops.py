import itertools

import pytest

import numba
import numpy as np
import pandas as pd
from pandas_ops.lex_ops import LexicographicIndex


@numba.njit
def no_args_foo(a, b):
    return np.sum(a) + np.sum(b)


@numba.njit
def sum_all(a, b, *args):
    return np.sum(a) + np.sum(b)


class TestLexicographicIndex:
    def __init__(self):
        self.a_min = 1
        self.a_max = 4
        self.b_min = 2
        self.b_max = 5
        self.c_min = 10
        self.c_max = 15
        self.X = pd.DataFrame(
            itertools.product(
                range(self.a_min, self.a_max + 1),
                range(self.b_min, self.b_max + 1),
                range(self.c_min, self.c_max + 1),
            ),
            columns=["a", "b", "c"],
        )
        self.lexidx = LexicographicIndex.from_df(self.X[["a", "b"]])


def test_index_correctly_constructed():
    test = TestLexicographicIndex()
    unique_diffs = np.unique(np.diff(test.lexidx.idx))
    assert len(unique_diffs) == 1, "Number of diffs not 1 on a grid."
    assert unique_diffs[0] == test.c_max - test.c_min + 1, "Unexpected diff size."
    assert len(test.lexidx) == (test.a_max - test.a_min + 1) * (
        test.b_max - test.b_min + 1
    ), "Wrong number of groups."


def test_vargs_presence_constrain():
    test = TestLexicographicIndex()
    with pytest.raises(AssertionError):
        test.lexidx.map(no_args_foo, test.X.a, test.X.b)


def test_cast_to_array_works():
    test = TestLexicographicIndex()
    res1 = test.lexidx.map(sum_all, test.X.b.to_numpy(), test.X.c.to_numpy())
    res2 = test.lexidx.map(sum_all, test.X.b, test.X.c)

    assert np.all(res1 == res2), "Some error in automatic calling of to_numpy()."


def test_map_results_correctness():
    test = TestLexicographicIndex()
    map_res = test.lexidx.map(sum_all, test.X.b, test.X.c)
    pandas_res = np.array(
        [
            sum_all(
                d.b.to_numpy(),
                d.c.to_numpy(),
            )
            for _, d in test.X.groupby(["a", "b"])
        ]
    )

    assert np.all(
        map_res == pandas_res
    ), "Trivial groupby and LexicographicIndex do not return the same results."


@numba.njit
def make_matrix(a, b, *args):
    return np.zeros(shape=(10, 20), dtype=np.int32)


def test_mapping_function_with_nd_array_values():
    test = TestLexicographicIndex()
    assert test.lexidx.map(make_matrix, test.X.b, test.X.c).shape == (
        len(test.lexidx),
        10,
        20,
    )


# TODO: test writing of data in RAM or something??

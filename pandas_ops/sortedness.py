"""
TODO: start using Michal's midia_cpp::argsort for parallelized argsort (numpy sucks here)
"""

import math
from math import inf

import numba
import numpy as np
import numpy.typing as npt
from pandas_ops.numba_ops import inputs_series_to_numpy


@numba.njit
def _assert_consecutive_ints(xx):
    i_prev = -1
    for i in xx:
        if i > i_prev + 1:
            return False
        i_prev = i
    return True


assert_consecutive_ints = inputs_series_to_numpy(_assert_consecutive_ints)


@numba.njit
def _is_strictly_increasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev >= x:
            return False
        x_prev = x
    return True


is_strictly_increasing = inputs_series_to_numpy(_is_strictly_increasing)


@numba.njit
def _is_nondecreasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev > x:
            return False
        x_prev = x
    return True


is_nondecreasing = inputs_series_to_numpy(_is_nondecreasing)


@numba.njit
def _count_sorted(xx: npt.NDArray):
    if len(xx) == 0:
        return 0
    cnt = 1
    prev = xx[0]
    for x in xx:
        if x != prev:
            cnt += 1
    return cnt


count_sorted = inputs_series_to_numpy(_count_sorted)


@numba.njit(boundscheck=True)
def _is_sorted_lexicographically(
    strictly: bool = True,
    *arrays: npt.NDArray,
) -> bool:
    """
    Check if input arrays considered row-wise are lexicographically sorted.

    E.g. arrays = (A, B) then A[i][0] > A[i-1][0] or A[i][0] == A[i-1][0] and B[i][0] > B[i-1][0] for strictly increasing or B[i][0] == B[i-1][0] if not.
    Hence, A and B are consecutive dimensions to check.

    Arguments:
        strictly (bool): Should they be strictly increasing?
        *arrays (npt.NDArray): Arrays of the same type and length.
    """
    for arr in arrays:
        assert arr.dtype == arrays[0].dtype
        assert len(arr) == len(arrays[0])

    if len(arr) == 1:
        return True

    prev = np.empty(len(arrays), dtype=arr.dtype)
    for j in range(len(arrays)):
        prev[j] = arrays[j][0]

    for i in range(1, len(arr)):
        already_strictly_bigger = False
        for j in range(len(arrays)):
            already_strictly_bigger |= arrays[j][i] > prev[j]
            if not already_strictly_bigger and (
                arrays[j][i] < prev[j] or strictly and (j == len(arrays) - 1)
            ):
                return False
            prev[j] = arrays[j][i]

    return True


is_sorted_lexicographically = inputs_series_to_numpy(_is_sorted_lexicographically)


def test_is_sorted_lexicograhically():
    assert not is_sorted_lexicographically(
        True, np.array([10, 10, 20, 30]), np.array([11, 11, 1, 2])
    )
    assert is_sorted_lexicographically(
        True, np.array([10, 11, 20, 30]), np.array([11, 11, 1, 2])
    )
    assert not is_sorted_lexicographically(True, np.array([11, 1]), np.array([11, 20]))


@inputs_series_to_numpy
@numba.njit
def count_intersection_of_sorted_arrays(xx: npt.NDArray, yy: npt.NDArray) -> int:
    """
    Count the number of common elements in two sorted arrays.
    """
    cnt = 0
    i = 0
    j = 0
    prev_x = -math.inf
    prev_y = -math.inf
    while i < len(xx) and j < len(yy):
        x = xx[i]
        y = yy[j]
        assert prev_x < x, "xx was not sorted"
        assert prev_y < y, "yy was not sorted"
        if x == y:
            i += 1
            j += 1
            cnt += 1
            prev_x = x
            prev_y = y
        if x < y:
            i += 1
            prev_x = x
        if y < x:
            j += 1
            prev_y = y
    return cnt


@inputs_series_to_numpy
@numba.njit
def get_intersection_of_sorted_arrays(xx: npt.NDArray, yy: npt.NDArray) -> npt.NDArray:
    """
    Count the number of common elements in two sorted arrays.
    """
    res = []
    i = 0
    j = 0
    prev_x = -math.inf
    prev_y = -math.inf
    while i < len(xx) and j < len(yy):
        x = xx[i]
        y = yy[j]
        assert prev_x < x, "xx was not sorted"
        assert prev_y < y, "yy was not sorted"
        if x == y:
            i += 1
            j += 1
            res.append(x)
            prev_x = x
            prev_y = y
        if x < y:
            i += 1
            prev_x = x
        if y < x:
            j += 1
            prev_y = y
    return np.array(res, dtype=xx.dtype)

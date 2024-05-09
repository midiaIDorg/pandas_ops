"""
TODO: start using Michal's midia_cpp::argsort for parallelized argsort (numpy sucks here)
"""

from math import inf

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def assert_consecutive_ints(xx):
    i_prev = -1
    for i in xx:
        if i > i_prev + 1:
            return False
        i_prev = i
    return True


@numba.njit
def is_strictly_increasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev >= x:
            return False
        x_prev = x
    return True


@numba.njit
def is_nondecreasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev > x:
            return False
        x_prev = x
    return True


@numba.njit
def count_sorted(xx: npt.NDArray):
    if len(xx) == 0:
        return 0
    cnt = 1
    prev = xx[0]
    for x in xx:
        if x != prev:
            cnt += 1
    return cnt


@numba.njit(boundscheck=True)
def is_sorted_lexicographically(
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


def test_is_sorted_lexicograhically():
    assert not is_sorted_lexicographically(
        True, np.array([10, 10, 20, 30]), np.array([11, 11, 1, 2])
    )
    assert is_sorted_lexicographically(
        True, np.array([10, 11, 20, 30]), np.array([11, 11, 1, 2])
    )
    assert not is_sorted_lexicographically(True, np.array([11, 1]), np.array([11, 20]))

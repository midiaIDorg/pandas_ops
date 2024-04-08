"""
TODO: start using Michal's midia_cpp::argsort for parallelized argsort (numpy sucks here)
"""

from cmath import inf

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


def is_sorted_lexicographically(*arrays):
    arrays = list(arrays)
    arrays.reverse()
    return assert_consecutive_ints(np.lexsort(arrays))


@numba.njit
def is_strictly_increasing(xx):
    x_prev = -inf
    for x in xx:
        if x_prev >= x:
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

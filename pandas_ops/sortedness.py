import numba
import numpy as np


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
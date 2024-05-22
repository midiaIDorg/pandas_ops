import numba
import numpy as np
import numpy.typing as npt
from pandas_ops.numba_ops import inputs_series_to_numpy


@inputs_series_to_numpy
@numba.njit(boundscheck=True)
def observe(unorded_ids: npt.NDArray, to_observe: npt.NDArray):
    for _id in unorded_ids:
        to_observe[_id] = True
    return to_observe.nonzero()[0].astype(unorded_ids.dtype)


@inputs_series_to_numpy
def get_unique(ids: npt.NDArray, upper_limit: int | None = None) -> npt.NDArray:
    """
    Return an array of sorted unique numbers.
    """
    if len(ids) == 0:
        return ids

    if upper_limit is not None:
        N = upper_limit
    else:
        N = int(np.max(ids)) + 1

    return observe(
        ids,
        to_observe=np.full(fill_value=False, dtype=bool, shape=N),
    )


@inputs_series_to_numpy
@numba.njit
def get_unique_sorted(sorted_ids: npt.NDArray):
    res = []
    prev = sorted_ids[0]
    for _id in sorted_ids:
        if _id != prev:
            res.append(_id)
        prev = _id
    return np.array(res, dtype=sorted_ids.dtype)

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def min_max(xx: npt.NDArray, *args):
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


@numba.njit
def sum_real_good(xx: npt.NDArray, *args):
    return np.sum(xx)


@numba.njit
def weighted_mean_and_var(
    xx: npt.NDArray, weights: npt.NDArray, *args
) -> tuple[float, float]:
    weights = weights.astype(np.float64) / weights.sum()
    _weighted_mean = np.average(
        xx,
        weights=weights,
    )  # and this is float
    _weighted_var = np.average(
        (xx - _weighted_mean) ** 2,
        weights=weights,
    )
    return _weighted_mean, _weighted_var


@numba.njit(boundscheck=True)
def count2D(
    xx: npt.NDArray,
    yy: npt.NDArray,
) -> tuple[npt.NDArray, float | int, float | int, float | int, float | int]:
    """
    Do not try to optimize that by multithreading please without thinking of race conditions.
    """
    assert len(xx) == len(yy)
    min_x, max_x = min_max(xx)
    min_y, max_y = min_max(yy)
    cnts = np.zeros(dtype=np.uint64, shape=(max_x + 1, max_y + 1))

    for i in range(len(xx)):
        cnts[xx[i], yy[i]] += 1

    return cnts, min_x, max_x, min_y, max_y


def quantiles(xx, bin_cnt=5):
    return np.quantile(xx, np.linspace(0, 1, bin_cnt + 1))

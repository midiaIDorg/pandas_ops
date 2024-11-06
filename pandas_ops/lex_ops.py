"""
Implementing indexing functionality for lexicogrpahically sorted data.
"""
from __future__ import annotations

import inspect
import typing
from math import inf

from numba_progress import ProgressBar

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas_ops.misc import cast_to_array_if_possible
from pandas_ops.sortedness import is_strictly_increasing


@numba.njit(parallel=True)
def get_changes(arr: npt.NDArray, res: npt.NDArray[np.bool_]):
    assert len(arr) == len(res)
    for i in numba.prange(1, len(arr)):
        res[i] = arr[i] != arr[i - 1] | res[i]


@numba.njit
def get_change_idxs(idxs):
    res = []
    for i, idx in enumerate(idxs):
        if idx:
            res.append(i)
    res.append(len(idxs))
    return res


def get_lex_index(*columns):
    for i in range(len(columns)):
        try:
            columns[i] = columns[i].to_numpy()
        except AttributeError:
            pass


@numba.njit(parallel=True)
def general_parallel_map(
    foo: numba.core.registry.CPUDispatcher,
    indices: npt.NDArray,
    progress_proxy: ProgressBar | None = None,
    progress_step: int = 1,
    *foo_args,
):
    """General spread of indpenendent tasks unto threads."""
    for i in numba.prange(len(indices) - 1):
        foo(i, indices, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(progress_step)


@numba.njit(parallel=True)
def general_parallel_map(
    foo: numba.core.registry.CPUDispatcher,
    indices: npt.NDArray,
    progress_proxy: ProgressBar | None = None,
    progress_step: int = 1,
    *foo_args,
) -> None:
    """General spread of independent tasks unto threads.

    Assumptions on foo:
        * accepts current chunk number `i` and table of indices `indices`.
        * handles saving of the results itself (output_array among *foo_args)
    """
    for i in numba.prange(len(indices) - 1):
        foo(i, indices, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(progress_step)


@numba.njit(parallel=True)
def simple_parallel_map(
    outputs: npt.NDArray,
    indices: npt.NDArray,
    foo: numba.core.registry.CPUDispatcher,
    *foo_args,
    progress_proxy: ProgressBar | None = None,
    progress_step: int = 1,
) -> None:
    """Simple spread of independent tasks unto threads.

    Assumptions on foo:
        * accepts start_idx and stop_idx as first two args
        * returns a 1D numpy array.
    """
    assert (
        len(outputs) == len(indices) - 1
    ), "Size of outputs incompatible with expected number of chunks."
    for i in numba.prange(len(indices) - 1):
        start_idx = indices[i]
        stop_idx = indices[i + 1]
        outputs[i] = foo(start_idx, stop_idx, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(progress_step)


@numba.njit
def eval_on_view(i, indices, arr, res, foo, *foo_args):
    start_idx = indices[i]
    stop_idx = indices[i + 1]
    res[i] = foo(arr[start_idx:stop_idx], *foo_args)


@numba.njit
def eval_on_views_and_save(
    i,
    indices,
    res,
    foo,
    arg0=None,
    arg1=None,
    arg2=None,
    arg3=None,
    arg4=None,
    arg5=None,
    arg6=None,
    arg7=None,
    arg8=None,
    arg9=None,
    *foo_args,
) -> None:
    """Disgusting workaround to make it possible to run numba on multiple variadic inputs."""
    start_idx = indices[i]
    stop_idx = indices[i + 1]
    res[i] = foo(
        arg0[start_idx:stop_idx] if arg0 is not None else None,
        arg1[start_idx:stop_idx] if arg1 is not None else None,
        arg2[start_idx:stop_idx] if arg2 is not None else None,
        arg3[start_idx:stop_idx] if arg3 is not None else None,
        arg4[start_idx:stop_idx] if arg4 is not None else None,
        arg5[start_idx:stop_idx] if arg5 is not None else None,
        arg6[start_idx:stop_idx] if arg6 is not None else None,
        arg7[start_idx:stop_idx] if arg7 is not None else None,
        arg8[start_idx:stop_idx] if arg8 is not None else None,
        arg9[start_idx:stop_idx] if arg9 is not None else None,
        *foo_args,
    )


@numba.njit
def eval_on_views(
    foo,
    start_idx,
    stop_idx,
    arg0=None,
    arg1=None,
    arg2=None,
    arg3=None,
    arg4=None,
    arg5=None,
    arg6=None,
    arg7=None,
    arg8=None,
    arg9=None,
    *foo_args,
):
    """Disgusting workaround to make it possible to run numba on multiple variadic inputs."""
    return foo(
        arg0[start_idx:stop_idx] if arg0 is not None else None,
        arg1[start_idx:stop_idx] if arg1 is not None else None,
        arg2[start_idx:stop_idx] if arg2 is not None else None,
        arg3[start_idx:stop_idx] if arg3 is not None else None,
        arg4[start_idx:stop_idx] if arg4 is not None else None,
        arg5[start_idx:stop_idx] if arg5 is not None else None,
        arg6[start_idx:stop_idx] if arg6 is not None else None,
        arg7[start_idx:stop_idx] if arg7 is not None else None,
        arg8[start_idx:stop_idx] if arg8 is not None else None,
        arg9[start_idx:stop_idx] if arg9 is not None else None,
        *foo_args,
    )


def has_varargs(func):
    """Check if any parameter of func is a var-positional (*args) type"""
    for param in inspect.signature(func).parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


NumpyType = typing.TypeVar("NumpyType", bound=np.generic)


class LexicographicIndex:
    """
    Build up an index mapping a lexicographically sorted dataset into contiguous chunks.
    """

    def __init__(self, *columns: npt.NDArray):
        idxs = np.zeros(shape=len(columns[0]), dtype=np.bool_)
        idxs[0] = True
        for i in range(len(columns)):
            get_changes(columns[i], idxs)
        self.idx = np.array(get_change_idxs(idxs), dtype=np.uint32)
        assert len(self.idx) > 0, "Produced an empty index."
        assert len(self.idx) > 1, "No chunks present."
        assert is_strictly_increasing(self.idx), "Index not strictly increasing. ABORT"

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> LexicographicIndex:
        return cls(*[df[c].to_numpy() for c in df.columns])

    @classmethod
    def from_array(cls, arr: npt.NDArray) -> LexicographicIndex:
        assert (
            len(arr.shape) == 2
        ), "Can instantiate LexicographicIndex from 2D arrays only."
        return cls(*[arr[:, j] for j in range(arr.shape[1])])

    def __len__(self) -> int:
        return len(self.idx) - 1

    def general_map(
        self,
        foo,
        *foo_args,
        progress_proxy: ProgressBar | None = None,
        progress_step: int = 1,
    ):
        """
        Apply `foo` that is `general_parallel_map` compatible, in as much as:

        * it takes as first argument the number in the indices array to consider.
        * it takes as second argument the indices array (self.idx from LexicographicIndex)
        * all rest is positionally passed in and no kwargs.
        * in particular, if you want results, `foo` must know where to put it.

        check out `self.map` for a simpler yet less general interface.

        Arguments:
            foo: njitted function, must define *args (variadic arguments).
            *foo_args: a number of positional arguments to the function: columns of the same size assumed.
            progress_proxy (ProgressBar|None): use external progress proxy.
            progress_step (int): Step for `progress_proxy.update`.
        """
        general_parallel_map(foo, self.idx, progress_proxy, progress_step, *foo_args)

    def map(
        self,
        foo: typing.Callable[..., npt.NDArray[NumpyType]],
        *foo_args: npt.NDArray,
        progress_proxy: ProgressBar | None = None,
        progress_step: int = 1,
    ) -> npt.NDArray[NumpyType]:
        """
        This function will apply the user defined njit-compiled `foo` to chunks defined by isoquants of the index.

        Arguments:
            foo: njitted function, must define *args (variadic arguments).
            *foo_args: a number of positional arguments to the function: columns of the same size assumed.
            progress_proxy (ProgressBar|None): use external progress proxy.
            progress_step (int): Step for `progress_proxy.update`.
        """
        assert len(foo_args) <= 10, f"`foo` must use less than 10 positional arguments."

        assert has_varargs(foo), "`foo` needs `*args`."

        assert len(self.idx) > 1, "No chunks present."

        foo_args = tuple(map(cast_to_array_if_possible, foo_args))

        # using magic of interpretation for what statically typed advanced languages would do with finger in butt...
        first_result = eval_on_views(foo, self.idx[0], self.idx[1], *foo_args)

        outputs = np.empty(
            dtype=first_result.dtype,
            shape=(len(self), len(first_result)),
        )

        simple_parallel_map(
            outputs,
            self.idx,
            eval_on_views,  # foo
            foo,  # foo_args*
            *foo_args,  # foo_args*
            progress_proxy=progress_proxy,
            progress_step=progress_step,
        )

        assert outputs[0] == first_result, "First eval not the same as first in batch."

        return outputs

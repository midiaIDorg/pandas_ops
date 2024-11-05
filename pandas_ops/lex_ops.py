"""
Implementing indexing functionality for lexicogrpahically sorted data.
"""
from __future__ import annotations

import inspect
from math import inf

from numba_progress import ProgressBar

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


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
def parallel_map(
    foo: numba.core.registry.CPUDispatcher,
    indices: npt.NDArray,
    progress_proxy: ProgressBar | None = None,
    *foo_args,
):
    """General spread of indpenendent tasks unto threads."""
    for i in numba.prange(len(indices) - 1):
        foo(i, indices, *foo_args)
        if progress_proxy is not None:
            progress_proxy.update(1)


@numba.njit
def simple_io(i, indices, arr, res, foo, *foo_args):
    start_idx = indices[i]
    stop_idx = indices[i + 1]
    res[i] = foo(arr[start_idx:stop_idx], *foo_args)


@numba.njit
def simple_max_10_args_io(
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
    """Disgusting workaround."""
    start_idx = indices[i]
    stop_idx = indices[i + 1]
    arg0_view = arg0[start_idx:stop_idx] if arg0 is not None else None
    arg1_view = arg1[start_idx:stop_idx] if arg1 is not None else None
    arg2_view = arg2[start_idx:stop_idx] if arg2 is not None else None
    arg3_view = arg3[start_idx:stop_idx] if arg3 is not None else None
    arg4_view = arg4[start_idx:stop_idx] if arg4 is not None else None
    arg5_view = arg5[start_idx:stop_idx] if arg5 is not None else None
    arg6_view = arg6[start_idx:stop_idx] if arg6 is not None else None
    arg7_view = arg7[start_idx:stop_idx] if arg7 is not None else None
    arg8_view = arg8[start_idx:stop_idx] if arg8 is not None else None
    arg9_view = arg9[start_idx:stop_idx] if arg9 is not None else None
    res[i] = foo(
        arg0_view,
        arg1_view,
        arg2_view,
        arg3_view,
        arg4_view,
        arg5_view,
        arg6_view,
        arg7_view,
        arg8_view,
        arg9_view,
        *foo_args,
    )


def has_varargs(func):
    # Get the signature of the function
    signature = inspect.signature(func)

    # Check if any parameter is a var-positional (*args) type
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


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
        desc: str | None = None,
    ):
        """
        Apply `foo` that is `parallel_map` compatible, in as much as:

        * it takes as first argument the number in the indices array to consider.
        * it takes as second argument the indices array (self.idx from LexicographicIndex)
        * all rest is positionally passed in and no kwargs.
        * in particular, if you want results, `foo` must know where to put it.

        check out `self.map` for a simpler interface.
        """
        if progress_proxy is None and desc is not None:
            with ProgressBar(total=len(self), desc=desc) as progress_proxy:
                parallel_map(foo, self.idx, progress_proxy, *foo_args)
        else:
            parallel_map(foo, self.idx, progress_proxy, *foo_args)

    def map(
        self,
        foo,
        input_array,
        output_array,
        *foo_args,
        progress_proxy: ProgressBar | None = None,
        desc: str | None = None,
    ) -> None:
        """
        Apply `foo` on data divided into chunks by the current index.
        """
        assert len(output_array) == len(
            self
        ), "We assume each group must result in one set of statistics."
        self.general_map(
            simple_io,
            input_array,
            output_array,
            foo,
            *foo_args,
            progress_proxy=progress_proxy,
            desc=desc,
        )

    def simple_map(
        self,
        foo,
        output_array,
        *foo_args,
        progress_proxy: ProgressBar | None = None,
        desc: str | None = None,
    ) -> None:
        """
        This function will apply the user defined njit-compiled `foo` to chunks defined by isoquants of the index.

        Arguments:
            foo: njitted function, must define *args (variadic arguments).
            output_array: array of size len(index)*len(foo_result). The user must make sure the size is as should (function results will be stored in rows of the `output_array`).
            *foo_args: a number of positional arguments to the function: columns of the same size assumed.
            progress_proxy: to reuse outside provided numba_progress progress proxy.
            desc: message for the new proxy if not using user provided proxy.
        """
        assert (
            len(foo_args) <= 10
        ), f"Supporting up to {len(foo_args)} arguments for `foo`."
        assert has_varargs(
            foo
        ), "You need to pass in a numba compiled function with *args."
        foo_args_arrays = []
        for arg in foo_args:
            try:
                arg = arg.to_numpy()
            except AttributeError:
                pass
            foo_args_arrays.append(arg)
        # we could inspect the number of arguments and choosing appropriately...
        self.general_map(
            simple_max_10_args_io,  # foo
            output_array,  # foo_args*
            foo,  # foo_args*
            *foo_args_arrays,  # remaining foo_args*
            progress_proxy=progress_proxy,
            desc=desc,
        )

    def typed_map(
        self,
        foo,
        arr,
        res_type,
        *foo_args,
        progress_proxy=None,
        desc: str | None = None,
    ) -> npt.NDArray:
        raise NotImplementedError
        output_array = np.empty(dtype=res_type, shape=len(self))
        self.map(foo, arr, output_array, *foo_args, progress_proxy, desc)
        return output_array

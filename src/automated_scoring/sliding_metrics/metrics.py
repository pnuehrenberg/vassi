import numpy as np
from numba import config, njit, prange
from numpy.typing import NDArray

# set the threading layer before any parallel target compilation
# this requires tbb!
config.THREADING_LAYER = "safe"  # type: ignore


@njit(parallel=True)
def mean(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.mean(array[t, f])
    return result


@njit(parallel=True)
def median(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.median(array[t, f])
    return result


@njit(parallel=True)
def min(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.min(array[t, f])
    return result


@njit(parallel=True)
def max(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.max(array[t, f])
    return result


@njit(parallel=True)
def quantile(array: NDArray, q: float) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.quantile(array[t, f], q)
    return result


@njit
def q01(array: NDArray) -> NDArray:
    return quantile(array, 0.01)


@njit
def q05(array: NDArray) -> NDArray:
    return quantile(array, 0.05)


@njit
def q10(array: NDArray) -> NDArray:
    return quantile(array, 0.10)


@njit
def q90(array: NDArray) -> NDArray:
    return quantile(array, 0.90)


@njit
def q95(array: NDArray) -> NDArray:
    return quantile(array, 0.95)


@njit
def q99(array: NDArray) -> NDArray:
    return quantile(array, 0.99)

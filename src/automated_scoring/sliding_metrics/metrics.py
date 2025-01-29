import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(parallel=True, cache=True)
def mean(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.mean(array[t, f])
    return result


@njit(parallel=True, cache=True)
def median(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.median(array[t, f])
    return result


@njit(parallel=True, cache=True)
def min(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.min(array[t, f])
    return result


@njit(parallel=True, cache=True)
def max(array: NDArray) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.max(array[t, f])
    return result


@njit(parallel=True, cache=True)
def quantile(array: NDArray, q: float) -> NDArray:
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.quantile(array[t, f], q)
    return result


@njit(cache=True)
def q01(array: NDArray) -> NDArray:
    return quantile(array, 0.01)


@njit(cache=True)
def q05(array: NDArray) -> NDArray:
    return quantile(array, 0.05)


@njit(cache=True)
def q10(array: NDArray) -> NDArray:
    return quantile(array, 0.10)


@njit(cache=True)
def q90(array: NDArray) -> NDArray:
    return quantile(array, 0.90)


@njit(cache=True)
def q95(array: NDArray) -> NDArray:
    return quantile(array, 0.95)


@njit(cache=True)
def q99(array: NDArray) -> NDArray:
    return quantile(array, 0.99)

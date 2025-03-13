import numpy as np
from numba import config, njit, prange
from numpy.typing import NDArray

# set the threading layer before any parallel target compilation
# this requires tbb!
config.THREADING_LAYER = "safe"  # type: ignore


@njit(parallel=True)
def mean(array: NDArray) -> NDArray:
    """
    Jitted function to calculate the mean along the last axis of a 3D array.

    Args:
        array: The input array
    """
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.mean(array[t, f])
    return result


@njit(parallel=True)
def median(array: NDArray) -> NDArray:
    """
    Jitted function to calculate the median along the last axis of a 3D array.

    Args:
        array: The input array
    """
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.median(array[t, f])
    return result


@njit(parallel=True)
def min(array: NDArray) -> NDArray:
    """
    Jitted function to calculate the minimum along the last axis of a 3D array.

    Args:
        array: The input array
    """
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.min(array[t, f])
    return result


@njit(parallel=True)
def max(array: NDArray) -> NDArray:
    """
    Jitted function to calculate the maximum along the last axis of a 3D array.

    Args:
        array: The input array
    """
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.max(array[t, f])
    return result


@njit(parallel=True)
def quantile(array: NDArray, q: float) -> NDArray:
    """
    Jitted function to calculate a quantile along the last axis of a 3D array.

    Args:
        array: The input array
        q: The quantile to calculate
    """
    assert array.ndim == 3
    result = np.zeros((array.shape[0], array.shape[1]), dtype=np.float64)
    for t in prange(array.shape[0]):
        for f in prange(array.shape[1]):
            result[t, f] = np.quantile(array[t, f], q)
    return result


@njit
def q01(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 1st percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.01)


@njit
def q05(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 5th percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.05)


@njit
def q10(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 10th percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.10)


@njit
def q90(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 90th percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.90)


@njit
def q95(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 95th percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.95)


@njit
def q99(array: NDArray) -> NDArray:
    """
    Convenience function to calculate the 99th percentile along the last axis of a 3D array.

    Args:
        array: The input array
    """
    return quantile(array, 0.99)

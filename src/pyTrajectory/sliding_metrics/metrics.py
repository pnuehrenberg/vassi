from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def mean(array: NDArray) -> NDArray:
    return np.mean(array, axis=-1, keepdims=True)


def median(array: NDArray) -> NDArray:
    return np.median(array, axis=-1, keepdims=True)


def min(array: NDArray) -> NDArray:
    return np.min(array, axis=-1, keepdims=True)


def max(array: NDArray) -> NDArray:
    return np.max(array, axis=-1, keepdims=True)


def quantiles(array: NDArray, quantiles: Iterable[float]) -> NDArray:
    last_dim_quantiles = np.quantile(array, tuple(quantiles), axis=-1)
    transposed_axes = tuple(np.roll(range(last_dim_quantiles.ndim), -1))
    last_dim_quantiles = last_dim_quantiles.transpose(*transposed_axes)
    return last_dim_quantiles


def q01(array: NDArray) -> NDArray:
    return quantiles(array, (0.01,))


def q05(array: NDArray) -> NDArray:
    return quantiles(array, (0.05,))


def q10(array: NDArray) -> NDArray:
    return quantiles(array, (0.1,))


def q90(array: NDArray) -> NDArray:
    return quantiles(array, (0.9,))


def q95(array: NDArray) -> NDArray:
    return quantiles(array, (0.95,))


def q99(array: NDArray) -> NDArray:
    return quantiles(array, (0.95,))

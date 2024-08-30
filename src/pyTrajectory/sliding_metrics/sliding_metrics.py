from collections.abc import Iterable

from numpy.typing import NDArray

from . import metrics
from .sliding_window import apply_to_sliding_windows


def sliding_mean(series: NDArray, window_size: int) -> NDArray:
    return apply_to_sliding_windows(series, window_size, metrics.mean)


def sliding_median(series: NDArray, window_size: int) -> NDArray:
    return apply_to_sliding_windows(series, window_size, metrics.median)


def sliding_min(series: NDArray, window_size: int) -> NDArray:
    return apply_to_sliding_windows(series, window_size, metrics.min)


def sliding_max(series: NDArray, window_size: int) -> NDArray:
    return apply_to_sliding_windows(series, window_size, metrics.max)


def sliding_quantiles(
    series: NDArray, window_size: int, quantiles: Iterable[float]
) -> NDArray:
    def _last_dim_quantiles(array: NDArray):
        return metrics.quantiles(array, quantiles)

    return apply_to_sliding_windows(series, window_size, _last_dim_quantiles)

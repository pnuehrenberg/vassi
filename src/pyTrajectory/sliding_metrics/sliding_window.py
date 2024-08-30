import numpy as np
from numpy.typing import NDArray

from ..utils import NDArray_to_NDArray


def apply_to_sliding_windows(
    series: NDArray,
    window_size: int,
    metric_func: NDArray_to_NDArray,
    *,
    window_slices: slice | list[slice] | None = None,
) -> NDArray:
    if window_size % 2 != 1:
        raise ValueError("window size must be odd")
    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        series,
        window_size,
        axis=0,
    )
    if sliding_windows.shape[-1] == 1:
        sliding_windows = sliding_windows.squeeze(axis=-1)
    num_selections = len(window_slices) if isinstance(window_slices, list) else 1
    sliding_metrics = np.zeros(
        (*sliding_windows.shape[:-1], num_selections), dtype=float
    )
    if window_slices is None:
        sliding_metric = metric_func(sliding_windows)
        sliding_metrics[..., : sliding_metric.shape[-1]] = sliding_metric
        sliding_metrics = sliding_metrics.squeeze(axis=-1)
    elif isinstance(window_slices, slice):
        sliding_metric = metric_func(sliding_windows[..., window_slices])
        sliding_metrics[..., : sliding_metric.shape[-1]] = sliding_metric
    elif isinstance(window_slices, list):
        for idx, window_slice in enumerate(window_slices):
            sliding_metric = metric_func(sliding_windows[..., window_slice])
            sliding_metrics[..., idx : (idx + sliding_metric.shape[-1])] = (
                sliding_metric
            )
    else:
        raise ValueError(f"Invalid window_slices argument {window_slices}.")
    padding = window_size // 2
    sliding_metrics = np.concatenate(
        [
            [sliding_metrics[0]] * padding,
            sliding_metrics,
            [sliding_metrics[-1]] * padding,
        ],
        axis=0,
    )
    return sliding_metrics


def apply_multiple_to_sliding_windows(
    series: NDArray,
    window_size: int,
    metric_funcs: list[NDArray_to_NDArray],
    *,
    window_slices: slice | list[slice] | None = None,
) -> NDArray:
    if window_size % 2 != 1:
        raise ValueError("window size must be odd")
    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        series,
        window_size,
        axis=0,
    )
    if sliding_windows.shape[-1] == 1:
        sliding_windows = sliding_windows.squeeze(axis=-1)
    num_selections = len(window_slices) if isinstance(window_slices, list) else 1
    sliding_metrics = np.zeros(
        (*sliding_windows.shape[:-1], num_selections, len(metric_funcs)), dtype=float
    )
    if window_slices is None:
        for metric_idx, metric_func in enumerate(metric_funcs):
            sliding_metric = metric_func(sliding_windows)
            sliding_metrics[..., : sliding_metric.shape[-1], metric_idx] = (
                sliding_metric
            )
        sliding_metrics = sliding_metrics.squeeze(axis=-2)
    elif isinstance(window_slices, slice):
        for metric_idx, metric_func in enumerate(metric_funcs):
            sliding_metric = metric_func(sliding_windows[..., window_slices])
            sliding_metrics[..., : sliding_metric.shape[-1], metric_idx] = (
                sliding_metric
            )
    elif isinstance(window_slices, list):
        for idx, window_slice in enumerate(window_slices):
            for metric_idx, metric_func in enumerate(metric_funcs):
                sliding_metric = metric_func(sliding_windows[..., window_slice])
                sliding_metrics[
                    ..., idx : (idx + sliding_metric.shape[-1]), metric_idx
                ] = sliding_metric
    else:
        raise ValueError(f"Invalid window_slices argument {window_slices}.")
    padding = window_size // 2
    sliding_metrics = np.concatenate(
        [
            [sliding_metrics[0]] * padding,
            sliding_metrics,
            [sliding_metrics[-1]] * padding,
        ],
        axis=0,
    )
    return sliding_metrics


def as_window_delta(metric_func: NDArray_to_NDArray) -> NDArray_to_NDArray:
    def decorated(array: NDArray) -> NDArray:
        half_window_size = array.shape[-1] // 2
        metric_first_half = metric_func(array[..., :half_window_size])
        metric_second_half = metric_func(array[..., -half_window_size:])
        return metric_second_half - metric_first_half

    decorated.__name__ = f"d_{metric_func.__name__}"
    return decorated

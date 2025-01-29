from collections.abc import Iterable
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from . import metrics


def apply_multiple_to_sliding_windows(
    series: NDArray,
    window_size: int,
    funcs: Iterable[Callable],
    slices: Optional[Iterable[slice]] = None,
) -> NDArray:
    sliding_window_view = np.lib.stride_tricks.sliding_window_view(
        series, window_size, axis=0
    )
    funcs = list(funcs)
    sliced = False
    if slices is None:
        slices = [slice(None, None)]
    else:
        sliced = True
        slices = list(slices)
    padding = window_size // 2
    result = np.zeros(
        (
            sliding_window_view.shape[0] + 2 * padding,
            sliding_window_view.shape[1],
            len(funcs),
            len(slices),
        )
    )
    for func_idx in range(len(funcs)):
        for slice_idx in range(len(slices)):
            result[padding:-padding, ..., func_idx, slice_idx] = funcs[func_idx](
                sliding_window_view[..., slices[slice_idx]]
            )
    result[:padding] = result[padding]
    result[-padding:] = result[-(padding + 1)]
    if len(funcs) == 1:
        result = result[..., 0, :]
    if not sliced and result.shape[-1] == 1:
        result = result[..., 0]
    return result


def sliding_mean(series: NDArray, window_size: int) -> NDArray:
    return apply_multiple_to_sliding_windows(series, window_size, [metrics.mean])


def sliding_median(series: NDArray, window_size: int) -> NDArray:
    return apply_multiple_to_sliding_windows(series, window_size, [metrics.median])


def sliding_min(series: NDArray, window_size: int) -> NDArray:
    return apply_multiple_to_sliding_windows(series, window_size, [metrics.min])


def sliding_max(series: NDArray, window_size: int) -> NDArray:
    return apply_multiple_to_sliding_windows(series, window_size, [metrics.max])


def sliding_quantile(series: NDArray, window_size: int, q: float) -> NDArray:
    return apply_multiple_to_sliding_windows(
        series, window_size, [lambda series: metrics.quantile(series, q)]
    )


def sliding_quantiles(
    series: NDArray, window_size: int, quantiles: Iterable[float]
) -> NDArray:
    return apply_multiple_to_sliding_windows(
        series,
        window_size,
        [lambda series, q=q: metrics.quantile(series, q) for q in quantiles],
    )

from collections.abc import Iterable
from typing import Callable, Optional

import numpy as np

from . import metrics


def apply_multiple_to_sliding_windows(
    array: np.ndarray,
    window_size: int,
    funcs: Iterable[Callable],
    slices: Optional[Iterable[slice]] = None,
) -> np.ndarray:
    """
    Apply multiple functions to sliding windows of an array.

    Parameters:
        array: The input array.
        window_size: The size of the sliding window.
        funcs: The functions to apply.
        slices: The slices to apply the functions to (slicing the moving window).
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if (ndim := array.ndim) == 1:
        array = array[:, np.newaxis]
    sliding_window_view = np.lib.stride_tricks.sliding_window_view(
        array, window_size, axis=0
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
    if ndim == 1 and len(funcs) == 1:
        return result[:, 0]
    return result


def sliding_mean(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the sliding mean of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
    """
    return apply_multiple_to_sliding_windows(array, window_size, [metrics.mean])


def sliding_median(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the sliding median of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
    """
    return apply_multiple_to_sliding_windows(array, window_size, [metrics.median])


def sliding_min(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the sliding minimum of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
    """
    return apply_multiple_to_sliding_windows(array, window_size, [metrics.min])


def sliding_max(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the sliding maximum of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
    """
    return apply_multiple_to_sliding_windows(array, window_size, [metrics.max])


def sliding_quantile(array: np.ndarray, window_size: int, q: float) -> np.ndarray:
    """
    Calculate a sliding quantile of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
        q: The quantile to calculate
    """
    return apply_multiple_to_sliding_windows(
        array, window_size, [lambda array: metrics.quantile(array, q)]
    )


def sliding_quantiles(
    array: np.ndarray, window_size: int, quantiles: Iterable[float]
) -> np.ndarray:
    """
    Calculate sliding quantiles of an array.

    Parameters:
        array: The input array
        window_size: The size of the sliding window
        quantiles: The quantiles to calculate
    """
    return apply_multiple_to_sliding_windows(
        array,
        window_size,
        [lambda array, q=q: metrics.quantile(array, q) for q in quantiles],
    )

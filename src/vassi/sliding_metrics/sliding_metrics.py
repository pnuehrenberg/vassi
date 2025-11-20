from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numba import (
    njit,  # pyright: ignore[reportUnknownVariableType]
    prange,
)


def _compile_slices_to_relative(
    window_slice: None | slice | Iterable[slice],
    window_size: int,
) -> np.ndarray:
    """
    Converts None, a single slice, or an iterable of slices into a
    Numba-compatible (S, 2) float64 array of relative [start, stop] pairs.
    The conversion is based on the provided `window_size`.
    """
    if window_slice is None:
        return np.empty((0, 2), dtype=np.float64)
    if isinstance(window_slice, slice):
        slices_list = [window_slice]
    else:
        slices_list = list(window_slice)

    s_dim = len(slices_list)
    if s_dim == 0:
        return np.empty((0, 2), dtype=np.float64)

    slices_arr = np.empty((s_dim, 2), dtype=np.float64)
    for i, sl in enumerate(slices_list):
        if not isinstance(sl, slice):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"window_slice must be a slice or iterable of slices, got {type(sl)}"
            )
        if sl.step is not None:
            raise ValueError("Numba kernel does not support slice steps")

        start_abs, stop_abs = sl.start, sl.stop

        # Resolve start: None -> 0, handle negatives
        if start_abs is None:
            start_abs = 0
        elif start_abs < 0:
            start_abs += window_size

        # Resolve stop: None -> window_size, handle negatives
        if stop_abs is None:
            stop_abs = window_size
        elif stop_abs < 0:
            stop_abs += window_size

        # Clamp to the valid absolute range [0, window_size]
        start_abs = max(0, min(start_abs, window_size))
        stop_abs = max(start_abs, min(stop_abs, window_size))

        # Convert to relative floats
        slices_arr[i, 0] = start_abs / window_size if window_size > 0 else 0.0
        slices_arr[i, 1] = stop_abs / window_size if window_size > 0 else 0.0

    return slices_arr


@njit(fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _quantile_interp(sorted_arr: np.ndarray, q: float) -> float:
    """
    Numba-friendly linear interpolation for a *sorted* array.
    """
    n = sorted_arr.shape[0]
    if n == 0:
        return np.nan
    idx_f = (n - 1) * q
    idx_i = int(idx_f)
    if idx_i == n - 1:
        return sorted_arr[idx_i]
    frac = idx_f - idx_i
    val1 = sorted_arr[idx_i]
    val2 = sorted_arr[idx_i + 1]
    return val1 + frac * (val2 - val1)


# Must be False to handle NaNs
@njit(fastmath=False)  # pyright: ignore[reportUntypedFunctionDecorator]
def _apply_relative_window_slice(
    window_slice: np.ndarray, slice_start_rel: float, slice_stop_rel: float
) -> np.ndarray:
    """
    Applies a relative float-based slice to a window. Guarantees
    at least one element is returned if the window is not empty.
    """
    w_len = window_slice.shape[0]
    if w_len == 0:
        return window_slice

    # Convert relative floats to absolute integer indices
    start_idx = int(np.floor(slice_start_rel * w_len))
    stop_idx = int(np.ceil(slice_stop_rel * w_len))

    # Guarantee: Retrieve at least one value if window is not empty
    if start_idx == stop_idx and w_len > 0:
        # If slice is so small it results in zero length, take one element
        if start_idx < w_len:
            stop_idx = start_idx + 1
        else:  # Handle case where start_idx is at the very end
            start_idx = w_len - 1

    # Clamp indices to the valid range of the current window
    start_idx = max(0, min(start_idx, w_len))
    stop_idx = max(start_idx, min(stop_idx, w_len))

    return window_slice[start_idx:stop_idx]


# Must be False to handle NaNs
@njit(fastmath=False)  # pyright: ignore[reportUntypedFunctionDecorator]
def _filter_and_sort_window(window_slice: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Reusable helper to filter NaNs from a window, check for all-NaNs,
    and return the sorted, clean array.
    """
    nan_count = 0
    for x in window_slice:
        if np.isnan(x):
            nan_count += 1
    current_window_size = window_slice.shape[0]
    if nan_count == current_window_size:
        return np.empty(0, dtype=window_slice.dtype), True
    w_real = current_window_size - nan_count
    clean_slice = np.empty(w_real, dtype=window_slice.dtype)
    idx = 0
    for x in window_slice:
        if not np.isnan(x):
            clean_slice[idx] = x
            idx += 1
    return np.sort(clean_slice), False


# Must be False to handle NaNs
@njit(fastmath=False)  # pyright: ignore[reportUntypedFunctionDecorator]
def _calc_mean_on_window(window: np.ndarray) -> float:
    """Calculates the mean of a 1D window, ignoring NaNs."""
    current_sum = 0.0
    valid_count = 0
    for x in window:
        if not np.isnan(x):
            current_sum += x
            valid_count += 1
    if valid_count > 0:
        return current_sum / valid_count
    return np.nan


# Must be False to handle NaNs
@njit(fastmath=False)  # pyright: ignore[reportUntypedFunctionDecorator]
def _calc_min_on_window(window: np.ndarray) -> float:
    """Calculates the min of a 1D window, ignoring NaNs."""
    current_min = np.inf
    found_valid = False
    for x in window:
        if not np.isnan(x):
            if x < current_min:
                current_min = x
            found_valid = True
    if found_valid:
        return current_min
    return np.nan


# Must be False to handle NaNs
@njit(fastmath=False)  # pyright: ignore[reportUntypedFunctionDecorator]
def _calc_max_on_window(window: np.ndarray) -> float:
    """Calculates the max of a 1D window, ignoring NaNs."""
    current_max = -np.inf
    found_valid = False
    for x in window:
        if not np.isnan(x):
            if x > current_max:
                current_max = x
            found_valid = True
    if found_valid:
        return current_max
    return np.nan


@njit(parallel=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _numba_sliding_quantiles_kernel(
    array_2d: np.ndarray,
    window_size: int,
    quantiles_arr: np.ndarray,
    slices_arr: np.ndarray,  # Shape (S, 2), dtype=float64
) -> np.ndarray:
    """
    The core Numba kernel (looper) for multiple quantiles (centered).
    """
    t_dim = array_2d.shape[0]
    k_dim = array_2d.shape[1]
    q_dim = quantiles_arr.shape[0]
    s_dim = slices_arr.shape[0]
    half_window = window_size // 2

    if s_dim == 0:
        out_arr = np.empty((t_dim, k_dim, q_dim, 1), dtype=array_2d.dtype)
    else:
        out_arr = np.empty((t_dim, k_dim, q_dim, s_dim), dtype=array_2d.dtype)

    for t in prange(t_dim):
        start = t - half_window
        end = t + half_window + 1
        start_clamped = max(0, start)
        end_clamped = min(t_dim, end)

        for k in range(k_dim):
            window = array_2d[start_clamped:end_clamped, k]

            if s_dim == 0:
                sorted_slice, all_nans = _filter_and_sort_window(window)
                if all_nans:
                    for q_idx in range(q_dim):
                        out_arr[t, k, q_idx, 0] = np.nan
                else:
                    for q_idx in range(q_dim):
                        q = quantiles_arr[q_idx]
                        out_arr[t, k, q_idx, 0] = _quantile_interp(sorted_slice, q)
            else:
                for s_idx in range(s_dim):
                    start_rel, stop_rel = slices_arr[s_idx, 0], slices_arr[s_idx, 1]
                    final_window = _apply_relative_window_slice(
                        window, start_rel, stop_rel
                    )
                    sorted_slice, all_nans = _filter_and_sort_window(final_window)
                    if all_nans:
                        for q_idx in range(q_dim):
                            out_arr[t, k, q_idx, s_idx] = np.nan
                    else:
                        for q_idx in range(q_dim):
                            q = quantiles_arr[q_idx]
                            out_arr[t, k, q_idx, s_idx] = _quantile_interp(
                                sorted_slice, q
                            )
    return out_arr


@njit(parallel=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _numba_sliding_quantile_kernel(
    array_2d: np.ndarray,
    window_size: int,
    quantile: float,
    slices_arr: np.ndarray,  # Shape (S, 2), dtype=float64
) -> np.ndarray:
    """
    The core Numba kernel for a single quantile (centered).
    """
    t_dim = array_2d.shape[0]
    k_dim = array_2d.shape[1]
    s_dim = slices_arr.shape[0]
    half_window = window_size // 2

    if s_dim == 0:
        out_arr = np.empty((t_dim, k_dim, 1), dtype=array_2d.dtype)
    else:
        out_arr = np.empty((t_dim, k_dim, s_dim), dtype=array_2d.dtype)

    for t in prange(t_dim):
        start = t - half_window
        end = t + half_window + 1
        start_clamped = max(0, start)
        end_clamped = min(t_dim, end)

        for k in range(k_dim):
            window = array_2d[start_clamped:end_clamped, k]

            if s_dim == 0:
                sorted_slice, all_nans = _filter_and_sort_window(window)
                if all_nans:
                    out_arr[t, k, 0] = np.nan
                else:
                    out_arr[t, k, 0] = _quantile_interp(sorted_slice, quantile)
            else:
                for s_idx in range(s_dim):
                    start_rel, stop_rel = slices_arr[s_idx, 0], slices_arr[s_idx, 1]
                    final_window = _apply_relative_window_slice(
                        window, start_rel, stop_rel
                    )
                    sorted_slice, all_nans = _filter_and_sort_window(final_window)
                    if all_nans:
                        out_arr[t, k, s_idx] = np.nan
                    else:
                        out_arr[t, k, s_idx] = _quantile_interp(sorted_slice, quantile)
    return out_arr


@njit(parallel=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _numba_sliding_mean_kernel(
    array_2d: np.ndarray,
    window_size: int,
    slices_arr: np.ndarray,  # Shape (S, 2), dtype=float64
) -> np.ndarray:
    """
    The core Numba kernel for sliding mean (centered).
    """
    t_dim = array_2d.shape[0]
    k_dim = array_2d.shape[1]
    s_dim = slices_arr.shape[0]
    half_window = window_size // 2

    if s_dim == 0:
        out_arr = np.empty((t_dim, k_dim, 1), dtype=array_2d.dtype)
    else:
        out_arr = np.empty((t_dim, k_dim, s_dim), dtype=array_2d.dtype)

    for t in prange(t_dim):
        start = t - half_window
        end = t + half_window + 1
        start_clamped = max(0, start)
        end_clamped = min(t_dim, end)

        for k in range(k_dim):
            window = array_2d[start_clamped:end_clamped, k]

            if s_dim == 0:
                out_arr[t, k, 0] = _calc_mean_on_window(window)
            else:
                for s_idx in range(s_dim):
                    start_rel, stop_rel = slices_arr[s_idx, 0], slices_arr[s_idx, 1]
                    final_window = _apply_relative_window_slice(
                        window, start_rel, stop_rel
                    )
                    out_arr[t, k, s_idx] = _calc_mean_on_window(final_window)
    return out_arr


@njit(parallel=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _numba_sliding_min_kernel(
    array_2d: np.ndarray,
    window_size: int,
    slices_arr: np.ndarray,  # Shape (S, 2), dtype=float64
) -> np.ndarray:
    """
    The core Numba kernel for sliding min (centered).
    """
    t_dim = array_2d.shape[0]
    k_dim = array_2d.shape[1]
    s_dim = slices_arr.shape[0]
    half_window = window_size // 2

    if s_dim == 0:
        out_arr = np.empty((t_dim, k_dim, 1), dtype=array_2d.dtype)
    else:
        out_arr = np.empty((t_dim, k_dim, s_dim), dtype=array_2d.dtype)

    for t in prange(t_dim):
        start = t - half_window
        end = t + half_window + 1
        start_clamped = max(0, start)
        end_clamped = min(t_dim, end)

        for k in range(k_dim):
            window = array_2d[start_clamped:end_clamped, k]

            if s_dim == 0:
                out_arr[t, k, 0] = _calc_min_on_window(window)
            else:
                for s_idx in range(s_dim):
                    start_rel, stop_rel = slices_arr[s_idx, 0], slices_arr[s_idx, 1]
                    final_window = _apply_relative_window_slice(
                        window, start_rel, stop_rel
                    )
                    out_arr[t, k, s_idx] = _calc_min_on_window(final_window)
    return out_arr


# ######################################################################
# ## 6. Sliding Max
# ######################################################################


@njit(parallel=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _numba_sliding_max_kernel(
    array_2d: np.ndarray,
    window_size: int,
    slices_arr: np.ndarray,  # Shape (S, 2), dtype=float64
) -> np.ndarray:
    """
    The core Numba kernel for sliding max (centered).
    """
    t_dim = array_2d.shape[0]
    k_dim = array_2d.shape[1]
    s_dim = slices_arr.shape[0]
    half_window = window_size // 2

    if s_dim == 0:
        out_arr = np.empty((t_dim, k_dim, 1), dtype=array_2d.dtype)
    else:
        out_arr = np.empty((t_dim, k_dim, s_dim), dtype=array_2d.dtype)

    for t in prange(t_dim):
        start = t - half_window
        end = t + half_window + 1
        start_clamped = max(0, start)
        end_clamped = min(t_dim, end)

        for k in range(k_dim):
            window = array_2d[start_clamped:end_clamped, k]

            if s_dim == 0:
                out_arr[t, k, 0] = _calc_max_on_window(window)
            else:
                for s_idx in range(s_dim):
                    start_rel, stop_rel = slices_arr[s_idx, 0], slices_arr[s_idx, 1]
                    final_window = _apply_relative_window_slice(
                        window, start_rel, stop_rel
                    )
                    out_arr[t, k, s_idx] = _calc_max_on_window(final_window)
    return out_arr


def _sliding_metric_wrapper(
    kernel_func: Callable[..., np.ndarray],
    array: np.ndarray,
    window_size: int,
    window_slice: None | slice | Iterable[slice],
    *,
    kernel_args: tuple[Any, ...],  # pyright: ignore[reportExplicitAny]
    is_plural_metric: bool,
) -> np.ndarray:
    """
    A reusable wrapper to handle the boilerplate of sliding metric calculations.

    This function performs:
    1. Input validation for the array and window_size.
    2. Compilation of user-friendly slices to relative float arrays.
    3. Reshaping of the input array from (T, ...) to (T, K).
    4. Invocation of the specialized Numba kernel.
    5. Reshaping of the output array to match the expected final shape.
    """
    # 1. Input Validation
    if array.ndim < 2:
        raise ValueError("Input array must be at least 2-dimensional (T, ...)")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for a centered window")

    # 2. Slice Compilation
    slices_arr = _compile_slices_to_relative(window_slice, window_size)
    s_dim = slices_arr.shape[0]

    # 3. Input Array Reshaping
    original_shape = array.shape
    t_dim = original_shape[0]
    other_dims = original_shape[1:]
    k_dim = np.prod(other_dims).item() if other_dims else 1
    array_2d = array.reshape(t_dim, k_dim)

    # 4. Invoke the Numba Kernel
    # The kernel expects arguments in the order: (array, window, *specific_args, slices)
    out_arr = kernel_func(array_2d, window_size, *kernel_args, slices_arr)

    # 5. Reshape Output
    if s_dim == 0:
        # No slices were used, the kernel added a singleton dimension we must remove.
        if is_plural_metric:
            # e.g., for quantiles, shape is (T, ..., Q)
            q_dim = kernel_args[0].shape[0]
            final_out_shape = (t_dim,) + other_dims + (q_dim,)
        else:
            # e.g., for mean, shape is just (T, ...)
            final_out_shape = original_shape
    else:
        # Slices were used, so an S dimension is added at the end.
        if is_plural_metric:
            # e.g., for quantiles, shape is (T, ..., Q, S)
            q_dim = kernel_args[0].shape[0]
            final_out_shape = (t_dim,) + other_dims + (q_dim, s_dim)
        else:
            # e.g., for mean, shape is (T, ..., S)
            final_out_shape = original_shape + (s_dim,)

    return out_arr.reshape(final_out_shape)


def sliding_quantiles(
    array: np.ndarray,
    window_size: int,
    quantiles: Iterable[float],
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate sliding quantiles of an array using a centered window.
    Output shape is (T, ..., Q) or (T, ..., Q, S) if slices are provided.

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        quantiles: The quantiles to calculate (e.g., [0.25, 0.5, 0.75]).
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ..., Q) or (T, ..., Q, S).
    """
    quantiles_arr = np.array(list(quantiles), dtype=np.float64)
    return _sliding_metric_wrapper(
        _numba_sliding_quantiles_kernel,
        array,
        window_size,
        window_slice,
        kernel_args=(quantiles_arr,),
        is_plural_metric=True,
    )


def sliding_quantile(
    array: np.ndarray,
    window_size: int,
    quantile: float,
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate a single sliding quantile of an array using a centered window.
    Output shape is (T, ...) or (T, ..., S) if slices are provided.

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        quantile: The quantile to calculate (float, e.g., 0.5).
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ...) or (T, ..., S).
    """
    return _sliding_metric_wrapper(
        _numba_sliding_quantile_kernel,
        array,
        window_size,
        window_slice,
        kernel_args=(quantile,),
        is_plural_metric=False,
    )


def sliding_mean(
    array: np.ndarray,
    window_size: int,
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate the sliding mean of an array using a centered window.
    Output shape is (T, ...) or (T, ..., S) if slices are provided.

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ...) or (T, ..., S).
    """
    return _sliding_metric_wrapper(
        _numba_sliding_mean_kernel,
        array,
        window_size,
        window_slice,
        kernel_args=(),
        is_plural_metric=False,
    )


def sliding_median(
    array: np.ndarray,
    window_size: int,
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate the sliding median of an array using a centered window.
    Output shape is (T, ...) or (T, ..., S) if slices are provided.

    This is a convenience wrapper for sliding_quantile(..., quantile=0.5).

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ...) or (T, ..., S).
    """
    # This function is already a clean wrapper, so it remains unchanged.
    return sliding_quantile(array, window_size, quantile=0.5, window_slice=window_slice)


def sliding_min(
    array: np.ndarray,
    window_size: int,
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate the sliding minimum of an array using a centered window.
    Output shape is (T, ...) or (T, ..., S) if slices are provided.

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ...) or (T, ..., S).
    """
    return _sliding_metric_wrapper(
        _numba_sliding_min_kernel,
        array,
        window_size,
        window_slice,
        kernel_args=(),
        is_plural_metric=False,
    )


def sliding_max(
    array: np.ndarray,
    window_size: int,
    *,
    window_slice: None | slice | Iterable[slice] = None,
) -> np.ndarray:
    """
    Calculate the sliding maximum of an array using a centered window.
    Output shape is (T, ...) or (T, ..., S) if slices are provided.

    Parameters:
        array: The input array (T, ...). Must be at least 2D.
        window_size: The size of the sliding window (int). Must be odd.
        window_slice: An optional slice, or iterable of slices, to apply
                      *within* each window. The slice is interpreted relative
                      to the `window_size` and is proportionally applied to
                      partial windows at the array edges.

    Returns:
        np.ndarray: The result with shape (T, ...) or (T, ..., S).
    """
    return _sliding_metric_wrapper(
        _numba_sliding_max_kernel,
        array,
        window_size,
        window_slice,
        kernel_args=(),
        is_plural_metric=False,
    )

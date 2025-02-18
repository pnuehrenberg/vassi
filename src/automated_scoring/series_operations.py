from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from . import math, sliding_metrics, utils


def sample_1d(
    array: NDArray, timestamps_array: NDArray, timestamps: NDArray, *, keep_dtype: bool
) -> NDArray:
    """
    Sample a 1D array at the specified timestamps using linear interpolation.

    Parameters
    ----------
    array: NDArray
        The array to sample.
    timestamps_array: NDArray
        The array of timestamps.
    timestamps: NDArray
        The timestamps to sample at.
    keep_dtype: bool
        Whether to keep the data type of the array.

    Returns
    -------
    NDArray
        The sampled array.
    """
    dtype = array.dtype
    interpolated = np.interp(timestamps, timestamps_array, array)
    if keep_dtype:
        return interpolated.astype(dtype)
    return interpolated


def sign_change_latency_1d(array: NDArray) -> NDArray:
    """
    Return the sign change latency of a 1D array.

    Sign change latency is the number of array values with the same sign after the previous sign change.

    Parameters
    ----------
    array: NDArray
        The array to calculate the sign change latency of.

    Returns
    -------
    NDArray
        The sign change latency of the array.
    """
    sign_change_idx = (
        np.argwhere(np.sign(array) != np.sign(math.shift(array, 1))).ravel() + 1
    )
    sign_change_idx[1:] -= sign_change_idx[:-1]
    sign_change_idx = sign_change_idx.tolist()
    last_change = 0
    if len(sign_change_idx) > 0:
        last_change = np.sum(sign_change_idx)
    sign_change_idx += [len(array) - last_change]
    return np.concatenate([np.arange(change) for change in sign_change_idx])


def filter_sliding_quantile_range_1d(
    array: NDArray, window_size: int, q_lower: float, q_upper: float
) -> NDArray:
    """
    Apply a sliding quantile filter to a 1D array.

    Parameters
    ----------
    array: NDArray
        The array to filter.
    window_size: int
        The size of the sliding window used to calculate the quantiles.
    q_lower: float
        The lower quantile of the filter.
    q_upper: float
        The upper quantile of the filter.

    Returns
    -------
    NDArray
        The filtered array.

    Raises
    ------
    ValueError
        If no values are within the quantile range.
    """
    quantiles = sliding_metrics.sliding_quantiles(
        array, window_size, (q_lower, q_upper)
    )
    in_quantile_range = (array >= quantiles[:, 0]) & (array <= quantiles[:, -1])
    idx = np.argwhere(in_quantile_range).ravel()
    if idx.size == 0:
        raise ValueError("at least one value required within quantile range")
    return sample_1d(array[idx], idx, np.arange(array.size), keep_dtype=True)


def smooth_1d(
    array: NDArray, filter_funcs: Iterable[utils.SmoothingFunction]
) -> NDArray:
    """
    Apply multiple filter functions to a 1D array.

    Each filter function should take a 1D array as input and return a 1D array.

    Parameters
    ----------
    array: NDArray
        The array to filter.
    filter_funcs: Iterable[utils.NDArray_to_NDArray]
        The filter functions to apply.

    Returns
    -------
    NDArray
        The filtered array.
    """
    for filter_func in filter_funcs:
        array = filter_func(array=array)
    return array


def sample(
    series: NDArray,
    timestamps_series: NDArray,
    timestamps: NDArray,
    *,
    keep_dtype: bool,
):
    """
    Sample a series at the specified timestamps using linear interpolation.

    Parameters
    ----------
    series: NDArray
        The series to sample.
    timestamps_series: NDArray
        The timestamps of the series.
    timestamps: NDArray
        The timestamps to sample at.
    keep_dtype: bool
        Whether to keep the data type of the series.

    Returns
    -------
    NDArray
        The sampled series.

    See also
    --------
    sample_1d
    """
    sampled = np.apply_along_axis(
        sample_1d,
        0,
        series,
        timestamps_array=timestamps_series,
        timestamps=timestamps,
        keep_dtype=keep_dtype,
    )
    if keep_dtype:
        return sampled.astype(series.dtype)
    return sampled


def sign_change_latency(series: NDArray):
    """
    Return the sign change latency of a series.

    Parameters
    ----------
    series: NDArray
        The series to calculate the sign change latency of.

    Returns
    -------
    NDArray
        The sign change latency of the series.

    See also
    --------
    sign_change_latency_1d
    """
    return np.apply_along_axis(
        sign_change_latency_1d,
        0,
        series,
    ).astype(series.dtype)


def filter_sliding_quantile_range(
    series: NDArray, window_size: int, q_lower: float, q_upper: float, copy: bool = True
):
    """
    Apply a sliding quantile filter to a series.

    Parameters
    ----------
    series: NDArray
        The series to filter.
    window_size: int
        The size of the sliding window used to calculate the quantiles.
    q_lower: float
        The lower quantile of the filter.
    q_upper: float
        The upper quantile of the filter.
    copy: bool
        Whether to copy the series before filtering.

    Returns
    -------
    NDArray
        The filtered series.

    See also
    --------
    filter_sliding_quantile_range_1d
    """
    if copy:
        series = deepcopy(series)
    return np.apply_along_axis(
        filter_sliding_quantile_range_1d,
        0,
        series,
        window_size=window_size,
        q_lower=q_lower,
        q_upper=q_upper,
    )


def smooth(
    series: NDArray,
    filter_funcs: Iterable[utils.SmoothingFunction],
    copy: bool = True,
) -> NDArray:
    """
    Apply multiple filter functions to a series.

    Parameters
    ----------
    series: NDArray
        The series to filter.
    filter_funcs: Iterable[utils.NDArray_to_NDArray]
        The filter functions to apply.
    copy: bool
        Whether to copy the series before filtering.

    Returns
    -------
    NDArray
        The filtered series.

    See also
    --------
    smooth_1d
    """
    if copy:
        series = deepcopy(series)
    return np.apply_along_axis(
        smooth_1d,
        0,
        series,
        filter_funcs=filter_funcs,
    )

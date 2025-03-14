from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from . import math, sliding_metrics, utils


def sample_1d(
    array: np.ndarray, timestamps_array: np.ndarray, timestamps: np.ndarray, *, keep_dtype: bool
) -> np.ndarray:
    dtype = array.dtype
    interpolated = np.interp(timestamps, timestamps_array, array)
    if keep_dtype:
        return interpolated.astype(dtype)
    return interpolated


def sign_change_latency_1d(array: np.ndarray) -> np.ndarray:
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
    array: np.ndarray, window_size: int, q_lower: float, q_upper: float
) -> np.ndarray:
    quantiles = sliding_metrics.sliding_quantiles(
        array, window_size, (q_lower, q_upper)
    )
    in_quantile_range = (array >= quantiles[:, 0]) & (array <= quantiles[:, -1])
    idx = np.argwhere(in_quantile_range).ravel()
    if idx.size == 0:
        raise ValueError("at least one value required within quantile range")
    return sample_1d(array[idx], idx, np.arange(array.size), keep_dtype=True)


def smooth_1d(
    array: np.ndarray, filter_funcs: Iterable[utils.SmoothingFunction]
) -> np.ndarray:
    for filter_func in filter_funcs:
        array = filter_func(array=array)
    return array


def sample(
    series: np.ndarray,
    timestamps_series: np.ndarray,
    timestamps: np.ndarray,
    *,
    keep_dtype: bool,
):
    """
    Samples a time series at specified timestamps.

    This function interpolates values from a time series at given timestamps. It uses linear interpolation to estimate values between the known data points.

    Parameters
    ----------
    series : numpy.ndarray
        The time series data to sample.
    timestamps_series : numpy.ndarray
        The timestamps corresponding to the time series data.
    timestamps : numpy.ndarray
        The timestamps at which to sample the time series.
    keep_dtype : bool
        Whether to preserve the original data type of the series. Defaults to False.

    Returns
    -------
    numpy.ndarray
        The sampled time series data.
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


def sign_change_latency(series: np.ndarray):
    return np.apply_along_axis(
        sign_change_latency_1d,
        0,
        series,
    ).astype(series.dtype)


def filter_sliding_quantile_range(
    series: np.ndarray, window_size: int, q_lower: float, q_upper: float, copy: bool = True
):
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
    series: np.ndarray,
    filter_funcs: Iterable[utils.SmoothingFunction],
    copy: bool = True,
) -> np.ndarray:
    if copy:
        series = deepcopy(series)
    return np.apply_along_axis(
        smooth_1d,
        0,
        series,
        filter_funcs=filter_funcs,
    )

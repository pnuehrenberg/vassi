from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

import pyTrajectory.math as math
import pyTrajectory.sliding_metrics as sliding_metrics

from . import utils


def sample_1d(
    array: NDArray, timestamps_array: NDArray, timestamps: NDArray, *, keep_dtype: bool
) -> NDArray:
    dtype = array.dtype
    interpolated = np.interp(timestamps, timestamps_array, array)
    if keep_dtype:
        return interpolated.astype(dtype)
    return interpolated


def sign_change_latency_1d(array: NDArray) -> NDArray:
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
    quantiles = sliding_metrics.sliding_quantiles(
        array, window_size, (q_lower, q_upper)
    )
    in_quantile_range = (array >= quantiles[:, 0]) & (array <= quantiles[:, -1])
    idx = np.argwhere(in_quantile_range).ravel()
    if idx.size == 0:
        raise ValueError("at least one value required within quantile range")
    return sample_1d(array[idx], idx, np.arange(array.size), keep_dtype=True)


def smooth_1d(array: NDArray, filter_funcs: list[utils.NDArray_to_NDArray]) -> NDArray:
    for filter_func in filter_funcs:
        array = filter_func(array)
    return array


def sample(
    series: NDArray,
    timestamps_series: NDArray,
    timestamps: NDArray,
    *,
    keep_dtype: bool,
):
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
    return np.apply_along_axis(
        sign_change_latency_1d,
        0,
        series,
    ).astype(series.dtype)


def filter_sliding_quantile_range(
    series: NDArray, window_size: int, q_lower: float, q_upper: float, copy: bool = True
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
    series: NDArray,
    filter_funcs: list[utils.NDArray_to_NDArray],
    copy: bool = True,
) -> NDArray:
    if copy:
        series = deepcopy(series)
    return np.apply_along_axis(
        smooth_1d,
        0,
        series,
        filter_funcs=filter_funcs,
    )
    # from scipy.signal import savgol_filter, medfilt
    # s = medfilt(s, median_filter_window_size)
    # s = savgol_filter(s, savgol_filter_window_size, savgol_filter_polyorder)
    # if force_positive:
    #     series[series < 0] = 0

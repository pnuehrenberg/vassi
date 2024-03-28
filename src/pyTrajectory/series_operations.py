import numpy as np
from scipy.signal import savgol_filter, medfilt
from copy import deepcopy

import pyTrajectory.series_math
# from .series_math import calculate_element_wise_magnitude


def get_sliding_quantiles(series, quantiles, window_size):
    series_sliding_windows = np.lib.stride_tricks.sliding_window_view(series, window_size)
    sliding_quantiles = np.quantile(series_sliding_windows, quantiles, axis=1).T
    sliding_quantiles = np.concatenate([
        np.tile(sliding_quantiles[0], window_size // 2).reshape(-1, len(quantiles)),
        sliding_quantiles,
        np.tile(sliding_quantiles[-1], window_size // 2).reshape(-1, len(quantiles))], axis=0)
    return sliding_quantiles


def filter_series(series, lower=0.05, upper=0.95, window_size=61, copy=True):
    if copy:
        series = deepcopy(series)
    shape = series.shape
    if len(shape) == 1:
        series = series[:, np.newaxis]
    for idx in np.ndindex(series.shape[1:]):
        s = series[(slice(None), *idx)]
        if len(s) < window_size:
            continue
        quantiles = get_sliding_quantiles(s, [lower, upper], window_size)
        condition = (s >= quantiles[:, 0]) & (s <= quantiles[:, -1])
        s[~condition] = np.nan
        series[(slice(None), *idx)] = s
    return series.reshape(shape)


def smooth_series(series,
                  use_median_filter=True,
                  median_filter_window_size=5,
                  use_savgol_filter=True,
                  savgol_filter_window_size=5,
                  savgol_filter_polyorder=1,
                  force_positive=False,
                  copy=True):
    if copy:
        series = deepcopy(series)
    shape = series.shape
    if len(shape) == 1:
        series = series[:, np.newaxis]
    for idx in np.ndindex(series.shape[1:]):
        s = series[(slice(None), *idx)]
        indices_finite = np.argwhere(np.isfinite(s)).ravel()
        if indices_finite.size == 0:
            continue
        length = len(s)
        s = s[indices_finite]
        if use_median_filter and len(s) > median_filter_window_size:
            s = medfilt(s, median_filter_window_size)
        if use_savgol_filter and len(s) > savgol_filter_window_size:
            s = savgol_filter(
                s, savgol_filter_window_size, savgol_filter_polyorder)
        s = np.interp(np.arange(length), indices_finite, s)
        series[(slice(None), *idx)] = s
    if force_positive:
        series[series < 0] = 0
    return series.reshape(shape)


def interpolate_series(series, time_stamps, time_stamps_target):
    series = deepcopy(series)
    shape = series.shape
    if len(shape) == 1:
        series = series[:, np.newaxis]
    series_interpolated = np.zeros((time_stamps_target.size, *series.shape[1:]))
    for idx in np.ndindex(series.shape[1:]):
        s = series[(slice(None), *idx)]
        series_interpolated[(slice(None), *idx)] = np.interp(time_stamps_target, time_stamps, s)
    return series_interpolated.reshape((time_stamps_target.size, *shape[1:]))

def sample_series(series, timestamps_series, timestamps):
    shape = series.shape
    if series.ndim == 1:
        series = series[:, np.newaxis]
    series_interpolated = np.zeros((timestamps.size, *series.shape[1:]), dtype=series.dtype)
    for idx in np.ndindex(series.shape[1:]):
        s = series[(slice(None), *idx)]
        series_interpolated[(slice(None), *idx)] = np.interp(timestamps, timestamps_series, s)
    return series_interpolated.reshape((timestamps.size, *shape[1:]))

def get_sliding_cumulative_distance(series, window_size):
    # this only makes sense on coordinate series
    series_sliding_windows = np.lib.stride_tricks.sliding_window_view(series, window_size, axis=0)
    sliding_distance = pyTrajectory.series_math.calculate_element_wise_magnitude(np.swapaxes(np.diff(series_sliding_windows, axis=-1), -1, -2)).sum(axis=-1)
    sliding_distance = np.concatenate([
        np.tile(sliding_distance[0], window_size // 2).reshape(window_size // 2, *sliding_distance.shape[1:]),
        sliding_distance,
        np.tile(sliding_distance[-1], window_size // 2).reshape(window_size // 2, *sliding_distance.shape[1:])], axis=0)
    return sliding_distance


def get_sliding_distance(series, window_size):
    # this only makes sense on coordinate series
    series_sliding_windows = np.lib.stride_tricks.sliding_window_view(series, window_size, axis=0)
    sliding_distance = pyTrajectory.series_math.calculate_element_wise_magnitude(series_sliding_windows[..., 0] - series_sliding_windows[..., -1])
    sliding_distance = np.concatenate([
        np.tile(sliding_distance[0], window_size // 2).reshape(window_size // 2, *sliding_distance.shape[1:]),
        sliding_distance,
        np.tile(sliding_distance[-1], window_size // 2).reshape(window_size // 2, *sliding_distance.shape[1:])], axis=0)
    return sliding_distance

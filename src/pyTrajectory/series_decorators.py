import numpy as np
from decorator import decorator

import pyTrajectory.series_operations  # avoid circular import
import pyTrajectory.series_math  # avoid circular import


def raw(func):
    if hasattr(func, '__wrapped__'):
        return raw(func.__wrapped__)
    return func


@decorator
def filter_output(func,
                  lower=0.05,
                  upper=0.95,
                  window_size=61,
                  *args,
                  **kwargs):
    series = func(*args, **kwargs)
    return pyTrajectory.series_operations.filter_series(series,
                                                        lower,
                                                        upper,
                                                        window_size)


@decorator
def absolute_output(func, *args, **kwargs):
    values = func(*args, **kwargs)
    return np.absolute(values)


@decorator
def norm_output(func, *args, **kwargs):
    values = func(*args, **kwargs)
    return pyTrajectory.series_math.calculate_unit_vectors(values)


@decorator
def smooth_output(func,
                  use_median_filter=True,
                  median_filter_window_size=5,
                  use_savgol_filter=True,
                  savgol_filter_window_size=5,
                  savgol_filter_polyorder=1,
                  force_positive=False,
                  *args, **kwargs):
    series = func(*args, **kwargs)
    return pyTrajectory.series_operations.smooth_series(
        series,
        use_median_filter=use_median_filter,
        median_filter_window_size=median_filter_window_size,
        use_savgol_filter=use_savgol_filter,
        savgol_filter_window_size=savgol_filter_window_size,
         savgol_filter_polyorder=savgol_filter_polyorder,
        force_positive=force_positive)

@decorator
def norm_input(func, *args, **kwargs):
    args = [pyTrajectory.series_math.calculate_unit_vectors(arg)
            for arg in args]
    return func(*args, **kwargs)

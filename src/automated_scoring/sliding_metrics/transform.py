from typing import TYPE_CHECKING, Iterable, Optional, Self, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, validate_data

from ..logging import set_logging_level
from ..utils import NDArray_to_NDArray, closest_odd_divisible, flatten
from .sliding_metrics import apply_multiple_to_sliding_windows


@overload
def get_window_slices(
    num_windows_per_scale: int,
    *,
    time_scales: Iterable[int],
) -> tuple[list[int], list[slice]]: ...


@overload
def get_window_slices(
    num_windows_per_scale: int,
    *,
    durations: NDArray,
    time_scale_quantiles: Iterable[float],
) -> tuple[list[int], list[slice]]: ...


def get_window_slices(
    num_windows_per_scale: int,
    *,
    time_scales: Optional[Iterable[int]] = None,
    durations: Optional[NDArray] = None,
    time_scale_quantiles: Optional[Iterable[float]] = None,
) -> tuple[list[int], list[slice]]:
    """
    Find consecutive window slices for time scales, either explicitly specified or derived from durations and quantiles.

    Args:
        num_windows_per_scale: Number of windows per time scale.
        time_scales: Explicit time scales.
        durations: Durations to calculate time scales from.
        time_scale_quantiles: Quantiles of the durations to derive time scales.

    Raises:
        ValueError: If neither :code:`time_scales` nor :code:`durations` and :code:`time_scale_quantiles` are specified.
        ValueError: If :code:`time_scale_quantiles` are specified but :code:`durations` are not.
    """
    if time_scales is None and durations is not None:
        if time_scale_quantiles is not None:
            time_scales = (
                np.quantile(durations, tuple(time_scale_quantiles)).astype(int).tolist()
            )
        else:
            raise ValueError("Specify time_scale_quantiles.")
    if time_scales is None:
        raise ValueError(
            "Specify either time_scales or durations and time_scale_quantiles"
        )
    time_scales_adjusted = [
        closest_odd_divisible(scale, num_windows_per_scale) for scale in time_scales
    ]
    if set(time_scales) != set(time_scales_adjusted):
        set_logging_level().warning(
            f"Time scales adjusted to match num_windows_per_scale: {time_scales} -> {time_scales_adjusted}."
        )
    time_scales = time_scales_adjusted
    window_slices = []
    max_time_scale = max(time_scales)
    for time_scale in time_scales:
        window_size = time_scale // num_windows_per_scale
        padding = (max_time_scale - time_scale) // 2
        for window_idx in range(num_windows_per_scale):
            start = padding + window_idx * window_size
            stop = start + window_size
            window_slices.append(slice(start, stop))
    return time_scales, window_slices


class SlidingWindowAggregator(BaseEstimator, TransformerMixin):
    """
    Sliding window aggregator for time series data.

    Args:
        metric_funcs: List of functions to apply to each window.
        window_size: Size of the sliding window.
        window_slices: List of slices to use for each window.
    """
    def __init__(
        self,
        metric_funcs: list[NDArray_to_NDArray],
        window_size: int | Iterable[int],
        window_slices: list[slice] | None = None,
    ):
        self.metric_funcs = metric_funcs
        if isinstance(window_size, int):
            self.window_size = window_size
        else:
            self.window_size = max(window_size)
        self.window_slices = window_slices

    def fit(self, X: NDArray | pd.DataFrame, y: None = None) -> Self:
        """
        This method is required by the sklearn API and does not perform any actual fitting.

        Args:
            X: Ignored.
            y: Ignored.
        """
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """
        Transform the input data by applying the metric functions to sliding windows.
        The transformed data is returned as a 2D array, flattened along all axes except the first.

        Args:
            X: Input data to transform.
        """
        # X is not typed in pandas
        X_numpy = validate_data(
            self,
            X,  # type: ignore
        )
        if TYPE_CHECKING:
            assert isinstance(X_numpy, np.ndarray)
        return flatten(
            apply_multiple_to_sliding_windows(
                X_numpy,
                self.window_size,
                self.metric_funcs,
                slices=self.window_slices,
            )
        )

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        """
        Get output feature names for transformation.


        Args:
            input_features: Input feature names.
        """
        # https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/preprocessing/_polynomial.py#L99
        input_features = _check_feature_names_in(self, input_features)
        if input_features is None:
            raise ValueError
        feature_names = []
        selection = (
            self.window_slices
            if isinstance(self.window_slices, list)
            else [
                self.window_slices,
            ]
        )
        for feature_name in input_features:
            for selection_slice in selection:
                for aggregation_func in self.metric_funcs:
                    name = f"{feature_name}-{aggregation_func.__name__}"
                    if selection_slice is not None:
                        start = selection_slice.start
                        if start is not None:
                            start -= self.window_size // 2
                        stop = selection_slice.stop
                        if stop is not None:
                            stop -= self.window_size // 2
                        name = f"{name}({start}:{stop})"
                    feature_names.append(name)
        return np.asarray(feature_names, dtype=object)

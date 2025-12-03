import hashlib
from collections.abc import Callable, Iterable
from functools import partial
from typing import Self, final, overload, override

import numpy as np
import pandas as pd
import sklearn.utils.validation as sklearn_validation
from sklearn.base import BaseEstimator, TransformerMixin

from .._utils import get_inner
from ..utils import closest_odd_divisible
from ..warnings import warn


def apply_multiple_to_sliding_windows(
    array: np.ndarray,
    window_size: int,
    funcs: Iterable[Callable[..., np.ndarray] | tuple[Callable[..., np.ndarray], int]],
    slices: slice | Iterable[slice] | None = None,
    *,
    indices: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply multiple functions to sliding windows of an array.

    Parameters:
        array: The input array.
        window_size: The size of the sliding window.
        funcs: The functions to apply.
        slices: The slices to apply the functions to (slicing the moving window).
    """
    funcs = list(funcs)
    if isinstance(slices, Iterable):
        slices = list(slices)
    result_shape = (
        array.shape[0] if indices is None else len(indices),
        *array.shape[1:],
        sum((func[1] if isinstance(func, tuple) else 1) for func in funcs),
    )
    if slices is not None:
        result_shape += (1 if isinstance(slices, slice) else len(slices),)
    else:
        result_shape += (1,)
    if out is None:
        out = np.zeros(result_shape)
    elif out.shape != result_shape:
        raise ValueError(f"Expected output shape {result_shape}, but got {out.shape}")
    feature_idx = 0
    for func_idx in range(len(funcs)):
        _func = funcs[func_idx]
        if isinstance(_func, tuple):
            func, num_features = _func
        else:
            func = _func
            num_features = 1
        func = partial(func, window_size=window_size, window_slice=slices)
        result = func(array)
        if indices is not None:
            result = result[indices]
        if num_features == 1:
            if slices is None:
                out[..., feature_idx : feature_idx + num_features] = result[
                    ..., np.newaxis, np.newaxis
                ]
            else:
                out[..., feature_idx : feature_idx + num_features, :] = result[
                    ..., np.newaxis, :
                ]
        else:
            if slices is None:
                out[..., feature_idx : feature_idx + num_features] = result[
                    ..., np.newaxis
                ]
            else:
                out[..., feature_idx : feature_idx + num_features, :] = result
        feature_idx += num_features
    return out


@overload
def get_window_slices(
    num_slices_per_window: int,
    *,
    windows: Iterable[int],
) -> tuple[list[int], list[slice]]: ...


@overload
def get_window_slices(
    num_slices_per_window: int,
    *,
    durations: np.ndarray,
    duration_quantiles: Iterable[float],
) -> tuple[list[int], list[slice]]: ...


def get_window_slices(
    num_slices_per_window: int,
    *,
    windows: Iterable[float] | None = None,
    durations: np.ndarray | None = None,
    duration_quantiles: Iterable[float] | None = None,
) -> tuple[list[int], list[slice]]:
    """
    Find consecutive window slices for time scales, either explicitly specified or derived from durations and quantiles.

    Parameters:
        num_slices_per_window: Number of slices per time window.
        windows: Explicit time windows.
        durations: Durations to calculate time scales from.
        duration_quantiles: Quantiles of the durations to derive time windows.

    Returns:
        A tuple containing the (adjusted) time scales and the corresponding window slices.

    Raises:
        ValueError: If neither :code:`windows` nor :code:`durations` and :code:`duration_quantiles` are specified.
        ValueError: If :code:`duration_quantiles` are specified but :code:`durations` are not.
    """
    if windows is None:
        if durations is None or duration_quantiles is None:
            raise ValueError(
                "Specify either windows or durations and duration_quantiles"
            )
        windows = [
            float(window)
            for window in np.quantile(durations, tuple(duration_quantiles))
        ]
    windows_adjusted = [
        closest_odd_divisible(scale, num_slices_per_window) for scale in windows
    ]
    if set(windows) != set(windows_adjusted):
        warn(
            f"Time scales adjusted to match num_slices_per_window: {windows} -> {windows_adjusted}."
        )
    windows = windows_adjusted
    window_slices: list[slice] = []
    max_window = max(windows)
    for window in windows:
        window_size = window // num_slices_per_window
        padding = (max_window - window) // 2
        for window_idx in range(num_slices_per_window):
            start = padding + window_idx * window_size
            stop = start + window_size
            window_slices.append(slice(start, stop))
    return windows, window_slices


@final
class SlidingWindowAggregator(BaseEstimator, TransformerMixin):
    """
    Sliding window aggregator for time series data.

    Parameters:
        metric_funcs: List of functions to apply to each window.
        window_size: Size of the sliding window.
        window_slices: List of slices to use for each window.

    See also:
        :func:`get_window_slices` to obtain window size(s) and corresponding slices.
    """

    def __init__(
        self,
        metric_funcs: Iterable[
            Callable[..., np.ndarray] | tuple[Callable[..., np.ndarray], int]
        ],
        window_size: int,
        *,
        window_slices: Iterable[slice] | None,
        keep_original: bool,
    ):
        self.metric_funcs = metric_funcs
        self.window_size = window_size
        self.window_slices = list(window_slices) if window_slices is not None else None
        self.keep_original = keep_original
        self.num_transformations = sum(
            func[1] if isinstance(func, tuple) else 1 for func in self.metric_funcs
        )

    @property
    def sha1(self):
        return hashlib.sha1(self._feature_names_out(["f"])).hexdigest()

    @override
    def __eq__(self, other: ...) -> bool:
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def fit(self, X: np.ndarray | pd.DataFrame, y: None = None) -> Self:
        """
        This method is required by the sklearn API and does not perform any actual fitting.

        Parameters:
            X: Ignored.
            y: Ignored.
        """
        del X, y  # unused parameters
        return self

    def get_num_features_out(self, num_features_in: int) -> int:
        return (
            num_features_in if self.keep_original else 0
        ) + num_features_in * self.num_transformations * (
            1 if self.window_slices is None else len(self.window_slices)
        )

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        indices: np.ndarray | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform the input data by applying the metric functions to sliding windows.
        The transformed data is returned as a 2D array, flattened along all axes except the first.

        Parameters:
            X: Input data to transform.

        Returns:
            Transformed data as a 2D array.
        """
        if isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError("Indices must be a 1D array")
            if np.isdtype(indices.dtype, "bool"):
                indices = np.argwhere(indices).ravel()
            elif not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Indices must be of type int64")
        # scikit-learn is not fully typed
        X_validated = np.asarray(
            sklearn_validation.validate_data(  # pyright: ignore[reportUnknownMemberType]
                self,
                X,  # pyright: ignore[reportArgumentType]
                ensure_all_finite="allow-nan",
            )
        )
        num_samples, num_features_in = X_validated.shape
        window_slices = self.window_slices
        if window_slices is not None:
            window_slices = list(window_slices)
        shape = (
            num_samples if indices is None else len(indices),
            self.get_num_features_out(num_features_in),
        )
        if out is None:
            out = np.zeros(shape)
        elif out.shape != shape:
            raise ValueError(f"Expected output shape {shape}, but got {out.shape}")
        if self.keep_original:
            out[:, :num_features_in] = X_validated
        _ = apply_multiple_to_sliding_windows(
            X_validated,
            self.window_size,
            self.metric_funcs,
            slices=window_slices,
            indices=indices,
            out=(out[:, num_features_in:] if self.keep_original else out).reshape(
                out.shape[0],
                X_validated.shape[1],
                self.num_transformations,
                1 if window_slices is None else len(window_slices),
            ),
        )
        return out.reshape(out.shape[0], -1)

    def _get_feature_name(
        self, func_name: str, feature_name: str, selection_slice: slice | None
    ) -> str:
        if selection_slice is None:
            return f"{func_name}(w={self.window_size})-{feature_name}"
        start = selection_slice.start
        stop = selection_slice.stop
        return f"{func_name}(w={self.window_size}|{start if start is not None else ''}:{stop if stop is not None else ''})-{feature_name}"

    def _feature_names_out(self, input_features: Iterable[str]) -> np.ndarray:
        feature_names: list[str] = []
        selection = (
            self.window_slices
            if isinstance(self.window_slices, Iterable)
            else [
                self.window_slices,
            ]
        )
        if self.keep_original:
            feature_names.extend(input_features)
        for feature_name in input_features:
            for selection_slice in selection:
                for aggregation_func in self.metric_funcs:
                    if isinstance(aggregation_func, tuple):
                        aggregation_func, num_features = aggregation_func
                        params = None
                        if isinstance(aggregation_func, partial):
                            if (
                                len(aggregation_func.keywords) > 0
                                and isinstance(
                                    values := next(
                                        iter(aggregation_func.keywords.values())
                                    ),
                                    Iterable,
                                )
                                and len(values := list(values)) == num_features
                            ):
                                params = values
                        for feature_idx in range(num_features):
                            param = None
                            if params is not None:
                                param = params[feature_idx]
                                param = (
                                    f"{param:.2g}"
                                    if isinstance(param, (int, float))
                                    else str(param)
                                )
                            feature_names.append(
                                self._get_feature_name(
                                    f"{get_inner(aggregation_func).__name__}({feature_idx if param is None else param})",  # pyright: ignore[reportUnknownArgumentType]
                                    feature_name,
                                    selection_slice,
                                )
                            )
                        continue
                    feature_names.append(
                        self._get_feature_name(
                            get_inner(aggregation_func).__name__,
                            feature_name,
                            selection_slice,
                        )
                    )
        return np.asarray(feature_names, dtype=str)

    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        """
        Get output feature names for transformation.

        Parameters:
            input_features: Input feature names.

        Returns:
            Output feature names as a 1D array.
        """
        # https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/preprocessing/_polynomial.py#L99
        input_features = sklearn_validation._check_feature_names_in(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
            self, input_features
        )
        if input_features is None:
            raise ValueError
        return self._feature_names_out(input_features)

from abc import ABC
from collections.abc import Callable
from functools import partial
from typing import Literal, override

import polars as pl
from pydantic.dataclasses import dataclass

from .._results import AnnotatedDatasetClassification
from .optuna_utils import ParameterSpace, Params


def without_postprocessing(
    dataset_classification: AnnotatedDatasetClassification, **_: ...
) -> AnnotatedDatasetClassification:
    return dataset_classification.discretize()


def _scorer(
    result: AnnotatedDatasetClassification,
    *,
    on: Literal["timestamp", "prediction", "annotation"] | None,
    foreground_only: bool,
) -> float:
    if on is None:
        scores = result.score()
    else:
        scores = result.f1_score(on=on)
    if foreground_only:
        scores = scores.filter(pl.col("category").is_in(result.foreground_categories))
    score = scores.get_column("score").mean()
    if not isinstance(score, float):
        raise ValueError(f"Expected float score, got {score}")
    return score


def _make_scorer(
    on: Literal["timestamp", "prediction", "annotation"] | None = None,
    foreground_only: bool = False,
) -> Callable[[AnnotatedDatasetClassification], float]:
    return partial(_scorer, on=on, foreground_only=foreground_only)


macro_f1_all_levels = _make_scorer()
macro_f1_foreground_all_levels = _make_scorer(foreground_only=True)
macro_f1_timestamp = _make_scorer(on="timestamp")
macro_f1_foreground_timestamp = _make_scorer(on="timestamp", foreground_only=True)


@dataclass
class ParamsFeatureAggregator(Params):
    sliding_metric: Literal["mean", "median"]
    window_size: int
    num_slices_per_window: int
    keep_original_features: bool

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        _ = space.suggest_categorical("sliding_metric", ["mean", "median"])
        _ = space.suggest_int("window_size", 11, 61, step=2)
        _ = space.suggest_int("num_slices_per_window", 1, 7, step=2)
        _ = space.suggest_bool("keep_original_features")


@dataclass
class ParamsQuantileRangeFilter(Params):
    window_size_lower: int
    window_size_upper: int
    lower: float
    upper: float

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        _ = space.suggest_int("window_size_lower", 3, 91, 2)
        _ = space.suggest_int("window_size_upper", 3, 91, 2)
        lower = space.suggest_float("lower", 0.0, 1.0)
        _ = space.suggest_float("upper", float(lower), 1.0)


@dataclass
class ParamsSmoothing(Params):
    sliding_metric: Literal["mean", "median"]
    window_size: int

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        _ = space.suggest_categorical("sliding_metric", ["mean", "median"])
        _ = space.suggest_int("window_size", 3, 91, 2)


@dataclass
class ParamsCategoryOutput(Params):
    use_quantile_range_filter: bool
    use_smoothing: bool
    use_thresholding: bool
    params_quantile_range_filter: ParamsQuantileRangeFilter | None = None
    params_smoothing: ParamsSmoothing | None = None
    threshold: float | None = None

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        use_quantile_range_filter = space.suggest_bool("use_quantile_range_filter")
        # by using .choice() on the nested subspaces, tell optuna to treat these parameters seperately depending on their parents' suggested values
        with use_quantile_range_filter.choice() as subspace:
            if use_quantile_range_filter:
                space.assign(
                    "params_quantile_range_filter",
                    ParamsQuantileRangeFilter.init_from(subspace),
                )
            use_smoothing = subspace.suggest_bool("use_smoothing")
            space.assign(
                "use_smoothing", use_smoothing.value
            )  # reuse the result value in this Params object
            with use_smoothing.choice() as subspace:
                if use_smoothing:
                    space.assign(
                        "params_smoothing", ParamsSmoothing.init_from(subspace)
                    )
                use_thresholding = subspace.suggest_bool("use_thresholding")
                space.assign("use_thresholding", use_thresholding.value)
                with (
                    use_thresholding as subspace
                ):  # here, no .choice(), this subspace is only "active" when use_thresholding==True
                    space.assign(
                        "threshold", subspace.suggest_float("threshold", 0.0, 1.0).value
                    )


@dataclass
class ParamsPipeline(Params, ABC):
    # note if you reimplement this for your own pipeline, these parameters should definitely exist to ensure compatibility
    min_samples_per_stratum: int
    balance_sample_weights: bool
    use_sliding_window_features: bool
    params_feature_aggregator: ParamsFeatureAggregator | None = None
    params_classifier: Params | None = None

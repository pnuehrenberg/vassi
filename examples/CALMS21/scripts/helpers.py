from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import override

import numpy as np

from vassi.classification import AnnotatedDatasetClassification
from vassi.classification.optimization.optuna_utils import ParameterSpace, Params
from vassi.classification.optimization.utils import (
    ParamsCategoryOutput,
    ParamsFeatureAggregator,
    ParamsPipeline,
)
from vassi.dataset import AnnotatedDataset
from vassi.features import BaseExtractor, Shaped
from vassi.sliding_metrics import (
    sliding_mean,
    sliding_median,
    sliding_quantile,
)

CATEGORIES = ("attack", "investigation", "mount", "none")


def sampling_function[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    *,
    random_state: int | np.random.Generator | None,
    params: ParamsPipeline,
) -> tuple[F, np.ndarray]:
    return dataset.subsample(
        extractor,
        size={
            category: min(30000, dataset.category_counts[category])
            for category in dataset.categories
        },
        min_samples_per_stratum=params.min_samples_per_stratum,
        random_state=random_state,
        reset=False,
        exclude_previously_sampled=False,
        store_indices=False,
        out=(None, None),
        ensure_sampling_at=None,
    )


def smooth_model_outputs(
    categories: Iterable[str], *, array: np.ndarray, params: CALMS21PipelineParams
) -> np.ndarray:
    probabilities_smoothed = array.copy()
    for idx, category in enumerate(sorted(categories)):
        if (
            not params.params_category_output
            or category not in params.params_category_output
        ):
            continue
        params_category = params.params_category_output[category]
        params_quantile_range_filter = params_category.params_quantile_range_filter
        params_smoothing = params_category.params_smoothing
        probabilities_category = array[:, idx]
        if params_category.use_quantile_range_filter and params_quantile_range_filter:
            q_lower = sliding_quantile(
                probabilities_category,
                params_quantile_range_filter.window_size_lower,
                params_quantile_range_filter.lower,
            )
            q_upper = sliding_quantile(
                probabilities_category,
                params_quantile_range_filter.window_size_upper,
                params_quantile_range_filter.upper,
            )
            probabilities_category = np.clip(probabilities_category, q_lower, q_upper)
        if params_category.use_smoothing and params_smoothing:
            if params_smoothing.sliding_metric == "mean":
                smoothing_function = sliding_mean
            elif params_smoothing.sliding_metric == "median":
                smoothing_function = sliding_median
            else:
                raise ValueError("expected on of 'mean', 'median'")
            probabilities_category = smoothing_function(
                probabilities_category, params_smoothing.window_size
            )
        probabilities_smoothed[:, idx] = probabilities_category
    return probabilities_smoothed


def postprocessing_function(
    result: AnnotatedDatasetClassification,
    *,
    params: ParamsPipeline,
):
    assert isinstance(params, CALMS21PipelineParams)
    return result.smooth(
        partial(
            smooth_model_outputs,
            result.categories,
            params=params,
        ),
        decision_thresholds=(
            {
                category: threshold
                for category, category_params in params.params_category_output.items()
                if category_params.use_thresholding
                and (threshold := category_params.threshold)
            }
            if params.params_category_output
            else {}
        ),
    )


@dataclass
class ParamsClassifier(Params, ABC):
    n_jobs: int
    n_estimators: int

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        space.assign("n_jobs", 72 // 4)  # physical cpus // n_jobs in optimization.py
        space.assign("n_estimators", 1000)
        # _ = space.suggest_categorical("n_estimators", [100, 200, 500, 1000])


@dataclass
class CALMS21PipelineParams(ParamsPipeline):
    # extend ParamsPipeline to provide params for custom postprocessing (see function above)
    params_category_output: dict[str, ParamsCategoryOutput] | None = None

    @override
    @classmethod
    def define(cls, space: ParameterSpace):
        _ = space.suggest_int("min_samples_per_stratum", 0, 30)
        _ = space.suggest_bool("balance_sample_weights")
        if space.suggest_bool("use_sliding_window_features"):
            space.assign(
                "params_feature_aggregator", ParamsFeatureAggregator.init_from(space)
            )
        space.assign(
            "params_category_output",
            {
                category: ParamsCategoryOutput.init_from(space.subspace(category))
                for category in CATEGORIES
            },
        )
        space.assign("params_classifier", ParamsClassifier.init_from(space))

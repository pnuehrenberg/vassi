from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import optuna

from vassi.classification import AnnotatedDatasetClassification
from vassi.classification.optimization import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    ParameterSpace,
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
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None,
) -> tuple[F, np.ndarray]:
    return dataset.subsample(
        extractor,
        size={
            category: min(30000, dataset.category_counts[category])
            for category in dataset.categories
        },
        min_samples_per_stratum=min_samples_per_stratum,
        random_state=random_state,
        reset=False,
        exclude_previously_sampled=False,
        store_indices=False,
        out=None,
        ensure_sampling_at=None,
    )


def smooth_model_outputs(
    categories: Iterable[str], *, array: np.ndarray, **kwargs: ...
) -> np.ndarray:
    probabilities_smoothed = np.zeros_like(array)
    for idx, category in enumerate(sorted(categories)):
        window_lower = kwargs[f"quantile_range_window_lower-{category}"]
        window_upper = kwargs[f"quantile_range_window_upper-{category}"]
        probabilities_category = array[:, idx]
        q_lower = probabilities_category
        if window_lower > 1:
            q_lower = sliding_quantile(
                probabilities_category,
                window_lower,
                kwargs[f"quantile_range_lower-{category}"],
            )
        q_upper = probabilities_category
        if window_upper > 1:
            q_upper = sliding_quantile(
                probabilities_category,
                window_upper,
                kwargs[f"quantile_range_upper-{category}"],
            )
        probabilities_category = np.clip(probabilities_category, q_lower, q_upper)
        smoothing_window = kwargs[f"smoothing_window-{category}"]
        if smoothing_window > 1:
            if (smoothing_function_kwarg := kwargs["smoothing_function"]) == "mean":
                smoothing_function = sliding_mean
            elif smoothing_function_kwarg == "median":
                smoothing_function = sliding_median
            else:
                raise ValueError(
                    f"Invalid smoothing function keyword argument: {smoothing_function_kwarg}"
                )
            probabilities_smoothed[:, idx] = smoothing_function(
                probabilities_category, smoothing_window
            )
        else:
            probabilities_smoothed[:, idx] = probabilities_category
    return probabilities_smoothed


def postprocessing_function(
    result: AnnotatedDatasetClassification,
    **postprocessing_function_kwargs: ...,
):
    smoothing_function_kwargs = dict(postprocessing_function_kwargs)
    decision_thresholds = {
        threshold.replace("decision_threshold-", "", 1): value
        for threshold in postprocessing_function_kwargs
        if threshold.startswith("decision_threshold-")
        and (value := smoothing_function_kwargs.pop(threshold))
    }
    return result.smooth(
        partial(smooth_model_outputs, result.categories, **smoothing_function_kwargs),
        decision_thresholds=decision_thresholds,
    )


postprocessing_parameters: dict[str, Callable[[optuna.trial.Trial], object]] = {
    "smoothing_function": CategoricalParameter("smoothing_function", ["mean", "median"])
}
for category in CATEGORIES:
    threshold = f"decision_threshold-{category}"
    postprocessing_parameters[threshold] = FloatParameter(threshold, 0.0, 1.0)
    for window in [
        f"quantile_range_window_lower-{category}",
        f"quantile_range_window_upper-{category}",
        f"smoothing_window-{category}",
    ]:
        postprocessing_parameters[window] = IntParameter(window, 1, 91, step=2)
    for quantile_range in [
        f"quantile_range_lower-{category}",
        f"quantile_range_upper-{category}",
    ]:
        postprocessing_parameters[quantile_range] = FloatParameter(
            quantile_range, 0.0, 1.0
        )

parameter_space = ParameterSpace(
    classifier_kwargs={"n_estimators": IntParameter("n_estimators", 1000, 1000)},
    balance_sample_weights=CategoricalParameter(
        "balance_sample_weights", [True, False]
    ),
    postprocessing_function_kwargs=postprocessing_parameters,
    sampling_function_kwargs={
        "min_samples_per_stratum": IntParameter("min_samples_per_stratum", 0, 30),
    },
)

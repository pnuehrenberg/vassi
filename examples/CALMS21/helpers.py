from functools import partial
from typing import Any, Iterable

import numpy as np
import optuna
from numpy.typing import NDArray

from automated_scoring.classification.postprocessing import PostprocessingParameters
from automated_scoring.dataset.types._mixins import (
    AnnotatedSampleableMixin,
    SampleableMixin,
)
from automated_scoring.features import BaseExtractor, Shaped
from automated_scoring.sliding_metrics import sliding_mean, sliding_quantile


def subsample_train[F: Shaped](
    sampleable: SampleableMixin,
    extractor: BaseExtractor[F],
    *,
    random_state,
    log,
) -> tuple[F, NDArray]:
    if not isinstance(sampleable, AnnotatedSampleableMixin):
        raise ValueError("sampleable must be annotated")
    return sampleable.subsample(
        extractor,
        {
            ("attack", "mount"): 1.0,
            ("none", "investigation"): 30000,
        },
        random_state=random_state,
        log=log,
    )


def smooth_model_outputs(postprocessing_parameters: dict[str, Any], *, array: NDArray):
    categories = ("attack", "investigation", "mount", "none")
    probabilities_smoothed = np.zeros_like(array)
    for idx, category in enumerate(categories):
        window_lower = postprocessing_parameters[
            f"quantile_range_window_lower-{category}"
        ]
        window_upper = postprocessing_parameters[
            f"quantile_range_window_upper-{category}"
        ]
        window_mean = postprocessing_parameters[f"mean_window-{category}"]
        probabilities_category = array[:, idx]
        q_lower = probabilities_category
        if window_lower > 1:
            q_lower = sliding_quantile(
                probabilities_category,
                window_lower,
                postprocessing_parameters[f"quantile_range_lower-{category}"],
            )
        q_upper = probabilities_category
        if window_upper > 1:
            q_upper = sliding_quantile(
                probabilities_category,
                window_upper,
                postprocessing_parameters[f"quantile_range_upper-{category}"],
            )
        probabilities_category = np.clip(probabilities_category, q_lower, q_upper)
        if window_mean > 1:
            probabilities_smoothed[:, idx] = sliding_mean(
                probabilities_category, window_mean
            )
        else:
            probabilities_smoothed[:, idx] = probabilities_category
    return probabilities_smoothed


def postprocessing(
    result,
    *,
    postprocessing_parameters: dict[str, Any],
    decision_thresholds: Iterable[float],
    default_decision: int | str,
):
    return result.smooth(
        partial(smooth_model_outputs, postprocessing_parameters),
        decision_thresholds=decision_thresholds,
        default_decision=default_decision,
    )


def suggest_postprocessing_parameters(
    trial: optuna.trial.Trial, *, categories: Iterable[str]
) -> PostprocessingParameters:
    parameters = {}
    for category in categories:
        parameters[f"quantile_range_window_lower-{category}"] = (
            trial.suggest_int(f"quantile_range_window_lower-{category}", 0, 90, step=2)
            + 1
        )
        parameters[f"quantile_range_lower-{category}"] = trial.suggest_float(
            f"quantile_range_lower-{category}", 0, 0.5
        )
        parameters[f"quantile_range_window_upper-{category}"] = (
            trial.suggest_int(f"quantile_range_window_upper-{category}", 0, 90, step=2)
            + 1
        )
        parameters[f"quantile_range_upper-{category}"] = trial.suggest_float(
            f"quantile_range_upper-{category}", 0.5, 1
        )
        parameters[f"mean_window-{category}"] = (
            trial.suggest_int(f"mean_window-{category}", 0, 90, step=2) + 1
        )
    return {
        "postprocessing_parameters": parameters,
        "decision_thresholds": [
            trial.suggest_float(f"threshold-{category}", 0, 1)
            for category in categories
        ],
    }

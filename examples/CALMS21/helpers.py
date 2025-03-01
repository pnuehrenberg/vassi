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
from automated_scoring.features import BaseExtractor, F
from automated_scoring.sliding_metrics import sliding_mean, sliding_quantile


def subsample_train(
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
        stratify=True,
        reset_previous_indices=False,
        exclude_previous_indices=False,
        store_indices=False,
        log=log,
    )


def smooth_model_outputs(postprocessing_parameters: dict[str, Any], *, array: NDArray):
    q_lower = sliding_quantile(
        array,
        postprocessing_parameters["quantile_range_window_lower"],
        postprocessing_parameters["quantile_range_lower"],
    )
    q_upper = sliding_quantile(
        array,
        postprocessing_parameters["quantile_range_window_upper"],
        postprocessing_parameters["quantile_range_upper"],
    )
    array_clipped = np.clip(array, q_lower, q_upper)
    array_smoothed = sliding_mean(
        array_clipped, postprocessing_parameters["mean_window"]
    )
    return array_smoothed


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
    return {
        "postprocessing_parameters": {
            "quantile_range_window_lower": trial.suggest_int(
                "quantile_range_window_lower", 2, 88, step=2
            )
            + 1,
            "quantile_range_lower": trial.suggest_float("quantile_range_lower", 0, 0.5),
            "quantile_range_window_upper": trial.suggest_int(
                "quantile_range_window_upper", 2, 88, step=2
            )
            + 1,
            "quantile_range_upper": trial.suggest_float("quantile_range_upper", 0.5, 1),
            "mean_window": trial.suggest_int("mean_window", 2, 88, step=2) + 1,
        },
        "decision_thresholds": [
            trial.suggest_float(f"threshold_{category}", 0, 1)
            for category in categories
        ],
    }

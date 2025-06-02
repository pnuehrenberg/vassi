from functools import partial
from typing import Any, Iterable

import numpy as np
import optuna
import pandas as pd

from vassi.classification.postprocessing import PostprocessingParameters
from vassi.classification.results import ClassificationResult, _NestedResult
from vassi.dataset.permute import permute_recipients
from vassi.dataset.types import AnnotatedDataset, AnnotatedGroup
from vassi.dataset.types.mixins import (
    AnnotatedSampleableMixin,
    SampleableMixin,
)
from vassi.features import BaseExtractor, Shaped
from vassi.sliding_metrics import (
    sliding_mean,
    sliding_median,
    sliding_quantile,
)


def subsample_train[F: Shaped](
    sampleable: SampleableMixin,
    extractor: BaseExtractor[F],
    *,
    random_state,
    log,
) -> tuple[F, np.ndarray]:
    if not isinstance(sampleable, AnnotatedSampleableMixin):
        raise ValueError("sampleable must be annotated")
    X, y = sampleable.subsample(
        extractor,
        {
            ("approach", "chase", "dart_bite", "lateral_display", "quiver"): 1.0,
            "frontal_display": 0.25,
            "none": 0.01,
        },
        random_state=random_state,
        log=log,
    )

    if not isinstance(sampleable, AnnotatedGroup | AnnotatedDataset):
        raise ValueError("sampleable must be group or dataset for permutation")

    sampling_frequency = {0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05}

    X_additional = [
        permute_recipients(sampleable, neighbor_rank=neighbor_rank).subsample(
            extractor,
            {
                ("approach", "chase", "dart_bite", "lateral_display", "quiver"): 1.0
                * sampling_frequency[neighbor_rank],
                "frontal_display": 0.25 * sampling_frequency[neighbor_rank],
            },
            random_state=random_state,
            log=log,
        )[0]  # only keep samples (X) but not labels (y)
        for neighbor_rank in sampling_frequency
    ]

    X_additional = extractor.concatenate(*X_additional, axis=0, ignore_index=True)
    # all corresponding labels are "none" because of switched recipients
    y_additional = np.repeat("none", X_additional.shape[0])

    return (
        extractor.concatenate(X, X_additional, axis=0, ignore_index=True),
        np.concatenate([y, y_additional]),
    )


def smooth_model_outputs(
    postprocessing_parameters: dict[str, Any], *, array: np.ndarray
):
    categories = (
        "approach",
        "chase",
        "dart_bite",
        "frontal_display",
        "lateral_display",
        "none",
        "quiver",
    )
    probabilities_smoothed = np.zeros_like(array)
    for idx, category in enumerate(categories):
        window_lower = postprocessing_parameters[
            f"quantile_range_window_lower-{category}"
        ]
        window_upper = postprocessing_parameters[
            f"quantile_range_window_upper-{category}"
        ]
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
        smoothing_window = postprocessing_parameters[f"smoothing_window-{category}"]
        if smoothing_window > 1:
            match postprocessing_parameters["smoothing_function"]:
                case "mean":
                    smoothing_function = sliding_mean
                case "median":
                    smoothing_function = sliding_median
                case _:
                    raise ValueError("Invalid smoothing function")
            probabilities_smoothed[:, idx] = smoothing_function(
                probabilities_category, smoothing_window
            )
        else:
            probabilities_smoothed[:, idx] = probabilities_category
    return probabilities_smoothed


def postprocessing(
    result: ClassificationResult | _NestedResult,
    *,
    postprocessing_parameters: dict[str, Any],
    decision_thresholds: Iterable[float],
    default_decision: int | str,
):
    return result.smooth(
        partial(smooth_model_outputs, postprocessing_parameters),
        decision_thresholds=decision_thresholds,
        default_decision=default_decision,
    ).remove_overlapping_predictions(
        priority_function=postprocessing_parameters["priority_function"],
        prefilter_recipient_bouts=postprocessing_parameters[
            "prefilter_recipient_bouts"
        ],
        max_bout_gap=postprocessing_parameters["max_bout_gap"],
        max_allowed_bout_overlap=postprocessing_parameters["max_allowed_bout_overlap"],
    )


def score_priority(
    observations: pd.DataFrame,
    *,
    weight_max_probability: float,
    weight_mean_probability: float,
) -> pd.Series:
    # lower is better
    if weight_max_probability + weight_mean_probability == 0:
        weight_max_probability = 1
        weight_mean_probability = 1
    return (
        (1 - observations["max_probability"]) * weight_max_probability
        + (1 - observations["mean_probability"]) * weight_mean_probability
    ) / (weight_max_probability + weight_mean_probability)


def suggest_postprocessing_parameters(
    trial: optuna.trial.Trial, *, categories: Iterable[str]
) -> PostprocessingParameters:
    weight_max_probability = trial.suggest_float("weight_max_probability", 0, 1)
    weight_mean_probability = 1 - weight_max_probability
    trial.set_user_attr("weight_mean_probability", weight_mean_probability)
    parameters = {
        "priority_function": partial(
            score_priority,
            weight_max_probability=weight_max_probability,
            weight_mean_probability=weight_mean_probability,
        ),
        "prefilter_recipient_bouts": trial.suggest_categorical(
            "prefilter_recipient_bouts", [True, False]
        ),
        "max_bout_gap": trial.suggest_int("max_bout_gap", 0, 120),
        "max_allowed_bout_overlap": trial.suggest_int(
            "max_allowed_bout_overlap", 0, 60
        ),
        "smoothing_function": trial.suggest_categorical(
            "smoothing_function", ["mean", "median"]
        ),
    }
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
        parameters[f"smoothing_window-{category}"] = (
            trial.suggest_int(f"smoothing_window-{category}", 0, 90, step=2) + 1
        )
    return {
        "postprocessing_parameters": parameters,
        "decision_thresholds": [
            trial.suggest_float(f"threshold-{category}", 0, 1)
            for category in categories
        ],
    }

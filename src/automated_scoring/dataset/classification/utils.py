from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .. import AnnotatedGroup, Dataset
from ..observations.bouts import aggregate_bouts
from ..observations.utils import (
    check_observations,
    ensure_matching_index_keys,
    ensure_single_index,
    infill_observations,
    remove_overlapping_observations,
    to_observations,
)
from ..utils import interval_contained, interval_overlap

if TYPE_CHECKING:
    from .results import DatasetClassificationResult


def to_predictions(
    y: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    category_names: Iterable[str],
    timestamps: NDArray[np.int64 | np.float64],
) -> pd.DataFrame:
    predictions = to_observations(y, category_names, timestamps=timestamps)
    probabilities = [
        y_proba[
            (timestamps >= prediction["start"]) & (timestamps <= prediction["stop"])
        ]
        for _, prediction in predictions.iterrows()
    ]
    predictions["mean_probability"] = [
        proba[:, proba.argmax(axis=1)].mean() for proba in probabilities
    ]
    predictions["max_probability"] = [
        proba[:, proba.argmax(axis=1)].max() for proba in probabilities
    ]
    return predictions


def to_prediction_dataset(
    dataset_classification_result: "DatasetClassificationResult",
    *,
    target: Literal["individuals", "dyads"],
) -> Dataset:
    categories = tuple(
        np.unique(list(dataset_classification_result.predictions["category"]))
    )
    return Dataset(
        {
            group_key: AnnotatedGroup(
                group_result.trajectories,
                target=target,
                observations=group_result.predictions,
                categories=categories,
            )
            for group_key, group_result in dataset_classification_result.classification_results.items()
        }
    )


def validate_predictions(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    on: Literal["predictions", "annotations"] = "predictions",
    key_columns: Iterable[str] = ("group", "actor", "recipient"),
):
    available_key_columns = []
    for column_name in key_columns:
        if column_name not in predictions:
            continue
        if column_name not in annotations:
            raise ValueError("columns do not match")
        available_key_columns.append(column_name)
    if len(available_key_columns) > 0:
        predictions, annotations = ensure_matching_index_keys(
            predictions, annotations, available_key_columns
        )
    predictions = check_observations(predictions, ("start", "stop", "category"))
    annotations = check_observations(annotations, ("start", "stop", "category"))
    stop: int = max(predictions["stop"].max(), annotations["stop"].max())  # type: ignore
    predictions = infill_observations(predictions, stop)
    annotations = infill_observations(annotations, stop)
    alternative_categories = []
    intervals_predictions = predictions[["start", "stop"]].to_numpy()
    intervals_annotations = annotations[["start", "stop"]].to_numpy()
    if on == "predictions":
        overlap = interval_overlap(
            intervals_predictions, intervals_annotations, mask_diagonal=False
        )
        for idx, (_, prediction) in enumerate(predictions.iterrows()):
            overlap_idx = np.argwhere(overlap[idx] > 0).ravel()
            overlap_duration = overlap[idx, overlap_idx]
            overlap_category = annotations.iloc[overlap_idx]["category"].tolist()
            alternative_categories.append(overlap_category[np.argmax(overlap_duration)])
        predictions["true_category"] = alternative_categories
        return predictions
    elif on == "annotations":
        overlap = interval_overlap(
            intervals_annotations, intervals_predictions, mask_diagonal=False
        )
        for idx, (_, annotation) in enumerate(annotations.iterrows()):
            overlap_idx = np.argwhere(overlap[idx]).ravel()
            overlap_duration = overlap[idx, overlap_idx]
            overlap_category = predictions.iloc[overlap_idx]["category"].tolist()
            alternative_categories.append(overlap_category[np.argmax(overlap_duration)])
        annotations["predicted_category"] = alternative_categories
        return annotations
    else:
        raise ValueError(
            f"invalid 'on' argument {on}. specify either 'predictions' or 'annotations'"
        )


def score_category_counts(
    annotations: pd.DataFrame, predictions: pd.DataFrame, categories: Iterable[str]
) -> NDArray:
    categories = tuple(categories)
    counts_annotated = np.asarray(
        [(annotations["category"] == category).sum() for category in categories]
    )
    counts_predicted = np.asarray(
        [(predictions["category"] == category).sum() for category in categories]
    )
    return np.minimum(counts_annotated, counts_predicted) / np.maximum(
        counts_annotated, counts_predicted
    )


def _filter_recipient_bouts(
    observations: pd.DataFrame,
    *,
    priority_func: Callable[[pd.DataFrame], Iterable[float]],
    max_bout_gap: float,
    max_allowed_bout_overlap: float,
):
    observations = check_observations(
        observations,
        required_columns=["start", "stop", "actor", "recipient", "category"],
        allow_overlapping=True,
        allow_unsorted=True,
    )
    observations = observations[observations["category"] != "none"]  # type: ignore
    observations = ensure_single_index(observations, index_keys=["actor"], drop=False)
    bouts = []
    for recipient, observations_recipient in observations.groupby("recipient"):
        if len(observations_recipient) == 0:
            continue
        bouts.append(
            aggregate_bouts(
                observations_recipient,
                max_bout_gap=max_bout_gap,
                index_keys=[],
            )
        )
    if len(bouts) == 0:
        return observations
    bouts = (
        remove_overlapping_observations(
            pd.concat(bouts, ignore_index=True),
            index_keys=[],
            priority_func=priority_func,
            max_allowed_overlap=max_allowed_bout_overlap,
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
    same_recipient = np.asarray(observations["recipient"])[:, np.newaxis] == np.asarray(
        bouts["recipient"]
    )
    contained_in_bout = np.sum(
        same_recipient
        & interval_contained(
            observations[["start", "stop"]].to_numpy(),
            bouts[["start", "stop"]].to_numpy(),
        ),
        axis=1,
    )
    assert np.isin(contained_in_bout, [0, 1]).all()
    idx = np.argwhere(contained_in_bout).ravel()
    indices = observations.index[idx]
    observations["in_bout"] = False
    observations.loc[indices, "in_bout"] = True
    observations = observations[observations["in_bout"]]  # type: ignore
    return (
        observations.drop(columns=["in_bout"])
        .sort_values("start")
        .reset_index(drop=True)
    )

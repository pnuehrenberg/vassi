from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Literal

import numpy as np
import pandas as pd

from ..dataset.observations import (
    aggregate_observations_as_bouts,
    assert_singular_index_combination,
    densify_observations,
    interval_overlap,
    remove_overlapping_observations,
    to_observations,
)


def to_predictions(
    labels: np.ndarray,
    y_proba: np.ndarray,
    timestamps: np.ndarray,
    categories: set[str],
) -> pd.DataFrame:
    """
    Convert the given predictions to a DataFrame.

    Parameters:
        y: The predicted integer labels.
        y_proba: The predicted probabilities.
        categories: Category names.
        timestamps: The timestamps.

    See also:
        :func:`to_observations`for more information how categories and labels are mapped.
    """
    predictions = to_observations(labels, categories, timestamps=timestamps)
    y_proba_predicted = y_proba[np.arange(len(labels)), labels]
    start_indices = np.searchsorted(
        timestamps, np.asarray(predictions["start"]), side="left"
    )
    end_indices = np.r_[start_indices[1:], len(labels)]
    durations = end_indices - start_indices
    predictions["mean_probability"] = (
        np.add.reduceat(y_proba_predicted, start_indices) / durations
    )
    predictions["max_probability"] = np.maximum.reduceat(
        y_proba_predicted, start_indices
    )
    return predictions


def validate_predictions(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    on: Literal["predictions", "annotations"] = "predictions",
    index_columns: Iterable[str] = ("group", "actor", "recipient"),
    background_category: str,
) -> pd.DataFrame:
    available_index_columns: list[str] = []
    for column_name in index_columns:
        if column_name not in predictions:
            continue
        if column_name not in annotations:
            raise ValueError("columns do not match")
        available_index_columns.append(column_name)
    if len(available_index_columns) > 0:
        predictions = assert_singular_index_combination(
            predictions, tuple(available_index_columns)
        )
        annotations = assert_singular_index_combination(
            annotations, tuple(available_index_columns)
        )
    if predictions.empty and annotations.empty:
        raise ValueError("No data to compare")
    elif predictions.empty:
        start, stop = annotations["start"].min(), annotations["stop"].max()
    else:
        start, stop = predictions["start"].min(), predictions["stop"].max()
    start = min(predictions["start"].min(), annotations["start"].min())
    stop = max(predictions["stop"].max(), annotations["stop"].max())
    predictions = densify_observations(
        predictions, time_range=(start, stop), background_category=background_category
    )
    annotations = densify_observations(
        annotations, time_range=(start, stop), background_category=background_category
    )
    if on == "predictions":
        target_df, source_df = predictions, annotations
        new_col_name = "true_category"
    else:
        target_df, source_df = annotations, predictions
        new_col_name = "predicted_category"
    target_intervals = np.asarray(target_df[["start", "stop"]])
    source_intervals = np.asarray(source_df[["start", "stop"]])
    source_categories_array = np.asarray(source_df["category"])
    overlap_matrix = interval_overlap(target_intervals, source_intervals)
    unique_categories, integer_labels = np.unique(
        source_categories_array, return_inverse=True
    )
    num_unique_categories = len(unique_categories)
    one_hot_matrix = integer_labels[:, np.newaxis] == np.arange(num_unique_categories)
    cumulative_overlap_matrix = overlap_matrix @ one_hot_matrix
    best_category_indices = np.argmax(cumulative_overlap_matrix, axis=1)
    matched_categories = unique_categories[best_category_indices]
    validated_df = target_df.copy()
    validated_df[new_col_name] = matched_categories
    return validated_df


def filter_observations_by_recipient_bouts(
    observations: pd.DataFrame,
    *,
    priority_function: Callable[[pd.DataFrame], Iterable[float]],
    max_bout_gap: float,
    max_bout_overlap: float,
    background_category: str,
) -> pd.DataFrame:
    bout_offset = 0
    bouts_by_recipient: list[pd.DataFrame] = []
    observations_by_recipient: list[pd.DataFrame] = []
    for recipient in observations["recipient"].unique():
        bouts_recipient, observations_recipient = aggregate_observations_as_bouts(
            observations.loc[observations["recipient"] == recipient, :],
            max_bout_gap=max_bout_gap,
            index_columns=("actor",),
            background_category=background_category,
        )
        bouts_recipient["bout"] += bout_offset
        observations_recipient["bout"] += bout_offset
        bout_offset += len(bouts_recipient)
        bouts_by_recipient.append(bouts_recipient)
        observations_by_recipient.append(observations_recipient)
    if len(bouts_by_recipient) <= 1:
        return observations
    non_overlapping_bouts = remove_overlapping_observations(
        pd.concat(bouts_by_recipient, ignore_index=True),
        index_columns=(),
        priority_function=priority_function,
        max_overlap=max_bout_overlap,
    )
    observations = pd.concat(observations_by_recipient, ignore_index=True)
    observations = observations.loc[
        np.isin(observations["bout"], non_overlapping_bouts["bout"]), :
    ]
    return observations.drop(columns=["bout"]).sort_values("start", ignore_index=True)

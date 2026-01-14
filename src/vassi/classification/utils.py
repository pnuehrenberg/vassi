from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import polars as pl
from tabularintervals import densify

from ..dataset.observations import (
    interval_overlap,
    to_observations,
)


def to_predictions(
    labels: np.ndarray,
    y_proba: np.ndarray,
    timestamps: np.ndarray,
    categories: set[str],
) -> pl.DataFrame:
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
    if predictions.is_empty():
        return predictions.with_columns(
            mean_probability=pl.lit(None).cast(pl.Float64),
            max_probability=pl.lit(None).cast(pl.Float64),
        )
    y_proba_predicted = y_proba[np.arange(len(labels)), labels]
    start_indices = np.searchsorted(
        timestamps, np.asarray(predictions.get_column("start")), side="left"
    )
    end_indices = np.r_[start_indices[1:], len(labels)]
    durations = end_indices - start_indices
    predictions = predictions.with_columns(
        mean_probability=(
            np.add.reduceat(y_proba_predicted, start_indices) / durations
        ),
        max_probability=(np.maximum.reduceat(y_proba_predicted, start_indices)),
    )
    return predictions


def validate_predictions(
    predictions: pl.DataFrame,
    annotations: pl.DataFrame,
    *,
    on: Literal["predictions", "annotations"] = "predictions",
    index_columns: Iterable[str] = ("group", "actor", "recipient"),
    background_category: str,
) -> pl.DataFrame:
    available_index_columns: list[str] = []
    for column_name in index_columns:
        if column_name not in predictions:
            continue
        if column_name not in annotations:
            raise ValueError("columns do not match")
        available_index_columns.append(column_name)
    if len(available_index_columns) > 0:
        if predictions.select(*available_index_columns).unique().height > 1:
            raise ValueError(
                f"predictions should have single combination of available identifier columns {available_index_columns}"
            )
        if annotations.select(*available_index_columns).unique().height > 1:
            raise ValueError(
                f"annotations should have single combination of available identifier columns {available_index_columns}"
            )
    if predictions.is_empty() and annotations.is_empty():
        raise ValueError("no data to compare")
    elif predictions.is_empty():
        start, stop = (
            annotations.get_column("start").min(),
            annotations.get_column("stop").max(),
        )
    elif annotations.is_empty():
        start, stop = (
            predictions.get_column("start").min(),
            predictions.get_column("stop").max(),
        )
    else:
        starts = [
            predictions.get_column("start").min(),
            annotations.get_column("start").min(),
        ]
        stops = [
            predictions.get_column("stop").max(),
            annotations.get_column("stop").max(),
        ]
        start = min([start for start in starts if start is not None])
        stop = min([stop for stop in stops if stop is not None])
    if (
        start is None
        or stop is None
        or not isinstance(start, int)
        or not isinstance(stop, int)
    ):
        raise ValueError(f"invalid input dataframes with [{start, stop}]")
    predictions = densify(
        predictions, within=(start, stop), category=background_category
    )
    annotations = densify(
        annotations, within=(start, stop), category=background_category
    )
    if on == "predictions":
        target_df, source_df = predictions, annotations
        new_col_name = "true_category"
    else:
        target_df, source_df = annotations, predictions
        new_col_name = "predicted_category"
    target_intervals = np.asarray(target_df.select("start", "stop"))
    source_intervals = np.asarray(source_df.select("start", "stop"))
    source_categories_array = np.asarray(source_df.get_column("category"))
    overlap_matrix = interval_overlap(target_intervals, source_intervals)
    unique_categories, integer_labels = np.unique(
        source_categories_array, return_inverse=True
    )
    num_unique_categories = len(unique_categories)
    one_hot_matrix = integer_labels[:, np.newaxis] == np.arange(num_unique_categories)
    cumulative_overlap_matrix = overlap_matrix @ one_hot_matrix
    best_category_indices = np.argmax(cumulative_overlap_matrix, axis=1)
    matched_categories = unique_categories[best_category_indices]
    validated_df = target_df.with_columns(
        pl.Series(matched_categories).alias(new_col_name)
    )
    return validated_df

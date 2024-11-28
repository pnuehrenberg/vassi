from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyTrajectory.dataset import AnnotatedGroup
from pyTrajectory.dataset.annotations.utils import (
    check_annotations,
    infill_annotations,
    to_annotations,
)
from pyTrajectory.dataset.types.dataset import Dataset
from pyTrajectory.dataset.utils import interval_overlap

if TYPE_CHECKING:
    from .results import DatasetClassificationResult


def to_predictions(
    y: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    category_names: Iterable[str],
    timestamps: NDArray[np.int64 | np.float64],
) -> pd.DataFrame:
    predictions = to_annotations(y, category_names, timestamps=timestamps)
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
                annotations=group_result.predictions,
                categories=categories,
            )
            for group_key, group_result in dataset_classification_result.classification_results.items()
        }
    )


def _ensure_matching_index_keys(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    index_keys: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predictions.set_index(index_keys)
    annotations = annotations.set_index(index_keys)
    if len(np.unique(predictions.index)) > 1:
        raise ValueError(
            "intervals cannot be matched with more than one unique key column combination in predictions"
        )
    if len(np.unique(annotations.index)) > 1:
        raise ValueError(
            "intervals cannot be matched with more than one unique key column combination in annotations"
        )
    return predictions.reset_index(drop=True), annotations.reset_index(drop=True)


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
        predictions, annotations = _ensure_matching_index_keys(
            predictions, annotations, available_key_columns
        )
    predictions = check_annotations(predictions, ("start", "stop", "category"))
    annotations = check_annotations(annotations, ("start", "stop", "category"))
    stop: int = max(predictions["stop"].max(), annotations["stop"].max())  # type: ignore
    predictions = infill_annotations(predictions, stop)
    annotations = infill_annotations(annotations, stop)
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

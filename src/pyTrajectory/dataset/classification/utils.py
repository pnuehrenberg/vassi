from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Optional,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from pyTrajectory.dataset import AnnotatedGroup
from pyTrajectory.dataset.annotations.utils import (
    check_annotations,
    infill_annotations,
)
from pyTrajectory.dataset.types.dataset import Dataset
from pyTrajectory.dataset.utils import interval_overlap

if TYPE_CHECKING:
    from .results import DatasetClassificationResult


def to_prediction_dataset(
    dataset_classification_result: DatasetClassificationResult,
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
        predictions = predictions.set_index(available_key_columns)
        annotations = annotations.set_index(available_key_columns)
        validated = []
        for key in np.unique(predictions.index):
            predictions_key: pd.DataFrame = predictions.loc[[key]]
            try:
                annotations_key: pd.DataFrame = annotations.loc[[key]]
            except KeyError:
                annotations_key = annotations[:0]  # type: ignore
            validated.append(
                validate_predictions(
                    predictions_key,
                    annotations_key,
                    on=on,
                    key_columns=key_columns,
                )
            )
        return pd.concat(validated, axis=0)
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


def interval_f1_score(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
    num_categories: int,
    average: Optional[Literal["micro", "macro", "weighted"]] = None,
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
        predictions = predictions.set_index(available_key_columns)
        annotations = annotations.set_index(available_key_columns)
        scores = []
        for key in np.unique(predictions.index):
            predictions_key: pd.DataFrame = predictions.loc[[key]]
            try:
                annotations_key: pd.DataFrame = annotations.loc[[key]]
            except KeyError:
                annotations_key = annotations[:0]  # type: ignore
            scores.append(
                interval_f1_score(
                    predictions_key,
                    annotations_key,
                    encode_func=encode_func,
                    num_categories=num_categories,
                    average=average,
                    on=on,
                    key_columns=key_columns,
                )
            )
        return np.concatenate(scores, axis=0)
    validated = validate_predictions(
        predictions, annotations, on=on, key_columns=key_columns
    )
    y_true = (
        validated["category"] if on == "annotations" else validated["true_category"]
    )
    y_pred = (
        validated["predicted_category"]
        if on == "annotations"
        else validated["category"]
    )
    score = f1_score(
        encode_func(np.asarray(y_true)),
        encode_func(np.asarray(y_pred)),
        labels=range(num_categories),
        average=average,  # type: ignore
        zero_division=np.nan,  # type: ignore
    )
    return np.asarray(score).reshape(1, -1)

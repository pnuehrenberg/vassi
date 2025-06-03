from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional, Protocol, Self

import loguru
import numpy as np
import pandas as pd

from ..data_structures.utils import get_interval_slice
from ..dataset.observations import (
    aggregate_bouts,
    check_observations,
    infill_observations,
    remove_overlapping_observations,
    to_observations,
)
from ..dataset.observations.utils import (
    ensure_matching_index_columns,
    ensure_single_index,
)
from ..dataset.utils import interval_contained, interval_overlap
from ..utils import to_int_seed


class Classifier(Protocol):
    """Protocol for classifiers.

    This protocol defines the methods that a classifier should implement.

    See also:
        :class:`~sklearn.base.ClassifierMixin` for the classifier interface in :mod:`~sklearn`.
    """

    def predict(self, *args, **kwargs) -> np.ndarray: ...

    def predict_proba(self, *args, **kwargs) -> np.ndarray: ...

    def get_params(self) -> dict[str, Any]: ...

    def fit(self, *args, **kwargs) -> Self: ...


def init_new_classifier(
    classifier: Classifier, random_state: Optional[np.random.Generator | int]
) -> Classifier:
    """
    Initialize a new classifier with the same parameters as the given classifier.

    Parameters:
        classifier: The classifier to copy the parameters from.
        random_state: The random state to use for the new classifier.
    """
    random_state = np.random.default_rng(random_state)
    params = classifier.get_params()
    params["random_state"] = to_int_seed(random_state)
    return type(classifier)(**params)


def fit_classifier(
    classifier: Classifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    log: Optional[loguru.Logger] = None,
):
    """
    Fit the given classifier to the given data.

    Parameters:
        classifier: The classifier to fit.
        X: The feature matrix.
        y: The target vector.
        sample_weight: The sample weights.
        log: The logger to use.
    """
    if sample_weight is None:
        return classifier.fit(X, y)
    return classifier.fit(X, y, sample_weight=sample_weight)


def to_predictions(
    y: np.ndarray,
    y_proba: np.ndarray,
    category_names: Iterable[str],
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Convert the given predictions to a DataFrame.

    Parameters:
        y: The target vector.
        y_proba: The predicted probabilities.
        category_names: The category names.
        timestamps: The timestamps.
    """
    predictions = to_observations(y, category_names, timestamps=timestamps)
    y_max_proba = y_proba.max(axis=1)
    mean_probability = []
    max_probability = []
    for start, stop in np.asarray(predictions[["start", "stop"]]):
        y_max_proba_interval = y_max_proba[get_interval_slice(timestamps, start, stop)]
        mean_probability.append(np.mean(y_max_proba_interval))
        max_probability.append(np.max(y_max_proba_interval))
    predictions["mean_probability"] = np.array(mean_probability, dtype=float)
    predictions["max_probability"] = np.array(max_probability, dtype=float)
    return predictions


def validate_predictions(
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    on: Literal["predictions", "annotations"] = "predictions",
    key_columns: Iterable[str] = ("group", "actor", "recipient"),
) -> pd.DataFrame:
    """
    Validate the predictions or annotations.

    This calculates the mean and maximum probabilities for each predicted interval
    and retrieves the corresponding ground truth category as the category of
    the annotated interval with the highest overlap (:code:`on="predictions"`),
    or correspondingly, the category of the predicted interval with the highest overlap (:code:`on="annotations"`).

    Parameters:
        predictions: The predictions.
        annotations: The annotations.
        on: The type of data to validate.
        key_columns: The key columns.

    Returns:
        The validated predictions or annotations.
    """
    available_index_columns = []
    for column_name in key_columns:
        if column_name not in predictions:
            continue
        if column_name not in annotations:
            raise ValueError("columns do not match")
        available_index_columns.append(column_name)
    if len(available_index_columns) > 0:
        predictions, annotations = ensure_matching_index_columns(
            predictions, annotations, tuple(available_index_columns)
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


def _filter_recipient_bouts(
    observations: pd.DataFrame,
    *,
    priority_function: Callable[[pd.DataFrame], Iterable[float]],
    max_bout_gap: float,
    max_allowed_bout_overlap: float,
) -> pd.DataFrame:
    observations = check_observations(
        observations,
        required_columns=["start", "stop", "actor", "recipient", "category"],
        allow_overlapping=True,
        allow_unsorted=True,
    )
    observations = observations[observations["category"] != "none"]  # type: ignore
    observations = ensure_single_index(
        observations, index_columns=("actor",), drop=False
    )
    bouts = []
    for recipient, observations_recipient in observations.groupby("recipient"):
        if len(observations_recipient) == 0:
            continue
        bouts.append(
            aggregate_bouts(
                observations_recipient,
                max_bout_gap=max_bout_gap,
                index_columns=(),
            )
        )
    if len(bouts) == 0:
        return observations
    bouts = remove_overlapping_observations(
        pd.concat(bouts, ignore_index=True),
        index_columns=(),
        priority_function=priority_function,
        max_allowed_overlap=max_allowed_bout_overlap,
    ).sort_values("start", ignore_index=True, inplace=False)
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
    observations = observations[observations["in_bout"]]
    observations = observations.drop(columns=["in_bout"], inplace=False).sort_values(
        "start", ignore_index=True, inplace=False
    )
    return observations

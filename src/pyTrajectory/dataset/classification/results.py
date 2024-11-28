from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...data_structures import Trajectory
from ..types.utils import Identity


@dataclass
class ClassificationResult:
    categories: Iterable[str]
    predictions: pd.DataFrame
    y_proba: NDArray
    y_pred_numeric: NDArray[np.integer]
    _y_proba_smoothed: Optional[NDArray] = None
    _y_pred_numeric_smoothed: Optional[NDArray[np.integer]] = None
    _annotations: Optional[pd.DataFrame] = None
    _y_true_numeric: Optional[NDArray[np.integer]] = None

    @property
    def y_proba_smoothed(self):
        if self._y_proba_smoothed is None:
            raise ValueError("no smoothing functions applied during classification")
        return self._y_proba_smoothed

    @property
    def y_pred_numeric_smoothed(self):
        if self._y_pred_numeric_smoothed is None:
            raise ValueError("no smoothing functions applied during classification")
        return self._y_pred_numeric_smoothed

    @property
    def annotations(self):
        if self._annotations is None:
            raise ValueError("classification on non-annotated sampleable")
        return self._annotations

    @property
    def y_true_numeric(self):
        if self._y_true_numeric is None:
            raise ValueError("classification on non-annotated sampleable")
        return self._y_true_numeric


class _NestedResult:
    classification_results: dict[
        Identity | tuple[Identity, Identity],
        "ClassificationResult | GroupClassificationResult",
    ]

    @property
    def y_proba(self) -> NDArray:
        y_proba = []
        for classification_result in self.classification_results.values():
            y_proba.append(classification_result.y_proba)
        return np.concatenate(y_proba, axis=0)

    @property
    def y_pred_numeric(self) -> NDArray:
        y_pred_numeric = []
        for classification_result in self.classification_results.values():
            y_pred_numeric.append(classification_result.y_pred_numeric)
        return np.concatenate(y_pred_numeric, axis=0)

    @property
    def y_proba_smoothed(self) -> NDArray:
        y_proba_smoothed = []
        for classification_result in self.classification_results.values():
            y_proba_smoothed.append(classification_result.y_proba_smoothed)
        return np.concatenate(y_proba_smoothed, axis=0)

    @property
    def y_pred_numeric_smoothed(self) -> NDArray:
        y_pred_numeric_smoothed = []
        for classification_result in self.classification_results.values():
            y_pred_numeric_smoothed.append(
                classification_result.y_pred_numeric_smoothed
            )
        return np.concatenate(y_pred_numeric_smoothed, axis=0)

    @property
    def y_true_numeric(self) -> NDArray:
        y_true_numeric = []
        for classification_result in self.classification_results.values():
            y_true_numeric.append(classification_result.y_true_numeric)
        return np.concatenate(y_true_numeric, axis=0)


@dataclass
class GroupClassificationResult(_NestedResult):
    classification_results: dict[  # type: ignore
        Identity | tuple[Identity, Identity], ClassificationResult
    ]
    trajectories: dict[Identity, Trajectory]

    @property
    def predictions(self) -> pd.DataFrame:
        predictions = []
        for key, classification_result in self.classification_results.items():
            predictions_key = classification_result.predictions
            if isinstance(key, tuple):
                predictions_key[["actor", "recipient"]] = key
            else:
                predictions_key["actor"] = key
            predictions.append(predictions_key)
        return pd.concat(predictions, axis=0)

    @property
    def annotations(self) -> pd.DataFrame:
        annotations = []
        for key, classification_result in self.classification_results.items():
            annotations_key = classification_result.annotations
            if isinstance(key, tuple):
                annotations_key[["actor", "recipient"]] = key
            else:
                annotations_key["actor"] = key
            annotations.append(annotations_key)
        return pd.concat(annotations, axis=0)


@dataclass
class DatasetClassificationResult(_NestedResult):
    classification_results: dict[Identity, GroupClassificationResult]  # type: ignore

    @property
    def predictions(self) -> pd.DataFrame:
        predictions = []
        for key, classification_result in self.classification_results.items():
            predictions_key = classification_result.predictions
            predictions_key["group"] = key
            predictions.append(predictions_key)
        return pd.concat(predictions, axis=0)

    @property
    def annotations(self) -> pd.DataFrame:
        annotations = []
        for key, classification_result in self.classification_results.items():
            annotations_key = classification_result.annotations
            annotations_key["group"] = key
            annotations.append(annotations_key)
        return pd.concat(annotations, axis=0)

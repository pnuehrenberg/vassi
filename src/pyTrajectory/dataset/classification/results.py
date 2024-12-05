from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from ...data_structures import Trajectory
from ...series_operations import smooth
from ...utils import NDArray_to_NDArray
from ..types.utils import DyadIdentity, Identity
from .utils import score_category_counts, to_predictions, validate_predictions


class _Result:
    def score_category_counts(self) -> NDArray:
        annotations: pd.DataFrame = self.annotations  # type: ignore
        predictions: pd.DataFrame = self.predictions  # type: ignore
        categories: Iterable[str] = self.categories  # type: ignore
        return score_category_counts(annotations, predictions, categories)

    def f1_score(
        self,
        per: Literal["timestamp", "annotation", "prediction"],
        *,
        average: Optional[Literal["micro", "macro", "weighted"]] = None,
        encode_func: Callable[[NDArray], NDArray[np.int64]],
    ) -> float | tuple[float, ...]:
        categories: tuple[str, ...] = tuple(self.categories)  # type: ignore
        if per == "timestamp":
            y_true = self.y_true_numeric  # type: ignore
            y_pred = self.y_pred_numeric  # type: ignore
        elif per == "annotation":
            annotations: pd.DataFrame = self.annotations  # type: ignore
            y_true = encode_func(annotations["category"].to_numpy())
            y_pred = encode_func(annotations["predicted_category"].to_numpy())
        elif per == "prediction":
            predictions: pd.DataFrame = self.predictions  # type: ignore
            y_true = encode_func(predictions["true_category"].to_numpy())
            y_pred = encode_func(predictions["category"].to_numpy())
        else:
            raise ValueError(
                f"'per' should be one of 'timestamp', 'annotation', 'prediction' and not '{per}'"
            )
        return f1_score(
            y_true,
            y_pred,
            labels=range(len(categories)),
            average=average,  # type: ignore
            zero_division=np.nan,  # type: ignore
        )

    def score(self, encode_func: Callable[[NDArray], NDArray[np.int64]]) -> NDArray:
        category_count_scores = self.score_category_counts()
        f1_per_timestamp = np.asarray(
            self.f1_score("timestamp", encode_func=encode_func)
        )
        f1_per_annotation = np.asarray(
            self.f1_score("annotation", encode_func=encode_func)
        )
        f1_per_prediction = np.asarray(
            self.f1_score("prediction", encode_func=encode_func)
        )
        return np.array(
            [
                category_count_scores,
                f1_per_timestamp,
                f1_per_annotation,
                f1_per_prediction,
            ]
        )


@dataclass
class ClassificationResult(_Result):
    categories: Iterable[str]
    predictions: pd.DataFrame
    timestamps: NDArray[np.int64]
    y_proba: NDArray
    y_pred_numeric: NDArray[np.int64]
    _y_proba_smoothed: Optional[NDArray] = None
    _y_pred_numeric_smoothed: Optional[NDArray[np.int64]] = None
    _annotations: Optional[pd.DataFrame] = None
    _y_true_numeric: Optional[NDArray[np.int64]] = None

    def smooth(self, label_smoothing_funcs: list[NDArray_to_NDArray]) -> None:
        self._y_proba_smoothed = smooth(
            self.y_proba, filter_funcs=label_smoothing_funcs
        )
        self._y_pred_numeric_smoothed = np.argmax(self._y_proba_smoothed, axis=1)
        self.predictions = to_predictions(
            self.y_pred_numeric_smoothed,
            self.y_proba_smoothed,
            category_names=self.categories,
            timestamps=self.timestamps,
        )
        if self.annotations is not None:
            self.predictions = validate_predictions(
                self.predictions, self.annotations, on="predictions"
            )
            self._annotations = validate_predictions(
                self.predictions, self.annotations, on="annotations"
            )

    @property
    def y_proba_smoothed(self):
        if self._y_proba_smoothed is None:
            raise ValueError("no smoothing functions applied")
        return self._y_proba_smoothed

    @property
    def y_pred_numeric_smoothed(self):
        if self._y_pred_numeric_smoothed is None:
            raise ValueError("no smoothing functions applied")
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


class _NestedResult(_Result):
    classification_results: dict[
        Identity | DyadIdentity,
        "ClassificationResult | GroupClassificationResult",
    ]

    def smooth(self, label_smoothing_funcs: list[NDArray_to_NDArray]) -> None:
        for classification_result in self.classification_results.values():
            classification_result.smooth(label_smoothing_funcs)

    @property
    def categories(self) -> tuple[str, ...]:
        categories = None
        for classification_result in self.classification_results.values():
            if categories is None:
                categories = tuple(classification_result.categories)
                continue
            if tuple(classification_result.categories) != categories:
                raise ValueError("classification results with missmatched categories")
        if categories is None:
            raise ValueError(
                "nested results should contain at least one classification result"
            )
        return categories

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
        Identity | DyadIdentity, ClassificationResult
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
        return pd.concat(predictions, axis=0, ignore_index=True)

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
        return pd.concat(annotations, axis=0, ignore_index=True)


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
        return pd.concat(predictions, axis=0, ignore_index=True)

    @property
    def annotations(self) -> pd.DataFrame:
        annotations = []
        for key, classification_result in self.classification_results.items():
            annotations_key = classification_result.annotations
            annotations_key["group"] = key
            annotations.append(annotations_key)
        return pd.concat(annotations, axis=0, ignore_index=True)

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Callable, Literal, Optional, Self, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from ..data_structures import Trajectory
from ..dataset import (
    AnnotatedDataset,
    AnnotatedGroup,
    GroupIdentifier,
    Identifier,
    IndividualIdentifier,
)
from ..dataset.observations import (
    infill_observations,
    remove_overlapping_observations,
)
from ..dataset.types import encode_categories
from ..logging import set_logging_level
from ..utils import SmoothingFunction
from .utils import (
    _filter_recipient_bouts,
    to_predictions,
    validate_predictions,
)


class _Result:
    def f1_score(
        self,
        on: Literal["timestamp", "annotation", "prediction"],
    ) -> pd.Series:
        categories: tuple[str, ...] = tuple(self.categories)  # type: ignore  # better done via abstract base class!
        encoding_function = partial(encode_categories, categories=categories)
        if on == "timestamp":
            y_true = self.y_true_numeric  # type: ignore
            y_pred = self.y_pred_numeric  # type: ignore
        elif on == "annotation":
            annotations: pd.DataFrame = self.annotations  # type: ignore
            y_true = encoding_function(annotations["category"].to_numpy())
            y_pred = encoding_function(annotations["predicted_category"].to_numpy())
        elif on == "prediction":
            predictions: pd.DataFrame = self.predictions  # type: ignore
            y_true = encoding_function(predictions["true_category"].to_numpy())
            y_pred = encoding_function(predictions["category"].to_numpy())
        else:
            raise ValueError(
                f"'on' should be one of 'timestamp', 'annotation', 'prediction' and not '{on}'"
            )
        scores = f1_score(
            y_true,
            y_pred,
            labels=range(len(categories)),
            average=None,  # type: ignore
            zero_division=1.0,  # type: ignore
        )
        return pd.Series(scores, index=categories, name=on)

    def score(self) -> pd.DataFrame:
        categories: tuple[str, ...] = tuple(self.categories)  # type: ignore
        levels = ("timestamp", "annotation", "prediction")
        return pd.DataFrame([self.f1_score(level) for level in levels])

    def _remove_overlapping_predictions(
        self,
        priority_func: Callable[[pd.DataFrame], Iterable[float]],
        *,
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        raise NotImplementedError(
            "this should be implemented by subclasses if applicable"
        )


def _smooth_probabilities(
    classification_result: "ClassificationResult",
    smoothing_func: SmoothingFunction,
    threshold: bool,
    decision_thresholds: Optional[Iterable[float]],
    default_decision: int | str,
) -> "ClassificationResult":
    """Wrapper for smoothing probabilities in a classification result. Used in parallel processing."""
    return classification_result.smooth(
        smoothing_func,
        threshold=threshold,
        decision_thresholds=decision_thresholds,
        default_decision=default_decision,
    )


def _threshold_probabilities(
    classification_result: "ClassificationResult",
    decision_thresholds: Optional[Iterable[float]],
    default_decision: int | str,
) -> "ClassificationResult":
    """Wrapper for thresholding probabilities in a classification result. Used in parallel processing."""
    return classification_result.threshold(
        decision_thresholds,
        default_decision=default_decision,
    )


@dataclass
class ClassificationResult(_Result):
    categories: tuple[str, ...]
    timestamps: NDArray[np.integer]
    y_proba: NDArray
    y_pred_numeric: NDArray[np.integer]
    _y_proba_smoothed: Optional[NDArray] = None
    _predictions: Optional[pd.DataFrame] = None
    _annotations: Optional[pd.DataFrame] = None
    _y_true_numeric: Optional[NDArray[np.integer]] = None

    def _apply_thresholds(
        self,
        decision_thresholds: Optional[Iterable[float]] = None,
        default_decision: int | str = "none",
    ) -> Self:
        if isinstance(default_decision, str):
            default_decision = list(self.categories).index(default_decision)
        probabilities = self.y_proba
        try:
            probabilities = self.y_proba_smoothed
        except ValueError:
            pass
        if decision_thresholds is None:
            self.y_pred_numeric = np.argmax(probabilities, axis=1)
            return self
        decision_thresholds = list(decision_thresholds)
        if len(decision_thresholds) != probabilities.shape[1]:
            raise ValueError(
                f"number of decision thresholds ({len(decision_thresholds)}) does not match number of categories ({probabilities.shape[1]})"
            )
        probabilities = deepcopy(probabilities)
        for idx, threshold in enumerate(decision_thresholds):
            probabilities[:, idx] = np.where(
                probabilities[:, idx] >= threshold, probabilities[:, idx], 0
            )
        # if no category surpasses threshold, force background category
        probabilities[probabilities.sum(axis=1) == 0, default_decision] = 1
        self.y_pred_numeric = np.argmax(probabilities, axis=1)
        return self

    def threshold(
        self,
        decision_thresholds: Optional[Iterable[float]] = None,
        *,
        default_decision: int | str = "none",
    ) -> Self:
        self._apply_thresholds(decision_thresholds, default_decision)
        probabilties = self.y_proba
        try:
            probabilties = self.y_proba_smoothed
        except ValueError:
            pass
        self._predictions = to_predictions(
            self.y_pred_numeric,
            probabilties,
            category_names=self.categories,
            timestamps=self.timestamps,
        )
        if self.annotations is not None:
            self._predictions = validate_predictions(
                self.predictions, self.annotations, on="predictions"
            )
            self._annotations = validate_predictions(
                self.predictions, self.annotations, on="annotations"
            )
        return self

    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        threshold: bool = True,
        decision_thresholds: Optional[Iterable[float]] = None,
        default_decision: int | str = "none",
    ) -> Self:
        self._y_proba_smoothed = smoothing_func(array=self.y_proba)
        if threshold:
            return self.threshold(
                decision_thresholds,
                default_decision=default_decision,
            )
        return self

    @property
    def y_proba_smoothed(self):
        if self._y_proba_smoothed is None:
            raise ValueError("no smoothing functions applied")
        return self._y_proba_smoothed

    @property
    def predictions(self):
        if self._predictions is None:
            raise ValueError("result not thresholded")
        return self._predictions

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


def _get_target(
    results: Iterable["GroupClassificationResult"]
    | Iterable["DatasetClassificationResult"],
) -> Literal["individual", "dyad"]:
    targets: list[Literal["individual", "dyad"]] = [result.target for result in results]
    target = targets[0]
    if any(target != result_target for result_target in targets):
        raise ValueError(
            "results with mixed targets (dyads and individuals) are not supported"
        )
    return target


@dataclass
class _NestedResult(_Result, ABC):
    classification_results: dict[
        Identifier,
        "ClassificationResult | GroupClassificationResult",
    ]
    target: Literal["individual", "dyad"]

    def _flat_classification_results(self) -> list["ClassificationResult"]:
        classification_results = []
        for classification_result in self.classification_results.values():
            if not isinstance(classification_result, _NestedResult):
                classification_results.append(classification_result)
                continue
            classification_results.extend(
                classification_result._flat_classification_results()
            )
        return classification_results

    def _set_classification_results(
        self, flat_classification_results: list["ClassificationResult"]
    ):
        for key in self.classification_results.keys():
            classification_result = self.classification_results[key]
            if not isinstance(classification_result, _NestedResult):
                self.classification_results[key] = flat_classification_results.pop(0)
                continue
            classification_result._set_classification_results(
                flat_classification_results
            )

    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        threshold: bool = True,
        decision_thresholds: Optional[Iterable[float]] = None,
        default_decision: int | str = "none",
    ) -> Self:
        classification_results = self._flat_classification_results()
        num_cpus = cpu_count()
        num_inner_threads = num_cpus // 4
        num_jobs = num_cpus // num_inner_threads
        with parallel_config(backend="loky", inner_max_num_threads=num_inner_threads):
            results = Parallel(n_jobs=num_jobs)(
                delayed(_smooth_probabilities)(
                    classification_result,
                    smoothing_func,
                    threshold,
                    decision_thresholds,
                    default_decision,
                )
                for classification_result in classification_results
            )
        classification_results: list[ClassificationResult] = cast(
            list[ClassificationResult], results
        )
        self._set_classification_results(classification_results)
        return self

    def threshold(
        self,
        decision_thresholds: Optional[Iterable[float]] = None,
        *,
        default_decision: int | str = "none",
    ) -> Self:
        classification_results = self._flat_classification_results()
        num_cpus = cpu_count()
        num_inner_threads = num_cpus // 4
        num_jobs = num_cpus // num_inner_threads
        with parallel_config(backend="loky", inner_max_num_threads=num_inner_threads):
            results = Parallel(n_jobs=num_jobs)(
                delayed(_threshold_probabilities)(
                    classification_result,
                    decision_thresholds,
                    default_decision,
                )
                for classification_result in classification_results
            )
        classification_results: list[ClassificationResult] = cast(
            list[ClassificationResult], results
        )
        self._set_classification_results(classification_results)
        return self

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

    def remove_overlapping_predictions(
        self,
        *,
        priority_func: Callable[[pd.DataFrame], Iterable[float]],
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        try:
            return self._remove_overlapping_predictions(
                priority_func,
                prefilter_recipient_bouts=prefilter_recipient_bouts,
                max_bout_gap=max_bout_gap,
                max_allowed_bout_overlap=max_allowed_bout_overlap,
            )
        except NotImplementedError:
            pass
        for classification_result in self.classification_results.values():
            try:
                classification_result._remove_overlapping_predictions(
                    priority_func,
                    prefilter_recipient_bouts=prefilter_recipient_bouts,
                    max_bout_gap=max_bout_gap,
                    max_allowed_bout_overlap=max_allowed_bout_overlap,
                )
            except NotImplementedError:
                pass
        return self

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
    def y_true_numeric(self) -> NDArray:
        y_true_numeric = []
        for classification_result in self.classification_results.values():
            y_true_numeric.append(classification_result.y_true_numeric)
        return np.concatenate(y_true_numeric, axis=0)

    @property
    @abstractmethod
    def annotations(self) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def predictions(self) -> pd.DataFrame: ...


@dataclass
class GroupClassificationResult(_NestedResult):
    classification_results: dict[  # type: ignore
        Identifier, ClassificationResult
    ]
    # trajectories: dict[IndividualIdentifier, Trajectory]
    individuals: tuple[IndividualIdentifier, ...]

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

    def _remove_overlapping_predictions(
        self,
        priority_func: Callable[[pd.DataFrame], Iterable[float]],
        *,
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        predictions = self.predictions
        if "recipient" not in predictions.columns:
            set_logging_level().warning(
                "individual predictions (not dyadic) cannot overlap"
            )
            return self
        predictions = predictions[predictions["category"] != "none"]
        for actor in self.individuals:
            predictions_actor = predictions.loc[
                predictions["actor"] == actor
            ].reset_index(drop=True, inplace=False)
            if TYPE_CHECKING:
                # reset_index with inplace=False not correctly detected by pyright
                assert predictions_actor is not None
            if prefilter_recipient_bouts:
                predictions_actor = _filter_recipient_bouts(
                    predictions_actor,
                    priority_func=priority_func,
                    max_bout_gap=max_bout_gap,
                    max_allowed_bout_overlap=max_allowed_bout_overlap,
                )
            predictions_actor = remove_overlapping_observations(
                predictions_actor,
                priority_func=priority_func,
                max_allowed_overlap=0,
                index_columns=(),
            )
            for recipient in self.individuals:
                if (actor, recipient) not in self.classification_results:
                    continue
                classification_result = self.classification_results[(actor, recipient)]
                predictions_dyad: pd.DataFrame = predictions_actor[
                    predictions_actor["recipient"] == recipient
                ].reset_index(drop=True)  # type: ignore
                predictions_dyad = infill_observations(
                    predictions_dyad,
                    observation_stop=classification_result.timestamps[-1],
                )
                try:
                    predictions_dyad.loc[:, "mean_probability"] = predictions_dyad[
                        "mean_probability"
                    ].fillna(value=0)
                except KeyError:
                    predictions_dyad["mean_probability"] = 0
                try:
                    predictions_dyad.loc[:, "max_probability"] = predictions_dyad[
                        "max_probability"
                    ].fillna(value=0)
                except KeyError:
                    predictions_dyad["max_probability"] = 0
                try:
                    predictions_dyad = validate_predictions(
                        predictions_dyad,
                        classification_result.annotations,
                        on="predictions",
                        key_columns=[],
                    )
                except ValueError:
                    pass
                drop = [
                    column
                    for column in ["actor", "recipient"]
                    if column in predictions_dyad.columns
                ]
                if len(drop) > 0:
                    predictions_dyad = predictions_dyad.drop(
                        columns=drop, inplace=False
                    )
                classification_result._predictions = predictions_dyad
        return self

    @classmethod
    def combine(
        cls, results: Iterable["GroupClassificationResult"]
    ) -> "GroupClassificationResult":
        target = _get_target(results)
        individuals: tuple[IndividualIdentifier, ...] = tuple(
            sorted(set.union(*[set(result.individuals) for result in results]))
        )
        classification_results = {}
        for result in results:
            for key, classification_result in result.classification_results.items():
                if key in classification_results:
                    raise ValueError(
                        "cannot combine GroupClassificationResults with multiple classification results for the same sampleable"
                    )
                classification_results[key] = classification_result
        return cls(
            classification_results={
                key: classification_results[key]
                for key in sorted(classification_results)
            },
            individuals=individuals,
            target=target,
        )


@dataclass
class DatasetClassificationResult(_NestedResult):
    classification_results: dict[GroupIdentifier, GroupClassificationResult]  # type: ignore

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

    def to_dataset(
        self,
        *,
        trajectories: Mapping[
            GroupIdentifier, Mapping[IndividualIdentifier, Trajectory]
        ],
        background_category: str,
    ) -> AnnotatedDataset:
        categories = tuple(np.unique(list(self.predictions["category"])))
        return AnnotatedDataset.from_groups(
            {
                identifier: AnnotatedGroup(
                    trajectories[identifier],
                    target=self.target,
                    observations=group_result.predictions,
                    categories=categories,
                    background_category=background_category,  # TODO: fix this
                )
                for identifier, group_result in self.classification_results.items()
            }
        )

    @classmethod
    def combine(
        cls, results: Iterable["DatasetClassificationResult"]
    ) -> "DatasetClassificationResult":
        target = _get_target(results)
        classification_results = {}
        for result in results:
            for (
                group_key,
                classification_result,
            ) in result.classification_results.items():
                if group_key in classification_results:
                    classification_results[group_key] = (
                        GroupClassificationResult.combine(
                            [classification_results[group_key], classification_result]
                        )
                    )
                else:
                    classification_results[group_key] = classification_result
        return cls(
            classification_results={
                group_key: classification_results[group_key]
                for group_key in sorted(classification_results)
            },
            target=target,
        )

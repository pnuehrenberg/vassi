import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
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
from ..io import load_data, save_data
from ..logging import set_logging_level
from ..utils import SmoothingFunction, available_resources, to_scalars
from .utils import (
    _filter_recipient_bouts,
    to_predictions,
    validate_predictions,
)


class BaseResult:
    """Base class for classification results."""

    def f1_score(
        self,
        on: Literal["timestamp", "annotation", "prediction"],
    ) -> pd.Series:
        """
        Calculate the F1 score for the given data.

        Parameters:
            on: The level on which to calculate the F1 scores.

                - :code:`"timestamp"`: Calculate the F1 scores across all samples (timestamps).
                - :code:`"annotation"`: Calculate the F1 scores based on the annotated intervals.
                - :code:`"prediction"`: Calculate the F1 scores based on the predicted intervals.
        """
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
        """
        Return a summary of the F1 scores for each category at different levels.

        Returns:
            A DataFrame containing the F1 scores for each category at different levels.

        See Also:
            :meth:`f1_score`
        """
        levels = ("timestamp", "annotation", "prediction")
        return pd.DataFrame([self.f1_score(level) for level in levels])

    def _remove_overlapping_predictions(
        self,
        priority_function: Callable[[pd.DataFrame], Iterable[float]],
        *,
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        raise NotImplementedError(
            "this should be implemented by subclasses if applicable"
        )

    def remove_overlapping_predictions(
        self,
        *,
        priority_function: Callable[[pd.DataFrame], Iterable[float]],
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        """
        Remove overlapping predictions from the classification result.

        This method is implemented by subclasses if applicable.

        Parameters:
            priority_function: A function that assigns a priority to each prediction.
            prefilter_recipient_bouts: Whether to prefilter recipient bouts.
            max_bout_gap: The maximum allowed gap between predictions.
            max_allowed_bout_overlap: The maximum allowed overlap between predictions.
        """
        return self._remove_overlapping_predictions(
            priority_function,
            prefilter_recipient_bouts=prefilter_recipient_bouts,
            max_bout_gap=max_bout_gap,
            max_allowed_bout_overlap=max_allowed_bout_overlap,
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
class ClassificationResult(BaseResult):
    categories: tuple[str, ...]
    timestamps: np.ndarray
    y_proba: np.ndarray
    y_pred_numeric: np.ndarray
    _y_proba_smoothed: Optional[np.ndarray] = None
    _predictions: Optional[pd.DataFrame] = None
    _annotations: Optional[pd.DataFrame] = None
    _y_true_numeric: Optional[np.ndarray] = None

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
    ) -> "ClassificationResult":
        """
        Apply decision thresholds to the classification result.

        Parameters:
            decision_thresholds: A list of decision thresholds for each category.
            default_decision: The default decision to apply if no category surpasses the threshold.
        """
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
        try:
            annotations = self.annotations
            self._predictions = validate_predictions(
                self.predictions, annotations, on="predictions"
            )
            self._annotations = validate_predictions(
                self.predictions, annotations, on="annotations"
            )
        except ValueError:
            pass
        return self

    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        threshold: bool = True,
        decision_thresholds: Optional[Iterable[float]] = None,
        default_decision: int | str = "none",
    ) -> "ClassificationResult":
        """
        Apply smoothing functions to the classification result.

        Parameters:
            smoothing_func: The smoothing function to apply.
            threshold: Whether to threshold the result.
            decision_thresholds: The decision thresholds to use.
            default_decision: The default decision to use.

        Returns:
            The smoothed classification result.
        """
        self._y_proba_smoothed = smoothing_func(array=self.y_proba)
        if threshold:
            return self.threshold(
                decision_thresholds,
                default_decision=default_decision,
            )
        return self

    @property
    def y_proba_smoothed(self):
        """Smoothed classification probabilities."""
        if self._y_proba_smoothed is None:
            raise ValueError("no smoothing functions applied")
        return self._y_proba_smoothed

    @property
    def predictions(self) -> pd.DataFrame:
        """
        Predicted intervals.

        Raises:
            ValueError: If the result is not thresholded.
        """
        if self._predictions is None:
            raise ValueError("result not thresholded")
        return self._predictions

    @property
    def annotations(self) -> pd.DataFrame:
        """
        Annotated intervals.

        Raises:
            ValueError: If the result is not annotated.
        """
        if self._annotations is None:
            raise ValueError("classification on non-annotated sampleable")
        return self._annotations

    @property
    def y_true_numeric(self):
        """
        True labels, represented as integers.

        Raises:
            ValueError: If the result is not annotated.
        """
        if self._y_true_numeric is None:
            raise ValueError("classification on non-annotated sampleable")
        return self._y_true_numeric

    def to_h5(self, data_file: str, data_path: str):
        """
        Save the classification results to an HDF5 file.

        Args:
            data_file (str): Path to the HDF5 file.
            data_path (str): Path within the HDF5 file to save the data.
        """
        data = {
            "categories": np.array(self.categories),
            "timestamps": self.timestamps,
            "y_proba": self.y_proba,
            "y_pred_numeric": self.y_pred_numeric,
        }
        if self._y_proba_smoothed is not None:
            data["_y_proba_smoothed"] = self._y_proba_smoothed
        if self._y_true_numeric is not None:
            data["_y_true_numeric"] = self._y_true_numeric
        save_data(data_file, data, os.path.join(data_path, "data"))
        if self._predictions is not None:
            self._predictions.to_hdf(
                data_file, key=os.path.join(data_path, "predictions")
            )
        if self._annotations is not None:
            self._annotations.to_hdf(
                data_file, key=os.path.join(data_path, "annotations")
            )

    @classmethod
    def from_h5(cls, data_file: str, data_path: str) -> "ClassificationResult":
        """
        Load classification results from an HDF5 file.

        Args:
            data_file (str): Path to the HDF5 file.
            data_path (str): Path within the HDF5 file to load the data.
        """
        _data = load_data(data_file, os.path.join(data_path, "data"))
        if TYPE_CHECKING:
            assert isinstance(_data, dict)
        data: dict[str, Any] = {**_data}
        data["categories"] = tuple(data["categories"])
        try:
            data["_predictions"] = pd.read_hdf(
                data_file, os.path.join(data_path, "predictions")
            )
        except KeyError:
            data["_predictions"] = None
        try:
            data["_annotations"] = pd.read_hdf(
                data_file, os.path.join(data_path, "annotations")
            )
        except KeyError:
            data["_annotations"] = None
        return cls(**data)


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
class _NestedResult(BaseResult, ABC):
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
        num_jobs, num_inner_threads = available_resources()
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
        num_jobs, num_inner_threads = available_resources()
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
        priority_function: Callable[[pd.DataFrame], Iterable[float]],
        prefilter_recipient_bouts: bool,
        max_bout_gap: float,
        max_allowed_bout_overlap: float,
    ) -> Self:
        try:
            return self._remove_overlapping_predictions(
                priority_function,
                prefilter_recipient_bouts=prefilter_recipient_bouts,
                max_bout_gap=max_bout_gap,
                max_allowed_bout_overlap=max_allowed_bout_overlap,
            )
        except NotImplementedError:
            pass
        for classification_result in self.classification_results.values():
            try:
                classification_result._remove_overlapping_predictions(
                    priority_function,
                    prefilter_recipient_bouts=prefilter_recipient_bouts,
                    max_bout_gap=max_bout_gap,
                    max_allowed_bout_overlap=max_allowed_bout_overlap,
                )
            except NotImplementedError:
                pass
        return self

    @property
    def y_proba(self) -> np.ndarray:
        y_proba = []
        for classification_result in self.classification_results.values():
            y_proba.append(classification_result.y_proba)
        return np.concatenate(y_proba, axis=0)

    @property
    def y_pred_numeric(self) -> np.ndarray:
        y_pred_numeric = []
        for classification_result in self.classification_results.values():
            y_pred_numeric.append(classification_result.y_pred_numeric)
        return np.concatenate(y_pred_numeric, axis=0)

    @property
    def y_proba_smoothed(self) -> np.ndarray:
        y_proba_smoothed = []
        for classification_result in self.classification_results.values():
            y_proba_smoothed.append(classification_result.y_proba_smoothed)
        return np.concatenate(y_proba_smoothed, axis=0)

    @property
    def y_true_numeric(self) -> np.ndarray:
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
        """Concatenated predictions of all classification results."""
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
        """Concatenated annotations of all classification results."""
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
        priority_function: Callable[[pd.DataFrame], Iterable[float]],
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
                    priority_function=priority_function,
                    max_bout_gap=max_bout_gap,
                    max_allowed_bout_overlap=max_allowed_bout_overlap,
                )
            predictions_actor = remove_overlapping_observations(
                predictions_actor,
                priority_function=priority_function,
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
        """
        Combine multiple :class:`GroupClassificationResult` into a single one.

        Parameters:
            results: Iterable of :class:`GroupClassificationResult` to combine.

        Returns:
            A new :class:`GroupClassificationResult` object containing the combined results.
        """
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

    def to_h5(self, data_file: str, data_path: str):
        """
        Save the GroupClassificationResult to an HDF5 file.

        Parameters:
            data_file: Path to the HDF5 file.
            data_path: Path within the HDF5 file to save the data.
        """
        identifiers = list(self.classification_results)
        save_data(
            data_file,
            {
                "individuals": np.array(self.individuals),
                "identifiers": np.array(identifiers),
                "target": np.array([self.target]),
            },
            data_path,
        )
        for identifier in identifiers:
            self.classification_results[identifier].to_h5(
                data_file, os.path.join(data_path, "results", str(identifier))
            )

    @classmethod
    def from_h5(cls, data_file: str, data_path: str) -> "GroupClassificationResult":
        """
        Load a :class:`GroupClassificationResult` from a HDF5 file.

        Parameters:
            data_file (str): The path to the HDF5 file.
            data_path (str): The path to the data within the HDF5 file.

        Returns:
            The loaded group classification result.
        """
        individuals = tuple(
            load_data(data_file, os.path.join(data_path, "individuals"))
        )
        target = to_scalars(load_data(data_file, os.path.join(data_path, "target")))[0]
        identifiers = [
            cast(
                Identifier,
                identifier if isinstance(identifier, str) else tuple(identifier),
            )
            for identifier in to_scalars(
                load_data(data_file, os.path.join(data_path, "identifiers"))
            )
        ]
        classification_results = {
            identifier: ClassificationResult.from_h5(
                data_file, os.path.join(data_path, "results", str(identifier))
            )
            for identifier in identifiers
        }
        return cls(classification_results, target, individuals)


@dataclass
class DatasetClassificationResult(_NestedResult):
    classification_results: dict[GroupIdentifier, GroupClassificationResult]  # type: ignore

    @property
    def predictions(self) -> pd.DataFrame:
        """Concatenated predictions of all groups."""
        predictions = []
        for key, classification_result in self.classification_results.items():
            predictions_key = classification_result.predictions
            predictions_key["group"] = key
            predictions.append(predictions_key)
        return pd.concat(predictions, axis=0, ignore_index=True)

    @property
    def annotations(self) -> pd.DataFrame:
        """Concatenated annotations of all groups."""
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
        """
        Convert the classification results to an annotated dataset.

        Parameters:
            trajectories: Mapping from group identifier to mapping from individual identifier to trajectory.
            background_category: Category to use as background category.

        Returns:
            Annotated dataset.
        """
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
        """
        Combine multiple dataset classification results into a single result.

        Parameters:
            results: The dataset classification results to combine.

        Returns:
            A new :class:`DatasetClassificationResult` with the combined results.
        """
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

    def to_h5(self, data_file: str, *, dataset_name: str):
        """
        Save the dataset classification result to an HDF5 file.

        Parameters:
            data_file: The path to the HDF5 file.
            dataset_name: The name of the dataset within the HDF5 file.
        """
        identifiers = list(self.classification_results)
        save_data(
            data_file,
            {"identifiers": np.array(identifiers), "target": np.array([self.target])},
            dataset_name,
        )
        for identifier in identifiers:
            self.classification_results[identifier].to_h5(
                data_file, os.path.join(dataset_name, "group_results", str(identifier))
            )

    @classmethod
    def from_h5(cls, data_file: str, *, dataset_name: str) -> "DatasetClassificationResult":
        """
        Load the dataset classification result from an HDF5 file.

        Parameters:
            data_file: The path to the HDF5 file.
            dataset_name: The name of the dataset within the HDF5 file.

        Returns:
            The loaded dataset classification result.
        """
        target = to_scalars(load_data(data_file, os.path.join(dataset_name, "target")))[
            0
        ]
        identifiers = to_scalars(
            load_data(data_file, os.path.join(dataset_name, "identifiers"))
        )
        classification_results = {
            identifier: GroupClassificationResult.from_h5(
                data_file, os.path.join(dataset_name, "group_results", str(identifier))
            )
            for identifier in identifiers
        }
        return cls(classification_results, target)

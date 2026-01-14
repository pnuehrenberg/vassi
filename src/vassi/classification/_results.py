from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from copy import copy, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal, Self, override

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score  # pyright: ignore[reportUnknownVariableType]
from tabularintervals import aggregate, densify, resolve

from ..dataset.types import WithCategories, get_validators
from ..distributed import get_process_state, limited_process_pool
from ..io.h5 import (
    get_h5_keys,
    read_h5_attrs,
    read_h5_data,
    write_h5_array,
    write_h5_attrs,
    write_h5_data,
)
from ..type_guards import is_tuple_of, is_valid_classification_data
from ..utils import SmoothingFunction
from ..warnings import warn
from .utils import (
    to_predictions,
    validate_predictions,
)


def _discretize[T: Classification](
    classification: T, decision_thresholds: Mapping[str, float] | None
) -> T:
    with get_process_state():
        return classification.discretize(decision_thresholds)


def _smooth[T: Classification](
    classification: T,
    smoothing_function: SmoothingFunction,
    decision_thresholds: Mapping[str, float] | None,
) -> T:
    with get_process_state():
        return classification.smooth(
            smoothing_function, decision_thresholds=decision_thresholds
        )


class BaseClassification(WithCategories, ABC):
    y_proba: np.ndarray
    y: np.ndarray
    predictions: pl.DataFrame
    _discretized: bool = False

    def __replace__(self, **changes: ...):
        new = copy(self)
        vars(new).update(changes)
        return new

    def _update_from(self, other: Self) -> None:
        # this should only be used in __init__
        vars(self).update(vars(other))

    @property
    def discretized(self) -> bool:
        return self._discretized

    @abstractmethod
    def update_predictions(self) -> Self: ...

    @abstractmethod
    def discretize(
        self,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self: ...

    @abstractmethod
    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self: ...

    @classmethod
    def confirm_instance(cls, classification: ...) -> Self:
        assert isinstance(classification, cls), (
            f"Expected classification to be an instance of {cls}, got {type(classification)}"
        )
        return classification

    @abstractmethod
    def to_h5(
        self, data_file: str | Path, *, data_path: str = ".", **attrs: ...
    ) -> None: ...

    @classmethod
    @abstractmethod
    def from_h5(
        cls, data_file: str | Path, *, data_path: str = "."
    ) -> tuple[Hashable, Self]: ...


class RequiresBaseClassification:
    _base: tuple[BaseClassification, type[BaseClassification]] | None = None

    @property
    def base(self) -> tuple[BaseClassification, type[BaseClassification]]:
        if self._base is not None:
            return self._base
        invalid_type_message = f"invalid use of {type(self)}, this mixin class should only be used in subclasses of {BaseClassification}"
        if not isinstance(self, BaseClassification):
            raise TypeError(invalid_type_message)
        base = None
        for _base in type(self).__bases__:
            if not issubclass(_base, BaseClassification):
                continue
            base = _base
            break
        if base is None:
            raise TypeError(invalid_type_message)
        self._base = self, base
        return self._base


class WithGroundTruth(RequiresBaseClassification, ABC):
    # this class should be used in combination with some BaseClassifcation,
    # so that categories and background_category are available

    _y_gt: np.ndarray | None = None
    _annotations: pl.DataFrame | None = None

    def with_ground_truth(
        self,
        y_gt: np.ndarray,
        annotations: pl.DataFrame,
    ):
        self._y_gt = y_gt.astype(np.int8)
        self._annotations = annotations

    @property
    def y_gt(self) -> np.ndarray:
        if self._y_gt is None:
            raise ValueError(f"y_gt not set, call with_ground_truth on {type(self)}")
        return self._y_gt

    @property
    def annotations(self) -> pl.DataFrame:
        if self._annotations is None:
            raise ValueError(
                f"Annotations are not set, call with_ground_truth on {type(self)}"
            )
        return self._annotations

    def f1_score(
        self,
        on: Literal["timestamp", "annotation", "prediction"],
    ) -> pl.DataFrame:
        base, _ = self.base
        if not base.discretized:
            raise ValueError(
                "f1_score is only available for discretized classifications"
            )
        if on == "timestamp":
            y_true = self.y_gt
            y_pred = base.y
        elif on == "annotation":
            annotations: pl.DataFrame = self.annotations
            y_true = base.encode(np.asarray(annotations.get_column("category")))
            y_pred = base.encode(
                np.asarray(annotations.get_column("predicted_category"))
            )
        elif on == "prediction":
            predictions: pl.DataFrame = base.predictions
            y_true = base.encode(np.asarray(predictions.get_column("true_category")))
            y_pred = base.encode(np.asarray(predictions.get_column("category")))
        else:
            raise ValueError(
                f"'on' should be one of 'timestamp', 'annotation', 'prediction' and not '{on}'"
            )
        scores = f1_score(
            y_true,
            y_pred,
            labels=range(
                len(base.categories)
            ),  # does not include -1 if background category is not included in categories
            average=None,  # pyright: ignore[reportArgumentType]
            zero_division=1.0,  # pyright: ignore[reportArgumentType]
        )
        return (
            pl.DataFrame({"score": scores, "category": sorted(base.categories)})
            .with_columns(pl.lit(on).alias("on"))
            .select("category", "on", "score")
        )

    def score(self) -> pl.DataFrame:
        levels = ("timestamp", "annotation", "prediction")
        return pl.concat(self.f1_score(level) for level in levels)

    @abstractmethod
    def validate(self) -> Self: ...


class ClassificationCollection[
    E: BaseClassification,
](BaseClassification, ABC):
    classifications: dict[Hashable, E]
    identifier_validator: Callable[..., Hashable]
    classification_validator: Callable[..., E]
    _level: tuple[str, ...]
    _minimal_classification_type: type[E]

    def __init__(
        self,
        classifications: Mapping[Hashable, BaseClassification],
        *,
        categories: set[str],
        background_category: str,
        identifier_validator: Callable[[object], Hashable],
        classification_validator: Callable[[object], E],
    ):
        self.with_categories(categories, background_category=background_category)
        self.identifier_validator = identifier_validator
        self.classification_validator = classification_validator
        self.classifications = {
            self.identifier_validator(identifier): self.classification_validator(
                classification
            )
            for identifier, classification in classifications.items()
        }
        self._discretized: bool = self._all_discretized
        if self.discretized:
            self.y_proba: np.ndarray = self._concatenate_numpy_attribute("y_proba")
            self.y: np.ndarray = self._concatenate_numpy_attribute("y")
            self.predictions: pl.DataFrame = self._concatenate_polars_attribute(
                "predictions"
            )

    @property
    def _all_discretized(self) -> bool:
        return all(
            classification.discretized
            for classification in self._flat_classifications()
        )

    def __getitem__(self, identifier: Hashable) -> E:
        return self.classifications[identifier]

    def __iter__(self) -> Iterator[tuple[Hashable, E]]:
        return iter(self.classifications.items())

    def __len__(self) -> int:
        return len(self.classifications)

    def identifiers(self) -> list[Hashable]:
        return list(self.classifications.keys())

    def _concatenate_numpy_attribute(self, attribute: str) -> np.ndarray:
        values: list[np.ndarray] = []
        for classification in self._flat_classifications():
            value = getattr(classification, attribute)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Attribute '{attribute}' is not a numpy array")
            values.append(value)
        return np.concatenate(values, axis=0)

    def _concatenate_polars_attribute(self, attribute: str) -> pl.DataFrame:
        all_values: list[pl.DataFrame] = []
        for (_, classification), identifier in zip(self, self.identifiers()):
            if isinstance(classification, ClassificationCollection):
                value = classification._concatenate_polars_attribute(attribute)
            else:
                value = getattr(classification, attribute)
                if not isinstance(value, pl.DataFrame):
                    raise ValueError(
                        f"Attribute '{attribute}' is not a polars DataFrame"
                    )
            all_values.append(
                value.with_columns(
                    pl.lit(i).alias(level)
                    for level, i in zip(
                        self._level,
                        identifier
                        if is_tuple_of(identifier, Hashable)
                        else [identifier],
                    )
                )
            )
        return pl.concat(all_values, how="vertical_relaxed")

    @property
    def _num_flat_classifications(self) -> int:
        num_flat = 0
        for _, classification in self:
            if isinstance(classification, ClassificationCollection):
                num_flat += classification._num_flat_classifications
            else:
                num_flat += 1
        return num_flat

    def _flat_classifications(self) -> list[Classification]:
        classifications: list[Classification] = []
        for _, classification in self:
            if isinstance(classification, ClassificationCollection):
                classifications.extend(classification._flat_classifications())
                continue
            if not isinstance(classification, Classification):
                raise TypeError(f"Expected Classification, got {type(classification)}")
            classifications.append(classification)
        return classifications

    @abstractmethod
    def update_from_flat_classifications(
        self, classifications: Sequence[Classification]
    ) -> Self: ...

    @override
    def update_predictions(self) -> Self:
        classifications = [
            classification.update_predictions()
            for classification in self._flat_classifications()
        ]
        return self.update_from_flat_classifications(classifications)

    @override
    def discretize(
        self,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        with limited_process_pool() as pool:
            classifications = [
                Classification.confirm_instance(classification)
                for classification in pool.map(
                    partial(_discretize, decision_thresholds=decision_thresholds),
                    self._flat_classifications(),
                )
            ]
        return self.update_from_flat_classifications(classifications)

    @override
    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        with limited_process_pool() as pool:
            classifications = [
                Classification.confirm_instance(classification)
                for classification in pool.map(
                    partial(
                        _smooth,
                        smoothing_function=smoothing_func,
                        decision_thresholds=decision_thresholds,
                    ),
                    self._flat_classifications(),
                )
            ]
        return self.update_from_flat_classifications(classifications)

    @abstractmethod
    def resolve_overlapping_predictions(
        self,
        priority: pl.Expr,
        *,
        filter_bouts_by_recipients: bool,
        max_bout_gap: int | None = None,
    ) -> Self: ...

    @classmethod
    @abstractmethod
    def combine(cls, *classifications: Self) -> Self: ...

    @override
    def to_h5(
        self, data_file: str | Path, *, data_path: str = ".", **attrs: ...
    ) -> None:
        for identifier, classification in self:
            classification.to_h5(
                data_file,
                data_path=str(PurePosixPath(data_path) / str(identifier)),
                identifier=identifier,
            )
        write_h5_attrs(data_file, data_path=data_path, **attrs)

    @override
    @classmethod
    def from_h5(
        cls, data_file: str | Path, *, data_path: str = "."
    ) -> tuple[Hashable, Self]:
        attrs = read_h5_attrs(data_file, data_path=data_path)
        identifier = attrs.get("identifier", None)
        classifications = [
            cls._minimal_classification_type.from_h5(
                data_file, data_path=str(PurePosixPath(data_path) / str(key))
            )
            for key in get_h5_keys(data_file, data_path=data_path)
        ]
        if len(classifications) == 0:
            raise ValueError("Expected at least one classification")
        _, first = classifications[0]
        classifications_dict = dict(classifications)
        if len(classifications_dict) != len(classifications):
            raise ValueError(
                "Could not load classifications with non-unique identifiers"
            )
        identifier_validator, classification_validator = get_validators(
            classifications_dict, minimal_value_type=cls._minimal_classification_type
        )
        return identifier, cls(
            classifications_dict,
            categories=set(first.categories),
            background_category=first.background_category,
            identifier_validator=identifier_validator,
            classification_validator=classification_validator,
        )


class Classification(BaseClassification):
    timestamps: np.ndarray

    def __init__(
        self,
        y_proba: np.ndarray,
        *,
        timestamps: np.ndarray,
        categories: set[str],
        background_category: str,
        discretize_on_init: bool,
        **_: ...,  # for compatibility with other classification types
    ):
        self.with_categories(categories, background_category=background_category)
        self.timestamps = timestamps
        self.y_proba: np.ndarray = y_proba
        if discretize_on_init:
            self._update_from(self.discretize())

    @override
    def update_predictions(self) -> Self:
        if not self.discretized:
            raise ValueError("Classification results are not discretized.")
        return replace(
            self,
            predictions=to_predictions(
                self.y,
                self.y_proba,
                self.timestamps,
                set(self.categories),
            ),
        )

    @override
    def discretize(
        self,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        default_decision_idx = self.category_index(self.background_category)
        if decision_thresholds is None:
            y = np.argmax(self.y_proba, axis=1).astype(np.int8)
        else:
            y_proba_thresh = self.y_proba.copy()
            for category, threshold in decision_thresholds.items():
                category_idx = self.category_index(category)
                y_proba_thresh[:, category_idx] = np.where(
                    self.y_proba[:, category_idx] > threshold,
                    self.y_proba[:, category_idx],
                    0,
                )
            y_proba_thresh[y_proba_thresh.sum(axis=1) == 0, default_decision_idx] = (
                1 if default_decision_idx is not None else 0
            )
            y = np.argmax(
                y_proba_thresh, axis=1
            )  # this is only true for non-all-zero rows
            y[
                (y_proba_thresh == 0).all(axis=1)
            ] = -1  # use -1 if background category is not included in categories
        return replace(self, y=y, _discretized=True).update_predictions()

    @override
    def smooth(
        self,
        smoothing_func: SmoothingFunction,
        *,
        decision_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        return replace(self, y_proba=smoothing_func(array=self.y_proba)).discretize(
            decision_thresholds
        )

    @override
    def to_h5(
        self, data_file: str | Path, *, data_path: str = ".", **attrs: ...
    ) -> None:
        write_h5_data(
            data_file,
            data={
                "categories": np.array(list(self.categories)),
                "background_category": np.array([self.background_category]),
                "timestamps": self.timestamps,
                "y_proba": self.y_proba,
                "y": self.y,
            },
            data_path=str(PurePosixPath(data_path) / "classification"),
        )
        self.predictions.to_pandas().to_hdf(
            data_file, key=str(PurePosixPath(data_path) / "predictions")
        )
        write_h5_attrs(data_file, data_path=data_path, **attrs)

    @override
    @classmethod
    def from_h5(
        cls, data_file: str | Path, *, data_path: str = "."
    ) -> tuple[Hashable, Self]:
        attrs = read_h5_attrs(data_file, data_path=data_path)
        identifier = attrs.get("identifier", None)
        if not isinstance(identifier, str) and isinstance(identifier, Iterable):
            identifier = tuple(identifier)
        if not isinstance(identifier, Hashable):
            raise ValueError(f"Invalid identifier {identifier}")
        _data = read_h5_data(
            data_file, data_path=str(PurePosixPath(data_path) / "classification")
        )
        annotations = None
        if "annotations" in get_h5_keys(data_file, data_path=data_path):
            annotations = pd.read_hdf(
                data_file, key=str(PurePosixPath(data_path) / "annotations")
            )
        data = {
            **{str(key): value for key, value in _data.items()},
            "categories": set(map(bytes.decode, _data["categories"].tolist())),
            "background_category": bytes.decode(_data["background_category"][0]),
            "y_gt": _data.get("y_gt", None),
            "annotations": annotations,
        }
        if not is_valid_classification_data(data):
            raise ValueError("Invalid classification data")

        predictions = pl.from_pandas(
            pd.read_hdf(data_file, key=str(PurePosixPath(data_path) / "predictions"))
        )
        if not isinstance(predictions, pl.DataFrame):
            raise ValueError(
                f"Expected predictions to be a DataFrame, got {type(predictions)}"
            )
        y = data.pop("y")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"Expected y to be a numpy array, got {type(y)}")
        return identifier, replace(
            cls(
                **data,
                discretize_on_init=False,
            ),
            _discretized=True,
            y=y,
            predictions=predictions,
        )


class AnnotatedClassification(WithGroundTruth, Classification):
    def __init__(
        self,
        y_proba: np.ndarray,
        *,
        timestamps: np.ndarray,
        categories: set[str],
        background_category: str,
        annotations: pl.DataFrame,
        y_gt: np.ndarray | None,
        discretize_on_init: bool,
    ):
        self.with_categories(categories, background_category=background_category)

        if y_gt is None:
            annotations = densify(
                annotations,
                within=tuple(timestamps[[0, -1]]),
                category=self.background_category,
            )
            duration = (
                annotations.select(
                    (pl.col("stop") - pl.col("start") + 1)
                    .cast(pl.Int64)
                    .alias("duration")
                )
                .get_column("duration")
                .to_numpy()
            )
            category = self.encode(np.asarray(annotations.get_column("category")))
            y_gt = np.repeat(category, duration)

        self.with_ground_truth(y_gt, annotations)

        super().__init__(
            y_proba,
            timestamps=timestamps,
            categories=categories,
            background_category=background_category,
            discretize_on_init=discretize_on_init,
        )

    @override
    def validate(self) -> Self:
        return replace(
            self,
            predictions=validate_predictions(
                self.predictions,
                self.annotations,
                on="predictions",
                background_category=self.background_category,
            ),
            _annotations=validate_predictions(
                self.predictions,
                self.annotations,
                on="annotations",
                background_category=self.background_category,
            ),
        )

    @override
    def update_predictions(self) -> Self:
        return super().update_predictions().validate()

    @override
    def to_h5(
        self, data_file: str | Path, *, data_path: str = ".", **attrs: ...
    ) -> None:
        super().to_h5(data_file, data_path=data_path, **attrs)
        write_h5_array(
            data_file,
            array=self.y_gt,
            data_path=str(PurePosixPath(data_path) / "classification" / "y_gt"),
            key="y_gt",
        )
        self.annotations.to_pandas().to_hdf(
            data_file, key=str(PurePosixPath(data_path) / "annotations")
        )


class GroupClassification(ClassificationCollection[Classification]):
    _target: Literal["individual", "dyad"]
    _level: tuple[str, ...]
    _minimal_classification_type: type[Classification] = Classification

    def __init__(
        self,
        classifications: Mapping[Hashable, Classification],
        *,
        categories: set[str],
        background_category: str,
        identifier_validator: Callable[[object], Hashable] | None = None,
        classification_validator: Callable[[object], Classification] | None = None,
    ):
        if identifier_validator is None or classification_validator is None:
            identifier_validator, classification_validator = get_validators(
                classifications,
                minimal_value_type=Classification,
            )
        self._target = (
            "individual"
            if isinstance(first_key := next(iter(classifications.keys())), str)
            or not isinstance(first_key, Iterable)
            else "dyad"
        )
        if self._target == "individual":
            self._level = ("actor",)
        else:
            self._level = ("actor", "recipient")
        super().__init__(
            classifications,
            categories=categories,
            background_category=background_category,
            identifier_validator=identifier_validator,
            classification_validator=classification_validator,
        )
        self.actors: frozenset[Hashable] = frozenset(
            identifier[0] if is_tuple_of(identifier, Hashable) else identifier
            for identifier in self.identifiers()
        )

    @override
    def update_from_flat_classifications(
        self, classifications: Sequence[Classification]
    ) -> Self:
        if len(classifications) != len(self):
            raise ValueError("Number of classifications does not match")
        return type(self)(
            {
                identifier: classification
                for (identifier, _), classification in zip(self, classifications)
            },
            categories=set(self.categories),
            background_category=self.background_category,
            identifier_validator=self.identifier_validator,
            classification_validator=self.classification_validator,
        )

    @override
    def resolve_overlapping_predictions(
        self,
        priority: pl.Expr,
        *,
        filter_bouts_by_recipients: bool,
        max_bout_gap: int | None = None,
    ) -> Self:
        if self._target != "dyad":
            warn(
                "Only nested, dyadic classifications can contain overlapping predictions, returning unchanged predictions"
            )
            return copy(self)
        predictions = self.predictions.filter(
            ~pl.col("category").eq(self.background_category)
        )
        if filter_bouts_by_recipients:
            if max_bout_gap is None:
                raise ValueError(
                    "max_bout_gap must be specified when prefiltering recipient bouts"
                )
            bouts, predictions = aggregate(
                predictions, max_gap=max_bout_gap, group_by=["actor"]
            )
            bouts = resolve(
                bouts, priority=pl.col("stop") - pl.col("start"), group_by=["actor"]
            )
            predictions = predictions.filter(
                pl.col("_bout_id").is_in(bouts["_bout_id"])
            ).drop("_bout_id")

        predictions = resolve(
            predictions,
            priority=priority,
            group_by=["actor"],
        )

        classifications: dict[Hashable, Classification] = {}
        for actor in self.actors:
            for dyad, classification in self:
                if is_tuple_of(dyad, Hashable):
                    _actor, recipient = dyad
                else:
                    raise ValueError(f"Invalid dyad identifier: {dyad}")
                if actor != _actor:
                    continue
                predictions_dyad = predictions.filter(
                    pl.col("actor").eq(actor) & pl.col("recipient").eq(recipient)
                )
                predictions_dyad = densify(
                    predictions_dyad,
                    within=tuple(classification.timestamps[[0, -1]]),
                    category=self.background_category,
                )
                duration = (
                    predictions_dyad.select(
                        (pl.col("stop") - pl.col("start") + 1)
                        .cast(pl.Int64)
                        .alias("duration")
                    )
                    .get_column("duration")
                    .to_numpy()
                )
                category = self.encode(
                    np.asarray(predictions_dyad.get_column("category"))
                )
                y = np.repeat(category, duration)
                predictions_dyad = to_predictions(
                    y,
                    classification.y_proba,
                    classification.timestamps,
                    set(classification.categories),
                )

                classifications[dyad] = replace(
                    classification, y=y, predictions=predictions_dyad
                )
        return type(self)(
            classifications,
            categories=set(self.categories),
            background_category=self.background_category,
        )

    @override
    @classmethod
    def combine(
        cls, *classifications: ClassificationCollection[Classification]
    ) -> Self:
        confirmed_classifications = [
            cls.confirm_instance(classification) for classification in classifications
        ]
        if len(confirmed_classifications) == 0:
            raise ValueError("No classifications provided")
        first = confirmed_classifications[0]
        target = first._target
        categories = first.categories
        background_category = first.background_category
        if len(confirmed_classifications) == 1:
            return cls(
                first.classifications,
                categories=set(first.categories),
                background_category=first.background_category,
                identifier_validator=first.identifier_validator,
                classification_validator=first.classification_validator,
            )
        combined_classifications: dict[Hashable, Classification] = {}
        for group_classification in confirmed_classifications:
            # TODO: this validation should happen in ClassificationCollection.__init__
            if group_classification._target != target:
                raise ValueError("Classifications with inconsistent target")
            if group_classification.categories != categories:
                raise ValueError("Classifications with inconsistent categories")
            if group_classification.background_category != background_category:
                raise ValueError(
                    "Classifications with inconsistent background category"
                )
            for identifier, classification in group_classification:
                if identifier in combined_classifications:
                    raise ValueError(f"Duplicate classification: {identifier}")
                combined_classifications[identifier] = classification
        return cls(
            combined_classifications,
            categories=set(categories),
            background_category=background_category,
            identifier_validator=first.identifier_validator,
            classification_validator=first.classification_validator,
        )


class AnnotatedGroupClassification(WithGroundTruth, GroupClassification):
    _minimal_classification_type: type[Classification] = AnnotatedClassification

    def __init__(
        self,
        classifications: Mapping[Hashable, AnnotatedClassification],
        *,
        categories: set[str],
        background_category: str,
        identifier_validator: Callable[[object], Hashable] | None = None,
        classification_validator: Callable[[object], AnnotatedClassification]
        | None = None,
    ):
        if identifier_validator is None or classification_validator is None:
            identifier_validator, classification_validator = get_validators(
                classifications,
                minimal_value_type=AnnotatedClassification,
            )
        super().__init__(
            classifications,
            categories=categories,
            background_category=background_category,
            identifier_validator=identifier_validator,
            classification_validator=classification_validator,
        )
        self.with_ground_truth(
            y_gt=self._concatenate_numpy_attribute("y_gt"),
            annotations=self._concatenate_polars_attribute("annotations"),
        )

    @override
    def update_from_flat_classifications(
        self, classifications: Sequence[Classification]
    ) -> Self:
        classifications = [
            AnnotatedClassification.confirm_instance(classification)
            for classification in classifications
        ]
        updated = super().update_from_flat_classifications(classifications)
        updated.with_ground_truth(
            y_gt=updated._concatenate_numpy_attribute("y_gt"),
            annotations=updated._concatenate_polars_attribute("annotations"),
        )
        return updated

    @override
    def validate(self) -> Self:
        return type(self)(
            {
                identifier: AnnotatedClassification.confirm_instance(
                    classification
                ).validate()
                for identifier, classification in self
            },
            categories=set(self.categories),
            background_category=self.background_category,
        )

    @override
    def resolve_overlapping_predictions(
        self,
        priority: pl.Expr,
        *,
        filter_bouts_by_recipients: bool,
        max_bout_gap: int | None = None,
    ) -> Self:
        return (
            super()
            .resolve_overlapping_predictions(
                priority=priority,
                filter_bouts_by_recipients=filter_bouts_by_recipients,
                max_bout_gap=max_bout_gap,
            )
            .validate()
        )


class DatasetClassification(ClassificationCollection[GroupClassification]):
    _minimal_classification_type: type[GroupClassification] = GroupClassification
    _level: tuple[str, ...]

    def __init__(
        self,
        classifications: Mapping[Hashable, GroupClassification],
        *,
        categories: set[str],
        background_category: str,
        identifier_validator: Callable[[object], Hashable] | None = None,
        classification_validator: Callable[[object], GroupClassification] | None = None,
    ):
        self._level = ("group",)
        if identifier_validator is None or classification_validator is None:
            identifier_validator, classification_validator = get_validators(
                classifications, minimal_value_type=GroupClassification
            )
        super().__init__(
            classifications,
            categories=categories,
            background_category=background_category,
            identifier_validator=identifier_validator,
            classification_validator=classification_validator,
        )

    @override
    def update_from_flat_classifications(
        self, classifications: Sequence[Classification]
    ) -> Self:
        if len(classifications) != self._num_flat_classifications:
            raise ValueError("Number of classifications does not match")
        structured_classifications: dict[Hashable, GroupClassification] = {}
        for identifier, classification in self:
            n = classification._num_flat_classifications
            structured_classifications[identifier] = (
                classification.update_from_flat_classifications(classifications[:n])
            )
            classifications = classifications[n:]
        return type(self)(
            structured_classifications,
            categories=set(self.categories),
            background_category=self.background_category,
            identifier_validator=self.identifier_validator,
            classification_validator=self.classification_validator,
        )

    @override
    def resolve_overlapping_predictions(
        self,
        priority: pl.Expr,
        *,
        filter_bouts_by_recipients: bool,
        max_bout_gap: int | None = None,
    ) -> Self:
        return type(self)(
            {
                identifier: classification.resolve_overlapping_predictions(
                    priority,
                    filter_bouts_by_recipients=filter_bouts_by_recipients,
                    max_bout_gap=max_bout_gap,
                )
                for identifier, classification in self
            },
            categories=set(self.categories),
            background_category=self.background_category,
            identifier_validator=self.identifier_validator,
            classification_validator=self.classification_validator,
        )

    @override
    @classmethod
    def combine(
        cls, *classifications: ClassificationCollection[GroupClassification]
    ) -> Self:
        confirmed_classifications = [
            cls.confirm_instance(classification) for classification in classifications
        ]
        if len(confirmed_classifications) == 0:
            raise ValueError("No classifications provided")
        first = confirmed_classifications[0]
        categories = first.categories
        background_category = first.background_category
        if len(confirmed_classifications) == 1:
            return cls(
                first.classifications,
                categories=set(first.categories),
                background_category=first.background_category,
                identifier_validator=first.identifier_validator,
                classification_validator=first.classification_validator,
            )
        classifications_by_identifier: dict[Hashable, list[GroupClassification]] = {}
        for dataset_classification in confirmed_classifications:
            if dataset_classification.categories != categories:
                raise ValueError("Classifications with inconsistent categories")
            if dataset_classification.background_category != background_category:
                raise ValueError(
                    "Classifications with inconsistent background category"
                )
            for identifier, group_classification in dataset_classification:
                if identifier not in classifications_by_identifier:
                    classifications_by_identifier[identifier] = [group_classification]
                    continue
                classifications_by_identifier[identifier].append(group_classification)
        combined_classifications: dict[Hashable, GroupClassification] = {}
        for identifier, group_classifications in classifications_by_identifier.items():
            if len(group_classifications) == 1:
                combined_classifications[identifier] = group_classifications[0]
                continue
            combined_classifications[identifier] = type(
                group_classifications[0]
            ).combine(*group_classifications)
        return cls(
            combined_classifications,
            categories=set(categories),
            background_category=background_category,
            identifier_validator=first.identifier_validator,
            classification_validator=first.classification_validator,
        )


class AnnotatedDatasetClassification(WithGroundTruth, DatasetClassification):
    _minimal_classification_type: type[GroupClassification] = (
        AnnotatedGroupClassification
    )

    def __init__(
        self,
        classifications: Mapping[Hashable, AnnotatedGroupClassification],
        *,
        categories: set[str],
        background_category: str,
        identifier_validator: Callable[[object], Hashable] | None = None,
        classification_validator: Callable[[object], AnnotatedGroupClassification]
        | None = None,
    ):
        if identifier_validator is None or classification_validator is None:
            identifier_validator, classification_validator = get_validators(
                classifications,
                minimal_value_type=AnnotatedGroupClassification,
            )
        super().__init__(
            classifications,
            categories=categories,
            background_category=background_category,
            identifier_validator=identifier_validator,
            classification_validator=classification_validator,
        )
        self.with_ground_truth(
            y_gt=self._concatenate_numpy_attribute("y_gt"),
            annotations=self._concatenate_polars_attribute("annotations"),
        )

    @override
    def update_from_flat_classifications(
        self, classifications: Sequence[Classification]
    ) -> Self:
        classifications = [
            AnnotatedClassification.confirm_instance(classification)
            for classification in classifications
        ]
        updated = super().update_from_flat_classifications(classifications)
        updated.with_ground_truth(
            y_gt=updated._concatenate_numpy_attribute("y_gt"),
            annotations=updated._concatenate_polars_attribute("annotations"),
        )
        return updated

    @override
    def validate(self) -> Self:
        return type(self)(
            {
                identifier: AnnotatedGroupClassification.confirm_instance(
                    classification
                ).validate()
                for identifier, classification in self
            },
            categories=set(self.categories),
            background_category=self.background_category,
        )

    @override
    def resolve_overlapping_predictions(
        self,
        priority: pl.Expr,
        *,
        filter_bouts_by_recipients: bool,
        max_bout_gap: int | None = None,
    ) -> Self:
        return (
            super()
            .resolve_overlapping_predictions(
                priority=priority,
                filter_bouts_by_recipients=filter_bouts_by_recipients,
                max_bout_gap=max_bout_gap,
            )
            .validate()
        )

from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate, Protocol, Self, overload

import numpy as np
from sklearn.utils.class_weight import (
    compute_sample_weight,  # pyright: ignore[reportUnknownVariableType]
)

from ..dataset import (
    AnnotatedDataset,
    AnnotatedDyad,
    AnnotatedGroup,
    AnnotatedIndividual,
    Dataset,
    Dyad,
    Group,
    Individual,
)
from ..dataset.types import WithCategories
from ..features import BaseExtractor, Shaped
from ..utils import to_int_seed
from ._results import (
    AnnotatedClassification,
    AnnotatedDatasetClassification,
    AnnotatedGroupClassification,
    Classification,
    DatasetClassification,
    GroupClassification,
)


class Classifier(Protocol):
    """Protocol for classifiers.

    This protocol defines the methods that a classifier should implement.

    See also:
        :class:`~sklearn.base.ClassifierMixin` for the classifier interface in :mod:`~sklearn`.
    """

    def predict(self, *args: ..., **kwargs: ...) -> np.ndarray: ...

    def predict_proba(self, *args: ..., **kwargs: ...) -> np.ndarray: ...

    def get_params(self) -> dict[str, object]: ...

    def fit(self, *args: ..., **kwargs: ...) -> Self: ...


def _predict_base_sampleable[F: Shaped](
    sampleable: Individual | Dyad,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> Classification:
    X = sampleable.sample_X(extractor, indices=None, out=None)
    return Classification(
        classifier.predict_proba(X),
        timestamps=sampleable.trajectory.timestamps,
        categories=categories,
        background_category=background_category,
        discretize_on_init=False,
    )


def _predict_annotated_base_sampleable[F: Shaped](
    sampleable: AnnotatedIndividual | AnnotatedDyad,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> AnnotatedClassification:
    X, y = sampleable.sample(extractor, indices=None, store_indices=False, out=None)
    return AnnotatedClassification(
        classifier.predict_proba(X),
        timestamps=sampleable.trajectory.timestamps,
        categories=categories,
        background_category=background_category,
        annotations=sampleable.observations,
        y_gt=y,
        discretize_on_init=False,
    )


def _predict_group[F: Shaped](
    sampleable: Group,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> GroupClassification:
    classifications = {
        identifier: _predict_base_sampleable(
            _sampleable,
            classifier,
            extractor,
            categories=categories,
            background_category=background_category,
        )
        for identifier, _sampleable in sampleable
    }
    return GroupClassification(
        classifications,
        categories=categories,
        background_category=background_category,
    )


def _predict_annotated_group[F: Shaped](
    sampleable: AnnotatedGroup,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> AnnotatedGroupClassification:
    classifications = {
        identifier: _predict_annotated_base_sampleable(
            _sampleable,
            classifier,
            extractor,
            categories=categories,
            background_category=background_category,
        )
        for identifier, _sampleable in sampleable
    }
    return AnnotatedGroupClassification(
        classifications,
        categories=categories,
        background_category=background_category,
    )


def _predict_dataset[F: Shaped](
    sampleable: Dataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> DatasetClassification:
    classifications = {
        identifier: _predict_group(
            group,
            classifier,
            extractor,
            categories=categories,
            background_category=background_category,
        )
        for identifier, group in sampleable
    }
    return DatasetClassification(
        classifications,
        categories=categories,
        background_category=background_category,
    )


def _predict_annotated_dataset[F: Shaped](
    sampleable: AnnotatedDataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> AnnotatedDatasetClassification:
    classifications = {
        identifier: _predict_annotated_group(
            group,
            classifier,
            extractor,
            categories=categories,
            background_category=background_category,
        )
        for identifier, group in sampleable
    }
    return AnnotatedDatasetClassification(
        classifications,
        categories=categories,
        background_category=background_category,
    )


# overloads for the generic predict function


@overload
def predict[F: Shaped](
    sampleable: AnnotatedIndividual,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str] | None = None,
    background_category: str | None = None,
) -> AnnotatedClassification: ...


@overload
def predict[F: Shaped](
    sampleable: Individual,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> Classification: ...


@overload
def predict[F: Shaped](
    sampleable: AnnotatedDyad,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str] | None = None,
    background_category: str | None = None,
) -> AnnotatedClassification: ...


@overload
def predict[F: Shaped](
    sampleable: Dyad,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> Classification: ...


@overload
def predict[F: Shaped](
    sampleable: AnnotatedGroup,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str] | None = None,
    background_category: str | None = None,
) -> AnnotatedGroupClassification: ...


@overload
def predict[F: Shaped](
    sampleable: Group,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> GroupClassification: ...


@overload
def predict[F: Shaped](
    sampleable: AnnotatedDataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str] | None = None,
    background_category: str | None = None,
) -> AnnotatedDatasetClassification: ...


@overload
def predict[F: Shaped](
    sampleable: Dataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str],
    background_category: str,
) -> DatasetClassification: ...


# actual implementation that directs to appropriate helper function


def predict[F: Shaped](
    sampleable: Individual
    | Dyad
    | Group
    | Dataset
    | AnnotatedIndividual
    | AnnotatedDyad
    | AnnotatedGroup
    | AnnotatedDataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    categories: set[str] | None = None,
    background_category: str | None = None,
) -> (
    Classification
    | AnnotatedClassification
    | GroupClassification
    | AnnotatedGroupClassification
    | DatasetClassification
    | AnnotatedDatasetClassification
):
    def _validate_category_metatdata() -> tuple[set[str], str]:
        if categories is None:
            raise ValueError(
                "Categories must be specified for non-annotated sampleables"
            )
        if background_category is None:
            raise ValueError(
                "Background category must be specified for non-annotated sampleables"
            )
        return categories, background_category

    def _get_category_metadata(
        sampleable: WithCategories,
    ) -> tuple[set[str], str]:
        _categories = sampleable.categories if categories is None else categories
        _background_category = (
            sampleable.background_category
            if background_category is None
            else background_category
        )
        classes = getattr(classifier, "classes_", None)
        if classes is None:
            raise ValueError("Classifier is not fitted")
        if len(_categories) != len(classes):
            raise ValueError(
                "Number of specified or annotated categories does not match classifier"
            )
        return set(_categories), _background_category

    match sampleable:
        case AnnotatedDataset():
            categories, background_category = _get_category_metadata(sampleable)
            return _predict_annotated_dataset(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()
        case Dataset():
            categories, background_category = _validate_category_metatdata()
            return _predict_dataset(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()
        case AnnotatedGroup():
            categories, background_category = _get_category_metadata(sampleable)
            return _predict_annotated_group(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()
        case Group():
            categories, background_category = _validate_category_metatdata()
            return _predict_group(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()
        case AnnotatedDyad() | AnnotatedIndividual():
            categories, background_category = _get_category_metadata(sampleable)
            return _predict_annotated_base_sampleable(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()
        case Dyad() | Individual():
            categories, background_category = _validate_category_metatdata()
            return _predict_base_sampleable(
                sampleable,
                classifier,
                extractor,
                categories=categories,
                background_category=background_category,
            ).discretize()


def _init_new_classifier(
    classifier: Classifier, random_state: np.random.Generator | int | None
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


def _fit_classifier(
    classifier: Classifier,
    X: Shaped,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
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


def k_fold_predict[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Classifier | type[Classifier],
    *,
    k: int,
    random_state: np.random.Generator | int | None = None,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...],
        tuple[F, np.ndarray],
    ],
    balance_sample_weights: bool = True,
    **sampling_function_kwargs: ...,
) -> AnnotatedDatasetClassification:
    """
    Run k-fold cross-validation on a dataset.

    Parameters:
        dataset: The dataset to cross-validate.
        extractor: The extractor to use.
        classifier: The classifier to use.
        k: The number of folds.
        random_state: The random state to use.
        sampling_function: The sampling function to use.
        balance_sample_weights: Whether to balance sample weights.
        log: The logger to use.

    Returns:
        The dataset classification result.
    """

    random_state = np.random.default_rng(random_state)
    if isinstance(classifier, type):
        classifier = _init_new_classifier(classifier(), random_state)
    fold_results: list[AnnotatedDatasetClassification] = []
    for dataset_fold_train, dataset_fold_val in dataset.k_fold(
        k, random_state=random_state
    ):
        X_train, y_train = sampling_function(
            dataset_fold_train,
            extractor,
            random_state=random_state,
            **sampling_function_kwargs,
        )
        sample_weight = None
        if balance_sample_weights:
            sample_weight = compute_sample_weight("balanced", y_train)
        classifier = _fit_classifier(
            _init_new_classifier(classifier, random_state),
            X_train,
            y_train,
            sample_weight=sample_weight,
        )
        fold_results.append(
            _predict_annotated_dataset(
                dataset_fold_val,
                classifier,
                extractor,
                categories=set(dataset.categories),
                background_category=dataset.background_category,
            )
        )
    return AnnotatedDatasetClassification.combine(*fold_results)

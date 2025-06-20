from __future__ import annotations

from typing import Iterable, Optional, overload

import loguru
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from ..dataset import (
    AnnotatedDataset,
    Dataset,
    Group,
    GroupIdentifier,
    Identifier,
)
from ..dataset.types import (
    AnnotatedMixin,
    BaseSampleable,
    EncodingFunction,
    SampleableMixin,
    SamplingFunction,
)
from ..features import BaseExtractor, Shaped
from ..logging import log_loop, log_time, set_logging_level
from ..utils import class_name
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import (
    Classifier,
    fit_classifier,
    init_new_classifier,
)


def _predict_sampleable[F: Shaped](
    sampleable: BaseSampleable,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    log: loguru.Logger,
) -> ClassificationResult:
    X, y = sampleable.sample(extractor)
    y_pred_numeric: np.ndarray = classifier.predict(X)
    y_proba: np.ndarray = classifier.predict_proba(X).astype(float)
    y_true_numeric: Optional[np.ndarray] = None
    if encoding_function is None and isinstance(sampleable, AnnotatedMixin):
        encoding_function = sampleable.encode
    elif encoding_function is None:
        raise ValueError(
            "encoding_function must be provided for non-annotated sampleables"
        )
    if y is not None:
        y_true_numeric = encoding_function(y)
    timestamps = sampleable.trajectory.timestamps
    annotations = None
    if isinstance(sampleable, AnnotatedMixin):
        annotations = sampleable.observations
        if categories is not None:
            log.warning(
                "ignoring categories parameter for annotated sampleable, using categories from sampleable instead"
            )
        categories = sampleable.categories
    if categories is None:
        raise ValueError("specify categories when classifying unannotated sampleables.")
    return ClassificationResult(
        categories=tuple(categories),
        timestamps=timestamps,  # type:ignore
        y_proba=y_proba,
        y_pred_numeric=y_pred_numeric,
        _y_proba_smoothed=None,
        _annotations=annotations,
        _y_true_numeric=y_true_numeric,
    ).threshold()


def _predict_group[F: Shaped](
    group: Group,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: loguru.Logger,
) -> GroupClassificationResult:
    if exclude is None:
        exclude = ()
    results: dict[Identifier, ClassificationResult] = {}
    for log, (identifier, sampleable) in log_loop(
        group,
        level="info",
        message="finished predictions",
        total=len(group.identifiers),
        log=log,
    ):
        if identifier in exclude:
            continue
        results[identifier] = _predict_sampleable(
            sampleable,
            classifier,
            extractor,
            encoding_function=encoding_function,
            categories=categories,
            log=log,
        )
    return GroupClassificationResult(
        classification_results=results,
        individuals=group.individuals,
        target=group.target,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="predicting on dataset",
)
def _predict[F: Shaped](
    dataset: Dataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: loguru.Logger,
) -> DatasetClassificationResult:
    if exclude is None:
        exclude = ()
    results: dict[GroupIdentifier, GroupClassificationResult] = {}
    for log, (identifier, group) in log_loop(
        dataset,
        level="info",
        message="finished predictions",
        name="group",
        total=len(dataset.identifiers),
        log=log,
    ):
        if identifier in exclude:
            continue
        results[identifier] = _predict_group(
            group,
            classifier,
            extractor,
            encoding_function=encoding_function,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    return DatasetClassificationResult(
        classification_results=results,
        target=dataset.target,
    )


@overload
def predict[F: Shaped](
    sampleable: BaseSampleable,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[loguru.Logger] = None,
) -> ClassificationResult: ...


@overload
def predict[F: Shaped](
    sampleable: Group,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[loguru.Logger] = None,
) -> GroupClassificationResult: ...


@overload
def predict[F: Shaped](
    sampleable: Dataset,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[loguru.Logger] = None,
) -> DatasetClassificationResult: ...


def predict[F: Shaped](
    sampleable: SampleableMixin,
    classifier: Classifier,
    extractor: BaseExtractor[F],
    *,
    encoding_function: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[loguru.Logger] = None,
) -> ClassificationResult | GroupClassificationResult | DatasetClassificationResult:
    """
    Run classification on a sampleable object.

    Parameters:
        sampleable: The sampleable object to classify.
        classifier: The classifier to use.
        extractor: The extractor to use.
        encoding_function: The encoding function to use.
        categories: The categories to use.
        exclude: The identifiers to exclude.
        log: The logger to use.

    Returns:
        The classification result.
    """
    if log is None:
        log = set_logging_level()
    if isinstance(sampleable, Dataset):
        return _predict(
            sampleable,
            classifier,
            extractor,
            encoding_function=encoding_function,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    if isinstance(sampleable, Group):
        return _predict_group(
            sampleable,
            classifier,
            extractor,
            encoding_function=encoding_function,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    if not isinstance(sampleable, BaseSampleable):
        raise TypeError(f"unsupported object of type {type(sampleable)}")
    if exclude is not None:
        log.warning("ignoring exclude parameter for single sampleable")
    return _predict_sampleable(
        sampleable,
        classifier,
        extractor,
        encoding_function=encoding_function,
        categories=categories,
        log=log,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="k-fold cross-validation",
)
def k_fold_predict[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Classifier,
    *,
    k: int,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    log: Optional[loguru.Logger],
) -> DatasetClassificationResult:
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
    encoding_function = dataset.encode
    if type(classifier) is type:
        classifier = init_new_classifier(classifier(), random_state)
    fold_results = []
    for log, (fold_train, fold_holdout) in log_loop(
        dataset.k_fold(k, random_state=random_state),
        level="info",
        message="finished",
        name="fold",
        total=k,
        log=log,
    ):
        X_train, y_train = log_time(
            sampling_function, level_finish="success", description="sampling"
        )(
            fold_train,
            extractor,
            random_state=random_state,
            log=log,
        )
        sample_weight = None
        if balance_sample_weights:
            sample_weight = compute_sample_weight(
                "balanced", encoding_function(y_train)
            )
        classifier = log_time(
            fit_classifier,
            level_finish="success",
            description=f"fitting {class_name(classifier)}",
        )(
            init_new_classifier(classifier, random_state),
            np.asarray(X_train),
            encoding_function(y_train),
            sample_weight=sample_weight,
            log=log,
        )
        fold_results.append(
            predict(
                fold_holdout,
                classifier,
                extractor,
                encoding_function=encoding_function,
                log=log,
            )
        )
    return DatasetClassificationResult.combine(fold_results)

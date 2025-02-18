from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_sample_weight

from ..dataset import (
    Dataset,
    Group,
    GroupIdentifier,
    Identifier,
)
from ..dataset.types._base_sampleable import BaseSampleable
from ..dataset.types._mixins import AnnotatedMixin, SampleableMixin
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..logging import log_loop, log_time, set_logging_level
from ..utils import class_name, ensure_generator
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import (
    Classifier,
    EncodingFunction,
    SamplingFunction,
    fit_classifier,
    init_new_classifier,
)

if TYPE_CHECKING:
    from loguru import Logger


def _predict_sampleable(
    sampleable: BaseSampleable,  # use base type instead
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    log: Logger,
) -> ClassificationResult:
    X, y = sampleable.sample(extractor)
    y_pred_numeric: NDArray = classifier.predict(X)
    y_proba: NDArray = classifier.predict_proba(X).astype(float)
    y_true_numeric: Optional[NDArray] = None
    if encode_func is None and isinstance(sampleable, AnnotatedMixin):
        encode_func = sampleable.encode
    elif encode_func is None:
        raise ValueError("encode_func must be provided for non-annotated sampleables")
    if y is not None:
        y_true_numeric = encode_func(y)
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


def _predict_group(
    group: Group,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Logger,
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
            encode_func=encode_func,
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
def _predict(
    dataset: Dataset,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Logger,
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
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    return DatasetClassificationResult(
        classification_results=results,
        target=dataset.target,
    )


@overload
def predict(
    sampleable: BaseSampleable,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[Logger],
) -> ClassificationResult: ...


@overload
def predict(
    sampleable: Group,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[Logger],
) -> GroupClassificationResult: ...


@overload
def predict(
    sampleable: Dataset,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[Logger],
) -> DatasetClassificationResult: ...


def predict(
    sampleable: SampleableMixin,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[Logger],
) -> ClassificationResult | GroupClassificationResult | DatasetClassificationResult:
    if log is None:
        log = set_logging_level()
    if isinstance(sampleable, Dataset):
        return _predict(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    if isinstance(sampleable, Group):
        return _predict_group(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
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
        encode_func=encode_func,
        categories=categories,
        log=log,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="k-fold cross-validation",
)
def k_fold_predict(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Classifier,
    *,
    k: int,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    log: Optional[Logger],
) -> DatasetClassificationResult:
    random_state = ensure_generator(random_state)
    if encode_func is None and isinstance(dataset, AnnotatedMixin):
        encode_func = dataset.encode
    elif encode_func is None:
        raise ValueError("encode_func must be provided for non-annotated datasets")
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
            sampling_func, level_finish="success", description="sampling"
        )(
            fold_train,
            extractor,
            random_state=random_state,
            log=log,
        )
        sample_weight = None
        if balance_sample_weights:
            sample_weight = compute_sample_weight("balanced", encode_func(y_train))
        classifier = log_time(
            fit_classifier,
            level_finish="success",
            description=f"fitting {class_name(classifier)}",
        )(
            init_new_classifier(classifier, random_state),
            np.asarray(X_train),
            encode_func(y_train),
            sample_weight=sample_weight,
            log=log,
        )
        fold_results.append(
            predict(
                fold_holdout,
                classifier,
                extractor,
                encode_func=encode_func,
                log=log,
            )
        )
    return DatasetClassificationResult.combine(fold_results)

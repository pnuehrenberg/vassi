from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence, overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_sample_weight

from ..dataset import (
    AnnotatedDyad,
    AnnotatedGroup,
    AnnotatedIndividual,
    Dataset,
    Dyad,
    Group,
    GroupIdentifier,
    Identifier,
    Individual,
)
from ..dataset.types._mixins import AnnotatedMixin
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import class_name, ensure_generator
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import Classifier, EncodingFunction, SamplingFunction, init_new_classifier

if TYPE_CHECKING:
    from loguru import Logger


def _predict_sampleable(
    sampleable: Dyad
    | AnnotatedDyad
    | Individual
    | AnnotatedIndividual,  # use base type instead
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    log: Logger,
) -> ClassificationResult:
    X, y = sampleable.sample(extractor, exclude=None)
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
    if isinstance(
        sampleable, (AnnotatedDyad, AnnotatedIndividual)
    ):  # use mixin instead
        annotations = sampleable.observations
        if categories is not None:
            log.warning(
                "ignoring categories parameter for annotated sampleable, using categories from sampleable instead"
            )
        categories = sampleable.categories
    if categories is None:
        raise ValueError("specify categories when classifying unannotated sampleables.")
    return ClassificationResult(
        categories=categories,
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
    results: dict[Identifier, ClassificationResult] = {}
    target = None
    for idx, (sampleable_key, sampleable) in enumerate(group._sampleables.items()):
        if exclude is not None and sampleable_key in exclude:
            continue
        if isinstance(sampleable, (Individual, AnnotatedIndividual)):
            if target is None:
                target = "individuals"
            elif target != "individuals":
                raise ValueError(
                    "unsupported group of mixed sampleables (dyads and individuals)"
                )
        elif isinstance(sampleable, (Dyad, AnnotatedDyad)):
            if target is None:
                target = "dyads"
            elif target != "dyads":
                raise ValueError(
                    "unsupported group of mixed sampleables (individuals and dyads)"
                )
        else:
            raise ValueError(f"unsupported sampleable of type {type(sampleable)}")
        log = log.bind(
            sublevel={"name": "", "step": idx + 1, "total": len(group.identifiers)}
        )
        results[sampleable_key] = _predict_sampleable(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            log=log,
        )
        log.trace(f"finished predictions on {class_name(sampleable)} {sampleable_key}")
    if target is None:
        raise ValueError("unsupported group with no sampleables")
    return GroupClassificationResult(
        classification_results=results,
        trajectories=group.trajectories,
        target=group.target,
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
    results: dict[GroupIdentifier, GroupClassificationResult] = {}
    for idx, group_id in enumerate(dataset.identifiers):
        log = log.bind(
            level={"name": "group", "step": idx + 1, "total": len(dataset.identifiers)}
        )
        if exclude is not None and group_id in exclude:
            continue
        group = dataset.select(group_id)
        if TYPE_CHECKING:
            assert isinstance(group, Group)
            assert isinstance(group_id, GroupIdentifier)
        results[group_id] = _predict_group(
            group,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
            log=log,
        )
        log.trace(f"finished predictions on {class_name(group)} {group_id}")
    if len(results) == 0:
        raise ValueError("unsupported dataset with no groups")
    targets: list[Literal["individual", "dyad"]] = [
        result.target for result in results.values()
    ]
    target = targets[0]
    if any(target != result_target for result_target in targets):
        raise ValueError(
            "unsupported dataset of groups with mixed targets (dyads and individuals)"
        )
    return DatasetClassificationResult(
        classification_results=results,
        target=target,
    )


@overload
def predict(
    sampleable: Dyad | AnnotatedDyad | Individual | AnnotatedIndividual,
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
    sampleable: Group | AnnotatedGroup,
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
    sampleable: (
        Dyad
        | AnnotatedDyad
        | Individual
        | AnnotatedIndividual
        | Group
        | AnnotatedGroup
        | Dataset
    ),
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[EncodingFunction] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identifier]] = None,
    log: Optional[Logger],
) -> ClassificationResult | GroupClassificationResult | DatasetClassificationResult:
    if log is None:
        log = logger
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
    if isinstance(sampleable, (Group, AnnotatedGroup)):
        return _predict_group(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
            log=log,
        )
    if not isinstance(
        sampleable, (Dyad, AnnotatedDyad, Individual, AnnotatedIndividual)
    ):
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


def k_fold_predict(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Classifier,
    *,
    k: int,
    exclude: Optional[Sequence[Identifier]] = None,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    log: Optional[Logger],
) -> DatasetClassificationResult:
    if log is None:
        log = logger
    random_state = ensure_generator(random_state)
    if encode_func is None and isinstance(dataset, AnnotatedMixin):
        encode_func = dataset.encode
    elif encode_func is None:
        raise ValueError("encode_func must be provided for non-annotated datasets")
    if type(classifier) is type:
        classifier = init_new_classifier(classifier(), random_state)
    log.info(f"running {k}-fold predict with {class_name(classifier)}")
    fold_results = []
    for fold_idx, (fold_train, fold_holdout) in enumerate(
        dataset.k_fold(k, exclude=exclude, random_state=random_state)
    ):
        _log = log.bind(fold={"name": "fold", "step": fold_idx + 1, "total": k})
        X_train, y_train = sampling_func(
            fold_train,
            extractor,
            random_state=random_state,
            log=_log,
        )
        _log.debug("finished sampling")
        sample_weight = None
        if balance_sample_weights:
            sample_weight = compute_sample_weight("balanced", encode_func(y_train))
        classifier = init_new_classifier(classifier, random_state).fit(
            np.asarray(X_train),
            encode_func(y_train),
            sample_weight=sample_weight,
        )
        _log.debug(f"fitted {class_name(classifier)}")
        fold_results.append(
            predict(
                fold_holdout,
                classifier,
                extractor,
                encode_func=encode_func,
                log=_log,
            )
        )
        _log.debug("finished predictions on holdout data")
    log.success(f"finished {k}-fold predict with {class_name(classifier)}")
    return DatasetClassificationResult.combine(fold_results)

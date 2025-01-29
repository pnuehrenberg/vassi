import warnings
from typing import Callable, Iterable, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_sample_weight

from ..dataset import (
    AnnotatedDyad,
    AnnotatedGroup,
    AnnotatedIndividual,
    Dataset,
    Dyad,
    Group,
    Individual,
)
from ..dataset.types.utils import DyadIdentity, Identity
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import ensure_generator, formatted_tqdm, warning_only
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import Classifier, SamplingFunction, init_new_classifier


def _predict_sampleable(
    sampleable: Dyad | AnnotatedDyad | Individual | AnnotatedIndividual,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    categories: Optional[Iterable[str]] = None,
) -> ClassificationResult:
    X, y = sampleable.sample(extractor)
    y_pred_numeric: NDArray = classifier.predict(X)
    y_proba: NDArray = classifier.predict_proba(X).astype(float)
    y_true_numeric: Optional[NDArray] = None
    if encode_func is None:
        encode_func = sampleable.encode
    if y is not None:
        y_true_numeric = encode_func(y)
    timestamps = sampleable.trajectory.timestamps
    annotations = None
    if isinstance(sampleable, (AnnotatedDyad, AnnotatedIndividual)):
        annotations = sampleable.observations
        if categories is not None:
            with warning_only():
                warnings.warn(
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
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
) -> GroupClassificationResult:
    results: dict[Identity | DyadIdentity, ClassificationResult] = {}
    target = None
    for sampleable_key, sampleable in group._sampleables.items():
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
        results[sampleable_key] = _predict_sampleable(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
        )
    if target is None:
        raise ValueError("unsupported group with no sampleables")
    return GroupClassificationResult(
        classification_results=results,
        trajectories=group.trajectories,
        target=target,
    )


def _predict(
    dataset: Dataset,
    classifier: Classifier,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
) -> DatasetClassificationResult:
    results: dict[Identity, GroupClassificationResult] = {}
    for group_key in dataset.group_keys:
        if exclude is not None and group_key in exclude:
            continue
        results[group_key] = _predict_group(
            dataset.select(group_key),
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
        )
    if len(results) == 0:
        raise ValueError("unsupported dataset with no groups")
    targets: list[Literal["individuals", "dyads"]] = [
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
    *args,
    **kwargs,
) -> ClassificationResult: ...


@overload
def predict(
    sampleable: Group | AnnotatedGroup,
    *args,
    **kwargs,
) -> GroupClassificationResult: ...


@overload
def predict(
    sampleable: Dataset,
    *args,
    **kwargs,
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
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
) -> ClassificationResult | GroupClassificationResult | DatasetClassificationResult:
    if isinstance(sampleable, Dataset):
        return _predict(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
        )
    if isinstance(sampleable, (Group, AnnotatedGroup)):
        return _predict_group(
            sampleable,
            classifier,
            extractor,
            encode_func=encode_func,
            categories=categories,
            exclude=exclude,
        )
    if not isinstance(
        sampleable, (Dyad, AnnotatedDyad, Individual, AnnotatedIndividual)
    ):
        raise TypeError(f"unsupported object of type {type(sampleable)}")
    if exclude is not None:
        with warning_only():
            warnings.warn("ignoring exclude parameter for single sampleable")
    return _predict_sampleable(
        sampleable,
        classifier,
        extractor,
        encode_func=encode_func,
        categories=categories,
    )


def k_fold_predict(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Classifier,
    *,
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    show_progress: bool = False,
) -> DatasetClassificationResult:
    random_state = ensure_generator(random_state)
    if type(classifier) is type:
        classifier = init_new_classifier(classifier(), random_state)
    fold_results = []
    k_fold = dataset.k_fold(k, exclude=exclude, random_state=random_state)
    for fold_train, fold_holdout in formatted_tqdm(
        k_fold, desc="k-fold predict", total=k, disable=not show_progress
    ):
        X_train, y_train = sampling_func(
            fold_train,
            extractor,
            random_state=random_state,
        )
        sample_weight = None
        if balance_sample_weights:
            sample_weight = compute_sample_weight("balanced", dataset.encode(y_train))
        classifier = init_new_classifier(classifier, random_state).fit(
            np.asarray(X_train),
            dataset.encode(y_train),
            sample_weight=sample_weight,
        )
        fold_results.append(
            predict(
                fold_holdout,
                classifier,
                extractor,
                encode_func=encode_func,
            )
        )
    return DatasetClassificationResult.combine(fold_results)

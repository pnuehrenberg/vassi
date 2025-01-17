from typing import Any, Callable, Iterable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from ..dataset import (
    AnnotatedDyad,
    AnnotatedIndividual,
    Dataset,
    Dyad,
    Group,
    Individual,
)
from ..dataset.types.utils import DyadIdentity, Identity
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import ensure_generator, formatted_tqdm, to_int_seed
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import SamplingFunction


def predict(
    classifier: Any,
    sampleable: Dyad | AnnotatedDyad | Individual | AnnotatedIndividual,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
    *,
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    categories: Optional[Iterable[str]] = None,
) -> ClassificationResult:
    if not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise ValueError(f"unsupported classifier of type {type(classifier)}")
    X, y = sampleable.sample(extractor, pipeline=pipeline, fit_pipeline=fit_pipeline)
    y_pred_numeric: NDArray = classifier.predict(X)
    y_proba: NDArray = classifier.predict_proba(X).astype(float)
    y_true_numeric: Optional[NDArray] = None
    if y is not None:
        y_true_numeric = encode_func(y)
    timestamps = sampleable.trajectory.timestamps
    annotations = None
    if isinstance(sampleable, (AnnotatedDyad, AnnotatedIndividual)):
        annotations = sampleable.observations
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


def predict_group(
    classifier: Any,
    group: Group,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
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
        results[sampleable_key] = predict(
            classifier,
            sampleable,
            extractor,
            encode_func,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
        )
    if target is None:
        raise ValueError("unsupported group with no sampleables")
    return GroupClassificationResult(
        classification_results=results,
        trajectories=group.trajectories,
        target=target,
    )


def predict_dataset(
    classifier: Any,
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
) -> DatasetClassificationResult:
    if encode_func is None:
        try:
            encode_func = dataset.encode
        except ValueError:
            pass
    if encode_func is None:
        raise ValueError("specify encode_func for non-annotated datasets")
    results: dict[Identity, GroupClassificationResult] = {}
    for group_key in dataset.group_keys:
        if exclude is not None and group_key in exclude:
            continue
        results[group_key] = predict_group(
            classifier,
            dataset.select(group_key),
            extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            encode_func=encode_func,
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


def k_fold_predict(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    # k fold paramters
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    # random_state is also used for sampling
    random_state: Optional[np.random.Generator | int] = None,
    # sampling parameters
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    # pipeline parameters are also used for sampling in classification
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    # encode_func required for k-fold prediction of datasets with non-annotated groups
    encode_func: Optional[Callable[[NDArray], NDArray[np.integer]]] = None,
    show_progress: bool = False,
) -> DatasetClassificationResult:
    def init_new_classifier[T](
        classifier: T, random_state: Optional[np.random.Generator | int]
    ) -> T:
        random_state = ensure_generator(random_state)
        params = classifier.get_params()  # type:ignore # see check below
        params["random_state"] = to_int_seed(random_state)
        return type(classifier)(**params)

    random_state = ensure_generator(random_state)
    if type(classifier) is type:
        classifier = init_new_classifier(classifier(), random_state)
    if not hasattr(classifier, "fit") or not hasattr(classifier, "get_params"):
        raise ValueError(f"unsupported classifier of type {type(classifier)}")
    fold_results = []
    k_fold = dataset.k_fold(k, exclude=exclude, random_state=random_state)
    if show_progress:
        k_fold = formatted_tqdm(k_fold, desc="k-fold predict", total=k)
    for fold_train, fold_holdout in k_fold:
        X_train, y_train = sampling_func(
            fold_train,
            extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
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
            predict_dataset(
                classifier,
                fold_holdout,
                extractor,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
                encode_func=encode_func,
            )
        )
    return DatasetClassificationResult.combine(fold_results)

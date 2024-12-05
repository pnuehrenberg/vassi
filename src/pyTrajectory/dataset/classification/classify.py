from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Optional,
)

import numpy as np
from numpy.typing import NDArray

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from .. import Dyad, Group, Individual
from ..types._sampleable import AnnotatedSampleable
from ..types.dataset import Dataset
from ..types.utils import DyadIdentity, Identity
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import to_predictions, validate_predictions


def classify(
    classifier,
    sampleable: Dyad | Individual,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
    *,
    pipeline=None,
    fit_pipeline=True,
    categories=None,
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
    if isinstance(sampleable, AnnotatedSampleable):
        annotations = sampleable.annotations
        categories = sampleable.categories
    if categories is None:
        raise ValueError("specify categories when classifying unannotated sampleables.")
    predictions = to_predictions(
        y_pred_numeric, y_proba, category_names=categories, timestamps=timestamps
    )
    if annotations is not None:
        predictions = validate_predictions(predictions, annotations, on="predictions")
        annotations = validate_predictions(predictions, annotations, on="annotations")
    return ClassificationResult(
        categories=categories,
        predictions=predictions,
        timestamps=timestamps,  # type:ignore
        y_proba=y_proba,
        y_pred_numeric=y_pred_numeric,
        _y_proba_smoothed=None,
        _y_pred_numeric_smoothed=None,
        _annotations=annotations,
        _y_true_numeric=y_true_numeric,
    )


def classify_group(
    classifier,
    group: Group,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    pipeline=None,
    fit_pipeline=True,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
) -> GroupClassificationResult:
    results: dict[Identity | DyadIdentity, ClassificationResult] = {}
    for sampleable_key, sampleable in group._sampleables.items():
        if exclude is not None and sampleable_key in exclude:
            continue
        if TYPE_CHECKING:
            assert isinstance(sampleable, (Individual, Dyad))
        results[sampleable_key] = classify(
            classifier,
            sampleable,
            extractor,
            encode_func,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
        )
    return GroupClassificationResult(
        classification_results=results,
        trajectories=group.trajectories,
    )


def classify_dataset(
    classifier,
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    pipeline=None,
    fit_pipeline=True,
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
        results[group_key] = classify_group(
            classifier,
            dataset.select(group_key),
            extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            encode_func=encode_func,
            exclude=exclude,
        )
    return DatasetClassificationResult(
        classification_results=results,
    )

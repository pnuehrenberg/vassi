from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
)

import numpy as np
from numpy.typing import NDArray

from pyTrajectory.dataset import Dyad, Group, Individual
from pyTrajectory.dataset.annotations.utils import (
    to_annotations,
)
from pyTrajectory.dataset.types._sampleable import AnnotatedSampleable
from pyTrajectory.dataset.types.dataset import Dataset
from pyTrajectory.dataset.types.utils import Identity
from pyTrajectory.features import DataFrameFeatureExtractor, FeatureExtractor
from pyTrajectory.series_operations import smooth
from pyTrajectory.utils import NDArray_to_NDArray

from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)
from .utils import validate_predictions


def classify(
    classifier,
    sampleable: Dyad | Individual,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    encode_func: Callable[[NDArray], NDArray[np.integer]],
    *,
    pipeline=None,
    fit_pipeline=True,
    label_smoothing_filter_funcs: Optional[list[NDArray_to_NDArray]] = None,
    categories=None,
) -> ClassificationResult:
    if not hasattr(classifier, "predict") or not hasattr(classifier, "predict_proba"):
        raise ValueError(f"unsupported classifier of type {type(classifier)}")
    X, y = sampleable.sample(extractor, pipeline=pipeline, fit_pipeline=fit_pipeline)
    y_pred_numeric: NDArray = classifier.predict(X)
    y_proba: NDArray = classifier.predict_proba(X).astype(float)
    y_true_numeric: Optional[NDArray] = None
    y_proba_smoothed: Optional[NDArray] = None
    y_pred_numeric_smoothed: Optional[NDArray] = None
    if y is not None:
        y_true_numeric = encode_func(y)
    if label_smoothing_filter_funcs is not None:
        y_proba_smoothed = smooth(y_proba, filter_funcs=label_smoothing_filter_funcs)
        y_pred_numeric_smoothed = np.argmax(y_proba_smoothed, axis=1)
    timestamps = sampleable.trajectory.timestamps
    annotations = None
    if isinstance(sampleable, AnnotatedSampleable):
        annotations = sampleable.annotations
        categories = sampleable.categories
    if categories is None:
        raise ValueError("specify categories when classifying unannotated sampleables.")
    labels = (
        y_pred_numeric_smoothed
        if y_pred_numeric_smoothed is not None
        else y_pred_numeric
    )
    probabilities = y_proba_smoothed if y_proba_smoothed is not None else y_proba
    predictions = to_annotations(labels, categories, timestamps=timestamps)
    probabilities = [
        probabilities[
            (timestamps >= prediction["start"]) & (timestamps <= prediction["stop"])
        ]
        for _, prediction in predictions.iterrows()
    ]
    predictions["mean_probability"] = [
        proba[:, proba.argmax(axis=1)].mean() for proba in probabilities
    ]
    predictions["max_probability"] = [
        proba[:, proba.argmax(axis=1)].max() for proba in probabilities
    ]
    if annotations is not None:
        predictions = validate_predictions(predictions, annotations, on="predictions")
        annotations = validate_predictions(predictions, annotations, on="annotations")
    return ClassificationResult(
        categories=categories,
        predictions=predictions,
        y_proba=y_proba,
        y_pred_numeric=y_pred_numeric,
        _y_proba_smoothed=y_proba_smoothed,
        _y_pred_numeric_smoothed=y_pred_numeric_smoothed,
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
    label_smoothing_filter_funcs: Optional[list[NDArray_to_NDArray]] = None,
    exclude: Optional[
        list[Identity]
        | list[tuple[Identity, Identity]]
        | list[Identity | tuple[Identity, Identity]]
    ] = None,
) -> GroupClassificationResult:
    results: dict[Identity | tuple[Identity, Identity], ClassificationResult] = {}
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
            label_smoothing_filter_funcs=label_smoothing_filter_funcs,
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
    label_smoothing_filter_funcs: Optional[list[NDArray_to_NDArray]] = None,
    exclude: Optional[
        list[Identity]
        | list[tuple[Identity, Identity]]
        | list[Identity | tuple[Identity, Identity]]
    ] = None,
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
            label_smoothing_filter_funcs=label_smoothing_filter_funcs,
            exclude=exclude,
        )
    return DatasetClassificationResult(
        classification_results=results,
    )

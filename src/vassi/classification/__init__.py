# from .optimize import optimize_decision_thresholds, optimize_smoothing
from ._predict import k_fold_predict, predict
from ._results import (
    AnnotatedClassification,
    AnnotatedDatasetClassification,
    AnnotatedGroupClassification,
    Classification,
    DatasetClassification,
    GroupClassification,
)
from .visualization import plot_classification_timeline, plot_confusion_matrix

__all__ = [
    "predict",
    "k_fold_predict",
    "AnnotatedClassification",
    "AnnotatedDatasetClassification",
    "AnnotatedGroupClassification",
    "Classification",
    "DatasetClassification",
    "GroupClassification",
    "plot_classification_timeline",
    "plot_confusion_matrix",
]

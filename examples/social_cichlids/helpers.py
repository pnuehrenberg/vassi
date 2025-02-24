import numpy as np
import pandas as pd

from automated_scoring.classification.optimize import OverlappingPredictionsKwargs
from automated_scoring.dataset import AnnotatedDataset, Dataset
from automated_scoring.dataset.sampling.permutation import permute_recipients
from automated_scoring.features import DataFrameFeatureExtractor, FeatureExtractor
from automated_scoring.sliding_metrics import sliding_median


def smooth(parameters, *, array):
    if parameters["median_filter_window"] <= 1:
        return array
    return sliding_median(array, parameters["median_filter_window"])


def score_priority(observations: pd.DataFrame) -> pd.Series:
    return (
        (1 - observations["max_probability"]) + (1 - observations["mean_probability"])
    ) / 2


overlapping_predictions_kwargs = OverlappingPredictionsKwargs(
    priority_func=score_priority,
    prefilter_recipient_bouts=True,
    max_bout_gap=60,
    max_allowed_bout_overlap=30,
)


def subsample_train(
    dataset: Dataset | AnnotatedDataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    random_state=None,
    log,
):
    if not isinstance(dataset, AnnotatedDataset):
        raise ValueError(
            f"helper function to sample annotated datasets, got invalid dataset of type {type(dataset)}"
        )

    X, y = dataset.subsample(
        extractor,
        {
            ("approach", "chase", "dart_bite", "lateral_display", "quiver"): 1.0,
            "frontal_display": 0.25,
            "none": 0.01,
        },
        random_state=random_state,
        stratify=True,
        reset_previous_indices=False,
        exclude_previous_indices=False,
        store_indices=False,
        log=log,
    )

    sampling_frequency = {0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05}

    X_additional = [
        permute_recipients(dataset, neighbor_rank=neighbor_rank).subsample(
            extractor,
            {
                ("approach", "chase", "dart_bite", "lateral_display", "quiver"): 1.0
                * sampling_frequency[neighbor_rank],
                "frontal_display": 0.25 * sampling_frequency[neighbor_rank],
            },
            random_state=random_state,
            stratify=True,
            reset_previous_indices=False,
            exclude_previous_indices=False,
            store_indices=False,
            log=log,
        )[0]  # only keep samples (X) but not labels (y)
        for neighbor_rank in sampling_frequency
    ]

    if isinstance(extractor, DataFrameFeatureExtractor):
        # we know that X_additional is a list of DataFrames
        X_additional = pd.concat(X_additional, axis=0, ignore_index=True)  # type: ignore
        X = pd.concat([X, X_additional], axis=0, ignore_index=True)  # type: ignore
    else:
        X_additional = np.concatenate(X_additional, axis=0)
        X = np.concatenate([X, X_additional], axis=0)

    y_additional = np.repeat(
        "none", len(X_additional)
    )  # all labels are "none" because of switched recipients

    return X, np.concatenate([y, y_additional])

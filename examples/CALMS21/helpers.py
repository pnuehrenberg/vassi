from numpy.typing import NDArray

from automated_scoring.dataset import AnnotatedDataset, Dataset
from automated_scoring.features import BaseExtractor, F
from automated_scoring.sliding_metrics import sliding_median


def smooth(parameters, *, array):
    median_filter_window = int(parameters["median_filter_window"])
    if median_filter_window <= 1:
        return array
    return sliding_median(array, median_filter_window)


def subsample_train(
    dataset: Dataset,
    extractor: BaseExtractor[F],
    *,
    random_state,
    log,
) -> tuple[F, NDArray]:
    if not isinstance(dataset, AnnotatedDataset):
        raise ValueError("dataset must be an annotated dataset")
    return dataset.subsample(
        extractor,
        {
            ("attack", "mount"): 1.0,
            ("none", "investigation"): 30000,
        },
        random_state=random_state,
        stratify=True,
        reset_previous_indices=False,
        exclude_previous_indices=False,
        store_indices=False,
        log=log,
    )

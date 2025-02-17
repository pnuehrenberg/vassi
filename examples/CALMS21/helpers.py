from typing import TYPE_CHECKING

from automated_scoring.dataset import AnnotatedDataset
from automated_scoring.features import DataFrameFeatureExtractor


def subsample_train(
    dataset,
    extractor,
    *,
    random_state=None,
    log,
):
    if TYPE_CHECKING:
        assert isinstance(dataset, AnnotatedDataset)
        assert isinstance(extractor, DataFrameFeatureExtractor)

    return dataset.subsample(
        extractor,
        {
            ("attack", "mount"): 1.0,
            ("none", "investigation"): 0.1,
        },
        random_state=random_state,
        stratify=True,
        reset_previous_indices=False,
        exclude_previous_indices=False,
        store_indices=False,
        exclude=None,
        log=log,
    )

from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ...features import DataFrameFeatureExtractor, FeatureExtractor

if TYPE_CHECKING:
    from ._sampleable import Sampleable
    from .group import Group


Identity = str | int


def get_concatenated_dataset(
    sampleables: dict[Identity | tuple[Identity, Identity], "Sampleable"]
    | Sequence["Group"]
    | Sequence["Sampleable"],
    feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    # subsample kwargs
    random_state: Optional[np.random.Generator | int] = None,
    stratify_by_groups: bool = True,
    store_indices: bool = False,
    exclude_stored_indices: bool = False,
    reset_stored_indices: bool = False,
    categories: Optional[list[str]] = None,
    try_even_subsampling: bool = True,
    # other kwargs
    sampling_type: Literal["sample", "subsample"],
    exclude: Optional[list[Identity] | list[tuple[Identity, Identity]]],
) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
    X = []
    y = []
    if isinstance(sampleables, dict):
        if exclude is None:
            exclude = []
        sampleables = [
            sampleable for key, sampleable in sampleables.items() if key not in exclude
        ]
        exclude = None
    for sampleable in sampleables:
        if sampling_type == "sample":
            _X, _y = sampleable.sample(
                feature_extractor,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
                exclude=exclude,
            )
        elif sampling_type == "subsample":
            _X, _y = sampleable.subsample(
                feature_extractor,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
                random_state=random_state,
                stratify_by_groups=stratify_by_groups,
                store_indices=store_indices,
                exclude_stored_indices=exclude_stored_indices,
                reset_stored_indices=reset_stored_indices,
                categories=categories,
                try_even_subsampling=try_even_subsampling,
                exclude=exclude,
            )
        X.append(_X)
        if _y is not None:
            y.append(_y)
    assert len(y) == 0 or len(y) == len(X)
    X = type(feature_extractor).concatenate(*X, axis=0)
    if len(y) > 0:
        y = np.concatenate(y)
    else:
        y = None
    return X, y

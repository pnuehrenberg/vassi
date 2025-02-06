from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import class_name
from ..utils import Identifier

if TYPE_CHECKING:
    from ._sampleable import Sampleable
    from .group import Group


def recursive_sampleables(sampleable, exclude: Optional[Iterable[Identifier]] = None):
    from . import AnnotatedDyad, AnnotatedIndividual, Dataset, Dyad, Group, Individual

    exclude = list(exclude) if exclude is not None else []
    if isinstance(sampleable, Dataset):
        sampleables = []
        for group_id in sampleable.identifiers:
            if group_id in exclude:
                continue
            sampleables.extend(
                recursive_sampleables(sampleable.select(group_id), exclude)
            )
        return sampleables
    if isinstance(sampleable, Group):
        sampleables = []
        for sampleable_id in sampleable.identifiers:
            if sampleable_id in exclude:
                continue
            sampleables.extend(
                recursive_sampleables(sampleable.select(sampleable_id), exclude)
            )
        return sampleables
    if isinstance(sampleable, (Individual, Dyad, AnnotatedDyad, AnnotatedIndividual)):
        return sampleable.sampling_targets
    raise ValueError(f"invalid input sampleable of type {type(sampleable)}")


def get_concatenated_dataset(
    sampleables: dict[Identifier, "Sampleable"]
    | Sequence["Group"]
    | Sequence["Sampleable"],
    feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
    *,
    # subsample kwargs
    size: Optional[int | float] = None,
    random_state: Optional[np.random.Generator | int] = None,
    stratify_by_groups: bool = True,
    store_indices: bool = False,
    exclude_stored_indices: bool = False,
    reset_stored_indices: bool = False,
    categories: Optional[list[str]] = None,
    try_even_subsampling: bool = True,
    # other kwargs
    sampling_type: Literal["sample", "subsample"],
) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
    X, y = [], []
    if isinstance(sampleables, dict):
        sampleables = list(sampleables.values())
    elif not isinstance(sampleables, Sequence):
        raise ValueError("sampleables must be a dict or a sequence")
    for idx, sampleable in enumerate(sampleables):
        if sampling_type == "sample":
            sampled_data = sampleable.sample(feature_extractor)
        elif sampling_type == "subsample":
            sampled_data = sampleable.subsample(
                feature_extractor,
                size,
                random_state=random_state,
                stratify_by_groups=stratify_by_groups,
                store_indices=store_indices,
                exclude_stored_indices=exclude_stored_indices,
                reset_stored_indices=reset_stored_indices,
                categories=categories,
                try_even_subsampling=try_even_subsampling,
            )
        else:
            raise ValueError("invalid samling type")
        X.append(sampled_data[0])
        y.append(sampled_data[1])
        logger.trace(
            f"[{idx + 1}/{len(sampleables)}] {"sub" if sampling_type == 'subsample' else ''}sampled {class_name(sampleable)} with {len(sampled_data[0])} samples"
        )
    if any([_y is None for _y in y]):
        y = []
    X = type(feature_extractor).concatenate(*X, axis=0)
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if len(y) > 0:
        y = np.concatenate(y)
    else:
        y = None
    return X, y

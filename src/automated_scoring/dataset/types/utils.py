# import multiprocessing
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import formatted_tqdm

if TYPE_CHECKING:
    from ._sampleable import Sampleable
    from .group import Group


Identity = str | int
DyadIdentity = tuple[Identity, Identity]


def recursive_sampleables(
    sampleable, exclude: Optional[Iterable[Identity | DyadIdentity]] = None
):
    from . import AnnotatedDyad, AnnotatedIndividual, Dataset, Dyad, Group, Individual

    exclude = list(exclude) if exclude is not None else []
    if isinstance(sampleable, Dataset):
        sampleables = []
        for group_key in sampleable.group_keys:
            if group_key in exclude:
                continue
            sampleables.extend(
                recursive_sampleables(sampleable.select(group_key), exclude)
            )
        return sampleables
    if isinstance(sampleable, Group):
        sampleables = []
        for key in sampleable.keys:
            if key in exclude:
                continue
            sampleables.extend(recursive_sampleables(sampleable.select(key), exclude))
        return sampleables
    if isinstance(sampleable, (Individual, Dyad, AnnotatedDyad, AnnotatedIndividual)):
        return sampleable.sampling_targets
    raise ValueError(f"invalid input sampleable of type {type(sampleable)}")


def _process(args):
    sampleable, kwargs = args
    if kwargs["sampling_type"] == "sample":
        return sampleable.sample(
            kwargs["feature_extractor"],
            pipeline=kwargs["pipeline"],
            fit_pipeline=kwargs["fit_pipeline"],
        )
    if kwargs["sampling_type"] == "subsample":
        return sampleable.subsample(
            kwargs["feature_extractor"],
            kwargs["size"],
            pipeline=kwargs["pipeline"],
            fit_pipeline=kwargs["fit_pipeline"],
            random_state=kwargs["random_state"],
            stratify_by_groups=kwargs["stratify_by_groups"],
            store_indices=kwargs["store_indices"],
            exclude_stored_indices=kwargs["exclude_stored_indices"],
            reset_stored_indices=kwargs["reset_stored_indices"],
            categories=kwargs["categories"],
            try_even_subsampling=kwargs["try_even_subsampling"],
        )
    raise ValueError("invalid samling type")


def get_concatenated_dataset(
    sampleables: dict[Identity | DyadIdentity, "Sampleable"]
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
    show_progress: bool = True,
) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
    kwargs = dict(
        feature_extractor=feature_extractor,
        size=size,
        random_state=random_state,
        stratify_by_groups=stratify_by_groups,
        store_indices=store_indices,
        exclude_stored_indices=exclude_stored_indices,
        reset_stored_indices=reset_stored_indices,
        categories=categories,
        try_even_subsampling=try_even_subsampling,
        sampling_type=sampling_type,
    )
    X, y = zip(
        *list(
            formatted_tqdm(
                map(
                    _process,
                    [(sampleable, kwargs) for sampleable in sampleables],
                ),
                total=len(sampleables),
                desc="sampling",
                disable=not show_progress,
            ),
        ),
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

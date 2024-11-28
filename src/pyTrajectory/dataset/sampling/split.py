from typing import Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ...utils import ensure_generator, sklearn_seed


def test_stratify(
    num_samples: int,
    size: float | int,
    stratify: NDArray,
    y: Optional[NDArray],
) -> NDArray | None:
    if isinstance(size, float):
        if size < 0 or size > 1:
            raise ValueError("size as float should be between 0 and 1 (inclusive)")
        elif size == 1:
            # valid, take entire samples
            return stratify
        size = int(size * num_samples)
    discard_size = num_samples - size
    if y is None:
        counts = np.unique(stratify, return_counts=True)[1]
    else:
        counts = np.unique(list(zip(y, stratify)), axis=0, return_counts=True)[1]
    num_classes = len(counts)  # classes used for stratification
    if counts.min() < 2 or size < num_classes or discard_size < num_classes:
        # The minimum number of groups for any class cannot be less than 2.
        # The test_size (and train_size) should be greater or equal to the number of classes.
        return None
    return stratify


@overload
def split(
    X: NDArray,
    *args,
    **kwargs,
) -> tuple[NDArray, NDArray | None, NDArray | None]: ...


@overload
def split(
    X: pd.DataFrame,
    *args,
    **kwargs,
) -> tuple[pd.DataFrame, NDArray | None, NDArray | None]: ...


def split(
    X: NDArray | pd.DataFrame,
    *,
    y: Optional[NDArray],
    idx: Optional[NDArray],
    size: int | float,
    stratify: Optional[NDArray] = None,
    random_state: Optional[np.random.Generator | int] = None,
) -> tuple[NDArray | pd.DataFrame, NDArray | None, NDArray | None]:
    if isinstance(size, int) and size >= len(X):
        size = 1.0
    elif isinstance(size, int):
        size = size / len(X)
    else:
        assert isinstance(size, float)
    if size == 1.0:
        return X, y, idx
    random_state = ensure_generator(random_state)
    inputs = [X]
    if has_y := (y is not None):
        inputs.append(y)
    if has_idx := (idx is not None):
        inputs.append(idx)
    outputs = train_test_split(
        *inputs,
        train_size=size,
        random_state=sklearn_seed(random_state),
        stratify=stratify,
    )
    X, _ = outputs[:2]  # type: ignore
    outputs = outputs[2:]
    if has_y:
        y, _ = outputs[:2]  # type: ignore
        outputs = outputs[2:]
    if has_idx:
        idx, _ = outputs[:2]  # type: ignore
    return X, y, idx

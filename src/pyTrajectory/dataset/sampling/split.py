from typing import Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from ...utils import ensure_generator, sklearn_seed


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
    if isinstance(size, float) and size == 1.0:
        return X, y, idx
    random_state = ensure_generator(random_state)
    inputs = [X]
    if has_y := y is not None:
        inputs.append(y)
    if has_idx := idx is not None:
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

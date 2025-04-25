from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Optional,
)

import loguru
import numpy as np
from sklearn.model_selection import train_test_split

from ..logging import set_logging_level
from ..utils import to_int_seed


def _get_indices_by_category(
    category: str | tuple[str, ...],
    *,
    indices: np.ndarray,
    y: np.ndarray,
    stratification_levels: Sequence[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, Sequence[np.ndarray | None]]:
    if isinstance(category, str):
        mask = y == category
    else:
        mask = np.isin(y, category)
    indices = indices[mask]
    y = y[mask]
    stratification_levels = [
        (level[mask] if level is not None else None) for level in stratification_levels
    ]
    return indices, y, stratification_levels


def get_available_indices(
    previous_indices: list[np.ndarray],
    *,
    indices: np.ndarray,
    y: Optional[np.ndarray],
    intervals: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    previous_indices_flat = np.concatenate(previous_indices)
    mask = ~np.isin(indices, previous_indices_flat)
    indices = indices[mask]
    if y is not None:
        y = y[mask]
    if intervals is not None:
        intervals = intervals[mask]
    return indices, y, intervals


def _subselect_indices(
    indices: np.ndarray,
    *,
    size: int | float,
    random_state: np.random.Generator,
    stratify: bool,
    stratification_levels: Optional[Iterable[np.ndarray | None]],
) -> np.ndarray:
    if len(indices) == 0:
        return indices
    if isinstance(size, int):
        size = float(np.clip(size / len(indices), 0, 1))
        return _subselect_indices(
            indices,
            size=size,
            random_state=random_state,
            stratify=stratify,
            stratification_levels=stratification_levels,
        )
    if TYPE_CHECKING:
        assert isinstance(size, float)
    if size == 0 or len(indices) * size < 1:
        return np.array([], dtype=int)
    if size == 1:
        return indices
    if not stratify:
        selected_indices = train_test_split(
            indices,
            train_size=size,
            random_state=to_int_seed(random_state),
        )[0]
        if TYPE_CHECKING:
            assert isinstance(selected_indices, np.ndarray)
        return selected_indices
    selected_indices = None
    if stratification_levels is None:
        raise ValueError("stratification levels must be provided when stratifying")
    for level in stratification_levels:
        try:
            selected_indices = train_test_split(
                indices,
                train_size=size,
                stratify=level,
                random_state=to_int_seed(random_state),
            )[0]
            break
        except ValueError:
            pass
    if selected_indices is None:
        raise ValueError("could not apply stratification")
    if TYPE_CHECKING:
        assert isinstance(selected_indices, np.ndarray)
    return selected_indices


def select_indices(
    indices: np.ndarray,
    y: Optional[np.ndarray],
    *,
    size: int | float | Mapping[str | tuple[str, ...], int | float],
    random_state: Optional[int | np.random.Generator],
    stratify: bool,
    stratification_levels: Sequence[np.ndarray | None],
    categories: Optional[tuple[str, ...]],
    log: Optional[loguru.Logger],
) -> np.ndarray:
    random_state = np.random.default_rng(random_state)
    if log is None:
        log = set_logging_level()
    if isinstance(size, float | int):
        return _subselect_indices(
            indices,
            size=size,
            random_state=random_state,
            stratify=stratify,
            stratification_levels=[*stratification_levels, y, None],
        )
    if not isinstance(size, Mapping) or len(size) == 0:
        raise ValueError(
            "size must be either a single value (float or int) or a mapping from categories (single: str, or multiple: tuple[str, ...]) to values (float or int)"
        )
    if categories is None:
        raise ValueError(
            "categories must be provided when subsampling with specified categories"
        )
    if y is None:
        raise ValueError(
            "y must be provided when subsampling with specified categories"
        )
    selected_indices = []
    for key, _size in size.items():
        if isinstance(key, str) and key not in categories:
            log.warning(f"attempting to subsample undefined category ({key})")
        elif isinstance(key, tuple) and not all(
            category in categories for category in key
        ):
            log.warning(
                f"attempting to subsample at least one undefined category ({', '.join(key)})"
            )
        indices_category, y_category, stratification_levels_category = (
            _get_indices_by_category(
                key, indices=indices, y=y, stratification_levels=stratification_levels
            )
        )
        selected_indices.append(
            _subselect_indices(
                indices_category,
                size=_size,
                random_state=random_state,
                stratify=stratify,
                stratification_levels=[
                    *stratification_levels_category,
                    y_category,
                    None,
                ],
            )
        )
    return np.concatenate(selected_indices)

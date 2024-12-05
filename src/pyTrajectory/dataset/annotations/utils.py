from functools import wraps
from typing import Callable, Iterable, Optional, ParamSpec

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..utils import interval_overlap

P = ParamSpec("P")


def with_duration(
    annotations_func: Callable[P, pd.DataFrame],
) -> Callable[P, pd.DataFrame]:
    @wraps(annotations_func)
    def decorated(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
        annotations = annotations_func(*args, **kwargs)
        annotations["duration"] = annotations["stop"] - annotations["start"] + 1
        return annotations

    decorated.__name__ = f"scl_{annotations_func.__name__}"
    return decorated


@with_duration
def to_annotations(
    y: NDArray[np.int64],
    category_names: Iterable[str],
    drop: Optional[Iterable[str]] = None,
    timestamps: Optional[NDArray[np.int64 | np.float64]] = None,
) -> pd.DataFrame:
    if not y.ndim == 1:
        raise ValueError("y should be a 1D array of category labels (int).")
    change_idx = np.argwhere((np.diff(y) != 0)).ravel()
    stop = np.asarray(change_idx.tolist() + [y.size - 1])
    start = np.asarray([0] + (change_idx + 1).tolist())
    categories = np.asarray(category_names)[y[start]]
    if timestamps is not None:
        start = timestamps[start]
        stop = timestamps[stop]
    # assert not (y[start[:-1]] == y[stop[:-1] + 1]).any() and not (y[start[1:] - 1] == y[stop[1:]]).any()
    annotations = pd.DataFrame({"start": start, "stop": stop, "category": categories})
    if drop is None:
        return annotations
    return annotations.set_index("category").drop(drop, axis="index").reset_index()


@with_duration
def infill_annotations(
    annotations: pd.DataFrame,
    observation_stop: Optional[int] = None,
) -> pd.DataFrame:
    if len(annotations) == 0:
        return pd.DataFrame(
            [pd.Series({"category": "none", "start": 0, "stop": observation_stop})]
        )
    insert_idx = (
        np.asarray(annotations["start"][1:]) - np.asarray(annotations["stop"][:-1]) > 1
    )
    padding: list[pd.Series] = []
    if (start := annotations["start"].min()) != 0:
        padding.append(pd.Series({"category": "none", "start": 0, "stop": start - 1}))
    if (
        observation_stop is not None
        and (stop := annotations["stop"].max()) != observation_stop
    ):
        padding.append(
            pd.Series({"category": "none", "start": stop + 1, "stop": observation_stop})
        )
    annotations_fill = pd.DataFrame(
        {
            "category": ["none"] * insert_idx.sum(),
            "start": np.asarray(annotations[:-1].loc[insert_idx, "stop"] + 1),
            "stop": np.asarray(annotations[1:].loc[insert_idx, "start"] - 1),
        }
    )
    annotations = (
        pd.concat(
            [
                pd.DataFrame(padding),
                annotations,
                annotations_fill,
            ],
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
    annotations.loc[annotations["stop"] > observation_stop, "stop"] = observation_stop
    annotations = annotations.loc[annotations["start"] <= observation_stop]
    return annotations


@with_duration
def check_annotations(
    annotations: pd.DataFrame,
    required_columns: Iterable[str],
    allow_overlapping: bool = False,
    allow_unsorted: bool = False,
) -> pd.DataFrame:
    missing_columns = [
        column for column in required_columns if column not in annotations.columns
    ]
    if len(missing_columns) > 0:
        raise ValueError(
            f"Annotations are missing required columns: {', '.join(missing_columns)}."
        )
    if allow_overlapping:
        return annotations
    if "start" not in required_columns or "stop" not in required_columns:
        raise ValueError(
            "Overlap can only be checked if both 'start' and 'stop' are required columns."
        )
    intervals = annotations[["start", "stop"]].to_numpy()
    overlap = interval_overlap(intervals, intervals, mask_diagonal=True)
    if np.any(overlap > 0):
        raise ValueError("Annotations contain overlapping intervals.")
    if np.any(np.diff(annotations["start"])) < 0:
        raise ValueError("Annotations are not sorted by 'start'.")
    return annotations

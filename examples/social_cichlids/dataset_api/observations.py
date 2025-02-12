import functools
from typing import (
    Callable,
    ParamSpec,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from automated_scoring.dataset.observations.utils import (
    check_observations,
)

P = ParamSpec("P")


def _with_duration(*args, func: Callable[P, pd.DataFrame], **kwargs) -> pd.DataFrame:
    observations = func(*args, **kwargs)
    duration = observations["stop"] - observations["start"] + 1
    if "duration" in observations.columns:
        observations.loc[:, "duration"] = duration
    else:
        observations["duration"] = duration
    return observations


def with_duration(func: Callable[P, pd.DataFrame]) -> Callable[P, pd.DataFrame]:
    result_func = functools.partial(_with_duration, func=func)
    decorated = functools.wraps(func)(result_func)
    return decorated


def ensure_single_index(
    observations: pd.DataFrame,
    *,
    index_columns: tuple[str, ...],
    drop: bool = True,
) -> pd.DataFrame:
    if len(index_columns) == 0:
        return observations
    observations = observations.set_index(list(index_columns))
    if len(np.unique(observations.index)) > 1:
        raise ValueError(
            "observations contain more than one unique index key combination"
        )
    return observations.reset_index(drop=drop)


def to_y(
    observations: pd.DataFrame,
    *,
    start: float = 0,
    stop: float = np.inf,
    dtype: type = str,
) -> NDArray:
    observations = check_observations(
        observations,
        required_columns=("start", "stop", "category"),
        allow_overlapping=False,
        allow_unsorted=False,
    ).copy()
    observations = observations.loc[observations["stop"] >= start]
    observations.loc[observations["start"] < start, "start"] = start
    if stop < np.inf:
        observations = observations.loc[observations["start"] <= stop]
        observations.loc[observations["stop"] > stop, "stop"] = stop
    intervals_float = np.array(observations[["start", "stop"]])
    intervals = intervals_float.astype(int)
    if not np.allclose(intervals, intervals_float, rtol=0):
        raise ValueError("start and stop columns must be integers")
    duration = intervals[:, 1] - intervals[:, 0] + 1
    return np.repeat(np.array(observations["category"], dtype=dtype), duration)

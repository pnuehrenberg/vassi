import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional, ParamSpec

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..utils import interval_contained, interval_overlap

P = ParamSpec("P")


def _with_duration(*args, func: Callable[P, pd.DataFrame], **kwargs) -> pd.DataFrame:
    observations = func(*args, **kwargs)
    duration = observations["stop"] - observations["start"] + 1
    if "duration" in observations.columns:
        observations.loc[:, "duration"] = duration
    else:
        observations["duration"] = duration
    for column in ["start", "stop", "duration"]:
        observations[column] = pd.to_numeric(observations[column], downcast="integer")
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
    validated_observations = observations.reset_index(drop=drop, inplace=False)
    if TYPE_CHECKING:
        # pyright does not correctly infer return type of reset_index with inplace=False
        assert validated_observations is not None
    return validated_observations


def to_y(
    observations: pd.DataFrame,
    *,
    start: int = 0,
    stop: Optional[int] = None,
    dtype: type = str,
) -> NDArray:
    observations = check_observations(
        observations,
        required_columns=("start", "stop", "category"),
        allow_overlapping=False,
        allow_unsorted=False,
    ).copy()
    for timestamp_dtype in observations[["start", "stop"]].dtypes:
        if "int" in str(timestamp_dtype):
            continue
        raise ValueError(f"start and stop columns must be integers, got {timestamp_dtype}")
    observations = observations.loc[observations["stop"] >= start]
    observations.loc[observations["start"] < start, "start"] = int(start)
    if stop is not None:
        observations = observations.loc[observations["start"] <= stop]
        observations.loc[observations["stop"] > stop, "stop"] = int(stop)
    intervals = np.array(observations[["start", "stop"]]).astype(int)
    duration = intervals[:, 1] - intervals[:, 0] + 1
    return np.repeat(np.array(observations["category"], dtype=dtype), duration)


@with_duration
def to_observations(
    y: NDArray[np.integer],
    category_names: Iterable[str],
    drop: Optional[Iterable[str]] = None,
    timestamps: Optional[NDArray[np.integer | np.floating]] = None,
) -> pd.DataFrame:
    if not y.ndim == 1:
        raise ValueError("y should be a 1D array of category labels (int).")
    change_idx = np.argwhere((np.diff(y) != 0)).ravel()
    stop = np.asarray(change_idx.tolist() + [len(y) - 1])
    start = np.asarray([0] + (change_idx + 1).tolist())
    categories = np.asarray(category_names)[y[start]]
    if timestamps is not None:
        start = timestamps[start]
        stop = timestamps[stop]
    observations = pd.DataFrame({"start": start, "stop": stop, "category": categories})
    if drop is None:
        return observations
    observations = observations.set_index("category").drop(
        drop, axis="index", inplace=False
    )
    observations = observations.reset_index(drop=False, inplace=False)
    return observations


@with_duration
def infill_observations(
    observations: pd.DataFrame,
    observation_stop: Optional[int] = None,
) -> pd.DataFrame:
    observations = check_observations(
        observations, required_columns=["category", "start", "stop"]
    )
    dtype_start = observations["start"].dtype
    dtype_stop = observations["stop"].dtype
    if observation_stop is None:
        observation_stop = np.max(observations["stop"])
    if len(observations) == 0:
        return pd.DataFrame(
            [pd.Series({"category": "none", "start": 0, "stop": observation_stop})]
        )
    insert_idx = (
        np.asarray(observations["start"][1:]) - np.asarray(observations["stop"][:-1])
        > 1
    )
    padding: list[pd.Series] = []
    if (start := observations["start"].min()) != 0:
        padding.append(pd.Series({"category": "none", "start": 0, "stop": start - 1}))
    if (stop := observations["stop"].max()) != observation_stop:
        padding.append(
            pd.Series({"category": "none", "start": stop + 1, "stop": observation_stop})
        )
    observations_fill = pd.DataFrame(
        {
            "category": ["none"] * insert_idx.sum(),
            "start": np.asarray(observations[:-1].loc[insert_idx, "stop"] + 1),
            "stop": np.asarray(observations[1:].loc[insert_idx, "start"] - 1),
        }
    )
    observations = pd.concat(
        [
            pd.DataFrame(padding),
            observations,
            observations_fill,
        ],
    ).sort_values("start", ignore_index=True, inplace=False)
    observations.loc[observations["stop"] > observation_stop, "stop"] = observation_stop
    observations = observations.loc[observations["start"] <= observation_stop]
    return observations


@with_duration
def check_observations(
    observations: pd.DataFrame,
    required_columns: Iterable[str],
    allow_overlapping: bool = False,
    allow_unsorted: bool = False,
) -> pd.DataFrame:
    missing_columns = [
        column for column in required_columns if column not in observations.columns
    ]
    if len(missing_columns) > 0:
        raise ValueError(
            f"observations are missing required columns: {', '.join(missing_columns)}."
        )
    if not allow_unsorted and (np.diff(observations["start"]) < 0).any():
        raise ValueError("observations are not sorted by 'start'")
    if allow_overlapping:
        return observations
    if "start" not in required_columns or "stop" not in required_columns:
        raise ValueError(
            "Overlap can only be checked if both 'start' and 'stop' are required columns."
        )
    intervals = observations[["start", "stop"]].to_numpy()
    overlap = interval_overlap(intervals, intervals, mask_diagonal=True)
    if np.any(overlap > 0):
        raise ValueError("observations contain overlapping intervals.")
    if np.any(np.diff(observations["start"])) < 0:
        raise ValueError("observations are not sorted by 'start'.")
    return observations


def ensure_matching_index_columns(
    observations: pd.DataFrame,
    reference_observations: pd.DataFrame,
    index_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        ensure_single_index(observations, index_columns=index_columns),
        ensure_single_index(reference_observations, index_columns=index_columns),
    )


@with_duration
def remove_overlapping_observations(
    observations: pd.DataFrame,
    *,
    index_columns: tuple[str, ...],
    priority_function: Callable[[pd.DataFrame], Iterable[float]],
    max_allowed_overlap: float,
    drop_overlapping: bool = True,
    drop_overlapping_column: bool = True,
) -> pd.DataFrame:
    observations = ensure_single_index(observations, index_columns=index_columns)
    observations = check_observations(
        observations,
        required_columns=["start", "stop"],
        allow_overlapping=True,
        allow_unsorted=True,
    )
    if np.any(np.unique(observations.index, return_counts=True)[1] > 1):
        raise ValueError("observations have duplicated index values")
    if "overlapping" in observations.columns:
        observations.loc[:, "overlapping"] = ""
    else:
        observations["overlapping"] = ""
    intervals = observations[["start", "stop"]].to_numpy()
    overlap = interval_overlap(intervals, intervals, mask_diagonal=False)
    contained = interval_contained(intervals, intervals)
    graph = nx.Graph((overlap > max_allowed_overlap) | contained)
    for overlapping_idx in nx.connected_components(graph):
        overlapping_idx = np.asarray(sorted(overlapping_idx))
        if len(overlapping_idx) == 1:
            idx = observations.index[overlapping_idx[0]]
            observations.at[idx, "overlapping"] = "no"
            continue
        observations_component = observations.iloc[overlapping_idx]
        priority = np.asarray(priority_function(observations_component))
        order = np.argsort(priority)  # lower is better!
        prioritized = int(observations_component.index[order][0])
        overlap_prioritized = interval_overlap(
            observations_component[["start", "stop"]].to_numpy(),
            observations_component.loc[[prioritized], ["start", "stop"]].to_numpy(),
            mask_diagonal=False,
        )
        overlaps = overlap_prioritized.ravel() > 0
        observations_component.loc[overlaps, "overlapping"] = "yes"
        observations_component.at[prioritized, "overlapping"] = "prioritized"
        observations_temp = observations_component.loc[
            observations_component["overlapping"] == ""
        ]
        if len(observations_temp) > 0:
            observations_temp = remove_overlapping_observations(
                observations_temp,
                index_columns=(),
                priority_function=priority_function,
                max_allowed_overlap=max_allowed_overlap,
                drop_overlapping=False,
                drop_overlapping_column=False,
            )
            observations_component.loc[np.array(observations_temp.index)] = (
                observations_temp
            )
        observations.loc[np.array(observations_component.index)] = (
            observations_component
        )
    if drop_overlapping:
        observations = observations[observations["overlapping"] != "yes"]
    if drop_overlapping_column:
        observations = observations.drop(columns=["overlapping"], inplace=False)
    return observations

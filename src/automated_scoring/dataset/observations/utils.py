from functools import wraps
from typing import Callable, Iterable, Optional, ParamSpec

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..utils import interval_contained, interval_overlap

P = ParamSpec("P")


def with_duration(
    observations_func: Callable[P, pd.DataFrame],
) -> Callable[P, pd.DataFrame]:
    # TODO this is not pickleable, follow implementation structure of features.decorators
    @wraps(observations_func)
    def decorated(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
        observations = observations_func(*args, **kwargs)
        duration = observations["stop"] - observations["start"] + 1
        if "duration" in observations.columns:
            observations.loc[:, "duration"] = duration
        else:
            observations["duration"] = duration
        return observations

    decorated.__name__ = observations_func.__name__
    return decorated


@with_duration
def to_observations(
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
    observations = pd.DataFrame({"start": start, "stop": stop, "category": categories})
    if drop is None:
        return observations
    return observations.set_index("category").drop(drop, axis="index").reset_index()


@with_duration
def infill_observations(
    observations: pd.DataFrame,
    observation_stop: Optional[int] = None,
) -> pd.DataFrame:
    observations = check_observations(
        observations, required_columns=["category", "start", "stop"]
    )
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
    observations = (
        pd.concat(
            [
                pd.DataFrame(padding),
                observations,
                observations_fill,
            ],
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
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


def ensure_single_index(
    observations: pd.DataFrame,
    *,
    index_keys: Iterable[str],
    drop: bool = True,
) -> pd.DataFrame:
    index_keys = list(index_keys)
    if len(index_keys) == 0:
        return observations
    observations = observations.set_index(index_keys)
    if len(np.unique(observations.index)) > 1:
        raise ValueError(
            "observations contain more than one unique index key combination"
        )
    return observations.reset_index(drop=drop)


def ensure_matching_index_keys(
    observations: pd.DataFrame,
    reference_observations: pd.DataFrame,
    index_keys: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        ensure_single_index(observations, index_keys=index_keys),
        ensure_single_index(reference_observations, index_keys=index_keys),
    )


@with_duration
def remove_overlapping_observations(
    observations: pd.DataFrame,
    *,
    index_keys: Iterable[str],
    priority_func: Callable[[pd.DataFrame], Iterable[float]],
    max_allowed_overlap: float,
    drop_overlapping: bool = True,
    drop_overlapping_column: bool = True,
) -> pd.DataFrame:
    observations = ensure_single_index(observations, index_keys=index_keys)
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
        priority = np.asarray(priority_func(observations_component))
        order = np.argsort(priority)  # lower is better!
        prioritized = observations_component.index[order[0]]
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
                index_keys=[],
                priority_func=priority_func,
                max_allowed_overlap=max_allowed_overlap,
                drop_overlapping=False,
                drop_overlapping_column=False,
            )
            observations_component.loc[observations_temp.index] = observations_temp
        observations.loc[observations_component.index] = observations_component
    if drop_overlapping:
        observations = observations[observations["overlapping"] != "yes"]  # type: ignore
    if drop_overlapping_column:
        observations = observations.drop(columns=["overlapping"])
    return observations

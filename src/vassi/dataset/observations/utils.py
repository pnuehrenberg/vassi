import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from ..utils import interval_contained, interval_overlap


def with_duration[**P](func: Callable[P, pd.DataFrame]) -> Callable[P, pd.DataFrame]:
    """Decorator to add a 'duration' column to the output of a function that returns a DataFrame."""

    @functools.wraps(func)
    def _with_duration(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
        observations = func(*args, **kwargs).copy()  # may be a slice etc.
        duration = observations["stop"] - observations["start"] + 1
        observations["duration"] = duration
        for column in ["start", "stop", "duration"]:
            values = pd.to_numeric(observations[column], downcast="integer")
            dtype = int if "int" in str(values.dtype) else float
            observations[column] = observations[column].astype(dtype)
        return observations

    return _with_duration


def ensure_single_index(
    observations: pd.DataFrame,
    *,
    index_columns: tuple[str, ...],
    drop: bool = True,
) -> pd.DataFrame:
    """
    Ensure that the observations DataFrame has a single index key combination.

    Parameters:
        observations: The observations to validate.
        index_columns: The columns to use as the index.
        drop: Whether to drop the index columns from the DataFrame.

    Returns:
        The validated observations.
    """
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
) -> np.ndarray:
    """
    Convert observations to a 1D array of category labels.

    Parameters:
        observations: Observations, requires columns "start", "stop", and "category".
        start: Start timestamp.
        stop: Stop timestamp.
        dtype: Data type of the output array.

    Returns:
        A 1D array of category labels.
    """
    observations = check_observations(
        observations,
        required_columns=("start", "stop", "category"),
        allow_overlapping=False,
        allow_unsorted=False,
    ).copy()
    for timestamp_dtype in observations[["start", "stop"]].dtypes:
        if "int" in str(timestamp_dtype):
            continue
        raise ValueError(
            f"start and stop columns must be integers, got {timestamp_dtype}"
        )
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
    y: np.ndarray,
    category_names: Iterable[str],
    drop: Optional[Iterable[str]] = None,
    timestamps: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Convert a 1D array of category labels to a DataFrame of observations.

    Parameters:
        y: A 1D array of category labels.
        category_names: Category names.
        drop: Categories that should be dropped from the resulting observations.
        timestamps : Timestamps that correspond to the category labels. If not provided, timestamps are starting from 0.

    Returns:
        Observations with columns "start", "stop", and "category".
    """
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
    *,
    background_category: str = "none",
) -> pd.DataFrame:
    """
    Infill observations with intervals of the background category.

    Parameters:
        observations: The observations to infill.
        observation_stop: The stop time of the observations. If none, the maximum stop time of the observations is used.
        background_category: The category to use for the background intervals.

    Returns:
        The infilled observations.
    """
    observations = check_observations(
        observations, required_columns=["category", "start", "stop"]
    )
    if observation_stop is None:
        observation_stop = np.max(observations["stop"])
    if len(observations) == 0:
        return pd.DataFrame(
            [
                pd.Series(
                    {
                        "category": background_category,
                        "start": 0,
                        "stop": observation_stop,
                    }
                )
            ]
        )
    insert_idx = (
        np.asarray(observations["start"][1:]) - np.asarray(observations["stop"][:-1])
        > 1
    )
    padding: list[pd.Series] = []
    if (start := observations["start"].min()) != 0:
        padding.append(
            pd.Series({"category": background_category, "start": 0, "stop": start - 1})
        )
    if (stop := observations["stop"].max()) != observation_stop:
        padding.append(
            pd.Series(
                {
                    "category": background_category,
                    "start": stop + 1,
                    "stop": observation_stop,
                }
            )
        )
    observations_fill = pd.DataFrame(
        {
            "category": [background_category] * insert_idx.sum(),
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
    """
    Checks that the observations are valid.

    Args:
        observations: The observations to check.
        required_columns: The columns that are required in the observations.
        allow_overlapping: Whether overlapping intervals are allowed.
        allow_unsorted: Whether unsorted intervals are allowed.

    Returns:
        The checked observations.

    Raises:
        ValueError: If the observations are missing required columns.
        ValueError: If the observations are not sorted by 'start'.
        ValueError: If the observations are overlapping.
    """
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
    """
    Validates if two sets of observations have matching index columns.

    Parameters:
        observations: The first set of observations.
        reference_observations: The second set of observations.
        index_columns: The columns to use as index.

    Returns:
        The validated observations.
    """
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
    """
    Removes overlapping observations.

    Parameters:
        observations: The set of observations.
        index_columns: The columns to use as index.
        priority_function: A function that assigns a priority to each observation, lower values indicate higher priority.
        max_allowed_overlap: The maximum allowed overlap between observations.
        drop_overlapping: Whether to drop overlapping observations.
        drop_overlapping_column: Whether to drop the overlapping column.

    Returns:
        Non-overlapping observations.
    """
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

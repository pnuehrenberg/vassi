from collections.abc import Callable, Iterable
from typing import overload

import networkx as nx
import numpy as np
import pandas as pd


def to_observations(
    labels: np.ndarray,
    categories: set[str],
    timestamps: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Convert a 1D array of category labels to a DataFrame of observations, with "start" and "stop" defining inclusive observation intervals.

    Parameters:
        labels: A 1D array of integer category labels.
        categories: Category names. :code:`sorted(categories).index(label)` yields the corresponding category from an integer label.
        timestamps: Timestamps that correspond to the integer category labels. If not provided, timestamps are starting from 0.

    Returns:
        Observations with columns "start", "stop", and "category".
    """
    if not labels.ndim == 1:
        raise ValueError("y should be a 1D array of category labels (int).")
    change_idx = np.argwhere((np.diff(labels) != 0)).ravel()
    stop = np.r_[change_idx, len(labels) - 1]
    start = np.r_[0, change_idx + 1]
    category_names = np.asarray(sorted(categories))[labels[start]]
    if timestamps is not None:
        start = timestamps[start]
        stop = timestamps[stop]
    observations = pd.DataFrame(
        {"start": start, "stop": stop, "category": category_names}
    )
    return observations


@overload
def _densify_intervals(
    intervals: np.ndarray,
    time_range: tuple[int, int],
    *,
    labels: np.ndarray,
    background_category: str,
    additional_columns: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]: ...


@overload
def _densify_intervals(
    intervals: np.ndarray,
    time_range: tuple[int, int],
    *,
    additional_columns: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, None, tuple[np.ndarray, ...]]: ...


def _densify_intervals(
    intervals: np.ndarray,
    time_range: tuple[int, int],
    labels: np.ndarray | None = None,
    background_category: str | None = None,
    *,
    additional_columns: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[np.ndarray, ...]]:
    num_intervals = intervals.shape[0]
    if labels is not None and labels.shape[0] != num_intervals:
        raise ValueError("labels size must match number of intervals")
    if not np.isclose(intervals % 1, 0, rtol=0, atol=1e-8).all():
        raise ValueError("Intervals must be integers with a timestep of 1")
    # early return if zero intervals
    if num_intervals == 0:
        intervals = np.asarray(time_range).reshape(1, 2)
        if labels is not None:
            labels = np.full(1, background_category)
        return intervals, labels, ()
    start = intervals[:, 0]
    stop = intervals[:, 1]
    # check assumptions
    if not (np.diff(start) >= 1).all():
        raise ValueError("Intervals must be sorted by start")
    if not (stop - start >= 0).all():
        raise ValueError("Intervals must have non-negative durations")
    if not (start[1:] - 1 >= stop[:-1]).all():
        raise ValueError("Intervals must not be strictly non-overlapping")
    gaps = np.argwhere(start[1:] - stop[:-1] > 1).ravel()
    gaps_start = stop[gaps] + 1
    gaps_stop = start[1:][gaps] - 1
    gaps = np.transpose([gaps_start, gaps_stop])
    pad_start = start[0] > time_range[0]
    pad_stop = stop[-1] < time_range[1]
    padding = np.zeros((sum([pad_start, pad_stop]), 2), dtype=int)
    if pad_start:
        padding[0] = time_range[0], start[0] - 1
    if pad_stop:
        padding[-1] = stop[-1] + 1, time_range[1]
    densified_intervals = np.concatenate([intervals, gaps, padding], axis=0)
    # sort by start
    sort_idx = np.argsort(densified_intervals[:, 0])
    densified_intervals = densified_intervals[sort_idx]
    # remove all that completely fall outside time range
    overlaps_range = (densified_intervals >= time_range[0]).any(axis=1) & (
        densified_intervals <= time_range[1]
    ).any(axis=1)
    densified_intervals = densified_intervals[overlaps_range]
    densified_labels = None
    if labels is not None:
        gaps_category = np.full(gaps.shape[0], background_category)
        gaps_padding = np.full(padding.shape[0], background_category)
        # concatenate and apply same sorting/filtering
        densified_labels = np.concatenate([labels, gaps_category, gaps_padding])[
            sort_idx
        ][overlaps_range]
    densified_additional_cols: list[np.ndarray] = []
    if additional_columns is None:
        additional_columns = []
    for column_array in additional_columns:
        # Check if dtype is integer; if so, cast to float to allow for np.nan
        processed_array = column_array
        if np.issubdtype(column_array.dtype, np.integer):
            processed_array = column_array.astype(np.float64)

        # Create gap and padding values as np.nan
        gaps_fill = np.full(gaps.shape[0], np.nan, dtype=processed_array.dtype)
        padding_fill = np.full(padding.shape[0], np.nan, dtype=processed_array.dtype)

        # Concatenate, sort, and filter just like the labels
        densified_col = np.concatenate([processed_array, gaps_fill, padding_fill])[
            sort_idx
        ][overlaps_range]
        densified_additional_cols.append(densified_col)
    # clip intervals that partially fall outside time range
    # note that these can only possibly be the very first or last interval
    # because intervals are strictly non-overlapping
    densified_intervals[densified_intervals < time_range[0]] = time_range[0]
    densified_intervals[densified_intervals > time_range[1]] = time_range[1]
    return densified_intervals, densified_labels, tuple(densified_additional_cols)


def densify_observations(
    observations: pd.DataFrame,
    *,
    time_range: tuple[int, int],
    background_category: str | None,
) -> pd.DataFrame:
    intervals = np.asarray(observations[["start", "stop"]])
    if "category" not in observations.columns:
        additional_col_names = [
            col for col in observations.columns if col not in ["start", "stop"]
        ]
        densified_intervals, _, densified_additional_cols = _densify_intervals(
            intervals,
            time_range,
            additional_columns=[
                np.asarray(observations[col]) for col in additional_col_names
            ],
        )
        return pd.DataFrame(
            {
                "start": densified_intervals[:, 0],
                "stop": densified_intervals[:, 1],
                **{
                    col: values
                    for col, values in zip(
                        additional_col_names, densified_additional_cols
                    )
                },
            }
        )
    if background_category is None:
        raise ValueError(
            "background_category must be specified for observations with 'category' column"
        )
    additional_col_names = [
        col for col in observations.columns if col not in ["start", "stop", "category"]
    ]
    labels = np.asarray(observations["category"])
    densified_intervals, densified_labels, densified_additional_cols = (
        _densify_intervals(
            intervals,
            time_range,
            labels=labels,
            background_category=background_category,
            additional_columns=[
                np.asarray(observations[col]) for col in additional_col_names
            ],
        )
    )
    return pd.DataFrame(
        {
            "start": densified_intervals[:, 0],
            "stop": densified_intervals[:, 1],
            "category": densified_labels,
            **{
                col: values
                for col, values in zip(additional_col_names, densified_additional_cols)
            },
        }
    )


def interval_overlap(
    intervals_a: np.ndarray,
    intervals_b: np.ndarray,
    element_wise: bool = False,
) -> np.ndarray:
    """
    Calculates the overlap between two sets of inclusive intervals.

    This function assumes intervals are inclusive, e.g., [start, stop].
    The calculation is equivalent to finding the intersection of half-open
    intervals [start, stop + 1).

    Parameters:
        intervals_a: A (N, 2) array of [start, stop] intervals.
        intervals_b: A (M, 2) array of [start, stop] intervals.
        element_wise: If True, perform element-wise overlap between two
                      (N, 2) arrays. If False, computes the (N, M)
                      all-vs-all overlap matrix.

    Returns:
        An array of overlap values. The shape is (N,) if element_wise,
        otherwise it's (N, M).
    """
    # Decompose intervals for clarity
    start_a, end_a = intervals_a[:, 0], intervals_a[:, 1]
    start_b, end_b = intervals_b[:, 0], intervals_b[:, 1]

    if not element_wise:
        # Use broadcasting to create the all-vs-all comparison matrices
        start_a = start_a[:, np.newaxis]
        end_a = end_a[:, np.newaxis]

    # The core overlap formula is simpler by thinking of the interval
    # as [start, end + 1).
    overlap_starts = np.maximum(start_a, start_b)
    overlap_ends = np.minimum(end_a + 1, end_b + 1)

    # Use np.maximum for efficient clipping of negative values to zero.
    return np.maximum(0, overlap_ends - overlap_starts)


def interval_contained(
    intervals_a: np.ndarray,  # The intervals that might be contained
    intervals_b: np.ndarray,  # The intervals they might be contained in
    element_wise: bool = False,
) -> np.ndarray:
    """
    Checks if intervals in `intervals_a` are contained within `intervals_b`.

    An interval A is contained in B if (start_A >= start_B) and (end_A <= end_B).

    Parameters:
        intervals_a: A (N, 2) array of [start, stop] intervals to check.
        intervals_b: A (M, 2) array of [start, stop] intervals to check against.
        element_wise: If True, performs an element-wise check.

    Returns:
        A boolean array indicating containment. The shape is (N,) if
        element_wise, otherwise it's (N, M).
    """
    # Decompose intervals for clarity
    start_a, end_a = intervals_a[:, 0], intervals_a[:, 1]
    start_b, end_b = intervals_b[:, 0], intervals_b[:, 1]

    if not element_wise:
        # Use broadcasting to set up the all-vs-all comparison
        start_a = start_a[:, np.newaxis]
        end_a = end_a[:, np.newaxis]

    # Direct boolean check is significantly faster than calculating overlap value
    return (start_a >= start_b) & (end_a <= end_b)


def _bout_aggregator(bout_data: pd.DataFrame, **_: ...) -> pd.Series:
    duration = np.asarray(bout_data["stop"] - bout_data["start"] + 1)
    aggregated_values: dict[str, object] = {}
    for column in bout_data.columns:
        values = np.asarray(bout_data[column])
        if column == "start":
            aggregated_values["start"] = np.min(values)
            continue
        if column == "stop":
            aggregated_values["stop"] = np.max(values)
            continue
        unique_values, inverse_idx = np.unique(values, return_inverse=True)
        if len(unique_values) == 1:
            aggregated_values[column] = unique_values[0]
            continue
        cumulative_duration = [
            np.sum(duration[inverse_idx == idx]) for idx in range(len(unique_values))
        ]
        try:
            aggregated_values[column] = np.average(
                unique_values,
                weights=cumulative_duration,
            )
            continue
        except TypeError:
            pass
        aggregated_values[column] = [
            (value, _cumulative_duration)
            for _cumulative_duration, value in sorted(
                zip(cumulative_duration, unique_values), reverse=True
            )
        ]
    return pd.Series(aggregated_values)


def assert_singular_index_combination(
    observations: pd.DataFrame, index_columns: tuple[str, ...]
) -> pd.DataFrame:
    if len(index_columns) == 0:
        return observations
    if observations[list(index_columns)].nunique().prod() != 1:
        raise ValueError(
            f"observations must be for a single combination of {index_columns}"
        )
    return observations


def aggregate_observations_as_bouts(
    observations: pd.DataFrame,
    *,
    max_bout_gap: float,
    index_columns: tuple[str, ...],
    background_category: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate observations (behavioral intervals) into bouts.

    Parameters:
        observations: The observations to aggregate.
        max_bout_gap: The maximum gap between observations to consider them part of the same bout. A gap of :code:`start[i] - stop[i - 1] == 1` implies that observations are adjacent.
        index_columns: The columns to use as the index, unique combinations should point to independent observations (e.g., of one individual).
        background_category: The category to use for background observations that are not part of any bout.

    Returns:
        The aggregated bouts.
    """
    observations = assert_singular_index_combination(observations, index_columns)
    observations = densify_observations(
        observations,
        time_range=(observations["start"].min(), observations["start"].max()),
        background_category=background_category,
    )
    observations["bout"] = -1
    observations_foreground = observations.loc[
        observations["category"] != background_category
    ]
    gaps = np.asarray(observations_foreground["start"][1:]) - np.asarray(
        observations_foreground["stop"][:-1]
    )
    is_bout = np.r_[False, gaps <= max_bout_gap]
    bout_idx = np.full(len(is_bout), -1, dtype=int)
    bout_idx[~is_bout] = np.arange(np.sum(~is_bout))
    bout_idx_filled: list[int] = []
    for idx in bout_idx:
        bout_idx_filled.append(idx if idx != -1 else bout_idx_filled[-1])
    observations_foreground.loc[:, "bout"] = bout_idx_filled
    bouts = observations_foreground.groupby("bout")[  # pyright: ignore[reportUnknownMemberType]
        observations_foreground.columns
    ].apply(_bout_aggregator)
    bouts = bouts.sort_values(by="start", ignore_index=True)
    observations.loc[observations["category"] != background_category, "bout"] = (
        observations_foreground["bout"]
    )
    return bouts, observations


def remove_overlapping_observations(
    observations: pd.DataFrame,
    *,
    index_columns: tuple[str, ...],
    priority_function: Callable[[pd.DataFrame], Iterable[float]],
    max_overlap: float,
) -> pd.DataFrame:
    """
    Removes overlapping observations using a robust iterative approach.

    Within groups of overlapping observations, this function iteratively identifies the one
    with the highest priority (lowest value from `priority_function`), keeps it, and
    discards all others that conflict with it. This process repeats until all overlaps
    within the group are resolved.

    Parameters:
        observations: The DataFrame of observations. Must contain 'start' and 'stop' columns.
        index_columns: The columns that define a unique entity. The function asserts that
                       it only processes one such entity at a time.
        priority_function: A function that accepts a DataFrame and returns an iterable of floats.
                           Lower values indicate higher priority.
        max_overlap: The maximum allowed overlap before two observations are considered conflicting.

    Returns:
        A DataFrame with overlapping observations resolved according to the priority function.
    """
    observations = assert_singular_index_combination(observations, index_columns)

    if observations.empty:
        return observations

    # Use a copy to not add the status column to the original DataFrame
    observations = observations.copy()

    # Using a Categorical dtype is highly memory-efficient for low-cardinality data.
    overlap_categories = ["", "no", "prioritized", "yes"]
    observations["overlapping"] = pd.Categorical(
        [""] * len(observations), categories=overlap_categories, ordered=False
    )

    intervals = observations[["start", "stop"]].to_numpy()

    # Build graph to find overlapping groups
    overlap_matrix = interval_overlap(intervals, intervals) > max_overlap
    contained_matrix = interval_contained(intervals, intervals)
    # An interval should not conflict with itself.
    np.fill_diagonal(overlap_matrix, False)
    np.fill_diagonal(contained_matrix, False)

    # An edge in the graph represents a conflict.
    graph = nx.from_numpy_array(overlap_matrix | contained_matrix)

    # --- 4. Iteratively Process Each Conflict Group (Component) ---
    for component_indices in nx.connected_components(graph):
        component_indices = sorted(list(component_indices))

        # Case 1: The "group" is a single observation with no conflicts.
        if len(component_indices) == 1:
            idx = observations.index[component_indices[0]]
            observations.at[idx, "overlapping"] = "no"
            continue

        # Case 2: A real conflict group that needs to be resolved.
        # This is the sub-DataFrame for the current group.
        component_df_iloc = observations.iloc[component_indices]

        # This iterative loop replaces the original recursion.
        while True:
            # Find all observations not yet marked as 'prioritized' (kept) or 'yes' (discarded).
            unprocessed_mask = component_df_iloc["overlapping"] == ""
            if not unprocessed_mask.any():
                break  # Exit loop if the component is fully resolved.

            unprocessed_observations = component_df_iloc[unprocessed_mask]

            # a. Find the highest-priority observation AMONG THE UNPROCESSED ONES.
            priorities = np.asarray(priority_function(unprocessed_observations))
            best_iloc_pos = np.argmin(priorities)
            best_candidate_index = unprocessed_observations.index[int(best_iloc_pos)]

            # b. Mark this winner as "prioritized" to keep it.
            observations.at[best_candidate_index, "overlapping"] = "prioritized"

            # c. Find and mark all observations that conflict with this winner.
            best_interval = observations.loc[
                [best_candidate_index], ["start", "stop"]
            ].to_numpy()
            unprocessed_intervals = unprocessed_observations[
                ["start", "stop"]
            ].to_numpy()

            conflicts_mask = (
                interval_overlap(unprocessed_intervals, best_interval) > max_overlap
            ).ravel() | (
                interval_contained(unprocessed_intervals, best_interval)
            ).ravel()

            conflicting_indices = unprocessed_observations.index[conflicts_mask]

            # d. Mark all conflicting observations (excluding the winner itself) as "yes" to discard them.
            observations.loc[
                observations.index.isin(conflicting_indices)  # pyright: ignore[reportUnknownMemberType]
                & (observations.index != best_candidate_index),
                "overlapping",
            ] = "yes"

            # e. Refresh the view of the component for the next iteration of the while loop.
            component_df_iloc = observations.iloc[component_indices]

    observations = observations[observations["overlapping"] != "yes"]
    observations = observations.drop(columns=["overlapping"])

    return observations

import numpy as np
import pandas as pd

from .utils import check_observations, ensure_single_index, with_duration


def _bout_aggregator(bout_data: pd.DataFrame) -> pd.Series:
    duration = np.asarray(bout_data["duration"])
    aggregated_values = {}
    for column in bout_data.columns:
        values = np.asarray(bout_data[column])
        if column == "start":
            aggregated_values["start"] = np.min(values)
            continue
        if column == "stop":
            aggregated_values["stop"] = np.max(values)
            continue
        unique_values, counts = np.unique(values, return_counts=True)
        if len(unique_values) == 1:
            aggregated_values[column] = unique_values[0]
            continue
        try:
            aggregated_values[column] = np.average(values, weights=duration)
            continue
        except TypeError:
            pass
        aggregated_values[column] = ", ".join(
            [
                str(value)
                for _, value in sorted(zip(counts, unique_values), reverse=True)
            ]
        )
    return pd.Series(aggregated_values)


@with_duration
def aggregate_bouts(
    observations: pd.DataFrame, *, max_bout_gap: float, index_columns: tuple[str, ...]
) -> pd.DataFrame:
    """
    Aggregate observations (behavioral intervals) into bouts.

    Parameters:
        observations: The observations to aggregate.
        max_bout_gap: The maximum gap between observations to consider them part of the same bout.
        index_columns: The columns to use as the index, unique combinations should point to independent observations (e.g., of one individual).

    Returns:
        The aggregated bouts.
    """
    observations = ensure_single_index(observations, index_columns=index_columns)
    observations = check_observations(
        observations, required_columns=["category", "start", "stop"]
    )
    observations = observations[observations["category"] != "none"].reset_index(
        drop=True
    )  # type: ignore
    gaps = np.asarray(observations["start"][1:]) - np.asarray(observations["stop"][:-1])
    is_bout = np.asarray([False] + list(gaps <= max_bout_gap))
    bout_idx = np.zeros(len(is_bout), dtype=int) - 1
    bout_idx[~is_bout] = np.arange(np.sum(~is_bout))
    bout_idx_filled = []
    for idx in bout_idx:
        bout_idx_filled.append(idx if idx != -1 else bout_idx_filled[-1])
    observations["bout"] = bout_idx_filled
    bouts = observations.groupby("bout").apply(_bout_aggregator)
    bouts = bouts.sort_values(by="start", ignore_index=True, inplace=False).drop(
        columns=["bout"], inplace=False
    )
    return bouts

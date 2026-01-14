import numpy as np
import polars as pl


def to_observations(
    labels: np.ndarray,
    categories: set[str],
    timestamps: np.ndarray | None = None,
) -> pl.DataFrame:
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
    if labels.size == 0:
        return pl.DataFrame(
            schema={"start": pl.Int64, "stop": pl.Int64, "category": pl.String},
        )
    change_idx = np.argwhere((np.diff(labels) != 0)).ravel()
    stop = np.r_[change_idx, len(labels) - 1]
    start = np.r_[0, change_idx + 1]
    category_names = np.asarray(sorted(categories))[labels[start]]
    if timestamps is not None:
        if not timestamps.shape == labels.shape:
            raise ValueError("timestamps should have the same shape as y.")
        start = timestamps[start]
        stop = timestamps[stop]
    return pl.DataFrame(
        {"start": start, "stop": stop, "category": category_names},
        schema_overrides={"start": pl.Int64, "stop": pl.Int64},
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

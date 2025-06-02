import numpy as np

IndividualIdentifier = str | int
DyadIdentifier = tuple[IndividualIdentifier, IndividualIdentifier]
Identifier = IndividualIdentifier | DyadIdentifier

GroupIdentifier = IndividualIdentifier
SubjectIdentifier = tuple[GroupIdentifier, IndividualIdentifier]


def get_actor(identifier: Identifier) -> IndividualIdentifier:
    """Return the actor identifier from a given identifier (the first if tuple).

    Parameters:
        identifier: The identifier to extract the actor from.

    Returns:
        The actor identifier.
    """
    if isinstance(identifier, tuple):
        return identifier[0]
    return identifier


def interval_overlap(
    intervals_1: np.ndarray,
    intervals_2: np.ndarray,
    clip_negative: bool = True,
    element_wise: bool = False,
    mask_diagonal: bool = True,
) -> np.ndarray:
    """
    Calculate the overlap between two sets of intervals.

    Parameters:
        intervals_1: The first set of intervals.
        intervals_2: The second set of intervals.
        clip_negative: Whether to clip negative overlaps to zero.
        element_wise: Whether to calculate element-wise overlap.
        mask_diagonal: Whether to mask the diagonal results with zero. Only applies when :code:`element_wise=True`.

    Returns:
        The overlap between the two sets of intervals.
    """
    if not element_wise:
        overlap = np.minimum(
            intervals_1[:, 1, np.newaxis] + 1, intervals_2[:, 1][np.newaxis] + 1
        ) - np.maximum(intervals_1[:, 0, np.newaxis], intervals_2[:, 0][np.newaxis])
        if mask_diagonal:
            overlap[np.diag_indices(min(overlap.shape))] = 0
    else:
        overlap = np.minimum(intervals_1[:, 1] + 1, intervals_2[:, 1] + 1) - np.maximum(
            intervals_1[:, 0], intervals_2[:, 0]
        )
    if clip_negative:
        overlap[overlap < 0] = 0
    return overlap


def interval_contained(
    intervals_1: np.ndarray,
    intervals_2: np.ndarray,
    element_wise: bool = False,
):
    """
    Check if intervals in :code:`intervals_1` are contained within intervals in :code:`intervals_2`.

    Parameters:
        intervals_1: The first set of intervals.
        intervals_2: The second set of intervals.
        element_wise: Whether to calculate containment element-wise.

    Returns:
        Whether intervals in :code:`intervals_1` are contained in :code:`intervals_2`.
    """
    overlap = interval_overlap(intervals_1, intervals_2, element_wise=element_wise)
    return overlap == np.diff(intervals_1, axis=1) + 1

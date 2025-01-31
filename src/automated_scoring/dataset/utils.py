import numpy as np
from numpy.typing import NDArray

IndividualIdentifier = str | int
DyadIdentifier = tuple[IndividualIdentifier, IndividualIdentifier]
Identifier = IndividualIdentifier | DyadIdentifier

GroupIdentifier = IndividualIdentifier
SubjectIdentifier = tuple[GroupIdentifier, IndividualIdentifier]


def get_actor(identifier: Identifier) -> IndividualIdentifier:
    if isinstance(identifier, tuple):
        return identifier[0]
    return identifier


def interval_overlap(
    intervals_1: NDArray,
    intervals_2: NDArray,
    clip_negative: bool = True,
    element_wise: bool = False,
    mask_diagonal: bool = True,
) -> NDArray:
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
    intervals_1: NDArray,
    intervals_2: NDArray,
    element_wise: bool = False,
):
    overlap = interval_overlap(intervals_1, intervals_2, element_wise=element_wise)
    return overlap == np.diff(intervals_1, axis=1) + 1

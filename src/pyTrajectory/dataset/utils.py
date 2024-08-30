import numpy as np
from numpy.typing import NDArray


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

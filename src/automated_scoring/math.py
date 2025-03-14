import numpy as np
from numba import config, njit

# set the threading layer before any parallel target compilation
# this requires tbb!
config.THREADING_LAYER = "safe"  # type: ignore


@njit
def subtract(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Subtract array_2 from array_1."""
    return array_2 - array_1


@njit
def unit_vector(vectors: np.ndarray) -> np.ndarray:
    """Vectors to unit vectors."""
    # with np.errstate(divide="ignore", invalid="ignore"):
    return vectors / np.expand_dims(magnitude(vectors), -1)


@njit
def magnitude(vectors: np.ndarray) -> np.ndarray:
    """Vector magnitudes."""
    return np.sqrt(np.sum(vectors**2, axis=-1))


@njit
def euclidean_distance(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Euclidean distance between two vector arrays."""
    return magnitude(array_2 - array_1)


@njit
def dot_product(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Dot product between two vector arrays."""
    return np.sum(vectors_1 * vectors_2, axis=-1)


@njit
def perp(vectors: np.ndarray) -> np.ndarray:
    """Perpendicular vectors (rotated counterclockwise)."""
    vectors_perp = np.zeros(vectors.shape)
    vectors_perp[..., 0] = -vectors[..., 1]
    vectors_perp[..., 1] = vectors[..., 0]
    return vectors_perp


@njit
def perp_dot_product(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Perpendicular dot product between two vector arrays."""
    return np.sum(vectors_1 * perp(vectors_2), axis=-1)


@njit
def scalar_projection(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Scalar projection of vectors_1 onto vectors_2."""
    # with np.errstate(divide="ignore", invalid="ignore"):
    return dot_product(vectors_1, vectors_2) / magnitude(vectors_2)


@njit
def projection(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Projection vectors of vectors_1 onto vectors_2."""
    return np.expand_dims(scalar_projection(vectors_1, vectors_2), -1) * unit_vector(
        vectors_2
    )


@njit
def scalar_rejection(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Scalar rejection of vectors_1 from vectors_2."""
    # with np.errstate(divide="ignore", invalid="ignore"):
    return perp_dot_product(vectors_1, vectors_2) / magnitude(vectors_2)


@njit
def rejection(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Rejection vectors of vectors_1 from vectors_2."""
    return np.expand_dims(scalar_rejection(vectors_1, vectors_2), -1) * unit_vector(
        perp(vectors_2)
    )


@njit
def rotate(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate vectors around angles in radians."""
    vectors_rotated = np.zeros(vectors.shape)
    vectors_rotated[..., 0] = vectors[..., 0] * np.cos(angles) - vectors[
        ..., 1
    ] * np.sin(angles)
    vectors_rotated[..., 1] = vectors[..., 0] * np.sin(angles) + vectors[
        ..., 1
    ] * np.cos(angles)
    return vectors_rotated


@njit
def as_angle(vectors: np.ndarray) -> np.ndarray:
    """Represent vectors as angles in radians on the unit circle.

    Note that input vectors do not need to be unit vectors.
    Zero-magnitude vectors result in np.nan values.
    """
    vectors = unit_vector(vectors)
    x = vectors[..., 0]
    y = vectors[..., 1]
    return np.arctan2(y, x)


@njit
def wrap_angle(radians: np.ndarray) -> np.ndarray:
    """Wrap angles in radians into the [-pi, pi] range."""
    return (radians + np.pi) % (2 * np.pi) - np.pi


@njit
def signed_angle(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Signed angles between vectors.

    See as_angle for behavior with zero-magnitude input vectors."""
    return wrap_angle(as_angle(vectors_2) - as_angle(vectors_1))


@njit
def unsigned_angle(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Unsigned angles between vectors.

    Returns np.nan values when either input is of zero-magnitude.
    """
    # slightly faster than np.abs(signed_angle(vectors_2, vectors_1))
    # with np.errstate(divide="ignore", invalid="ignore"):
    return np.acos(
        dot_product(vectors_1, vectors_2)
        / (magnitude(vectors_1) * magnitude(vectors_2))
    )


@njit
def as_unit_vector(radians: np.ndarray) -> np.ndarray:
    """Unit vectors representing angles in radians on the unit circle."""
    unit_vectors = np.zeros((*radians.shape, 2))
    unit_vectors[..., 0] = np.cos(radians)
    unit_vectors[..., 1] = np.sin(radians)
    return unit_vectors


@njit
def shift(array: np.ndarray, step: int) -> np.ndarray:
    """Similar to np.roll on axis 0 (shift to right with step > 0, shift to left with step < 0).

    Values are filled with the last value (shift to left) or the first value (shift to right), no wrapping."""
    if step == 0:
        return array.copy()
    array_shifted = np.zeros(array.shape, dtype=array.dtype)
    if step > 0:
        array_shifted[step:] = array[:-step]
        array_shifted[:step] = array[0]
        return array_shifted
    array_shifted[:step] = array[-step:]
    array_shifted[step:] = array[-1]
    return array_shifted

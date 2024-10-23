import numpy as np
from numba import jit
from numpy.typing import NDArray


@jit(nopython=True, cache=True)
def subtract(array_1: NDArray, array_2: NDArray) -> NDArray:
    """Subtract array_2 from array_1."""
    return array_2 - array_1


@jit(nopython=True, cache=True)
def unit_vector(vectors: NDArray) -> NDArray:
    """Vectors to unit vectors."""
    return vectors / magnitude(vectors)[..., np.newaxis]


@jit(nopython=True, cache=True)
def magnitude(vectors: NDArray) -> NDArray:
    """Vector magnitudes."""
    return np.sqrt(np.sum(vectors**2, axis=-1))


@jit(nopython=True, cache=True)
def euclidean_distance(array_1: NDArray, array_2: NDArray) -> NDArray:
    """Euclidean distance between two vector arrays."""
    return magnitude(array_2 - array_1)


@jit(nopython=True, cache=True)
def dot_product(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Dot product between two vector arrays."""
    return np.sum(vectors_1 * vectors_2, axis=-1)


@jit(nopython=True, cache=True)
def perp(vectors: NDArray) -> NDArray:
    """Perpendicular vectors (rotated counterclockwise)."""
    vectors_perp = np.zeros(vectors.shape)
    vectors_perp[..., 0] = -vectors[..., 1]
    vectors_perp[..., 1] = vectors[..., 0]
    return vectors_perp


@jit(nopython=True, cache=True)
def perp_dot_product(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Perpendicular dot product between two vector arrays."""
    return np.sum(vectors_1 * perp(vectors_2), axis=-1)


@jit(nopython=True, cache=True)
def scalar_projection(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Scalar projection of vectors_1 onto vectors_2."""
    return dot_product(vectors_1, vectors_2) / magnitude(vectors_2)


@jit(nopython=True, cache=True)
def projection(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Projection vectors of vectors_1 onto vectors_2."""
    return scalar_projection(vectors_1, vectors_2)[..., np.newaxis] * unit_vector(
        vectors_2
    )


@jit(nopython=True, cache=True)
def scalar_rejection(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Scalar rejection of vectors_1 from vectors_2."""
    return perp_dot_product(vectors_1, vectors_2) / magnitude(vectors_2)


@jit(nopython=True, cache=True)
def rejection(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Rejection vectors of vectors_1 from vectors_2."""
    return scalar_rejection(vectors_1, vectors_2)[..., np.newaxis] * unit_vector(
        perp(vectors_2)
    )


@jit(nopython=True, cache=True)
def rotate(vectors: NDArray, angles: NDArray) -> NDArray:
    """Rotate vectors around angles in radians."""
    vectors_rotated = np.zeros(vectors.shape)
    vectors_rotated[..., 0] = vectors[..., 0] * np.cos(angles) - vectors[
        ..., 1
    ] * np.sin(angles)
    vectors_rotated[..., 1] = vectors[..., 0] * np.sin(angles) + vectors[
        ..., 1
    ] * np.cos(angles)
    return vectors_rotated


@jit(nopython=True, cache=True)
def as_angle(vectors):
    """Represent vectors as angles in radians on the unit circle.

    Note that input vectors do not need to be unit vectors.
    Zero-magnitude vectors result in np.nan values.
    """
    vectors = unit_vector(vectors)
    x = vectors[..., 0]
    y = vectors[..., 1]
    return np.arctan2(y, x)


@jit(nopython=True, cache=True)
def wrap_angle(radians: NDArray) -> NDArray:
    """Wrap angles in radians into the [-pi, pi] range."""
    return (radians + np.pi) % (2 * np.pi) - np.pi


@jit(nopython=True, cache=True)
def signed_angle(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Signed angles between vectors.

    See as_angle for behavior with zero-magnitude input vectors."""
    return wrap_angle(as_angle(vectors_2) - as_angle(vectors_1))


@jit(nopython=True, cache=True)
def unsigned_angle(vectors_1: NDArray, vectors_2: NDArray) -> NDArray:
    """Unsigned angles between vectors.

    Returns np.nan values when either input is of zero-magnitude.
    """
    # slightly faster than np.abs(signed_angle(vectors_2, vectors_1))
    return np.acos(
        dot_product(vectors_1, vectors_2)
        / (magnitude(vectors_1) * magnitude(vectors_2))
    )


@jit(nopython=True, cache=True)
def as_unit_vector(radians: NDArray) -> NDArray:
    """Unit vectors representing angles in radians on the unit circle."""
    unit_vectors = np.zeros((*radians.shape, 2))
    unit_vectors[..., 0] = np.cos(radians)
    unit_vectors[..., 1] = np.sin(radians)
    return unit_vectors


@jit(nopython=True, cache=True)
def shift(array: NDArray, step: int) -> NDArray:
    """Similar to np.roll on axis 0 (shift to right with step > 0, shift to left with step < 0).

    Values are filled with the last value (shift to left) or the first value (shift to right), no wrapping."""
    if step == 0:
        return array.copy()
    array_shifted = np.zeros_like(array)
    if step > 0:
        array_shifted[step:] = array[:-step]
        array_shifted[:step] = array[0]
        return array_shifted
    array_shifted[:step] = array[-step:]
    array_shifted[step:] = array[-1]
    return array_shifted

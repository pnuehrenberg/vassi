import numpy as np
from numba import (
    njit,  # pyright: ignore[reportUnknownVariableType]
    prange,
)


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def magnitude(vectors: np.ndarray) -> np.ndarray:
    """
    Vector magnitudes. (Numba-optimized, handles only N-D)

    - If input is N-D (shape ...xMxN, D), returns N-D array (shape ...xMxN).
    """

    assert vectors.ndim > 1

    # Store the original shape and calculate the reshaped dimensions
    original_shape = vectors.shape
    d = original_shape[-1]
    output_shape = original_shape[:-1]

    # Calculate N manually
    n = 1
    for dim in output_shape:
        n *= dim
    n = int(n)  # Ensure N is an integer

    # Reshape into a 2D view (N, D)
    vectors_2d = vectors.reshape(n, d)

    # Run the fast 2D parallel loop
    out_1d = np.empty(n, dtype=vectors.dtype)
    for i in prange(n):
        sum_sq = 0.0
        for j in range(d):
            sum_sq += vectors_2d[i, j] ** 2
        out_1d[i] = np.sqrt(sum_sq)

    # Reshape the 1D result back to the correct output shape
    return out_1d.reshape(output_shape)


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def wrap_angle(radians: np.ndarray) -> np.ndarray:
    """
    Wrap angles in radians into the [-pi, pi] range. (Numba-optimized loop)
    """
    out = np.empty_like(radians)
    radians_flat = radians.flat
    out_flat = out.flat

    n = radians.size

    for i in prange(n):
        out_flat[i] = (radians_flat[i] + np.pi) % (2 * np.pi) - np.pi

    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def unit_vector(vectors: np.ndarray) -> np.ndarray:
    """
    Vectors to unit vectors. (Numba-optimized, N-dimensional)

    Replaces np.errstate with an explicit check,
    returning np.nan for zero-magnitude vectors.
    """
    # Ensure input is at least 2D
    assert vectors.ndim > 1

    original_shape = vectors.shape
    d = original_shape[-1]  # dimension of vectors
    output_shape = original_shape[:-1]

    n = 1  # number of vectors
    for dim in output_shape:
        n *= dim
    n = int(n)

    # Reshape view
    vectors_2d = vectors.reshape(n, d)
    # Create output array with the *original* shape, then reshape for assignment
    out_array = np.empty_like(vectors)
    out_2d = out_array.reshape(n, d)

    # Main parallel loop over each vector
    for i in prange(n):
        # 1. Calculate magnitude for this vector
        sum_sq = 0.0
        for j in range(d):
            sum_sq += vectors_2d[i, j] ** 2
        mag = np.sqrt(sum_sq)

        # 2. Divide, with zero-check
        if mag == 0.0:
            for j in range(d):
                out_2d[i, j] = np.nan
        else:
            for j in range(d):
                out_2d[i, j] = vectors_2d[i, j] / mag

    return out_array  # Return the array in its original shape


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def as_angle(vectors: np.ndarray) -> np.ndarray:
    """
    Represent vectors as angles in radians. (Numba-optimized, N-D)

    Fuses magnitude check (from unit_vector) with np.arctan2.
    Returns np.nan for zero-magnitude vectors.
    Assumes inputs are 2D vectors (last dimension is 2).
    """
    assert vectors.shape[-1] == 2
    assert vectors.ndim > 1

    original_shape = vectors.shape
    d = 2  # We know d is 2
    output_shape = original_shape[:-1]

    n = 1  # number of vectors
    for dim in output_shape:
        n *= dim
    n = int(n)

    # Reshape views
    vectors_2d = vectors.reshape(n, d)

    # Create 1D output array
    out_1d = np.empty(n, dtype=vectors.dtype)

    # Main parallel loop over each vector
    for i in prange(n):
        x = vectors_2d[i, 0]
        y = vectors_2d[i, 1]

        # 1. Calculate squared magnitude to check for zero
        sum_sq = x**2 + y**2

        # 2. Check for zero vector
        if sum_sq == 0.0:
            out_1d[i] = np.nan
        else:
            out_1d[i] = np.arctan2(y, x)

    return out_1d.reshape(output_shape)


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def perp(vectors: np.ndarray) -> np.ndarray:
    """
    Perpendicular vectors (rotated counterclockwise). (Numba-optimized)
    Assumes inputs are 2D vectors (last dimension is 2).
    """
    assert vectors.shape[-1] == 2
    assert vectors.ndim > 1

    # Numba is very fast at compiling this kind of logic
    vectors_perp = np.empty_like(vectors)

    # Get flat views for simple parallel looping
    n = int(vectors.size / 2)  # Total number of vectors
    vectors_flat = vectors.reshape(n, 2)
    perp_flat = vectors_perp.reshape(n, 2)

    for i in prange(n):
        perp_flat[i, 0] = -vectors_flat[i, 1]
        perp_flat[i, 1] = vectors_flat[i, 0]

    return vectors_perp  # Returned in original shape


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def rotate(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotate vectors around angles in radians. (Numba-optimized)
    Assumes inputs are 2D vectors (last dimension is 2).
    """
    assert vectors.shape[-1] == 2
    assert vectors.shape[:-1] == angles.shape
    assert vectors.ndim > 1

    original_shape = vectors.shape
    d = 2  # We know d is 2
    output_shape = original_shape[:-1]

    n = 1  # number of vectors
    for dim in output_shape:
        n *= dim
    n = int(n)

    # Reshape views
    vectors_2d = vectors.reshape(n, d)
    angles_1d = angles.reshape(n)  # Angles is 1D in the reshaped view

    # Create output array
    out_array = np.empty_like(vectors)
    out_2d = out_array.reshape(n, d)

    # Main parallel loop
    for i in prange(n):
        v_x = vectors_2d[i, 0]
        v_y = vectors_2d[i, 1]
        angle = angles_1d[i]

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        out_2d[i, 0] = v_x * cos_a - v_y * sin_a
        out_2d[i, 1] = v_x * sin_a + v_y * cos_a

    return out_array


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def as_unit_vector(radians: np.ndarray) -> np.ndarray:
    """
    Unit vectors representing angles in radians on the unit circle. (Numba-optimized)
    """
    # Create the output array with the new shape
    output_shape = (*radians.shape, 2)
    unit_vectors = np.empty(output_shape, dtype=radians.dtype)

    # Get flat views for simple parallel looping
    n = radians.size
    radians_flat = radians.flat
    # Reshape the output to (n, 2) for assignment
    unit_vectors_flat = unit_vectors.reshape(n, 2)

    for i in prange(n):
        rad = radians_flat[i]
        unit_vectors_flat[i, 0] = np.cos(rad)
        unit_vectors_flat[i, 1] = np.sin(rad)

    return unit_vectors


def shift(array: np.ndarray, step: int) -> np.ndarray:
    """
    Similar to :func:`numpy.roll` on axis 0 (shift to right with step > 0, shift to left with step < 0).

    Values are filled with the last value (shift to left) or the first value (shift to right), no wrapping.

    Parameters:
        array: The array to shift.
        step: The number of positions to shift the array.
    """
    # maybe move this where pad_values is located
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

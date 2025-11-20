import numpy as np
from numba import (
    njit,  # pyright: ignore[reportUnknownVariableType]
    prange,
)


def element_wise_subtract(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """
    Batched element-wise vector subtraction (v2 - v1).
    Takes (T, N, D) vs (T, N, D) and returns (T, N, D).

    This is a high-performance wrapper for the native NumPy ufunc.
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3, "Inputs must be 3D (T, N, D)"
    assert vectors_1.shape[0] == vectors_2.shape[0], "T dimension must match"
    assert vectors_1.shape[1] == vectors_2.shape[1], (
        "N dimension must match for element-wise"
    )
    assert vectors_1.shape[2] == vectors_2.shape[2], "D dimension must match"

    return vectors_2 - vectors_1


def broadcasted_subtract(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """
    Batched-broadcasting vector subtraction (v2 - v1).
    Takes (T, N, D) vs (T, M, D) and returns (T, N, M, D).

    This is a high-performance wrapper for the native NumPy ufunc.
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3, (
        "Inputs must be 3D (T, N, D) / (T, M, D)"
    )
    assert vectors_1.shape[0] == vectors_2.shape[0], "T dimension must match"
    assert vectors_1.shape[2] == vectors_2.shape[2], "D dimension must match"

    # Prepare arrays for broadcasting
    v1_exp = np.expand_dims(vectors_1, 2)  # (T, N, 1, D)
    v2_exp = np.expand_dims(vectors_2, 1)  # (T, 1, M, D)

    # NumPy's ufunc is the fastest way to do this
    return v2_exp - v1_exp  # (T, N, M, D)


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def element_wise_euclidean_distance(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched element-wise euclidean distance.
    Takes (T, N, D) vs (T, N, D) and returns (T, N).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[1] == vectors_2.shape[1]  # N
    assert vectors_1.shape[2] == vectors_2.shape[2]  # D

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    d_dim = vectors_1.shape[2]

    out = np.empty((t_dim, n_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            sum_sq_diff = 0.0
            for d_idx in range(d_dim):
                diff = vectors_2[t, n_idx, d_idx] - vectors_1[t, n_idx, d_idx]
                sum_sq_diff += diff**2
            out[t, n_idx] = np.sqrt(sum_sq_diff)
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def broadcasted_euclidean_distance(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched-broadcasting euclidean distance.
    Takes (T, N, D) vs (T, M, D) and returns (T, N, M).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[2] == vectors_2.shape[2]  # D

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    m_dim = vectors_2.shape[1]
    d_dim = vectors_1.shape[2]

    out = np.empty((t_dim, n_dim, m_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            for m_idx in range(m_dim):
                sum_sq_diff = 0.0
                for d_idx in range(d_dim):
                    diff = vectors_2[t, m_idx, d_idx] - vectors_1[t, n_idx, d_idx]
                    sum_sq_diff += diff**2
                out[t, n_idx, m_idx] = np.sqrt(sum_sq_diff)
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def element_wise_signed_angle(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched element-wise signed angle (v2 relative to v1).
    Takes (T, N, 2) vs (T, N, 2) and returns (T, N).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[1] == vectors_2.shape[1]  # N
    assert vectors_1.shape[2] == 2 and vectors_2.shape[2] == 2  # D must be 2

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]

    out = np.empty((t_dim, n_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            x1 = vectors_1[t, n_idx, 0]
            y1 = vectors_1[t, n_idx, 1]
            x2 = vectors_2[t, n_idx, 0]
            y2 = vectors_2[t, n_idx, 1]

            sum_sq_1 = x1**2 + y1**2
            sum_sq_2 = x2**2 + y2**2

            if sum_sq_1 == 0.0 or sum_sq_2 == 0.0:
                out[t, n_idx] = np.nan
            else:
                angle_1 = np.arctan2(y1, x1)
                angle_2 = np.arctan2(y2, x2)
                diff = angle_2 - angle_1
                out[t, n_idx] = (diff + np.pi) % (2 * np.pi) - np.pi
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def broadcasted_signed_angle(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched-broadcasting signed angle (v2 relative to v1).
    Takes (T, N, 2) vs (T, M, 2) and returns (T, N, M).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[2] == 2 and vectors_2.shape[2] == 2  # D must be 2

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    m_dim = vectors_2.shape[1]

    out = np.empty((t_dim, n_dim, m_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            for m_idx in range(m_dim):
                x1 = vectors_1[t, n_idx, 0]
                y1 = vectors_1[t, n_idx, 1]
                x2 = vectors_2[t, m_idx, 0]
                y2 = vectors_2[t, m_idx, 1]

                sum_sq_1 = x1**2 + y1**2
                sum_sq_2 = x2**2 + y2**2

                if sum_sq_1 == 0.0 or sum_sq_2 == 0.0:
                    out[t, n_idx, m_idx] = np.nan
                else:
                    angle_1 = np.arctan2(y1, x1)
                    angle_2 = np.arctan2(y2, x2)
                    diff = angle_2 - angle_1
                    out[t, n_idx, m_idx] = (diff + np.pi) % (2 * np.pi) - np.pi
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def element_wise_alignment(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """
    Batched element-wise alignment (1 - unsigned_angle / pi).
    Takes (T, N, D) vs (T, N, D) and returns (T, N).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[1] == vectors_2.shape[1]  # N
    assert vectors_1.shape[2] == vectors_2.shape[2]  # D

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    d_dim = vectors_1.shape[2]

    out = np.empty((t_dim, n_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            sum_prod, sum_sq_1, sum_sq_2 = 0.0, 0.0, 0.0
            for d_idx in range(d_dim):
                v1_j = vectors_1[t, n_idx, d_idx]
                v2_j = vectors_2[t, n_idx, d_idx]
                sum_prod += v1_j * v2_j
                sum_sq_1 += v1_j * v1_j
                sum_sq_2 += v2_j * v2_j

            mag_prod = np.sqrt(sum_sq_1 * sum_sq_2)

            if mag_prod == 0.0:
                out[t, n_idx] = np.nan
            else:
                cosine_angle = sum_prod / mag_prod
                if cosine_angle > 1.0:
                    cosine_angle = 1.0
                elif cosine_angle < -1.0:
                    cosine_angle = -1.0
                angle = np.arccos(cosine_angle)
                out[t, n_idx] = 1.0 - (angle / np.pi)
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def broadcasted_alignment(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """
    Batched-broadcasting alignment (1 - unsigned_angle / pi).
    Takes (T, N, D) vs (T, M, D) and returns (T, N, M).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[2] == vectors_2.shape[2]  # D

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    m_dim = vectors_2.shape[1]
    d_dim = vectors_1.shape[2]

    out = np.empty((t_dim, n_dim, m_dim), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            for m_idx in range(m_dim):
                sum_prod, sum_sq_1, sum_sq_2 = 0.0, 0.0, 0.0
                for d_idx in range(d_dim):
                    v1_j = vectors_1[t, n_idx, d_idx]
                    v2_j = vectors_2[t, m_idx, d_idx]
                    sum_prod += v1_j * v2_j
                    sum_sq_1 += v1_j * v1_j
                    sum_sq_2 += v2_j * v2_j

                mag_prod = np.sqrt(sum_sq_1 * sum_sq_2)

                if mag_prod == 0.0:
                    out[t, n_idx, m_idx] = np.nan
                else:
                    cosine_angle = sum_prod / mag_prod
                    if cosine_angle > 1.0:
                        cosine_angle = 1.0
                    elif cosine_angle < -1.0:
                        cosine_angle = -1.0
                    angle = np.arccos(cosine_angle)
                    out[t, n_idx, m_idx] = 1.0 - (angle / np.pi)
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def element_wise_projected_components(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched element-wise scalar projection and rejection [proj, rej].
    Takes (T, N, 2) vs (T, N, 2) and returns (T, N, 2).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[1] == vectors_2.shape[1]  # N
    assert vectors_1.shape[2] == 2 and vectors_2.shape[2] == 2  # D must be 2

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]

    out = np.empty((t_dim, n_dim, 2), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            v1_x = vectors_1[t, n_idx, 0]
            v1_y = vectors_1[t, n_idx, 1]
            v2_x = vectors_2[t, n_idx, 0]
            v2_y = vectors_2[t, n_idx, 1]

            sum_sq = v2_x**2 + v2_y**2
            mag = np.sqrt(sum_sq)

            if mag == 0.0:
                out[t, n_idx, 0] = np.nan
                out[t, n_idx, 1] = np.nan
            else:
                proj_scalar = (v1_x * v2_x + v1_y * v2_y) / mag
                rej_scalar = (v1_x * -v2_y + v1_y * v2_x) / mag
                out[t, n_idx, 0] = proj_scalar
                out[t, n_idx, 1] = rej_scalar
    return out


@njit(parallel=True, fastmath=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def broadcasted_projected_components(
    vectors_1: np.ndarray, vectors_2: np.ndarray
) -> np.ndarray:
    """
    Batched-broadcasting scalar projection and rejection [proj, rej].
    Takes (T, N, 2) vs (T, M, 2) and returns (T, N, M, 2).
    """
    assert vectors_1.ndim == 3 and vectors_2.ndim == 3
    assert vectors_1.shape[0] == vectors_2.shape[0]  # T
    assert vectors_1.shape[2] == 2 and vectors_2.shape[2] == 2  # D must be 2

    t_dim = vectors_1.shape[0]
    n_dim = vectors_1.shape[1]
    m_dim = vectors_2.shape[1]

    out = np.empty((t_dim, n_dim, m_dim, 2), dtype=vectors_1.dtype)

    for t in prange(t_dim):
        for n_idx in range(n_dim):
            for m_idx in range(m_dim):
                v1_x = vectors_1[t, n_idx, 0]
                v1_y = vectors_1[t, n_idx, 1]
                v2_x = vectors_2[t, m_idx, 0]
                v2_y = vectors_2[t, m_idx, 1]

                sum_sq = v2_x**2 + v2_y**2
                mag = np.sqrt(sum_sq)

                if mag == 0.0:
                    out[t, n_idx, m_idx, 0] = np.nan
                    out[t, n_idx, m_idx, 1] = np.nan
                else:
                    proj_scalar = (v1_x * v2_x + v1_y * v2_y) / mag
                    rej_scalar = (v1_x * -v2_y + v1_y * v2_x) / mag
                    out[t, n_idx, m_idx, 0] = proj_scalar
                    out[t, n_idx, m_idx, 1] = rej_scalar
    return out


def subtract(
    vectors_1: np.ndarray,
    vectors_2: np.ndarray,
    *,
    element_wise: bool,
    flat: bool,
) -> np.ndarray:
    """
    Wrapper for batched vector subtraction (v2 - v1).

    Dispatches to element-wise (T,N,D) or broadcasted (T,N,M,D) version.

    Parameters:
        vectors_1: The first array of vectors.
        vectors_2: The second array of vectors.
        element_wise: If True, perform element-wise operation (N==M).
                      If False (default), perform broadcasted operation.
        flat: If True, reshape the output to (T, -1).
    """
    if element_wise:
        result = element_wise_subtract(vectors_1, vectors_2)
    else:
        result = broadcasted_subtract(vectors_1, vectors_2)

    if flat:
        return result.reshape(result.shape[0], -1)
    return result


def euclidean_distance(
    vectors_1: np.ndarray,
    vectors_2: np.ndarray,
    *,
    element_wise: bool,
    flat: bool,
) -> np.ndarray:
    """
    Wrapper for batched euclidean distance.

    Dispatches to element-wise (T,N) or broadcasted (T,N,M) version.

    Parameters:
        vectors_1: The first array of vectors.
        vectors_2: The second array of vectors.
        element_wise: If True, perform element-wise operation (N==M).
                      If False (default), perform broadcasted operation.
        flat: If True, reshape the output to (T, -1).
    """
    if element_wise:
        result = element_wise_euclidean_distance(vectors_1, vectors_2)
    else:
        result = broadcasted_euclidean_distance(vectors_1, vectors_2)

    if flat:
        return result.reshape(result.shape[0], -1)
    return result


def signed_angle(
    vectors_1: np.ndarray,
    vectors_2: np.ndarray,
    *,
    element_wise: bool,
    flat: bool,
) -> np.ndarray:
    """
    Wrapper for batched signed angle (v2 relative to v1).

    Dispatches to element-wise (T,N) or broadcasted (T,N,M) version.

    Parameters:
        vectors_1: The first array of vectors.
        vectors_2: The second array of vectors.
        element_wise: If True, perform element-wise operation (N==M).
                      If False (default), perform broadcasted operation.
        flat: If True, reshape the output to (T, -1).
    """
    if element_wise:
        result = element_wise_signed_angle(vectors_1, vectors_2)
    else:
        result = broadcasted_signed_angle(vectors_1, vectors_2)

    if flat:
        return result.reshape(result.shape[0], -1)
    return result


def alignment(
    vectors_1: np.ndarray,
    vectors_2: np.ndarray,
    *,
    element_wise: bool,
    flat: bool,
) -> np.ndarray:
    """
    Wrapper for batched alignment (1 - unsigned_angle / pi).

    Dispatches to element-wise (T,N) or broadcasted (T,N,M) version.

    Parameters:
        vectors_1: The first array of vectors.
        vectors_2: The second array of vectors.
        element_wise: If True, perform element-wise operation (N==M).
                      If False (default), perform broadcasted operation.
        flat: If True, reshape the output to (T, -1).
    """
    if element_wise:
        result = element_wise_alignment(vectors_1, vectors_2)
    else:
        result = broadcasted_alignment(vectors_1, vectors_2)

    if flat:
        return result.reshape(result.shape[0], -1)
    return result


def projected_components(
    vectors_1: np.ndarray,
    vectors_2: np.ndarray,
    *,
    element_wise: bool,
    flat: bool,
) -> np.ndarray:
    """
    Wrapper for batched scalar projection and rejection [proj, rej].

    Dispatches to element-wise (T,N,2) or broadcasted (T,N,M,2) version.

    Parameters:
        vectors_1: The first array of vectors.
        vectors_2: The second array of vectors.
        element_wise: If True, perform element-wise operation (N==M).
                      If False (default), perform broadcasted operation.
        flat: If True, reshape the output to (T, -1).
    """
    if element_wise:
        result = element_wise_projected_components(vectors_1, vectors_2)
    else:
        result = broadcasted_projected_components(vectors_1, vectors_2)

    if flat:
        return result.reshape(result.shape[0], -1)
    return result

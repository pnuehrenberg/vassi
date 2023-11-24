import numpy as np

import pyTrajectory.series_decorators
# from .series_decorators import norm_input, absolute_output


def wrap_angles(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def calculate_element_wise_magnitude(vectors):
    return np.sqrt(np.square(vectors).sum(axis=-1))


def calculate_unit_vectors(vectors):
    return vectors / calculate_element_wise_magnitude(vectors)[..., np.newaxis]


def calculate_pairwise_dot_product(vectors_1, vectors_2):
    # input must be broadcastable
    return np.sum(vectors_1 * vectors_2, axis=-1)


def calculate_scalar_projection(vectors_1, vectors_2):
    # projection of vectors_1 on vectors_2
    return calculate_pairwise_dot_product(vectors_1, vectors_2) / calculate_element_wise_magnitude(vectors_2)


def element_wise_broadcast(array_1, array_2, func):
    # n x m (array_1) and n x k (array_2) -> n x m x k
    return func(array_1[:, :, np.newaxis], array_2[:, np.newaxis])


def calculate_pairwise_signed_angle_between(vectors_1, vectors_2, pairwise=True):
    # caluculate signed angles between two series of vectors
    # pairwise = row wise, otherwise braodcasting
    angles_1 = np.arctan2(vectors_1[..., 1], vectors_1[..., 0])
    angles_2 = np.arctan2(vectors_2[..., 1], vectors_2[..., 0])
    if pairwise:
        # broadcastable without reshaping
        return wrap_angles(angles_1 - angles_2)
    # n x m (angles_1) and n x k (angles_2) -> n x m x k (or not pairwise)
    return wrap_angles(element_wise_broadcast(angles_1, angles_2, np.subtract))


@pyTrajectory.series_decorators.absolute_output
def calculate_pairwise_unsigned_angle_between(vectors_1, vectors_2, pairwise=True):
    # caluculate unsigned angles between two series of vectors
    return calculate_pairwise_signed_angle_between(vectors_1, vectors_2, pairwise=pairwise)


def angles_to_unit_vectors(angles):
    return np.transpose([np.cos(angles), np.sin(angles)])


def unit_vectors_to_angles(unit_vectors):
    return np.arctan2(*unit_vectors[..., ::-1].T)


def rotate_unit_vectors(unit_vectors, angles):
    unit_vectors_rotated = np.zeros(unit_vectors.shape)
    unit_vectors_rotated[..., 0] = unit_vectors[..., 0] * np.cos(angles) - unit_vectors[..., 1] * np.sin(angles)
    unit_vectors_rotated[..., 1] = unit_vectors[..., 0] * np.sin(angles) + unit_vectors[..., 1] * np.cos(angles)
    return unit_vectors_rotated

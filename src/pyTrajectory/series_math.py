import numpy as np

from .series_decorators import norm_input, absolute_output


def wrap_angles(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def calculate_element_wise_magnitude(vectors):
    return np.sqrt(np.square(vectors).sum(axis=-1))


def calculate_unit_vectors(vectors):
    return vectors / calculate_element_wise_magnitude(vectors)[..., np.newaxis]


@norm_input
def calculate_pairwise_signed_angle_between(vectors_1, vectors_2):
    return wrap_angles(np.arctan2(*vectors_1[:, ::-1].T) - np.arctan2(*vectors_2[:, ::-1].T))


@absolute_output
def calculate_pairwise_unsigned_angle_between(vectors_1, vectors_2):
    return calculate_pairwise_signed_angle_between(vectors_1, vectors_2)


def angles_to_unit_vectors(angles):
    return np.transpose([np.cos(angles), np.sin(angles)])


def unit_vectors_to_angles(unit_vectors):
    return np.arctan2(*unit_vectors[..., ::-1].T)


def rotate_unit_vectors(unit_vectors, angles):
    unit_vectors_rotated = np.zeros(unit_vectors.shape)
    unit_vectors_rotated[..., 0] = unit_vectors[..., 0] * np.cos(angles) - unit_vectors[..., 1] * np.sin(angles)
    unit_vectors_rotated[..., 1] = unit_vectors[..., 0] * np.sin(angles) + unit_vectors[..., 1] * np.cos(angles)
    return unit_vectors_rotated

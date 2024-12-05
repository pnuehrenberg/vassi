import hashlib
import json
import warnings
from contextlib import contextmanager
from typing import Iterable, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

Keypoint = int
KeypointPair = tuple[Keypoint, Keypoint]
Keypoints = Iterable[Keypoint]
KeypointPairs = Iterable[KeypointPair]


def hash_dict(dictionary: dict) -> str:
    return hashlib.sha1(
        json.dumps(dictionary, sort_keys=True).encode("utf-8")
    ).hexdigest()


class To_NDArray(Protocol):
    __name__: str

    def __call__(self, *args, **kwargs) -> NDArray: ...


class NDArray_to_NDArray(Protocol):
    __name__: str

    def __call__(self, array: NDArray, *args, **kwargs) -> NDArray: ...


class PairedNDArrays_to_NDArray(Protocol):
    __name__: str

    def __call__(
        self, array_1: NDArray, array_2: NDArray, /, *args, **kwargs
    ) -> NDArray: ...


def flatten(array: NDArray, ensure_2d: bool = True) -> NDArray:
    if array.ndim == 2:
        return array
    if array.ndim == 3 or ensure_2d:
        return array.reshape(array.shape[0], -1)
    # keep last dim
    return array.reshape(array.shape[0], -1, array.shape[-1])


def perform_operation(
    operation_func: PairedNDArrays_to_NDArray,
    array_1: NDArray,
    array_2: NDArray,
    element_wise: bool,
    flat: bool,
    expand_dims_for_broadcasting: bool = True,
):
    if (
        not element_wise
        and expand_dims_for_broadcasting
        and array_1.ndim == array_2.ndim
    ):
        array_1 = np.expand_dims(array_1, axis=2)
        array_2 = np.expand_dims(array_2, axis=1)
    if not element_wise and array_1.ndim != array_2.ndim:
        raise NotImplementedError(
            f"Broadcasting not implemented for inputs of shapes {array_1.shape} and {array_2.shape}."
        )
    try:
        np.broadcast_shapes(array_1.shape, array_2.shape)
    except ValueError:
        raise ValueError(
            f"Can not broadcast inputs of shapes {array_1.shape} and {array_2.shape}."
        )
    result = operation_func(array_1, array_2)
    if flat:
        return flatten(result)
    return result


def pad_values(array: NDArray, step: int, value: int | float | str) -> NDArray:
    if value == "same":
        value = array[step]
    if step >= 0:
        array[:step] = value
    if step < 0:
        array[-step:] = value
    return array


def ensure_generator(
    random_state: Optional[np.random.Generator | int],
) -> np.random.Generator:
    if not isinstance(random_state, np.random.Generator):
        return np.random.default_rng(seed=random_state)
    return random_state


def to_int_seed(random_state: np.random.Generator) -> int:
    # max int for sklearn random_state as int
    return int(random_state.integers(4294967295))


def closest_odd_divisible(number: float, divisor: int) -> int:
    if divisor % 2 == 0:
        raise ValueError("Closest odd number can not have an even divisor.")
    remainder = number % divisor
    closest_smaller_number = number - remainder
    while closest_smaller_number % 2 == 0:
        closest_smaller_number -= divisor
    closest_greater_number = (number + divisor) - remainder
    while closest_greater_number % 2 == 0:
        closest_greater_number += divisor
    if (number - closest_smaller_number) > (closest_greater_number - number):
        return int(closest_greater_number)
    return int(closest_smaller_number)


@contextmanager
def warning_only():
    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return str(msg) + "\n"

    formatwarning = warnings.formatwarning
    warnings.formatwarning = custom_formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = formatwarning

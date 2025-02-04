import hashlib
import json
import warnings
from contextlib import contextmanager
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

Keypoint = int
KeypointPair = tuple[Keypoint, Keypoint]
Keypoints = Iterable[Keypoint]
KeypointPairs = Iterable[KeypointPair]


def hash_dict(dictionary: dict) -> str:
    """
    Return a hash (digest) of a dictionary. All values must be json serializable.

    Parameters
    ----------
    dictionary : dict
        The dictionary.

    Returns
    -------
    str
        The hash.
    """
    return hashlib.sha1(
        json.dumps(dictionary, sort_keys=True).encode("utf-8")
    ).hexdigest()


class To_NDArray(Protocol):
    """
    Protocol for a function that takes any values and returns an NDArray.
    """

    def __call__(self, *args, **kwargs) -> NDArray: ...


class NDArray_to_NDArray(Protocol):
    """
    Protocol for a function that takes an NDArray and returns an NDArray.
    """

    __name__: str  # named for sliding window aggregation (alternatively, use a dictionary naming scheme as implemented for the fature extractor)

    def __call__(self, array: NDArray, *args, **kwargs) -> NDArray: ...


class PairedNDArrays_to_NDArray(Protocol):
    """
    Protocol for a function that takes two NDArrays and returns an NDArray.
    """

    def __call__(
        self, array_1: NDArray, array_2: NDArray, /, *args, **kwargs
    ) -> NDArray: ...


class SmoothingFunction(Protocol):
    """
    Protocol for a function that takes any positional arguments and a NDArray and returns an NDArray.
    """

    def __call__(self, *args, array: NDArray, **kwargs) -> NDArray: ...


def flatten(array: NDArray, ensure_2d: bool = True) -> NDArray:
    """
    Flatten an array.

    If the array is 2D, it is returned as is.
    If the array is 3D or ensure_2d is True, it is reshaped to 2D.
    Otherwise, the array is reshaped to 3D with the first and last dimension kept.

    Parameters
    ----------
    array: NDArray
        The array to flatten.
    ensure_2d: bool, optional
        Whether to ensure that the array is 2D.

    Returns
    -------
    NDArray
        The flattened array.

    Raises
    ------
    ValueError
        If the array has less than 2 dimensions.
    """
    if array.ndim < 2:
        raise ValueError("Can only flatten arrays with at least 2 dimensions.")
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
    """
    Perform an operation on two arrays.

    Parameters
    ----------
    operation_func: PairedNDArrays_to_NDArray
        The operation function.
    array_1: NDArray
        The first array.
    array_2: NDArray
        The second array.
    element_wise: bool
        Whether to perform the operation element-wise.
    flat: bool
        Whether to flatten the result.
    expand_dims_for_broadcasting: bool, optional
        Whether to expand the dimensions for broadcasting.

    Returns
    -------
    NDArray
        The result of the operation.

    Raises
    ------
    NotImplementedError
        If the operation is not implemented for the shapes of the input arrays.
    ValueError
        If the input arrays are not broadcastable.
    """
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
    """
    Left pad (step >= 0) or right pad (step < 0) the array with the specified value.

    Parameters
    ----------
    array: NDArray
        The array to pad.
    step: int
        The number of elements to pad.
    value: int | float | str
        The value to pad with. If "same", the value of the first or last non-padded element is used.

    Returns
    -------
    NDArray
        The padded array.
    """
    if value == "same":
        value = array[step]
    if step >= 0:
        array[:step] = value
    if step < 0:
        array[-step:] = value
    return array


def ensure_generator(
    random_state: np.random.Generator | int | None,
) -> np.random.Generator:
    """
    Ensure that the random state is a numpy random generator.

    Parameters
    ----------
    random_state: np.random.Generator | int | None
        The random state.

    Returns
    -------
    np.random.Generator
        The random generator.
    """
    if not isinstance(random_state, np.random.Generator):
        return np.random.default_rng(seed=random_state)
    return random_state


def to_int_seed(random_state: np.random.Generator) -> int:
    """
    Convert a numpy random generator to an integer seed.

    Parameters
    ----------
    random_state: np.random.Generator
        The random generator.

    Returns
    -------
    int
        The integer seed.
    """
    # max int for sklearn random_state as int
    return int(random_state.integers(4294967295))


def closest_odd_divisible(number: float, divisor: int) -> int:
    """
    Find the closest odd number that is divisible by the specified divisor.

    Parameters
    ----------
    number: float
        The number.
    divisor: int
        The divisor.

    Returns
    -------
    int
        The closest odd number.

    Raises
    ------
    ValueError
        If the divisor is even.
    """
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
    """
    Context manager to customize the warning format. Only the warning message is displayed, but not the stack trace.
    """

    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return str(msg) + "\n"

    formatwarning = warnings.formatwarning
    warnings.formatwarning = custom_formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = formatwarning


def formatted_tqdm(arg, description_width="15%", bar_width="30%", **kwargs):
    """
    A tqdm wrapper that formats the description and bar width in Jupyter notebooks.
    """
    bar = tqdm(arg, **kwargs)
    if not hasattr(bar, "container"):
        # not an ipywidget tqdm bar
        return bar
    if description_width is not None:
        bar.container.children[0].layout.width = description_width  # type: ignore (see check above)
    if bar_width is not None:
        bar.container.children[1].layout.width = bar_width  # type: ignore (see check above)
    return bar

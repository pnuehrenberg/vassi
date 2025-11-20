import functools
from collections.abc import Callable

import numpy as np
import psutil


def get_inner[T, **P](func: Callable[P, T] | functools.partial[T]) -> Callable[P, T]:
    """
    Return the innermost function of a callable, functool partial, or recursive partial.

    Parameters:
        func: The callable to get the innermost function from.

    Returns:
        The innermost function.
    """
    if not isinstance(func, functools.partial):
        return func
    if len(func.args) == 0:
        return func.func
    return get_inner(func.args[0])


def check_memory_for_array(
    shape: tuple[int, ...], dtype: np.typing.DTypeLike = np.float64
) -> None:
    """
    Checks if there is enough available memory to create a NumPy array
    of a given shape and data type.

    Args:
        shape: The shape of the prospective array.
        dtype: The data type of the prospective array.

    Raises:
        MemoryError: If the required memory exceeds the available memory.
    """
    # Calculate the required memory in bytes
    num_elements = np.prod(shape)
    item_size = np.dtype(dtype).itemsize
    required_memory = num_elements * item_size

    # Get the available memory in bytes
    available_memory = psutil.virtual_memory().available

    if required_memory > available_memory:
        raise MemoryError(
            f"Required memory for array ({required_memory / 1e9:.2f} GB) exceeds available memory ({available_memory / 1e9:.2f} GB)."
        )

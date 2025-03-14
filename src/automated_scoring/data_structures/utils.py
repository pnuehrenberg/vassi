from collections.abc import Generator, Iterable
from contextlib import contextmanager
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np

Value = np.ndarray | int | float | str | np.integer | np.floating
MultipleValues = Iterable[Value]


def validate_keys(
    keys: Iterable[str],
    keys_reference: Iterable[str],
    allow_missing: bool,
) -> bool:
    """
    Validates that a set of keys conforms to a reference set, optionally allowing missing keys.

    Args:
        keys: The set of keys to validate.
        keys_reference: The reference set of valid keys.
        allow_missing: Whether to allow missing keys.

    Raises:
        KeyError: If any keys are undefined or if missing keys are not allowed.
    """
    keys = set(keys)
    keys_reference = set(keys_reference)
    undefined_keys = keys.difference(keys_reference)
    missing_keys = keys_reference.difference(keys)
    if (count := len(undefined_keys)) > 0:
        raise KeyError(
            f"undefined {'keys' if count > 1 else 'key'}: {', '.join(undefined_keys)}"
        )
    if (count := len(missing_keys)) > 0 and not allow_missing:
        raise KeyError(
            f"missing {'keys' if count > 1 else 'key'}: {', '.join(missing_keys)}"
        )
    return count == 0


def validated_length(*values: Value) -> int | None:
    """
    Returns the length of all iterable values if they are all the same length. Returns :code:`None` if all values are scalars.

    Args:
        *values: An iterable of values to check the length of.

    Raises:
        ValueError: If the values have unequal lengths or contain unsized iterables objects.
    """
    try:
        lengths = set([len(value) for value in values if isinstance(value, Iterable)])
    except TypeError as e:
        if str(e) != "len() of unsized object":
            raise e
        raise ValueError("values contain unsized objects.")
    if len(lengths) == 0:
        return None
    if len(lengths) != 1:
        raise ValueError("values have unequal lengths.")
    return lengths.pop()


def validate_timestamps(timestamps: Value) -> bool:
    """
    Validates that a sequence of timestamps contains no duplicates. Always returns :code:`True` if the timestamps are valid, otherwise raises a :code:`ValueError`.

    Args:
        timestamps: The timestamps to validate.

    Raises:
        ValueError: If timestamps is a scalar.
        ValueError: If any timestamps are duplicated.
    """
    if not isinstance(timestamps, Iterable):
        raise ValueError("duplicated timestamps from broadcasting singular timestamp")
    counts = np.unique(timestamps, return_counts=True)[1]
    if (counts > 1).any():
        raise ValueError("duplicated timestamps")
    return True


def greatest_common_denominator(
    values: Iterable[float],
    return_inverse: bool = True,
) -> float:
    """
    Finds the greatest common denominator (GCD) of a list of numbers and optionally returns its inverse.

    Args:
        values: The values to find the GCD of.
        return_inverse: Whether to return the inverse of the GCD.
    """
    denominators = [Fraction(value).limit_denominator().denominator for value in values]
    common_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    if return_inverse:
        return 1 / common_denominator
    return common_denominator


class OutOfInterval(Exception):
    """An error raised when a value is outside of an acceptable timestamp interval."""

    pass


def get_interval_slice(
    timestamps: np.ndarray, start: int | float, stop: int | float
) -> slice:
    """
    Gets a slice object representing the indices of timestamps within a specified interval.

    Args:
        timestamps: The timestamps to get the slice for.
        start: The start of the interval (inclusive).
        stop: The end of the interval (inclusive).
    """
    interval_indices = np.argwhere((timestamps >= start) & (timestamps <= stop)).ravel()
    if interval_indices.size == 0:
        return slice(0, 0)
    return slice(interval_indices[0], interval_indices[-1] + 1)


@contextmanager
def writeable(*arrays: np.ndarray) -> Generator:
    """
    A context manager that temporarily makes numpy arrays writeable.

    This context manager is useful when you need to modify the contents of a read-only array within a specific block of code. It ensures that the writeable flag is restored to its original state after the block, even if exceptions occur.

    Args:
        arrays: The arrays to make writeable.

    Raises:
        Exception: Any exception raised within the context is re-raised.
    """
    writeable: list[bool] = []
    for array in arrays:
        writeable.append(array.flags.writeable)
        array.flags.writeable = True
    try:
        yield
    except Exception as e:
        raise e
    finally:
        for array, was_writeable in zip(arrays, writeable):
            array.flags.writeable = was_writeable

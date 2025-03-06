from collections.abc import Generator, Iterable
from contextlib import contextmanager
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np
from numpy.typing import NDArray

Value = np.ndarray | int | float | str | np.integer | np.floating
MultipleValues = Iterable[Value]


def validate_keys(
    keys: Iterable[str],
    keys_reference: Iterable[str],
    allow_missing: bool,
) -> bool:
    """
    Validates that a set of keys conforms to a reference set, optionally allowing missing keys.

    This function checks if a given set of keys is a subset of a reference set. It raises a KeyError if any keys are undefined (not in the reference set) or if any keys are missing from the input set and `allow_missing` is False.

    Parameters
    ----------
    keys : Iterable[str]
        The set of keys to validate.
    keys_reference : Iterable[str]
        The reference set of valid keys.
    allow_missing : bool
        Whether to allow missing keys (default is False).

    Returns
    -------
    bool
        True if all keys are valid (or missing keys are allowed), False otherwise.

    Raises
    ------
    KeyError
        If any keys are undefined or if missing keys are not allowed.
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
    Returns the length of all iterable values if they are all the same length, otherwise raises a ValueError.

    Parameters
    ----------
    values : Iterable
        An iterable of values to check the length of.

    Returns
    -------
    int or None
        The length of the values if all iterable values have the same length, otherwise None.

    Raises
    ------
    ValueError
        If the values have unequal lengths or contain unsized iterables objects.
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
    Validates that a sequence of timestamps contains no duplicates.

    This function checks for duplicate timestamps within a given iterable. It leverages NumPy's efficient unique value counting to identify any repetitions.

    Parameters
    ----------
    timestamps : Value
        An iterable containing the timestamps to validate.

    Returns
    -------
    bool
        True if the timestamps are valid (no duplicates), False otherwise.

    Raises
    ------
    ValueError
        If the input is not iterable or if any timestamps are duplicated.
    """
    if not isinstance(timestamps, Iterable):
        raise ValueError("duplicated timestamps from broadcasting singular timestamp")
    counts = np.unique(timestamps, return_counts=True)[1]
    if (counts > 1).any():
        raise ValueError("duplicated timestamps")
    return True


def greatest_common_denominator(
    values: list[int | float] | NDArray[np.int64 | np.float64],
    return_inverse: bool = True,
) -> float:
    """
    Finds the greatest common denominator (GCD) of a list of numbers and optionally returns its inverse.

    Parameters
    ----------
    values : list of int or float or numpy.ndarray
        A list or NumPy array of numerical values.
    return_inverse : bool, optional
        Whether to return the inverse of the greatest common denominator. Defaults to True.

    Returns
    -------
    float
        The greatest common denominator or its inverse.
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
    timestamps: NDArray[np.floating | np.integer], start: int | float, stop: int | float
) -> slice:
    """
    Gets a slice object representing the indices of timestamps within a specified interval.

    Parameters
    ----------
    timestamps : numpy.ndarray
        An array of timestamps.
    start : int or float
        The start of the interval (inclusive).
    stop : int or float
        The end of the interval (inclusive).

    Returns
    -------
    slice
        A slice object representing the indices of the timestamps within the interval.
    """
    interval_indices = np.argwhere((timestamps >= start) & (timestamps <= stop)).ravel()
    if interval_indices.size == 0:
        return slice(0, 0)
    return slice(interval_indices[0], interval_indices[-1] + 1)


@contextmanager
def writeable(*arrays: NDArray) -> Generator:
    """
    A context manager that temporarily makes NumPy arrays writeable.

    This context manager is useful when you need to modify the contents of a read-only NumPy array within a specific block of code. It ensures that the writeable flag is restored to its original state after the block, even if exceptions occur.

    Parameters
    ----------
    arrays : tuple of numpy.ndarray
        The NumPy arrays to make writeable.

    Yields
    ------
    None
        The context manager yields control to the enclosed block of code.

    Raises
    ------
    Exception
        Any exception raised within the context is re-raised.
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

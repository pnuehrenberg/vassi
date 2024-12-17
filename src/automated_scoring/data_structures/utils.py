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
    Validates that all keys are present in the reference keys.

    Parameters
    ----------
    keys : Iterable[str]
        The keys to validate.
    keys_reference : Iterable[str]
        The reference keys.
    allow_missing : bool
        Whether to allow missing reference keys.

    Returns
    -------
    bool
        Whether the keys are valid.

    Raises
    ------
    KeyError
        If the keys contain undefined keys or missing reference keys when allow_missing is False.
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
    Validates that all iterable values have the same length.

    Parameters
    ----------
    values : Value
        The values to validate.

    Returns
    -------
    int | None
        The validated length of the iterable values or None if all values are not iterables.

    Raises
    ------
    ValueError
        If iterable values are unsized or have unequal lengths.
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
    Validates that timestamps are unique and non-singular.

    Parameters
    ----------
    timestamps : NDArray, int, float
        The timestamps to validate.

    Returns
    -------
    bool
        Whether the timestamps are valid. This will always be True, because invalid timestamps raise exceptions.

    Raises
    ------
    ValueError
        If the timestamps are singular or duplicated.
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
    Calculates the greatest common denominator of a list of values.

    Parameters
    ----------
    values : list[int | float] | NDArray
        The values to calculate the greatest common denominator of.
    return_inverse : bool, optional
        Whether to return the inverse of the greatest common denominator, by default True.

    Returns
    -------
    float
        The greatest common denominator of the values or its inverse if return_inverse is True.
    """
    denominators = [Fraction(value).limit_denominator().denominator for value in values]
    common_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    if return_inverse:
        return 1 / common_denominator
    return common_denominator


class OutOfInterval(Exception):
    """
    An exception to be raised when a timestamp is outside a given interval.
    """

    pass


def get_interval_slice(
    timestamps: NDArray[np.floating | np.integer], start: int | float, stop: int | float
) -> slice:
    """
    Returns a slice of the timestamps that fall within the given interval.

    Parameters
    ----------
    timestamps : NDArray[np.floating | np.integer]
        The timestamps to slice.
    start : int | float
        The start of the interval, inclusive.
    stop : int | float
        The end of the interval, inclusive.

    Returns
    -------
    slice
        The slice of the timestamps that fall within the interval.
    """
    interval_indices = np.argwhere((timestamps >= start) & (timestamps <= stop)).ravel()
    if interval_indices.size == 0:
        return slice(0, 0)
    return slice(interval_indices[0], interval_indices[-1] + 1)


@contextmanager
def writeable(*arrays: NDArray) -> Generator:
    """
    Context manager to temporarily make arrays writeable.

    Parameters
    ----------
    arrays : NDArray
        The arrays to make writeable.

    Yields
    ------
    Generator
        A generator that yields when the context is entered and exits when the context is exited.
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

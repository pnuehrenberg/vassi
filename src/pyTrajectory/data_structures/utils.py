from collections.abc import Generator, Iterable
from contextlib import contextmanager
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

MultipleStrings = list[str] | tuple[str, ...]
Value = np.ndarray | int | float | str | np.integer | np.floating
MultipleValues = list[Value] | tuple[Value, ...]


def validate_keys(
    keys: Iterable[str],
    keys_reference: Iterable[str],
    allow_missing: bool,
) -> bool:
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


def validate_timestamps(timestamps: Value):
    if not isinstance(timestamps, Iterable):
        raise ValueError("duplicated timestamps from broadcasting singular timestamp")
    counts = np.unique(timestamps, return_counts=True)[1]
    if (counts > 1).any():
        raise ValueError("duplicated timestamps")
    return True


def is_str_iterable(
    key: Any,
) -> tuple[Literal[True], MultipleStrings] | tuple[Literal[False], Any]:
    if not isinstance(key, list | tuple):
        return False, key
    if all([isinstance(element, str) for element in key]):
        return True, key
    return False, key


def is_slice_str(
    key: Any,
) -> tuple[Literal[True], tuple[slice, str]] | tuple[Literal[False], Any]:
    if not isinstance(key, tuple):
        return False, key
    if len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], str):
        return True, key
    return False, key


def is_slice_str_iterable(
    key: Any,
) -> tuple[Literal[True], tuple[slice, MultipleStrings]] | tuple[Literal[False], Any]:
    if not isinstance(key, tuple):
        return False, key
    if len(key) == 2 and isinstance(key[0], slice) and is_str_iterable(key[1]):
        return True, key
    return False, key


def is_int_str(
    key: Any,
) -> tuple[Literal[True], tuple[int, str]] | tuple[Literal[False], Any]:
    if not isinstance(key, tuple):
        return False, key
    if (
        len(key) == 2
        and isinstance(key[0], int | np.integer)
        and isinstance(key[1], str)
    ):
        return True, key
    return False, key


def is_int_str_iterable(
    key: Any,
) -> tuple[Literal[True], tuple[int, MultipleStrings]] | tuple[Literal[False], Any]:
    if not isinstance(key, tuple):
        return False, key
    if (
        len(key) == 2
        and isinstance(key[0], int | np.integer)
        and is_str_iterable(key[1])
    ):
        return True, key
    return False, key


def is_value(value: Any) -> tuple[Literal[True], Value] | tuple[Literal[False], Any]:
    if isinstance(value, Value):
        return True, value
    return False, value


def is_value_iterable(
    key: Any,
) -> tuple[Literal[True], MultipleValues] | tuple[Literal[False], Any]:
    if not isinstance(key, list | tuple):
        return False, key
    if all([isinstance(element, Value) for element in key]):
        return True, key
    return False, key


def greatest_common_denominator(
    values: list[int | float] | NDArray[np.int64 | np.float64],
    return_inverse: bool = True,
):
    denominators = [Fraction(value).limit_denominator().denominator for value in values]
    common_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    return 1 / common_denominator


class OutOfInterval(Exception):
    pass


def get_interval_slice(
    timestamps: NDArray[np.floating | np.integer], start: int | float, stop: int | float
) -> slice:
    interval_indices = np.argwhere((timestamps >= start) & (timestamps <= stop)).ravel()
    if interval_indices.size == 0:
        return slice(0, 0)
    return slice(interval_indices[0], interval_indices[-1] + 1)


@contextmanager
def writeable(*arrays: NDArray) -> Generator:
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

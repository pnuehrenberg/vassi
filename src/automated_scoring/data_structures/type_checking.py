from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from .utils import MultipleValues, Value


def is_str_iterable(
    value: Any,
) -> tuple[Literal[True], Iterable[str]] | tuple[Literal[False], Any]:
    """
    Checks if a value is an iterable of strings.

    This function determines if a given value is an iterable and if all its elements are strings. It returns a tuple indicating the result and the original value.

    :param value: The value to check.
    """
    if not isinstance(value, Iterable):
        return False, value
    if all([isinstance(element, str) for element in value]):
        return True, value
    return False, value


def is_slice_str(
    value: Any,
) -> tuple[Literal[True], tuple[slice, str]] | tuple[Literal[False], Any]:
    """
    Checks if a value is a tuple containing a slice and a string.

    :param value: The value to check.
    """
    if not isinstance(value, tuple):
        return False, value
    if len(value) == 2 and isinstance(value[0], slice) and isinstance(value[1], str):
        return True, value
    return False, value


def is_slice_str_iterable(
    value: Any,
) -> tuple[Literal[True], tuple[slice, Iterable[str]]] | tuple[Literal[False], Any]:
    """
    Checks if a value is a tuple containing a slice and an iterable of strings.

    This function is designed to validate if a given value adheres to a specific structure: a tuple where the first element is a slice object and the second element is an iterable containing only strings. It returns a tuple indicating the validation result and the original value.

    :param value: The value to check.
    """
    if not isinstance(value, tuple):
        return False, value
    if len(value) == 2 and isinstance(value[0], slice) and is_str_iterable(value[1]):
        return True, value
    return False, value


def is_int_str(
    value: Any,
) -> tuple[Literal[True], tuple[int, str]] | tuple[Literal[False], Any]:
    """
    Checks if a value is a tuple containing an integer and a string.

    :param value: The value to check.
    """
    if not isinstance(value, tuple):
        return False, value
    if (
        len(value) == 2
        and isinstance(value[0], int | np.integer)
        and isinstance(value[1], str)
    ):
        return True, value
    return False, value


def is_int_str_iterable(
    value: Any,
) -> tuple[Literal[True], tuple[int, Iterable[str]]] | tuple[Literal[False], Any]:
    """
    Checks if a value is a tuple containing an integer and an iterable of strings.

    This function validates if a given value is a tuple with two elements: an integer and an iterable of strings. It returns a tuple indicating the result and the original value.

    :param value: The value to check.
    """
    if not isinstance(value, tuple):
        return False, value
    if (
        len(value) == 2
        and isinstance(value[0], int | np.integer)
        and is_str_iterable(value[1])
    ):
        return True, value
    return False, value


def is_value(value: Any) -> tuple[Literal[True], Value] | tuple[Literal[False], Any]:
    """
    Checks if a given value is an instance of the `Value` type union.

    This function determines whether the input `value` is a `Value` object. If it is, it returns `True` along with the `Value` object; otherwise, it returns `False` and the original `value`.

    :param value: The value to check.
    """
    if isinstance(value, Value):
        return True, value
    return False, value


def is_value_iterable(
    value: Any,
) -> tuple[Literal[True], MultipleValues] | tuple[Literal[False], Any]:
    """
    Checks if a value is iterable and contains only valid values.

    This function determines if a given value is iterable and if all its elements are valid values according to the `is_value` function. It returns a tuple indicating whether the value is iterable and valid, along with the original value.

    :param value: The value to check for iterability and validity of its elements.
    """
    if not isinstance(value, Iterable):
        return False, value
    if any([not is_value(_value) for _value in value]):
        return False, value
    return True, value

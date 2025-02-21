from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from .utils import MultipleValues, Value


def is_str_iterable(
    value: Any,
) -> tuple[Literal[True], Iterable[str]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a value is an iterable of strings.
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
    Helper function to validate that a value a tuple of a slice and a string.
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
    Helper function to validate that a value a tuple of a slice and an iterable of strings.
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
    Helper function to validate that a value a tuple of an integer and a string.
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
    Helper function to validate that a value is a tuple of an integer and an iterable of strings.
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
    """Helper function to ensure that a value is valid."""
    if isinstance(value, Value):
        return True, value
    return False, value


def is_value_iterable(
    value: Any,
) -> tuple[Literal[True], MultipleValues] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a value is an iterable of values.
    """
    if not isinstance(value, Iterable):
        return False, value
    if any([not is_value(_value) for _value in value]):
        return False, value
    return True, value

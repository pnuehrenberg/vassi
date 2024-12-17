from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from .utils import MultipleValues, Value


def is_str_iterable(
    key: Any,
) -> tuple[Literal[True], Iterable[str]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key is an iterable of strings.
    """
    if not isinstance(key, Iterable):
        return False, key
    if all([isinstance(element, str) for element in key]):
        return True, key
    return False, key


def is_slice_str(
    key: Any,
) -> tuple[Literal[True], tuple[slice, str]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key a tuple of a slice and a string.
    """
    if not isinstance(key, tuple):
        return False, key
    if len(key) == 2 and isinstance(key[0], slice) and isinstance(key[1], str):
        return True, key
    return False, key


def is_slice_str_iterable(
    key: Any,
) -> tuple[Literal[True], tuple[slice, Iterable[str]]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key a tuple of a slice and an iterable of strings.
    """
    if not isinstance(key, tuple):
        return False, key
    if len(key) == 2 and isinstance(key[0], slice) and is_str_iterable(key[1]):
        return True, key
    return False, key


def is_int_str(
    key: Any,
) -> tuple[Literal[True], tuple[int, str]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key a tuple of an integer and a string.
    """
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
) -> tuple[Literal[True], tuple[int, Iterable[str]]] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key is a tuple of an integer and an iterable of strings.
    """
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
    """Helper function to ensure that a value is valid."""
    if isinstance(value, Value):
        return True, value
    return False, value


def is_value_iterable(
    key: Any,
) -> tuple[Literal[True], MultipleValues] | tuple[Literal[False], Any]:
    """
    Helper function to validate that a key is an iterable of values.
    """
    if not isinstance(key, list | tuple):
        return False, key
    if all([isinstance(element, Value) for element in key]):
        return True, key
    return False, key

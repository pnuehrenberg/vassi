from collections.abc import Iterable, Sequence
from typing import TypeGuard

import numpy as np

from ..type_guards import is_iterable_of
from .utils import MultipleValues, Value


def confirm_str_iterable(
    value: ...,
) -> TypeGuard[Iterable[str]]:
    """
    Checks if a value is an iterable of strings.

    Parameters:
        value: The value to check.
    """
    if is_iterable_of(value, str):
        return True
    return False


def confirm_slice_str(
    value: ...,
) -> TypeGuard[tuple[slice, str]]:
    """
    Checks if a value is a tuple containing a slice and a string.

    Parameters:
        value: The value to check.
    """
    if not isinstance(value, Sequence):
        return False
    if (
        len(value) == 2
        and isinstance(value[0], slice)
        and isinstance(value[1], str)
        and isinstance(value, tuple)
    ):
        return True
    return False


def confirm_slice_str_iterable(
    value: ...,
) -> TypeGuard[tuple[slice, Iterable[str]]]:
    """
    Checks if a value is a tuple containing a slice and an iterable of strings.

    Parameters:
        value: The value to check.
    """
    if not isinstance(value, Sequence):
        return False
    if (
        len(value) == 2
        and isinstance(value[0], slice)
        and confirm_str_iterable(value[1])
        and isinstance(value, tuple)
    ):
        return True
    return False


def confirm_int_str(
    value: ...,
) -> TypeGuard[tuple[int, str]]:
    """
    Checks if a value is a tuple containing an integer and a string.

    Parameters:
        value: The value to check.
    """
    if not isinstance(value, Sequence):
        return False
    if (
        len(value) == 2
        and isinstance(value[0], int | np.integer)
        and isinstance(value[1], str)
        and isinstance(value, tuple)
    ):
        return True
    return False


def confirm_int_str_iterable(
    value: ...,
) -> TypeGuard[tuple[int, Iterable[str]]]:
    """
    Checks if a value is a tuple containing an integer and an iterable of strings.

    Parameters:
        value: The value to check.
    """
    if not isinstance(value, Sequence):
        return False
    if (
        len(value) == 2
        and isinstance(value[0], int | np.integer)
        and confirm_str_iterable(value[1])
        and isinstance(value, tuple)
    ):
        return True
    return False


def confirm_value(
    value: ...,
) -> TypeGuard[Value]:
    """
    Checks if a given value is an instance of the :code:`Value` type union.

    Parameters:
        value: The value to check.
    """
    if isinstance(value, Value):
        return True
    return False


def confirm_value_iterable(
    value: ...,
) -> TypeGuard[MultipleValues]:
    """
    Checks if a value is iterable and contains only valid values (instances of :code:`Value`).

    Parameters:
        value: The value to check.
    """
    if not isinstance(value, Iterable):
        return False
    if not all(confirm_value(_value) for _value in value):
        return False
    return True

import functools
from collections.abc import Callable
from typing import Concatenate

import numpy as np

from .. import series_operations
from ..data_structures import Trajectory

PREFIXES = {
    "_as_absolute": "abs_",
    "_as_sign_change_latency": "scl_",
    "_reversed_dyad": "REVERSED",
    None: "",
}


def get_prefix[T, **P](func: Callable[P, T] | functools.partial[T]) -> str:
    """
    Return the prefix of the feature function.

    Prefixes are obtained from the global :code:`PREFIXES` dictionary. If a decorator is not found in the dictionary, an empty string is returned.

    Parameters:
        func: The feature function to get the prefix from.

    Returns:
        The prefix of the feature function.
    """
    global PREFIXES
    if not isinstance(func, functools.partial):
        return ""
    prefix = ""
    name = getattr(func.func, "__name__", None)
    prefix = PREFIXES.get(name, "")
    return f"{prefix}{get_prefix(func.args[0])}"


def _as_absolute[**P](
    func: Callable[P, np.ndarray], *args: P.args, **kwargs: P.kwargs
) -> np.ndarray:
    """
    Helper function that takes the wrapped function as a keyword argument
    and its arguments, then computes the absolute value of the result.
    """
    return np.abs(func(*args, **kwargs))


def as_absolute[**P](func: Callable[P, np.ndarray]) -> Callable[P, np.ndarray]:
    """
    Decorator to convert a feature function to an absolute value feature function.

    This decorator is fully type-hinted using ParamSpec to preserve the
    signature of the decorated function.
    """
    result_func = functools.partial(_as_absolute, func)
    decorated = functools.wraps(func)(result_func)
    return decorated


def _as_sign_change_latency[**P](
    func: Callable[P, np.ndarray], *args: P.args, **kwargs: P.kwargs
) -> np.ndarray:
    return series_operations.sign_change_latency(func(*args, **kwargs))


def as_sign_change_latency[**P](
    func: Callable[P, np.ndarray],
) -> Callable[P, np.ndarray]:
    """
    Decorator to convert a feature function to a sign change latency feature function.
    """
    result_func = functools.partial(_as_sign_change_latency, func)
    decorated = functools.wraps(func)(result_func)
    return decorated


def _reversed_dyad[**P](
    func: Callable[Concatenate[Trajectory, P], np.ndarray],
    trajectory: Trajectory,
    *args: P.args,
    **kwargs: P.kwargs,
) -> np.ndarray:
    trajectory_other = kwargs.get("trajectory_other", None)
    if not isinstance(trajectory_other, Trajectory):
        raise ValueError(
            f"the reversed_dyad decorator only supports relational feature functions with trajectory_other of type {Trajectory}"
        )
    kwargs["trajectory_other"] = trajectory
    return func(trajectory_other, *args, **kwargs)


def reversed_dyad[**P](
    func: Callable[Concatenate[Trajectory, P], np.ndarray],
) -> Callable[Concatenate[Trajectory, P], np.ndarray]:
    """
    Decorator to reverse the dyad that serves as input to a dyadic feature function.
    """
    result_func = functools.partial(_reversed_dyad, func)
    decorated = functools.wraps(func)(result_func)
    return decorated

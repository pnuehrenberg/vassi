import functools
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from .. import series_operations
from ..logging import set_logging_level
from .utils import (
    DataFrameFeature,
    Feature,
    get_feature_names,
    prune_feature_names,
)

PREFIXES = {
    "_as_absolute": "abs_",
    "_as_sign_change_latency": "scl_",
    "_as_dataframe": "",
}


def get_inner(func: Feature | functools.partial) -> Feature:
    """
    Return the innermost feature function.

    Parameters:
        func: The feature function to get the innermost function from.

    Returns:
        The innermost feature function.
    """
    if not isinstance(func, functools.partial):
        return func
    return get_inner(func.keywords["func"])


def get_prefix(func: Feature | functools.partial) -> str:
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
    if not hasattr(func.func, "__name__"):
        raise ValueError(f"{func} is not a valid named feature.")
    if func.func.__name__ in PREFIXES:
        prefix = PREFIXES[func.func.__name__]
    else:
        print("decorator", func.func.__name__, "has no prefix")
    return f"{prefix}{get_prefix(func.keywords['func'])}"


def _as_absolute(*args, func: Feature, **kwargs) -> np.ndarray:
    return np.abs(func(*args, **kwargs))


def as_absolute(func: Feature) -> Feature:
    """
    Decorator to convert a feature function to an absolute value feature function.
    """
    result_func = functools.partial(_as_absolute, func=func)
    decorated = functools.wraps(func)(result_func)
    return decorated


def _as_sign_change_latency(*args, func: Feature, **kwargs) -> np.ndarray:
    return series_operations.sign_change_latency(func(*args, **kwargs))


def as_sign_change_latency(func: Feature) -> Feature:
    """
    Decorator to convert a feature function to a sign change latency feature function.
    """
    result_func = functools.partial(_as_sign_change_latency, func=func)
    decorated = functools.wraps(func)(result_func)
    return decorated


def _as_dataframe(
    *args,
    func: Feature,
    keep: Optional[Iterable[str] | str],
    discard: Optional[Iterable[str] | str],
    **kwargs,
) -> pd.DataFrame:
    if "flat" in kwargs and not kwargs.pop("flat"):
        set_logging_level().warning(
            "Ignoring argument flat=False. Dataframe features are always flat."
        )
    feature = get_inner(func)
    prefix = get_prefix(func)
    names = [f"{prefix}{name}" for name in get_feature_names(feature, **kwargs)]
    pruned_names = prune_feature_names(names, keep=keep, discard=discard)
    drop = [name for name in names if name not in pruned_names]
    return pd.DataFrame(
        func(*args, flat=True, **kwargs),
        columns=pd.Index(names),
    ).drop(labels=drop, axis="columns", inplace=False)


def as_dataframe(
    func: Feature,
    keep: Optional[Iterable[str] | str] = None,
    discard: Optional[Iterable[str] | str] = None,
) -> DataFrameFeature:
    """
    Decorator to convert a feature function to a dataframe feature function.

    Parameters:
        func: The feature function to convert to a dataframe feature function.
        keep: Iterable of feature names (or patterns within names) to keep regardless of :code:`discard`.
        discard: Iterable of feature names (or patterns within names) to discard.
    """
    result_func = functools.partial(
        _as_dataframe, func=func, keep=keep, discard=discard
    )
    decorated = functools.wraps(func)(result_func)
    return decorated

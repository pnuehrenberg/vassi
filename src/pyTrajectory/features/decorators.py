from typing import Optional, overload
from collections.abc import Iterable

from functools import wraps

import warnings
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from .. import series_operations
from ..utils import warning_only
from ..data_structures import Trajectory, InstanceCollection
from .utils import (
    Feature,
    DataFrameFeature,
    get_feature_names,
    prune_feature_names,
)


def as_absolute(func: Feature) -> Feature:
    @overload
    def decorated(trajectory: Trajectory, *args, **kwargs) -> NDArray: ...

    @overload
    def decorated(collection: InstanceCollection, *args, **kwargs) -> NDArray: ...

    @wraps(func)
    def decorated(arg1: Trajectory | InstanceCollection, *args, **kwargs) -> NDArray:
        return np.abs(func(arg1, *args, **kwargs))

    decorated.__name__ = f"abs_{func.__name__}"
    return decorated


def as_sign_change_latency(func: Feature) -> Feature:
    @overload
    def decorated(trajectory: Trajectory, *args, **kwargs) -> NDArray: ...

    @overload
    def decorated(collection: InstanceCollection, *args, **kwargs) -> NDArray: ...

    @wraps(func)
    def decorated(arg1: Trajectory | InstanceCollection, *args, **kwargs) -> NDArray:
        return series_operations.sign_change_latency(func(arg1, *args, **kwargs))

    decorated.__name__ = f"scl_{func.__name__}"
    return decorated


def as_dataframe(
    func: Feature,
    keep: Optional[Iterable[str] | str] = None,
    discard: Optional[Iterable[str] | str] = None,
) -> DataFrameFeature:
    @overload
    def decorated(trajectory: Trajectory, *args, **kwargs) -> pd.DataFrame: ...

    @overload
    def decorated(collection: InstanceCollection, *args, **kwargs) -> pd.DataFrame: ...

    @wraps(func)
    def decorated(
        arg1: Trajectory | InstanceCollection, *args, **kwargs
    ) -> pd.DataFrame:
        if "flat" in kwargs and not kwargs.pop("flat"):
            with warning_only():
                warnings.warn("Ignoring argument flat=False.")
        names = get_feature_names(func, **kwargs)
        pruned_names = prune_feature_names(names, keep=keep, discard=discard)
        drop = [name for name in names if name not in pruned_names]
        return pd.DataFrame(
            func(arg1, *args, flat=True, **kwargs),
            columns=pd.Index(names),
        ).drop(labels=drop, axis="columns")

    decorated.__name__ = func.__name__
    return decorated

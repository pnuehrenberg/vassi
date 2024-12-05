import functools
from inspect import signature
from typing import Any, Callable, Iterable, Optional, Protocol, overload

import pandas as pd
from numpy.typing import NDArray

from ..data_structures import InstanceCollection, Trajectory
from ..utils import KeypointPair, KeypointPairs, Keypoints


class Feature(Protocol):
    __name__: str

    @overload
    def __call__(self, collection: InstanceCollection, *args, **kwargs) -> NDArray: ...

    @overload
    def __call__(self, trajectory: Trajectory, *args, **kwargs) -> NDArray: ...


class DataFrameFeature(Protocol):
    __name__: str

    @overload
    def __call__(
        self, collection: InstanceCollection, *args, **kwargs
    ) -> pd.DataFrame: ...

    @overload
    def __call__(self, trajectory: Trajectory, *args, **kwargs) -> pd.DataFrame: ...


def recursive_name(func: Feature | DataFrameFeature | functools.partial) -> str:
    if not isinstance(func, functools.partial):
        return func.__name__
    decorator_name = func.func.__name__
    func_name = recursive_name(func.keywords["func"])
    return decorator_name + func_name


def pair(keypoint_pair: KeypointPair) -> str:
    return f"{keypoint_pair[0]}_{keypoint_pair[1]}"


def names(
    func_name: str,
    keypoints: Keypoints | None = None,
    keypoint_pairs: KeypointPairs | None = None,
) -> list[str]:
    if keypoints is not None:
        return [f"{func_name}-{keypoint}" for keypoint in keypoints]
    if keypoint_pairs is not None:
        return [
            f"{func_name}-{pair(keypoint_pair)}" for keypoint_pair in keypoint_pairs
        ]
    return [func_name]


def relational_names(
    func_name: str,
    *,
    keypoints_1: Keypoints | None = None,
    keypoints_2: Keypoints | None = None,
    keypoint_pairs_1: KeypointPairs | None = None,
    keypoint_pairs_2: KeypointPairs | None = None,
    element_wise: bool | None = False,
) -> list[str]:
    def name(*args: str | int) -> str:
        return f"{func_name}-{'-'.join([str(arg) for arg in args])}"

    # keypoints and keypoints
    if element_wise and keypoints_1 is not None and keypoints_2 is not None:
        return [
            name(keypoint_1, keypoint_2)
            for keypoint_1, keypoint_2 in zip(keypoints_1, keypoints_2)
        ]
    if keypoints_1 is not None and keypoints_2 is not None:
        return [
            name(keypoint_1, keypoint_2)
            for keypoint_1 in keypoints_1
            for keypoint_2 in keypoints_2
        ]
    # keypoint pairs and keypoint pairs
    if element_wise and keypoint_pairs_1 is not None and keypoint_pairs_2 is not None:
        return [
            name(pair(keypoint_pair_1), pair(keypoint_pair_2))
            for keypoint_pair_1, keypoint_pair_2 in zip(
                keypoint_pairs_1, keypoint_pairs_2
            )
        ]
    if keypoint_pairs_1 is not None and keypoint_pairs_2 is not None:
        return [
            name(pair(keypoint_pair_1), pair(keypoint_pair_2))
            for keypoint_pair_1 in keypoint_pairs_1
            for keypoint_pair_2 in keypoint_pairs_2
        ]
    # keypoint pairs and keypoint
    if element_wise and keypoint_pairs_1 is not None and keypoints_2 is not None:
        return [
            name(pair(keypoint_pair), keypoint)
            for keypoint_pair, keypoint in zip(keypoint_pairs_1, keypoints_2)
        ]
    if keypoint_pairs_1 is not None and keypoints_2 is not None:
        return [
            name(pair(keypoint_pair), keypoint)
            for keypoint_pair in keypoint_pairs_1
            for keypoint in keypoints_2
        ]
    if element_wise and keypoints_1 is not None and keypoint_pairs_2 is not None:
        return [
            name(keypoint, pair(keypoint_pair))
            for keypoint, keypoint_pair in zip(keypoints_1, keypoint_pairs_2)
        ]
    if keypoints_1 is not None and keypoint_pairs_2 is not None:
        return [
            name(keypoint, pair(keypoint_pair))
            for keypoint in keypoints_1
            for keypoint_pair in keypoint_pairs_2
        ]
    raise NotImplementedError("Invalid input.")


def feature_names(
    func: Callable,
    relational: bool,
    dyadic: bool = False,
    suffixes: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> list[str]:
    def get_param(param: str):
        if param not in kwargs:
            return None
        return kwargs[param]

    def apply_suffixes(names: Iterable[str]) -> list[str]:
        if suffixes is None:
            return list(names)
        return [f"{name}-{suffix}" for name in names for suffix in suffixes]

    func_name = recursive_name(func)
    if dyadic:
        func_name = f"dyadic_{func_name}"
    if (step := get_param("step")) is not None:
        func_name = f"{func_name}_t({step})"
    if relational:
        params = {
            param: get_param(param)
            for param in [
                "keypoints_1",
                "keypoints_2",
                "keypoint_pairs_1",
                "keypoint_pairs_2",
                "element_wise",
            ]
        }
        return apply_suffixes(relational_names(func_name, **params))
    params = {param: get_param(param) for param in ["keypoints", "keypoint_pairs"]}
    return apply_suffixes(names(func_name, **params))


def get_feature_names(func, **kwargs) -> list[str]:
    relational = any([("keypoints_" in kwarg) for kwarg in kwargs]) or any(
        [("keypoint_pairs_" in kwarg) for kwarg in kwargs]
    )
    suffixes = None
    if "suffixes" in kwargs:
        suffixes = kwargs.pop("suffixes")
    if suffixes is None:
        suffixes = signature(func).parameters.get("suffixes")
        if suffixes is not None:
            suffixes = suffixes.default
    return feature_names(func, relational=relational, suffixes=suffixes, **kwargs)


def prune_feature_names(
    names: Iterable[str],
    *,
    keep: Optional[Iterable[str] | str] = None,
    discard: Optional[Iterable[str] | str] = None,
) -> list[str]:
    def _as_str_list(arg: Iterable[str] | str | None) -> list[str]:
        if isinstance(arg, list):
            return arg
        if isinstance(arg, str):
            return [arg]
        return []

    keep = _as_str_list(keep)
    discard = _as_str_list(discard)
    has_keep_names = len(keep) > 0
    has_discard_names = len(discard) > 0
    return [
        name
        for name in names
        if not (has_discard_names and any(_discard in name for _discard in discard))
        or (has_keep_names and any(_keep in name for _keep in keep))
    ]

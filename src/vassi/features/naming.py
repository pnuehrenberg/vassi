import functools
from collections.abc import Callable, Iterable
from inspect import signature
from itertools import product
from typing import overload

import numpy as np
import pandas as pd

from .._utils import get_inner
from ..logging import set_logging_level  # pyright: ignore[reportUnknownVariableType]
from ..type_guards import is_iterable_of, is_iterable_of_tuple
from .modifiers import get_prefix
from .utils import Shaped


def _as_dataframe[**P](
    func: Callable[P, np.ndarray],
    keep: Iterable[str] | str | None,
    discard: Iterable[str] | str | None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> pd.DataFrame:
    if kwargs.get("flat", True) is False:
        set_logging_level().warning(
            "Ignoring argument flat=False. Dataframe features are always flat."
        )

    names = generate_feature_names(func, *args, **kwargs)
    pruned_names = prune_feature_names(names, keep=keep, discard=discard)
    kwargs["flat"] = True
    return pd.DataFrame(
        func(*args, **kwargs)[:, np.isin(names, pruned_names)],
        columns=pd.Index(pruned_names),
    )


class _AsDataFrameDecorator[**P]:
    keep: Iterable[str] | str | None
    discard: Iterable[str] | str | None

    def __init__(
        self,
        keep: Iterable[str] | str | None = None,
        discard: Iterable[str] | str | None = None,
    ):
        self.keep = keep
        self.discard = discard

    def __call__(self, func: Callable[P, np.ndarray]) -> Callable[P, pd.DataFrame]:
        result_func = functools.partial(_as_dataframe, func, self.keep, self.discard)
        decorated = functools.wraps(func)(result_func)
        return decorated


@overload
def as_dataframe[**P](
    func: Callable[P, np.ndarray],
    *,
    keep: Iterable[str] | str | None = None,
    discard: Iterable[str] | str | None = None,
) -> Callable[P, pd.DataFrame]: ...


@overload
def as_dataframe[**P](
    func: None = None,
    *,
    keep: Iterable[str] | str | None = None,
    discard: Iterable[str] | str | None = None,
) -> _AsDataFrameDecorator[P]: ...


def as_dataframe[**P](
    func: Callable[P, np.ndarray] | None = None,
    *,
    keep: Iterable[str] | str | None = None,
    discard: Iterable[str] | str | None = None,
) -> Callable[P, pd.DataFrame] | _AsDataFrameDecorator[P]:
    """
    Decorator to convert a feature function to a dataframe feature function.

    Parameters:
        func: The feature function to convert to a dataframe feature function.
        keep: Iterable of feature names (or patterns within names) to keep regardless of :code:`discard`.
        discard: Iterable of feature names (or patterns within names) to discard.
    """
    if func is not None:
        return _AsDataFrameDecorator(keep=keep, discard=discard)(func)
    return _AsDataFrameDecorator(keep=keep, discard=discard)


def _feature_names[**P](
    func: Callable[P, Shaped],
    relational: bool,
    dyadic: bool,
    suffixes: Iterable[str] | None,
    *_: P.args,
    **kwargs: P.kwargs,
) -> list[str]:
    """
    Generate full feature names for a given function.

    Parameters:
        func: The function to generate feature names for.
        relational: Whether the function is relational.
        dyadic: Whether the function is dyadic.
        suffixes: Optional suffixes to append to the feature names.

    Returns:
        The full feature names.
    """

    def _name(*args: int | tuple[int, int]) -> str:
        nonlocal func_name
        return f"{func_name}-{'-'.join([str(arg) if isinstance(arg, int) else '_'.join(map(str, arg)) for arg in args])}"

    def _generate_names(
        *args: Iterable[int] | Iterable[tuple[int, int]], element_wise: bool = False
    ) -> list[str]:
        nonlocal func_name
        if len(args) == 1:
            return list(map(_name, args[0]))
        if not len(args) == 2:
            raise ValueError("relational names require exactly two arguments")
        arg_1 = list(args[0])
        arg_2 = list(args[1])
        if element_wise:
            if not len(arg_1) == len(arg_2):
                raise ValueError("element_wise requires arguments of the same length")
            return list(map(_name, arg_1, arg_2))
        return list(map(_name, *zip(*product(arg_1, arg_2))))

    func_name = f"{get_prefix(func)}{get_inner(func).__name__}"
    reversed = "REVERSED" in func_name
    if reversed and not dyadic:
        raise ValueError("invalid @reversed_dyad for non-dyadic feature function")
    if dyadic:
        func_name = (
            f"dyad{'(r)' if reversed else ''}-{func_name.replace('REVERSED', '')}"
        )
    if isinstance(step := kwargs.get("step"), int):
        func_name = f"{func_name}({step})"
    elif step is not None:
        raise ValueError("step must be an integer")
    if relational:
        element_wise = kwargs.get("element_wise", False)
        if not isinstance(element_wise, bool):
            raise ValueError("element_wise must be a boolean")
        if (
            keypoints_1 := kwargs.get("keypoints_1")
        ) is not None and not is_iterable_of(keypoints_1, int):
            raise ValueError("keypoints_1 must be an iterable of integers or None")
        if (
            keypoints_2 := kwargs.get("keypoints_2")
        ) is not None and not is_iterable_of(keypoints_2, int):
            raise ValueError("keypoints_2 must be an iterable of integers or None")
        if (
            keypoint_pairs_1 := kwargs.get("keypoint_pairs_1")
        ) is not None and not is_iterable_of_tuple(keypoint_pairs_1, int):
            raise ValueError("keypoint_pairs_1 must be an iterable of integers or None")
        if (
            keypoint_pairs_2 := kwargs.get("keypoint_pairs_2")
        ) is not None and not is_iterable_of_tuple(keypoint_pairs_2, int):
            raise ValueError("keypoint_pairs_2 must be an iterable of integers or None")

        if keypoints_1 is not None and keypoints_2 is not None:
            names = _generate_names(keypoints_1, keypoints_2, element_wise=element_wise)
        elif keypoints_1 is not None and keypoint_pairs_2 is not None:
            names = _generate_names(
                keypoints_1, keypoint_pairs_2, element_wise=element_wise
            )
        elif keypoint_pairs_1 is not None and keypoint_pairs_2 is not None:
            names = _generate_names(
                keypoint_pairs_1, keypoint_pairs_2, element_wise=element_wise
            )
        elif keypoint_pairs_1 is not None and keypoints_2 is not None:
            names = _generate_names(
                keypoint_pairs_1, keypoints_2, element_wise=element_wise
            )
        else:
            raise ValueError(
                "invalid relational combination of keypoints_1/keypoint_pairs_1 and keypoints_2/keypoint_pairs_2"
            )
    else:
        if (keypoints := kwargs.get("keypoints")) is not None and not is_iterable_of(
            keypoints, int
        ):
            raise ValueError("keypoints must be an iterable of integers or None")
        if (
            keypoint_pairs := kwargs.get("keypoint_pairs")
        ) is not None and not is_iterable_of_tuple(keypoint_pairs, int):
            raise ValueError("keypoint_pairs must be an iterable of integers or None")
        if keypoints is not None:
            names = _generate_names(keypoints)
        elif keypoint_pairs is not None:
            names = _generate_names(keypoint_pairs)
        else:
            names = [func_name]
    if suffixes is None:
        return names
    return [f"{name}-{suffix}" for name in names for suffix in suffixes]


def generate_feature_names[**P](
    func: Callable[P, Shaped],
    *args: P.args,
    **kwargs: P.kwargs,
) -> list[str]:
    """
    Entrypoint for feature name generation.

    Parameters:
        func: The feature function to generate names for.
        *args: Additional positional arguments to pass to the feature function.
        **kwargs: Additional keyword arguments to pass to the feature function.

    Returns:
        A list of feature names corresponding to the function.

    See also:
        - :func:`feature_names` to generate feature names, using:
        - :func:`names` to generate feature names, or,
        - :func:`relational_names` to generate feature names for relational features
    """
    relational = any([("keypoints_" in kwarg) for kwarg in kwargs]) or any(
        [("keypoint_pairs_" in kwarg) for kwarg in kwargs]
    )
    dyadic = kwargs.get("trajectory_other") is not None
    if (suffixes := kwargs.pop("suffixes", None)) is not None and not is_iterable_of(
        suffixes, str
    ):
        raise ValueError("suffixes must be an iterable of strings")
    if suffixes is None:
        # otherwise, use the default suffixes from the function signature
        suffixes = signature(func).parameters.get("suffixes")
        if suffixes is not None:
            if not is_iterable_of(suffixes.default, str):
                raise ValueError(
                    "feature function with invalid default suffixes, suffixes must be an iterable of strings"
                )
            suffixes = suffixes.default
    return _feature_names(func, relational, dyadic, suffixes, *args, **kwargs)


def prune_feature_names(
    names: Iterable[str],
    *,
    keep: Iterable[str] | str | None = None,
    discard: Iterable[str] | str | None = None,
) -> list[str]:
    """Discard (or keep) feature names based on a list of names to keep or discard.

    Parameters
        names: The feature names to prune.
        keep: A list of feature name patterns to keep, irregardless of the match :code:`discard`.
        discard: A list of feature name patterns to discard.

    Returns
        A list of feature names after pruning.
    """

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

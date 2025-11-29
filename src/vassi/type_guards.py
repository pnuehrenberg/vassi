from collections.abc import Iterable, Mapping
from typing import Literal, TypedDict, TypeGuard

import numpy as np
import pandas as pd


def is_iterable_of[T](obj: ..., t: type[T]) -> TypeGuard[Iterable[T]]:
    return isinstance(obj, Iterable) and all(isinstance(item, t) for item in obj)


def is_tuple_of[T](obj: ..., t: type[T]) -> TypeGuard[tuple[T, T]]:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:  # pyright: ignore[reportUnknownArgumentType]
        return False
    return all(isinstance(item, t) for item in obj)  # pyright: ignore[reportUnknownVariableType]


def is_iterable_of_tuple[T](obj: ..., t: type[T]) -> TypeGuard[Iterable[tuple[T, T]]]:
    return isinstance(obj, Iterable) and all(is_tuple_of(item, t) for item in obj)


def is_valid_features_config(
    obj: ...,
) -> TypeGuard[
    Mapping[Literal["individual", "dyad"], Iterable[tuple[str, Mapping[str, object]]]]
]:
    """
    Checks if an object is a valid features configuration.

    A valid features configuration is a mapping with keys "individual" and/or "dyad".
    The values associated with these keys must be an iterable of tuples, where each
    tuple contains a string and a mapping of string to object.
    """
    if not isinstance(obj, Mapping):
        return False
    if not all(key in ["individual", "dyad"] for key in obj.keys()):  # pyright: ignore[reportUnknownVariableType]
        return False

    for features_iterable in obj.values():
        if not isinstance(features_iterable, Iterable):
            return False
        for item in features_iterable:
            if not isinstance(item, tuple) or len(item) != 2:  # pyright: ignore[reportUnknownArgumentType]
                return False

            name, config = item  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(name, str):
                return False
            if not isinstance(config, Mapping):
                return False
            if not all(isinstance(key, str) for key in config.keys()):  # pyright: ignore[reportUnknownVariableType]
                return False

    return True


class ClassificationData(TypedDict):
    categories: set[str]
    background_category: str
    timestamps: np.ndarray
    y_proba: np.ndarray
    y: np.ndarray
    y_gt: np.ndarray | None
    annotations: pd.DataFrame | None


def is_valid_classification_data(obj: ...) -> TypeGuard[ClassificationData]:
    if not isinstance(obj, Mapping):
        print("not a mapping")
        return False
    required_keys = {
        "categories",
        "background_category",
        "timestamps",
        "y_proba",
        "y",
        "y_gt",
        "annotations",
    }
    if not required_keys.issubset(obj.keys()):  # pyright: ignore[reportUnknownArgumentType]
        return False
    categories = obj.get("categories")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(categories, set):
        return False
    if not all(isinstance(category, str) for category in categories):  # pyright: ignore[reportUnknownVariableType]
        return False
    if not isinstance(obj.get("background_category"), str):  # pyright: ignore[reportUnknownMemberType]
        return False
    for key in ["timestamps", "y_proba", "y"]:
        if not isinstance(obj.get(key), np.ndarray):  # pyright: ignore[reportUnknownMemberType]
            return False
    y_gt = obj.get("y_gt", None)  # pyright: ignore[reportUnknownMemberType]
    if y_gt is not None and not isinstance(y_gt, np.ndarray):
        return False
    annotations = obj.get("annotations", None)  # pyright: ignore[reportUnknownMemberType]
    if annotations is not None and not isinstance(annotations, pd.DataFrame):
        return False
    return True


def is_mapping_of[TK, TV](
    obj: ...,
    tk: type[TK],
    tv: type[TV],
) -> TypeGuard[Mapping[TK, TV]]:
    if not isinstance(obj, Mapping):
        return False
    for key, value in obj.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(key, tk):
            return False
        if not isinstance(value, tv):
            return False
    return True


def is_mapping_of_mappings_of[TK1, TK2, TV](
    obj: ...,
    tk_1: type[TK1],
    tk_2: type[TK2],
    tv: type[TV],
) -> TypeGuard[Mapping[TK1, Mapping[TK2, TV]]]:
    if not isinstance(obj, Mapping):
        return False
    for key, value in obj.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(key, tk_1):
            return False
        if not is_mapping_of(value, tk_2, tv):
            return False
    return True

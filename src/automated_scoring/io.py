import os
import pickle
import tempfile
from collections.abc import ItemsView, Mapping
from typing import Any, Literal, Optional, overload

import h5py
import numpy as np
import pandas as pd
import yaml
from numpy.dtypes import StringDType  # type: ignore

from .data_structures.trajectory import Trajectory
from .dataset.types import (
    AnnotatedDataset,
    Dataset,
    Group,
)
from .dataset.utils import (
    GroupIdentifier,
    IndividualIdentifier,
)
from .logging import set_logging_level


def remove_cache(cache_file: str) -> bool:
    """
    Helper function to remove a cache file.

    Returns whether the file was successfully removed (:code:`False` if the file does not exist).

    Args:
        cache_file: The path to the cache file.
    """
    try:
        os.remove(cache_file)
        return True
    except FileNotFoundError:
        return False


def to_cache(
    obj: Any, cache_file: Optional[str] = None, directory: Optional[str] = None
) -> str:
    """
    Helper function to write an object to a cache file using pickle.

    Args:
        obj: The object to write.
        cache_file: The path to the cache file.
    """
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    else:
        directory = "."
    if cache_file is None:
        _, cache_file = tempfile.mkstemp(suffix=".cache", dir=directory)
    with open(cache_file, "wb") as cached:
        pickle.dump(obj, cached)
    return cache_file


def from_cache(cache_file: str):
    """
    Helper function to read an object from a cache file using pickle.

    Args:
        cache_file: The path to the cache file.
    """
    if not os.path.isfile(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found")
    with open(cache_file, "rb") as cached:
        return pickle.load(cached)


class _NoAliasDumper(yaml.SafeDumper):
    """
    Helper class to dump yaml without aliases.
    """

    def ignore_aliases(self, data):
        return True


def _construct_yaml_tuple(self, node):
    """
    Helper function to construct a tuple from a yaml sequence.
    """
    seq = self.construct_sequence(node)
    if seq and isinstance(seq, list):
        return tuple(seq)
    return seq


class _TupleLoader(yaml.SafeLoader):
    """
    Helper class to load all sequences in yaml as tuples.
    """

    pass


_TupleLoader.add_constructor("tag:yaml.org,2002:seq", _construct_yaml_tuple)


def to_yaml(dump: Any, *, file_name: str) -> None:
    with open(file_name, "w") as yaml_file:
        yaml_file.write(yaml.dump(dump, Dumper=_NoAliasDumper, sort_keys=False))


def from_yaml(file_name: str) -> Any:
    with open(file_name, "r") as yaml_file:
        return yaml.load(yaml_file, Loader=_TupleLoader)


def _is_string_array(array: np.ndarray):
    """
    Returns whether an array is of dtype np.dtypes.StringDType or np.str_.
    """
    if isinstance(array.dtype, StringDType):
        return True
    if array.dtype.type == np.str_:
        return True
    return False


BaseData = dict[str, np.ndarray] | np.ndarray
Data = BaseData | dict[str, "Data"]


def load_data(
    data_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> Data:
    def read_dataset(dataset: h5py.Dataset) -> np.ndarray:
        if dataset.dtype == "O":
            value = dataset.asstr()[:]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"invalid dataset value of type {type(value)}")
            return value.astype(StringDType)
        return dataset[:]

    if exclude is None:
        exclude = []
    with h5py.File(data_file, "r") as h5_file:
        if data_path is not None:
            h5_data = h5_file[data_path]
            if isinstance(h5_data, h5py.Group):
                h5_data = h5_data.items()
        else:
            h5_data = h5_file.items()
            data_path = ""
        if isinstance(h5_data, h5py.Dataset):
            return read_dataset(h5_data)
        if not isinstance(h5_data, ItemsView):
            raise ValueError(f"{data_path} is a {type(h5_data)} and not a h5py.Group")
        data: Data = {}
        for key, value in h5_data:
            if key in exclude:
                continue
            if isinstance(value, h5py.Group):
                data[key] = load_data(
                    data_file, os.path.join(data_path, key), exclude=exclude
                )
                continue
            data[key] = read_dataset(value)
    return data


def save_data(
    data_file: str,
    data: Mapping[str, np.ndarray],
    data_path: str | None = None,
    exclude: list[str] | None = None,
) -> None:
    if exclude is None:
        exclude = []
    with h5py.File(data_file, "a") as h5_file:
        if data_path is not None:
            if data_path in h5_file:
                h5_data = h5_file[data_path]
                if not isinstance(h5_data, h5py.Group):
                    raise ValueError("cannot overwrite non-group element with group")
            else:
                h5_data = h5_file.create_group(data_path)
        else:
            h5_data = h5_file
        for key, value in data.items():
            dtype = None
            if _is_string_array(value):
                dtype = h5py.string_dtype()
                value = value.astype(dtype)
            if key not in h5_data:
                h5_data.create_dataset(key, data=value, dtype=dtype)
                continue
            h5_dataset = h5_data[key]
            if not isinstance(h5_dataset, h5py.Dataset):
                raise ValueError("cannot overwrite non-dataset element with data")
            if h5_dataset.shape == value.shape:
                h5_dataset[:] = value
                continue
            del h5_data[key]
            h5_data.create_dataset(key, data=value, dtype=dtype)


def save_trajectories(
    trajectory_file: str,
    trajectories: dict[int | str, Trajectory],
    prefix: str | None = None,
    exclude: list[str] | None = None,
) -> None:
    if prefix is None:
        prefix = ""
    identities = np.asarray(list(trajectories.keys()))
    save_data(
        trajectory_file, {"_identities": np.asarray(identities)}, data_path=prefix
    )
    for identity in identities:
        if str(identity) == "_identity":
            raise ValueError("invalid use of reserved identifier '_identity'")
        save_data(
            trajectory_file,
            trajectories[identity].data,
            os.path.join(prefix, str(identity)),
            exclude=exclude,
        )


def load_trajectories(
    trajectory_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> dict[IndividualIdentifier, Trajectory]:
    # delayed import to avoid circular imports
    from .data_structures.trajectory import Trajectory

    with h5py.File(trajectory_file, "r") as h5_file:
        if data_path is not None:
            h5_data = h5_file[data_path]
        else:
            h5_data = h5_file
            data_path = ""
        if not isinstance(h5_data, (h5py.File, h5py.Group)):
            raise ValueError(
                f"cannot read trajectories from {data_path} of type {type(h5_data)}"
            )
        identities = np.asarray(list(h5_data.keys()))
    try:
        identities = load_data(trajectory_file, os.path.join(data_path, "_identities"))
        if not isinstance(identities, np.ndarray):
            raise ValueError(f"invalid identities of type {type(identities)}")
    except KeyError:
        # fall back to h5_data keys as identities
        pass
    trajectories = {}
    for identity in np.asarray(identities).tolist():
        data = load_data(
            trajectory_file, os.path.join(data_path, str(identity)), exclude=exclude
        )
        if isinstance(data, dict):
            trajectories[identity] = Trajectory(
                data={key: np.asarray(value) for key, value in data.items()}
            )
            continue
        raise ValueError(f"invalid trajectory data for trajectory {identity}")
    return trajectories


def save_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    directory: str = ".",
    observation_suffix: Literal["annotations", "predictions"] = "annotations",
) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    observation_file = os.path.join(
        directory, f"{dataset_name}_{observation_suffix}.csv"
    )
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    observations = None
    if isinstance(dataset, AnnotatedDataset):
        observations = dataset.observations
    if observations is not None:
        observations.to_csv(observation_file, index=False)
    save_data(trajectory_file, {"_groups": np.asarray(dataset.identifiers)})
    for identifier, group in dataset:
        if str(identifier) == "_groups":
            raise ValueError("group identifier uses reserved key '_groups'")
        save_trajectories(trajectory_file, group.trajectories, prefix=str(identifier))


@overload
def load_dataset(
    dataset_name: str,
    *,
    directory: str = ".",
    target: Literal["individual", "dyad"],
    load_observations: Literal[True] = True,
    categories: Optional[tuple[str, ...]] = None,
    background_category: str,
    observation_suffix: str = "annotations",
) -> AnnotatedDataset: ...


@overload
def load_dataset(
    dataset_name: str,
    *,
    directory: str = ".",
    target: Literal["individual", "dyad"],
    load_observations: Literal[False],
    categories: Optional[tuple[str, ...]] = None,
    background_category: str,
    observation_suffix: str = "annotations",
) -> Dataset: ...


def load_dataset(
    dataset_name: str,
    *,
    directory: str = ".",
    target: Literal["individual", "dyad"],
    load_observations: bool = True,
    categories: Optional[tuple[str, ...]] = None,
    background_category: str,
    observation_suffix: str = "annotations",
) -> AnnotatedDataset | Dataset:
    observation_file = os.path.join(
        directory, f"{dataset_name}_{observation_suffix}.csv"
    )
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    observations = None
    if load_observations and os.path.exists(observation_file):
        observations = pd.read_csv(observation_file)
    elif load_observations:
        raise FileNotFoundError(f"{observation_file} does not exist.")
    groups: dict[GroupIdentifier, Group] = {}
    identifiers = load_data(trajectory_file, "_groups")
    if not isinstance(identifiers, np.ndarray):
        raise ValueError(
            f"invalid dataset file with group identifiers of type {type(groups)}"
        )
    for identifier in identifiers.tolist():
        group = Group(
            load_trajectories(trajectory_file, str(identifier)),
            target=target,
        )
        groups[identifier] = group
    if observations is not None:
        if categories is None:
            categories = tuple(
                sorted(set([*np.unique(observations["category"]), background_category]))
            )
            set_logging_level().warning(
                f"Loading categories ({', '.join(categories)}) from observations file, specify categories argument if incomplete."
            )
        return AnnotatedDataset(
            groups,
            target=target,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )
    return Dataset(groups, target=target)

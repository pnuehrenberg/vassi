import os
from collections.abc import ItemsView
from typing import TYPE_CHECKING, Literal, Mapping, Optional

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from numpy.dtypes import StringDType  # type: ignore
from numpy.typing import NDArray

from .data_structures.trajectory import Trajectory
from .dataset import (
    AnnotatedGroup,
    Dataset,
    Group,
    GroupIdentifier,
    IndividualIdentifier,
)


def _is_string_array(array: NDArray):
    """
    Returns whether an array is of dtype np.dtypes.StringDType or np.str_.
    """
    if isinstance(array.dtype, StringDType):
        return True
    if array.dtype.type == np.str_:
        return True
    return False


BaseData = dict[str, NDArray] | NDArray
Data = BaseData | dict[str, "Data"]


def load_data(
    data_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> Data:
    """
    Load data from a HDF5 file that was saved with save_data, save_trajectories or save_dataset.

    Parameters
    ----------
    data_file: str
        The file to load data from.
    data_path: str, optional
        The path to the data to load.
    exclude: list[str], optional
        The keys to exclude from the data.

    Returns
    -------
    Data
        The loaded data.

    Raises
    ------
    ValueError
        If the data is not a valid data structure.
    """

    def read_dataset(dataset: h5py.Dataset) -> NDArray:
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
    data: Mapping[str, NDArray],
    data_path: str | None = None,
    exclude: list[str] | None = None,
) -> None:
    """
    Save data to a HDF5 file.

    Parameters
    ----------
    data_file: str
        The file to save data to.
    data: Mapping[str, NDArray]
        The data to save.
    data_path: str, optional
        The path to the data to save.
    exclude: list[str], optional
        The keys to exclude from the data.

    Raises
    ------
    ValueError
        If existing data (at data_path) that is not a h5py.Group should be overwritten.
    ValueError
        If existing data (at data_path) that is not a h5py.Dataset is overwritten with data.
    """
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
    """
    Save trajectories to a HDF5 file.

    Parameters
    ----------
    trajectory_file: str
        The file to save trajectories to.
    trajectories: dict[int | str, Trajectory]
        The trajectories to save.
    prefix: str, optional
        The prefix to use as data path in the HDF5 file.
    exclude: list[str], optional
        The keys (identities) to exclude from the trajectories when saving.

    Raises
    ------
    ValueError
        If the trajectories dict containts a key (identity) that is '_identity'.
    """
    if prefix is None:
        prefix = ""
    identities = np.asarray(list(trajectories.keys()))
    save_data(
        trajectory_file, {"_identities": np.asarray(identities)}, data_path=prefix
    )
    for identity in identities:
        if str(identity) == "_identity":
            raise ValueError(f"invalid use of reserved key {identity}")
        save_data(
            trajectory_file,
            trajectories[identity].data,
            os.path.join(prefix, str(identity)),
            exclude=exclude,
        )


def load_trajectories(
    trajectory_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> dict[IndividualIdentifier, Trajectory]:
    """
    Load trajectories from a HDF5 file that was saved with save_trajectories.

    Parameters
    ----------
    trajectory_file: str
        The file to load trajectories from.
    data_path: str, optional
        The path (in the HDF5 file) to the trajectories to load.
    exclude: list[str], optional
        The keys (identities) to exclude from the trajectories when loading.

    Returns
    -------
    dict[Identity, Trajectory]
        The loaded trajectories.

    Raises
    ------
    ValueError
        If the trajectories are not a valid trajectory data structure.
    """
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
    for identity in identities.tolist():  # type: ignore  # see check above
        data = load_data(
            trajectory_file, os.path.join(data_path, str(identity)), exclude=exclude
        )
        if isinstance(data, dict) and all(
            [isinstance(value, np.ndarray) for value in data.values()]
        ):
            trajectories[identity] = Trajectory(data=data)  # type: ignore  # see above
            continue
        raise ValueError(f"invalid trajectory data for trajectory {identity}")
    return trajectories


def save_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    directory: str = ".",
    observation_type: Literal["annotations", "predictions"] = "annotations",
) -> None:
    """
    Save a dataset (all groups with respective trajectories) to HDF5 file file in the specified directory.

    If the dataset has observations (either annotations or predictions), they are saved to a CSV file alongside the trajectories.

    Parameters
    ----------
    dataset: Dataset
        The dataset to save.
    dataset_name: str
        The name of the dataset.
    directory: str, optional
        The directory to save the dataset to.
    observation_type: Literal["annotations", "predictions"], optional
        The type of observations to save. This is used as the file name suffix.

    Raises
    ------
    ValueError
        If the dataset containts a group key that is '_groups'.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    observation_file = os.path.join(directory, f"{dataset_name}_{observation_type}.csv")
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    observations = dataset.get_observations()
    if observations is not None:
        observations.to_csv(observation_file, index=False)
    group_keys = (
        list(dataset.groups.keys())
        if isinstance(dataset.groups, dict)
        else list(range(len(dataset.groups)))
    )
    save_data(trajectory_file, {"_groups": np.asarray(group_keys)})
    for group_key, group in (
        dataset.groups.items()
        if isinstance(dataset.groups, dict)
        else enumerate(dataset.groups)
    ):
        if str(group_key) == "_groups":
            raise ValueError(f"invalid use of reserved key {group_key}")
        save_trajectories(trajectory_file, group.trajectories, prefix=str(group_key))


def load_dataset(
    dataset_name: str,
    *,
    directory: str = ".",
    target: Literal["individuals", "dyads"],
    load_observations: bool = True,
    categories: Optional[tuple[str, ...]] = None,
    observation_type: Literal["annotations", "predictions"] = "annotations",
) -> Dataset:
    """
    Load a dataset (all groups with respective trajectories) from the HDF5 file with dataset_name in the specified directory.

    Groups are created with either individuals or dyads as target.
    If observations should be loaded, the list of categories is read from the observations file. If other categories should be
    included, specify them as categories argument.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.
    directory: str, optional
        The directory to load the dataset from.
    target: Literal["individuals", "dyads"], optional
        The target of the groups.
    load_observations: bool, optional
        Whether to load observations.
    categories: tuple[str, ...], optional
        The categories to include in the observations.
    observation_type: Literal["annotations", "predictions"], optional
        The type of observations to load. This is used as the file name suffix.

    Returns
    -------
    Dataset
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the observations file does not exist but load_observations is True.
    ValueError
        If the dataset file contains invalid data.
    """
    observation_file = os.path.join(directory, f"{dataset_name}_{observation_type}.csv")
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    observations = None
    if load_observations and os.path.exists(observation_file):
        observations = pd.read_csv(observation_file).set_index("group")
    elif load_observations:
        raise FileNotFoundError(f"{observation_file} does not exist.")
    groups: dict[GroupIdentifier, Group | AnnotatedGroup] = {}
    group_keys = load_data(trajectory_file, "_groups")
    if not isinstance(group_keys, np.ndarray):
        raise ValueError(f"invalid dataset file with group keys of type {type(groups)}")
    if categories is None and observations is not None:
        categories = tuple(np.unique(observations["category"]))
        logger.warning(
            f"Loading categories ({", ".join(categories)}) from observations file, specify categories argument if incomplete."
        )
    for group_key in group_keys.tolist():
        if observations is not None:
            if TYPE_CHECKING:
                assert categories is not None
            trajectories = load_trajectories(trajectory_file, str(group_key))
            str_identities = any(
                [isinstance(identity, str) for identity in trajectories.keys()]
            )
            observations_group = observations.loc[group_key].reset_index()
            if str_identities:
                observations_group["actor"] = np.asarray(
                    observations_group["actor"]
                ).astype(StringDType)
                observations_group["actor"] = observations_group["actor"].astype(
                    pd.CategoricalDtype()
                )
            if str_identities and "recipient" in observations_group.columns:
                observations_group["recipient"] = np.asarray(
                    observations_group["recipient"]
                ).astype(StringDType)
                observations_group["recipient"] = observations_group[
                    "recipient"
                ].astype(pd.CategoricalDtype())
            group = AnnotatedGroup(
                trajectories,
                target=target,
                observations=observations_group,
                categories=categories,
            )
        else:
            group = Group(
                load_trajectories(trajectory_file, str(group_key)),
                target=target,
            )
        groups[group_key] = group
    return Dataset(groups)

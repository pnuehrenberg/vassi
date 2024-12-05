import os
from collections.abc import ItemsView
from typing import Literal, Mapping, Optional

import h5py
import numpy as np
import pandas as pd
from numpy.dtypes import StringDType  # type: ignore
from numpy.typing import NDArray

from .data_structures.trajectory import Trajectory
from .dataset import AnnotatedGroup, Dataset, Group
from .dataset.types.utils import Identity


def is_string_array(array: NDArray):
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
):
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
            if is_string_array(value):
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
):
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
) -> dict[Identity, Trajectory]:
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
    annotation_type: Literal["annotations", "predictions"] = "annotations",
):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    annotation_file = os.path.join(directory, f"{dataset_name}_{annotation_type}.csv")
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    annotations = dataset.get_annotations()
    if annotations is not None:
        annotations.to_csv(annotation_file, index=False)
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
    load_annotations: bool = True,
    categories: Optional[tuple[str, ...]] = None,
    annotation_type: Literal["annotations", "predictions"] = "annotations",
) -> Dataset:
    annotation_file = os.path.join(directory, f"{dataset_name}_{annotation_type}.csv")
    trajectory_file = os.path.join(directory, f"{dataset_name}_trajectories.h5")
    annotations = None
    if load_annotations and os.path.exists(annotation_file):
        annotations = pd.read_csv(annotation_file).set_index("group")
    elif load_annotations:
        raise FileNotFoundError(f"{annotation_file} does not exist.")
    groups: dict[Identity, Group | AnnotatedGroup] = {}
    group_keys = load_data(trajectory_file, "_groups")
    if not isinstance(group_keys, np.ndarray):
        raise ValueError(f"invalid dataset file with group keys of type {type(groups)}")
    for group_key in group_keys.tolist():
        if annotations is not None:
            if categories is None:
                raise ValueError("specify categories")
            trajectories = load_trajectories(trajectory_file, str(group_key))
            str_identities = any(
                [isinstance(identity, str) for identity in trajectories.keys()]
            )
            annotations_group = annotations.loc[group_key].copy()
            if str_identities:
                annotations_group["actor"] = np.asarray(
                    annotations_group["actor"]
                ).astype(StringDType)
                annotations_group["actor"] = annotations_group["actor"].astype(
                    pd.CategoricalDtype()
                )
            if str_identities and "recipient" in annotations_group.columns:
                annotations_group["recipient"] = np.asarray(
                    annotations_group["recipient"]
                ).astype(StringDType)
                annotations_group["recipient"] = annotations_group["recipient"].astype(
                    pd.CategoricalDtype()
                )
            group = AnnotatedGroup(
                trajectories,
                target=target,
                annotations=annotations_group,
                categories=categories,
            )
        else:
            group = Group(
                load_trajectories(trajectory_file, str(group_key)),
                target=target,
            )
        groups[group_key] = group
    return Dataset(groups)

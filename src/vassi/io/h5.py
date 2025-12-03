from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from ..data_structures import Trajectory
from ..type_guards import is_mapping_of


def write_h5_array(
    data_file: str | Path, *, array: np.ndarray, data_path: str, **attrs: ...
) -> None:
    data_file = Path(data_file)
    with h5py.File(str(data_file), "a") as h5_file:
        if data_path in h5_file:
            h5_data = h5_file[data_path]
            if not isinstance(h5_data, h5py.Dataset):
                raise ValueError("cannot overwrite non-dataset element with array")
            if h5_data.shape != array.shape:
                raise ValueError(
                    "cannot overwrite dataset with array of different shape"
                )
            h5_data[:] = array
        else:
            if np.issubdtype(array.dtype, np.str_):
                array = array.astype(np.bytes_)
            h5_data = h5_file.create_dataset(
                data_path,
                data=array,
            )
        for key, attr in attrs.items():
            try:
                h5_data.attrs[key] = attr
            except TypeError:
                attr = np.array(attr).astype(np.bytes_)
                h5_data.attrs[key] = attr


def write_h5_attrs(data_file: str | Path, *, data_path: str, **attrs: ...) -> None:
    with h5py.File(str(data_file), "a") as h5_file:
        if data_path not in h5_file:
            raise ValueError(
                f"cannot write attributes to non-existent path '{data_path}'"
            )
        h5_data = h5_file[data_path]
        for key, attr in attrs.items():
            try:
                h5_data.attrs[key] = attr
            except TypeError:
                attr = np.array(attr).astype(np.bytes_)
                h5_data.attrs[key] = attr


def read_h5_attrs(data_file: str | Path, *, data_path: str) -> dict[str, object]:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"file {data_file} does not exist")
    with h5py.File(str(data_file), "r") as h5_file:
        if data_path not in h5_file:
            raise ValueError(
                f"cannot read attributes from non-existent path '{data_path}'"
            )
        h5_data = h5_file[data_path]
        attrs: dict[str, object] = dict(h5_data.attrs.items())
    for key, value in attrs.items():
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.bytes_):
                attrs[key] = list(map(bytes.decode, value.tolist()))
            else:
                attrs[key] = value.tolist()
    return attrs


def write_h5_data(
    data_file: str | Path,
    *,
    data: Mapping[str, np.ndarray],
    data_path: str | None,
    **attrs: ...,
) -> None:
    if data_path is None:
        data_path = ""
    posix_data_path = PurePosixPath(data_path)
    for key, array in data.items():
        write_h5_array(
            data_file, array=array, data_path=str(posix_data_path / str(key)), key=key
        )
    write_h5_attrs(data_file, data_path=data_path, **attrs)


def read_h5_array(
    data_file: str | Path, *, data_path: str
) -> tuple[None | object, np.ndarray]:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"file {data_file} does not exist")
    with h5py.File(str(data_file), "r") as h5_file:
        if data_path not in h5_file:
            raise KeyError(f"dataset {data_path} not found in file {data_file}")
        h5_data = h5_file[data_path]
        if not isinstance(h5_data, h5py.Dataset):
            raise ValueError(
                f"element {data_path} in file {data_file} is not a dataset"
            )
        key = None
        if "key" in h5_data.attrs:
            key = h5_data.attrs["key"]
        return key, h5_data[:]


def get_h5_keys(data_file: str | Path, *, data_path: str | None) -> list[str]:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"file {data_file} does not exist")
    with h5py.File(str(data_file), "r") as h5_file:
        if data_path is None:
            return list(h5_file.keys())
        posix_data_path = PurePosixPath(data_path)
        h5_data = h5_file[data_path]
        if not isinstance(h5_data, h5py.Group):
            raise ValueError(f"element {data_path} in file {data_file} is not a group")
        return [
            str((posix_data_path / key).relative_to(posix_data_path))
            for key in h5_data.keys()
        ]


def read_h5_data(
    data_file: str | Path, *, data_path: str | None
) -> dict[object, np.ndarray]:
    if data_path is None:
        data_path = ""
    posix_data_path = PurePosixPath(data_path)
    keys = get_h5_keys(data_file, data_path=data_path)
    data = [
        read_h5_array(data_file, data_path=str(posix_data_path / str(key)))
        for key in keys
    ]
    data_dict = dict(data)
    if len(data_dict) != len(data):
        raise ValueError("can only read data from h5 with unique key attributes")
    return data_dict


def load_trajectories_legacy(
    trajectory_file: str | Path,
) -> dict[Hashable, dict[Hashable, Trajectory]]:
    trajectories: dict[Hashable, dict[Hashable, Trajectory]] = {}
    _, groups = read_h5_array(trajectory_file, data_path="_groups")
    if groups.dtype == "O":
        groups = np.array(list(map(bytes.decode, groups)))
    for group in groups:
        trajectories[group] = {}
        _, identities = read_h5_array(
            trajectory_file,
            data_path=str(PurePosixPath(".") / str(group) / "_identities"),
        )
        if identities.dtype == "O":
            identities = np.array(list(map(bytes.decode, identities)))
        for identifier in identities:
            keys = get_h5_keys(
                trajectory_file,
                data_path=str(PurePosixPath(".") / str(group) / str(identifier)),
            )
            trajectories[group][identifier] = Trajectory(
                data={
                    key: read_h5_array(
                        trajectory_file,
                        data_path=str(
                            PurePosixPath(".") / str(group) / str(identifier) / key
                        ),
                    )[1]
                    for key in keys
                }
            )
    return trajectories


def load_trajectories(
    trajectory_file: str | Path,
) -> dict[Hashable, dict[Hashable, Trajectory]]:
    trajectories: dict[Hashable, dict[Hashable, Trajectory]] = {}
    groups = read_h5_attrs(trajectory_file, data_path=".")["groups"]
    if not isinstance(groups, Iterable):
        raise ValueError("expected groups attribute to be iterable")
    for group in groups:
        trajectories[group] = {}
        identifiers = read_h5_attrs(
            trajectory_file,
            data_path=str(PurePosixPath(".") / str(group)),
        )["identifiers"]
        if not isinstance(identifiers, Iterable):
            raise ValueError("expected identifiers attribute to be iterable")
        for identifier in identifiers:
            data = dict(
                read_h5_data(
                    trajectory_file,
                    data_path=str(PurePosixPath(".") / str(group) / str(identifier)),
                )
            )
            if not is_mapping_of(data, str, np.ndarray):
                raise ValueError(
                    "expected data to be a mapping of strings to numpy arrays"
                )
            trajectories[group][identifier] = Trajectory(data=data)
    return trajectories


def save_trajectories(
    trajectories: dict[Hashable, dict[Hashable, Trajectory]],
    trajectory_file: str | Path,
) -> None:
    write_h5_attrs(
        trajectory_file, data_path=".", groups=np.array(list(trajectories.keys()))
    )
    for group, trajectories_group in trajectories.items():
        for identifier, trajectory in trajectories_group.items():
            write_h5_data(
                trajectory_file,
                data_path=str(PurePosixPath(".") / str(group) / str(identifier)),
                data=trajectory.data,
            )
        write_h5_attrs(
            trajectory_file,
            data_path=str(PurePosixPath(".") / str(group)),
            identifiers=np.array(list(trajectories_group.keys())),
        )

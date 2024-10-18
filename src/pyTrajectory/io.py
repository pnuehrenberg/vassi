import os
from collections.abc import ItemsView
from typing import Mapping

import h5py
from numpy.typing import NDArray

from .data_structures.trajectory import Trajectory


def load_data(
    data_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> dict[str, NDArray]:
    if exclude is None:
        exclude = []
    with h5py.File(data_file, "r") as h5_file:
        if data_path is not None:
            h5_data = h5_file[data_path]
            if isinstance(h5_data, h5py.Group):
                h5_data = h5_data.items()
        else:
            h5_data = h5_file.items()
        if not isinstance(h5_data, ItemsView):
            raise ValueError(f"{data_path} is a {type(h5_data)} and not a h5py.Group")
        data: dict[str, NDArray] = {}
        for key, value in h5_data:
            if key in exclude:
                continue
            data[key] = value[:]
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
            h5_data = h5_file.create_group(data_path)
        else:
            h5_data = h5_file
        for key, value in data.items():
            h5_data.create_dataset(key, data=value)


def save_trajectories(
    trajectory_file: str,
    trajectories: dict[str, Trajectory],
    prefix: str | None = None,
    exclude: list[str] | None = None,
):
    if prefix is None:
        prefix = ""
    for identity in trajectories:
        save_data(
            trajectory_file,
            trajectories[identity].data,
            os.path.join(prefix, str(identity)),
            exclude=exclude,
        )


def load_trajectories(
    trajectory_file: str, data_path: str | None = None, exclude: list[str] | None = None
) -> dict[str, Trajectory]:
    with h5py.File(trajectory_file, "r") as h5_file:
        if data_path is not None:
            h5_data = h5_file[data_path]
            if not isinstance(h5_data, h5py.Group):
                raise ValueError(
                    f"{data_path} is a {type(h5_data)} and not a h5py.Group"
                )
        else:
            h5_data = h5_file
            data_path = ""
        identities: list[str] = list(h5_data.keys())
    return {
        identity: Trajectory(
            data=load_data(
                trajectory_file, os.path.join(data_path, str(identity)), exclude=exclude
            )
        )
        for identity in identities
    }

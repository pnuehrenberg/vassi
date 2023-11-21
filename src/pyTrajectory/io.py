import os
import h5py

from pyTrajectory.trajectory import Trajectory
from pyTrajectory.trajectory_linking import link_trajectories


def load_data(data_file, data_path=None, exclude=None):
    if exclude is None:
        exclude = []
    with h5py.File(data_file, 'r') as h5_file:
        if data_path is not None:
            h5_data = h5_file[data_path].items()
        else:
            h5_data = h5_file.items()
        data = {}
        for key, value in h5_data:
            if key in exclude:
                continue
            data[key] = value[:]
    return data


def save_data(data_file, data, data_path=None, exclude=None):
    if exclude is None:
        exclude = []
    with h5py.File(data_file, 'a') as h5_file:
        if data_path is not None:
            h5_data = h5_file.create_group(data_path)
        else:
            h5_data = h5_file
        for key, value in data.items():
            if value is None:
                continue
            h5_data.create_dataset(key, data=value)


def save_trajectories(trajectory_file, trajectories, prefix=None, exclude=None):
    if prefix is None:
        prefix = ''
    for identity in trajectories:
        save_data(trajectory_file,
                  trajectories[identity].data,
                  os.path.join(prefix, str(identity)),
                  exclude=exclude)


def load_trajectories(trajectory_file, prefix=None, exclude=None):
    trajectories = {}
    with h5py.File(trajectory_file, 'r') as h5_file:
        if prefix is not None:
            h5_data = h5_file[prefix]
        else:
            h5_data = h5_file
        identities = list(h5_data.keys())
    if prefix is None:
        prefix = ''
    trajectories = {}
    for identity in identities:
        data = load_data(trajectory_file,
                         os.path.join(prefix, str(identity)),
                         exclude=exclude)
        trajectories[identity] = Trajectory(data=data)
    return trajectories

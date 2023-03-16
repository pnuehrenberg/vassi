import _pickle
import deepdish as dd

import pyTrajectory.trajectory


def save_trajectories(trajectories, file_name):
    trajectory_data = [trajectory.data for trajectory in trajectories]
    dd.io.save(file_name, trajectory_data)
    return True


def load_trajectories(file_name):
    trajectory_data = dd.io.load(file_name)
    trajectories = [pyTrajectory.trajectory.Trajectory().load(file_path=None, data=trajectory_data)
                    for trajectory_data in trajectory_data]
    return trajectories


def load(file_name):
    with open(file_name, 'rb') as fid:
        dump = _pickle.load(fid)
    return dump

def save(dump, file_name):
    with open(file_name, 'wb') as fid:
        _pickle.dump(dump, fid)
    return True

import numpy as np

import pyTrajectory.config


def format_arg(arg):
    if arg is None:
        return None
    if type(arg) in [int, float]:
        return arg
    arg = np.asarray(arg)
    if len(arg.shape) == 0:
        return arg.item()
    if len(arg.shape) > 1 and arg.shape[0] == 1:
        return arg[0]
    return arg


class Instance(object):

    def __init__(self, **kwargs):
        for key, arg in kwargs.items():
            if key not in pyTrajectory.config.cfg.trajectory_keys:
                raise KeyError
            setattr(self, key, format_arg(arg))
        for key in set(pyTrajectory.config.cfg.trajectory_keys) - set(kwargs.keys()):
            setattr(self, key, None)

    def __getitem__(self, key):
        if key not in pyTrajectory.config.cfg.trajectory_keys:
            raise KeyError
        return getattr(self, key)

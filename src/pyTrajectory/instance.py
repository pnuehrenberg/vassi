import numpy as np

import .config


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
            if key not in config.KEYS:
                raise NotImplementedError
            setattr(self, key, format_arg(arg))
        for key in set(config.KEYS) - set(kwargs.keys()):
            setattr(self, key, None)

    def __getitem__(self, key):
        if key not in config.KEYS:
            raise NotImplementedError
        return getattr(self, key)

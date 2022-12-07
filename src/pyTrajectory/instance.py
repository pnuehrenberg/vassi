import numpy as np


KEYS = ('time_stamp',
        'position',
        'pose',
        'segmentation',
        'bbox',
        'score',
        'category')


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

    __slots__ = KEYS

    def __init__(self, **kwargs):
        for key, arg in kwargs.items():
            if key not in self.__slots__:
                raise NotImplementedError
            setattr(self, key, format_arg(arg))
        for key in set(self.__slots__) - set(kwargs.keys()):
            setattr(self, key, None)

    def __getitem__(self, key):
        if key not in self.__slots__:
            raise NotImplementedError
        return getattr(self, key)

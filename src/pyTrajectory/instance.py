import numpy as np


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

    __slots__ = ('time_stamp',
                 'position',
                 'pose',
                 'segmentation',
                 'bbox',
                 'score',
                 'category')

    def __init__(self,
                 time_stamp=None,
                 position=None,
                 pose=None,
                 segmentation=None,
                 bbox=None,
                 score=None,
                 category=None):
        self.time_stamp = format_arg(time_stamp)
        self.position = format_arg(position)
        self.pose = format_arg(pose)
        self.segmentation = format_arg(segmentation)
        self.bbox = format_arg(bbox)
        self.score = format_arg(score)
        self.category = format_arg(category)

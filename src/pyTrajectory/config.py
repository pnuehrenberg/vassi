import yaml

from copy import deepcopy
from typing import get_type_hints

import pyTrajectory.instance


class Config(object):

    def __init__(self, **kwargs):
        for key, arg in kwargs.items():
            setattr(self, key, arg)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __call__(self):
        cfg = deepcopy(self)
        for key, value in cfg.items():
            if not isinstance(value, Config):
                continue
            cfg[key] = value()
        return cfg.__dict__

    def __str__(self):
        return yaml.dump(self())

    def __getitem__(self, key):
        if not key in self.keys():
            raise KeyError
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if not key in self.keys():
            raise KeyError
        self.__dict__[key] = value

    def check_type_hint(self, value, value_type):
        if not callable(value):
            return False
        for argument, argument_type in get_type_hints(value).items():
            if not issubclass(value_type, argument_type):
                return False
        return True

    def apply(self, configurable):
        cfg = deepcopy(self)
        for key, value in cfg.items():
            if not callable(value):
                continue
            if isinstance(value, Config):
                cfg[key] = value.apply(configurable)
                continue
            if not cfg.check_type_hint(value, type(configurable)):
                continue
            cfg[key] = value(configurable)
        return cfg


cfg = Config()

# define data keys
cfg.trajectory_keys = \
    ('time_stamp',
     'pose',
     'bbox',
     'score',
     'category')

# these keys are used for visualization and analyses, modify accordingly
cfg.key_time_stamp = 'time_stamp'
cfg.key_category = 'category'
cfg.key_score = 'score'
cfg.key_box = 'bbox'
cfg.key_keypoints = 'pose'
cfg.key_keypoints_line = 'pose'

# config for visualization figures
cfg.figure = Config()
cfg.figure.figsize =(1, 1)
cfg.figure.dpi = 600
cfg.figure.padding = 1
cfg.figure.width = 200
cfg.figure.height = 200

# config for instance visualization
cfg.instance = Config()

cfg.instance.box = Config()
cfg.instance.box.linestyle = '--'
cfg.instance.box.linewidth = 0.5
cfg.instance.box.joinstyle = 'round'
cfg.instance.box.capstyle = 'round'
cfg.instance.box.edgecolor = 'k'
cfg.instance.box.facecolor = (0, 0, 0, 0)
cfg.instance.box.zorder = 0

cfg.instance.keypoints_line = Config()
cfg.instance.keypoints_line.alpha = 0.6
cfg.instance.keypoints_line.color = 'k'
cfg.instance.keypoints_line.capstyle = 'round'
cfg.instance.keypoints_line.zorder = 0

cfg.instance.keypoints = Config()
cfg.instance.keypoints.s = 3
cfg.instance.keypoints.edgecolor = 'k'
cfg.instance.keypoints.lw = 0.5

def get_keypoints_facecolor(instance: pyTrajectory.instance.Instance):
    global cfg
    return ['r'] + ['w'] * (len(instance[cfg.key_keypoints]) - 1)

cfg.instance.keypoints.facecolor = get_keypoints_facecolor

cfg.instance.label = Config()
cfg.instance.label.size = 5

# config for trajectory visualization
cfg.trajectory = Config()

cfg.trajectory.box = Config()
cfg.trajectory.box.edgecolor = (0, 0, 0, 0.1)
cfg.trajectory.box.facecolor = (0, 0, 0, 0)
cfg.trajectory.box.lw = 0.1

cfg.trajectory.keypoints = Config()
cfg.trajectory.keypoints.lw = 0.1
cfg.trajectory.keypoints.alpha = 0.1
cfg.trajectory.keypoints.capstyle = 'round'

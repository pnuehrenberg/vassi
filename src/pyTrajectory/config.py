from copy import deepcopy

from .instance import Instance


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
        return self.__dict__

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
            if type(value) is Config:
                cfg[key] = value.apply(configurable)
                continue
            if not cfg.check_type_hint(value, type(configurable)):
                continue
            cfg[key] = value(configurable)
        return cfg


cfg = Config()

cfg.trajectory_keys = \
    ('time_stamp',
     'pose',
     'bbox',
     'score',
     'category')

cfg.key_time_stamp = 'time_stamp'
cfg.key_category = 'category'
cfg.key_score = 'score'
cfg.key_box = 'bbox'
cfg.key_keypoints = 'pose'
cfg.key_keypoints_line = 'pose'

cfg.box = Config()
cfg.box.linestyle = '--'
cfg.box.linewidth = 0.5
cfg.box.joinstyle = 'round'
cfg.box.capstyle = 'round'
cfg.box.edgecolor = 'k'
cfg.box.facecolor = (0, 0, 0, 0)
cfg.box.zorder = 0

cfg.keypoints_line = Config()
cfg.keypoints_line.alpha = 0.6
cfg.keypoints_line.color = 'k'
cfg.keypoints_line.capstyle = 'round'
cfg.keypoints_line.zorder = 0

cfg.keypoints = Config()
cfg.keypoints.s = 3
cfg.keypoints.edgecolor = 'k'
cfg.keypoints.lw = 0.5

def get_keypoints_facecolor(instance: pyTrajectory.instance.Instance):
    global cfg
    return ['r'] + ['w'] * (len(instance[cfg.key_keypoints]) - 1)

cfg.keypoints.facecolor = get_keypoints_facecolor

cfg.label = Config()
cfg.label.size = 5

cfg.figure = Config()
cfg.figure.figsize =(1, 1)
cfg.figure.dpi = 600
cfg.figure.padding = 1
cfg.figure.width = 200
cfg.figure.height = 200

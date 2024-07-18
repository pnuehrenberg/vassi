import yaml

from copy import deepcopy

import pyTrajectory.instance


class Config:
    """A configuration object for storing key-value pairs.

    This class allows for the creation and manipulation of configuration objects.

    Examples
    --------
    >>> cfg = Config(key1="value1", key2=[1.0, 2.0])
    >>> cfg["key1"]
    'value1'
    >>> cfg.keys()
    dict_keys(['key1', 'key2'])
    >>> cfg.values()
    dict_values(['value1', [1.0, 2.0]])
    >>> cfg.items()
    dict_items([('key1', 'value1'), ('key2', [1.0, 2.0])])

    >>> cfg_copy = cfg.copy()
    >>> cfg_copy["key1"] = "new_value"
    >>> cfg["key1"]
    'value1'
    >>> cfg_copy["key1"]
    'new_value'
    >>> print(cfg)
    key1: value1
    key2:
    - 1.0
    - 2.0
    <BLANKLINE>

    Config objects can also be set up without passing arguments by using assigning attributes:

    >>> cfg = Config()
    >>> cfg.key1 = "value1"
    >>> cfg.key2 = [1.0, 2.0]
    >>> cfg.nested = Config(subkey1="subvalue1", subkey2="subvalue2")

    Accessing and setting values is also possible with the __getitem__ and __setitem__ methods.

    >>> cfg["nested"]["subkey1"]
    'subvalue1'
    >>> cfg["nested"]["subkey1"] = "subvalue1_modified"
    >>> print(cfg)
    key1: value1
    key2:
    - 1.0
    - 2.0
    nested:
      subkey1: subvalue1_modified
      subkey2: subvalue2
    <BLANKLINE>
    """

    def __init__(self, **kwargs):
        """Initialize a Config object with the provided key-value pairs."""
        for key, arg in kwargs.items():
            setattr(self, key, arg)

    def keys(self):
        """Return the keys of the configuration object."""
        return self.__dict__.keys()

    def values(self):
        """Return the values of the configuration object."""
        return self.__dict__.values()

    def items(self):
        """Return the items (key-value pairs) of the configuration object."""
        return self.__dict__.items()

    def __call__(self):
        """Return a deep copy of the configuration object."""
        cfg = deepcopy(self)
        for key, value in cfg.items():
            if not isinstance(value, Config):
                continue
            cfg[key] = value()
        return cfg.__dict__

    def __str__(self):
        """Return a string representation of the configuration object."""
        return yaml.dump(self())

    def __getitem__(self, key):
        """Get the value associated with the specified key."""
        if key not in self.keys():
            raise KeyError
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """Set the value associated with the specified key."""
        if key not in self.keys():
            raise KeyError
        self.__dict__[key] = value

    def copy(self):
        """Return a deep copy of the configuration object."""
        return deepcopy(self)


cfg = Config()

cfg.trajectory_keys = ((),)  # define data keys
# the following are predifined keys and used for visualization and analyses
# import and assign accordingly
cfg.key_time_stamp = (None,)  # depreated!
cfg.key_timestamp = (None,)
cfg.key_category = (None,)
cfg.key_score = (None,)
cfg.key_box = (None,)
cfg.key_keypoints = (None,)

cfg.timestep = None

cfg.vis = Config()
cfg.vis.padding = 0

# config for visualization figures
cfg.vis.figure = Config()
cfg.vis.figure.figsize = (1, 1)
cfg.vis.figure.dpi = 600

# config for instance visualization
cfg.vis.instance = Config()

cfg.vis.instance.box = Config()
cfg.vis.instance.box.linestyle = "-"
cfg.vis.instance.box.linewidth = 0.5
cfg.vis.instance.box.joinstyle = "round"
cfg.vis.instance.box.capstyle = "round"
cfg.vis.instance.box.edgecolor = "k"
cfg.vis.instance.box.facecolor = (0, 0, 0, 0)
cfg.vis.instance.box.zorder = 0

cfg.vis.instance.line = Config()
cfg.vis.instance.line.alpha = 0.6
cfg.vis.instance.line.color = "k"
cfg.vis.instance.line.capstyle = "round"
cfg.vis.instance.line.zorder = 0

cfg.vis.instance.points = Config()
cfg.vis.instance.points.s = 3
cfg.vis.instance.points.edgecolor = "k"
cfg.vis.instance.points.lw = 0.5
cfg.vis.instance.points.facecolor = (0, 0, 0, 0)

cfg.vis.instance.label = Config()
cfg.vis.instance.label.size = 5

# config for trajectory visualization
cfg.vis.trajectory = Config()

cfg.vis.trajectory.boxes = Config()
cfg.vis.trajectory.boxes.edgecolor = (0, 0, 0, 0.1)
cfg.vis.trajectory.boxes.facecolor = (0, 0, 0, 0)
cfg.vis.trajectory.boxes.lw = 0.1

cfg.vis.trajectory.lines = Config()
cfg.vis.trajectory.lines.color = (0, 0, 0, 0.1)
cfg.vis.trajectory.lines.capstyle = "round"
cfg.vis.trajectory.lines.lw = 0.1

cfg.vis.trajectory.points = Config()
cfg.vis.trajectory.points.sizes = 0.5
cfg.vis.trajectory.points.edgecolor = (0, 0, 0, 0.1)
cfg.vis.trajectory.points.facecolor = (0, 0, 0, 1)
cfg.vis.trajectory.points.lw = 0

if __name__ == "__main__":
    import doctest

    doctest.testmod()

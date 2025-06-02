from copy import deepcopy
from typing import Literal, Optional, overload

import yaml


class BaseConfig:
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

    Config objects can also be set up without passing arguments by assigning attributes:

    >>> cfg = Config()
    >>> cfg.key1 = "value1"
    >>> cfg.key2 = [1.0, 2.0]
    >>> cfg.nested = Config(subkey1="subvalue1", subkey2="subvalue2")

    Accessing and setting values is also possible with the :meth:`__getitem__` and :meth:`__setitem__` methods.

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

    @overload
    def __call__(self, *, as_frozenset: Literal[False] = False) -> dict: ...

    @overload
    def __call__(self, *, as_frozenset: Literal[True]) -> frozenset: ...

    @overload
    def __call__(self, *, as_frozenset: bool = False) -> dict | frozenset: ...

    def __call__(self, *, as_frozenset: bool = False) -> dict | frozenset:
        """Return a dictionary (or frozenset) representation of a deep copy of the configuration object."""
        cfg = {}
        for key, value in self.items():
            if isinstance(value, BaseConfig):
                cfg[key] = value(as_frozenset=as_frozenset)
                continue
            elif as_frozenset and isinstance(value, dict):
                cfg[key] = frozenset(value)
            else:
                cfg[key] = value
        if as_frozenset:
            return frozenset(cfg.items())
        return cfg

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

    def __eq__(self, other):
        if not isinstance(other, BaseConfig):
            return False
        return self(as_frozenset=True) == other(as_frozenset=True)

    def copy(self):
        """Return a deep copy of the configuration object."""
        return deepcopy(self)


class Config(BaseConfig):
    """
    Config for *automated-scoring*.

    Configurations derived from this class are used to define data access (via keys) for the structures implemented in :mod:`automated_scoring.data_structures`.

    Parameters:
        trajectory_keys: Tuple of keys used to specify trajectory data (e.g., positions, postures, timestamps, bounding boxes, individual identities).
        key_identity: Key used to specify identities, if applicable.
        key_timestamp: Key used to specify timestamps, required for timestamped data (e.g., :class:`~automated_scoring.data_structures.trajectory.Trajectory`).
        key_category: Key used to specify categories, for example if collections contain instances of different categories (e.g., different species).
        key_score: Key used to specify scores, for example for detection confidence for each instance in a collection.
        key_box: Key used to specify bounding boxes.
        key_keypoints: Key used to specify keypoints (i.e., posture data).
        timestep: Timestep used for temporal data (here, usually in video frames).
        hash_decimals: Number of decimal places used for hashing.
    """

    def __init__(
        self,
        *,
        trajectory_keys: tuple[str, ...] = tuple(),
        key_identity: Optional[str] = None,
        key_timestamp: Optional[str] = None,
        key_category: Optional[str] = None,
        key_score: Optional[str] = None,
        key_box: Optional[str] = None,
        key_keypoints: Optional[str] = None,
        timestep: Optional[int | float] = None,
        hash_decimals: int = 9,
        **kwargs,
    ):
        self.trajectory_keys = trajectory_keys
        self.key_identity = key_identity
        self.key_timestamp = key_timestamp
        self.key_category = key_category
        self.key_score = key_score
        self.key_box = key_box
        self.key_keypoints = key_keypoints
        self.timestep = timestep
        self.hash_decimals = hash_decimals
        super().__init__(**kwargs)


cfg = Config()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

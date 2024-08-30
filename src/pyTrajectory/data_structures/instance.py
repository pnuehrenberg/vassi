import numpy as np


from . import utils
from .base import ConfiguredData


class Instance(ConfiguredData):
    def __init__(self, cfg=None, from_scalars: bool = False, **kwargs):
        self._cfg = cfg
        self._data = {}
        for key in set(self.keys()) - set(kwargs.keys()):
            raise ValueError(f"missing key: {key}")
        for key, value in kwargs.items():
            if key not in self.keys():
                raise KeyError(f"key: {key} not in defined keys: {self.keys()}")
            if not from_scalars and not isinstance(value, np.ndarray):
                raise ValueError(
                    f"value for {key} is not numpy array but {type(value)}"
                )
            elif from_scalars:
                value = np.asarray(value)[np.newaxis, ...]
            if not value.ndim >= 1 and value.shape[0] == 1:
                raise ValueError(
                    f"value for {key} should at least be 1D with shape (1, ...)."
                )
            if not value.flags.owndata:
                value = value.copy()
            self._data[key] = value

    def __getitem__(self, key) -> utils.Value:
        return self.get_value(key)

    def __setitem__(self, key: str, value: utils.Value) -> None:
        _value = self.get_value(key)
        with utils.writeable(_value):
            _value[:] = value

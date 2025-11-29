import numpy as np

from ..config import Config
from ..config import cfg as CFG
from .base import ConfiguredData
from .utils import Value, writeable


class Instance(ConfiguredData):
    """
    Represents an instance of configured data, handling data initialization and access.

    Implements the :code:`__getitem__` and :code:`__setitem__` methods to provide dictionary-like access to the data.

    Parameters:
        cfg: The configuration object.
        from_scalars: Whether to convert scalar inputs to arrays.
        **kwargs: Keyword arguments representing the data to initialize the instance with.

    Raises:
        ValueError: If a required key is missing.
        ValueError: If a value is not a numpy array when :code:`from_scalars=False`.
        ValueError: If a value does not have at least one dimension, or if a value does not own its data.
        KeyError: If a key in :code:`kwargs` is not a defined key.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        from_scalars: bool = False,
        **kwargs: ...,
    ):
        if cfg is None:
            cfg = CFG.copy()
        self._cfg: Config | None = cfg
        self._data: dict[str, np.ndarray] | None = {}
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

    def __getitem__(self, key: str) -> np.ndarray:
        return self._get_value(key)

    def __setitem__(self, key: str, value: Value) -> None:
        _value = self._get_value(key)
        with writeable(_value):
            _value[:] = value

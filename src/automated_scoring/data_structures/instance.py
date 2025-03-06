import numpy as np
from numpy.typing import NDArray

from . import utils
from .base import ConfiguredData


class Instance(ConfiguredData):
    """
    Represents an instance of configured data, handling data initialization and access.

    Parameters
    ----------
    cfg : config.Config, optional
        The configuration object (default is None).
    from_scalars : bool, optional
        Whether to convert scalar inputs to arrays (default is False).
    **kwargs : dict
        Keyword arguments representing the data to initialize the instance with.

    Raises
    ------
    ValueError
        If a required key is missing, or if a value is not a NumPy array when `from_scalars` is False, or if a value does not have at least one dimension, or if a value does not own its data.
    KeyError
        If a key in `kwargs` is not a defined key.
    """

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

    def __getitem__(self, key: str) -> NDArray:
        """
        Gets the value associated with a given key.

        Parameters
        ----------
        key : str
            The key to retrieve the value for.

        Returns
        -------
        NDArray
            The value associated with the key.
        """
        return self._get_value(key)

    def __setitem__(self, key: str, value: utils.Value) -> None:
        """
        Sets the value of an instance attribute.

        Parameters
        ----------
        key : str
            The name of the attribute to set.
        value : utils.Value
            The new value for the attribute.
        """
        _value = self._get_value(key)
        with utils.writeable(_value):
            _value[:] = value

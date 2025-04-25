import hashlib
from collections.abc import Iterable
from typing import Optional

import numpy as np

from .. import config
from ..utils import hash_dict


class ConfiguredData:
    """Represents a configured data object with associated configuration and data."""

    _data: Optional[dict[str, np.ndarray]]
    _cfg: config.Config

    @property
    def sha1(self) -> str:
        """Calculates the SHA1 hash of the configured data."""
        items = {
            key: hashlib.sha1(
                np.round(value, decimals=self.cfg.hash_decimals)
            ).hexdigest()
            for key, value in self.items(copy=False)
        }
        items["cfg"] = hash_dict(self.cfg())
        return hash_dict(items)

    def __hash__(self) -> int:
        return hash(self.sha1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def cfg(self) -> config.Config:
        """Property that returns the configuration object."""
        return self._cfg

    def keys(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> tuple[str, ...]:
        """
        Returns the keys of the trajectory data, excluding specified keys.

        Parameters:
            exclude: Keys to exclude from the returned set of keys; defaults to None.

        Returns:
            A tuple of strings representing the keys of the trajectory data.

        Raises:
            ValueError: If the configuration is not initialized.
        """
        if self.cfg is None:
            raise ValueError("not initialized")
        if exclude is None:
            exclude = []
        return tuple(set(self.cfg.trajectory_keys).difference(exclude))

    def _get_value(
        self,
        key: str,
        *,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Gets a value from the internal data store.

        Parameters:
            key: The key to retrieve the value for.
            copy: Whether to return a copy of the data.

        Raises:
            ValueError: If the internal data store is not initialized.
        """
        if self._data is None:
            raise ValueError("not initialized")
        value = self._data[key]
        if copy:
            return value.copy()
        else:
            value = value.view()
            value.flags.writeable = False
        return value

    def values(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
        copy: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """
        Returns the values of the configured data.

        Parameters:
            exclude: Keys to exclude from the returned values; defaults to None.
            copy: Whether to return a copy of the data; defaults to True.

        Returns:
            The values of the configured data.
        """
        return tuple(
            self._get_value(key, copy=copy) for key in self.keys(exclude=exclude)
        )

    def items(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
        copy: bool = True,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """
        Returns a tuple of (key, value) pairs for the data, optionally excluding some keys.

        Parameters:
            exclude: Keys to exclude from the returned items; defaults to None.
            copy: Whether to return a copy of the data; defaults to True.

        Returns:
            A tuple of (key, value) pairs for the data.
        """
        keys = self.keys(exclude=exclude)
        values = self.values(exclude=exclude, copy=copy)
        return tuple((key, value) for key, value in zip(keys, values))

    @property
    def data(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary containing the a copy of the stored data.

        Returns:
            A dictionary containing the a copy of the stored data.

        Raises:
            ValueError: If the data has not been initialized.
        """
        if self._data is None:
            raise ValueError("not initialized")
        return {key: value for key, value in self.items()}

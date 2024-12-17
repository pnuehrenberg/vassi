import hashlib
from collections.abc import Iterable
from typing import Optional

from numpy.typing import NDArray

from .. import config
from ..utils import hash_dict


class ConfiguredData:
    """Base class for data structures that are configured with a config."""

    _data: Optional[dict[str, NDArray]] = None
    _cfg: Optional[config.Config] = None

    @property
    def sha1(self) -> str:
        """The SHA1 hash (digest) of the data structure."""
        items = {
            key: hashlib.sha1(value).hexdigest()
            for key, value in self.items(copy=False)
        }
        items["cfg"] = hash_dict(self.cfg())
        return hash_dict(items)

    def __hash__(self) -> int:
        """Return the hash of the data structure."""
        return hash(self.sha1)

    def __eq__(self, other: object) -> bool:
        """Return whether the data structure is equal to another object."""
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def __str__(self) -> str:
        """Return a string representation of the data structure."""
        return self.__repr__()

    @property
    def cfg(
        self,
    ) -> config.Config:
        """Configuration of the data structure."""
        if self._cfg is None:
            return config.cfg
        return self._cfg

    def keys(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> tuple[str, ...]:
        """Return the keys of the data structure.

        Parameters
        ----------
        exclude: Iterable[str], optional
            Keys to exclude.

        Returns
        -------
        tuple[str, ...]
            Keys of the data structure.

        Raises
        ------
        ValueError
            If the data structure is not initialized.
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
    ) -> NDArray:
        """Return the value of the data structure for a given key.

        Parameters
        ----------
        key: str
            Key of the value.
        copy: bool, optional
            Whether to copy the value.

        Returns
        -------
        NDArray
            Value of the data structure.

        Raises
        ------
        ValueError
            If the data structure is not initialized.
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
    ) -> tuple[NDArray, ...]:
        """Return the values of the data structure.

        Parameters
        ----------
        exclude: Iterable[str], optional
            Keys to exclude.
        copy: bool, optional
            Whether to copy the values.

        Returns
        -------
        tuple[NDArray, ...]
            Values of the data structure.
        """
        return tuple(
            self._get_value(key, copy=copy) for key in self.keys(exclude=exclude)
        )

    def items(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
        copy: bool = True,
    ) -> tuple[tuple[str, NDArray], ...]:
        """Return the items of the data structure.

        Parameters
        ----------
        exclude: Iterable[str], optional
            Keys to exclude.
        copy: bool, optional
            Whether to copy the items.

        Returns
        -------
        tuple[tuple[str, NDArray], ...]
            Items of the data structure.
        """
        keys = self.keys(exclude=exclude)
        values = self.values(exclude=exclude, copy=copy)
        return tuple((key, value) for key, value in zip(keys, values))

    @property
    def data(self) -> dict[str, NDArray]:
        """Return the data of the data structure as a dictionary.

        Returns
        -------
        dict[str, NDArray]
            Data of the data structure.

        Raises
        ------
        ValueError
            If the data structure is not initialized.
        """
        if self._data is None:
            raise ValueError("not initialized")
        return {key: value for key, value in self.items()}

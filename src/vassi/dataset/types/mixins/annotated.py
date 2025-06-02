from abc import ABC, abstractmethod
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd


class EncodingFunction(Protocol):
    """
    Protocol for category encoding functions.

    Parameters:
        y (:class:`~numpy.ndarray`): The input array to be encoded.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The encoded array.
    """

    def __call__(
        self,
        y: np.ndarray,
        *args: ...,
        **kwargs: ...,
    ) -> np.ndarray: ...


def encode_categories(y: np.ndarray, *, categories: tuple[str, ...]) -> np.ndarray:
    """
    Encode categories to integers.

    Parameters:
        y: The input array to be encoded.
        categories: The categories to encode.

    Returns:
        The encoded array.
    """
    y_numeric = np.zeros_like(y, dtype=int)
    for category in categories:
        y_numeric[y == category] = categories.index(category)
    return y_numeric


class AnnotatedMixin(ABC):
    """
    Mixin for annotated datasets.

    Parameters:
        categories: The categories of the dataset.
        background_category: The background category of the dataset.
    """

    def __init__(
        self,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ):
        self.background_category = background_category
        if background_category not in categories:
            categories = tuple(list(categories) + [background_category])
        self.categories = tuple(sorted(categories))
        self._foreground_categories = tuple(
            category for category in categories if category != background_category
        )
        self._encode = partial(encode_categories, categories=self.categories)
        # call finalize init to handle observations in subclasses
        # some additional attributes may be needed for other

    @property
    def encode(self) -> EncodingFunction:
        """Function to encode category names to integers."""
        return self._encode

    @property
    def foreground_categories(self) -> tuple[str, ...]:
        """The categories of the dataset excluding the background category."""
        return self._foreground_categories

    @property
    def category_counts(self) -> dict[str, int]:
        """Counts of each category in the sampleable."""
        y = self.sample_y()
        return {category: int((y == category).sum()) for category in self.categories}

    def sample_y(self) -> np.ndarray:
        """Return the target labels of the entire sampleable (as category names)."""
        return self._sample_y()

    @property
    def observations(self) -> pd.DataFrame:
        """Return the observations of the sampleable."""
        return self._get_observations()

    @abstractmethod
    def _finalize_init(self, observations: pd.DataFrame) -> None: ...

    @abstractmethod
    def _get_observations(self) -> pd.DataFrame: ...

    @abstractmethod
    def _sample_y(self) -> np.ndarray: ...

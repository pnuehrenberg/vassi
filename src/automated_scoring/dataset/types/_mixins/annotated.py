from abc import ABC, abstractmethod
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class EncodingFunction(Protocol):
    def __call__(
        self,
        y: NDArray,
        *args: ...,
        **kwargs: ...,
    ) -> NDArray[np.int64]: ...


def encode_categories(y: NDArray, *, categories: tuple[str, ...]) -> NDArray:
    y_numeric = np.zeros_like(y, dtype=int)
    for category in categories:
        y_numeric[y == category] = categories.index(category)
    return y_numeric


class AnnotatedMixin(ABC):
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
        self.foreground_categories = tuple(
            category for category in categories if category != background_category
        )
        self._encode = partial(encode_categories, categories=self.categories)
        # call finalize init to handle observations in subclasses
        # some additional attributes may be needed for other

    @property
    def encode(self) -> EncodingFunction:
        return self._encode

    @property
    def category_counts(self) -> dict[str, int]:
        y = self.sample_y()
        return {category: int((y == category).sum()) for category in self.categories}

    def sample_y(self) -> NDArray:
        return self._sample_y()

    @property
    def observations(self) -> pd.DataFrame:
        return self._get_observations()

    @abstractmethod
    def _finalize_init(self, observations: pd.DataFrame) -> None: ...

    @abstractmethod
    def _get_observations(self) -> pd.DataFrame: ...

    @abstractmethod
    def _sample_y(self) -> NDArray: ...

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
        # call finalize init to handle observations in subclasses
        # some additional attributes may be needed for other

    def encode(self, y: NDArray) -> NDArray:
        y_numeric = np.zeros(len(y), dtype=int)
        for category in self.categories:
            y_numeric[y == category] = self.categories.index(category)
        return y_numeric

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

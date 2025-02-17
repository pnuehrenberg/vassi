from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import Identifier
from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin

if TYPE_CHECKING:
    from loguru import Logger


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    if TYPE_CHECKING:

        def sample(
            self,
            extractor: FeatureExtractor | DataFrameFeatureExtractor,
            *,
            exclude: Sequence[Identifier] | None,
        ) -> tuple[pd.DataFrame | NDArray, NDArray]: ...

        def subsample(
            self,
            extractor: FeatureExtractor | DataFrameFeatureExtractor,
            size: int | float | Mapping[str | tuple[str, ...], int | float],
            *,
            random_state: int | np.random.Generator | None,
            stratify: bool,
            reset_previous_indices: bool,
            exclude_previous_indices: bool,
            store_indices: bool,
            exclude: Sequence[Identifier] | None,
            log: Optional["Logger"],
        ) -> tuple[pd.DataFrame | NDArray, NDArray]: ...

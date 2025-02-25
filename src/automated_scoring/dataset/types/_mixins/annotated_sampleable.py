from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from ....features import BaseExtractor, F
from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin

if TYPE_CHECKING:
    from loguru import Logger


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    if TYPE_CHECKING:

        def sample(
            self,
            extractor: BaseExtractor[F],
        ) -> tuple[F, NDArray]: ...

        def subsample(
            self,
            extractor: BaseExtractor[F],
            size: int | float | Mapping[str | tuple[str, ...], int | float],
            *,
            random_state: int | np.random.Generator | None,
            stratify: bool,
            reset_previous_indices: bool,
            exclude_previous_indices: bool,
            store_indices: bool,
            log: Optional[Logger],
        ) -> tuple[F, NDArray]: ...

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import numpy as np

from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin

if TYPE_CHECKING:
    from loguru import Logger

    from ....features import BaseExtractor, Shaped


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    if TYPE_CHECKING:

        def sample[F: Shaped](
            self,
            extractor: BaseExtractor[F],
        ) -> tuple[F, np.ndarray]: ...

        def subsample[F: Shaped](
            self,
            extractor: BaseExtractor[F],
            size: int | float | Mapping[str | tuple[str, ...], int | float],
            *,
            random_state: Optional[int | np.random.Generator] = None,
            stratify: bool = True,
            reset_previous_indices: bool = False,
            exclude_previous_indices: bool = False,
            store_indices: bool = False,
            log: Optional["Logger"] = None,
        ) -> tuple[F, np.ndarray]: ...

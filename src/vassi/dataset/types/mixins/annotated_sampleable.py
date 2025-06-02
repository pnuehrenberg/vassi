from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import loguru
import numpy as np

from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin

if TYPE_CHECKING:
    from ....features import BaseExtractor, Shaped


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    """
    Mixin for sampleable objects with annotations, only used for type checking.

    Intersection of :class:`~vassi.dataset.types.mixins.annotated.AnnotatedMixin` and :class:`~vassi.dataset.types.mixins.sampleable.SampleableMixin`.
    """

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
            log: Optional[loguru.Logger] = None,
        ) -> tuple[F, np.ndarray]: ...

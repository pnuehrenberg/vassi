import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyTrajectory.utils import warning_only

from ..data_structures import Trajectory
from . import features, temporal_features, utils, decorators

FeatureCategory = Literal["individual"] | Literal["dyadic"]


def _get_func(func_name: str) -> utils.Feature:
    try:
        return getattr(features, func_name)
    except AttributeError:
        try:
            return getattr(temporal_features, func_name)
        except AttributeError:
            pass
    raise ValueError(
        f"{func_name} not implemented in features.features or features.temporal_features"
    )


class BaseExtractor:
    _feature_funcs_individual: list[tuple[Callable, dict[str, Any]]] = []
    _feature_funcs_dyadic: list[tuple[Callable, dict[str, Any]]] = []
    _feature_names_individual: list[str] = []
    _feature_names_dyadic: list[str] = []
    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
    )

    def __init__(
        self,
        *,
        features: list[tuple[utils.Feature, dict[str, Any]]] | None = None,
        dyadic_features: list[tuple[utils.Feature, dict[str, Any]]] | None = None,
    ):
        if features is not None:
            self._init_features(features, category="individual")
        if dyadic_features is not None:
            self._init_features(dyadic_features, category="dyadic")

    def _get_func(self, func_name: str) -> utils.Feature:
        return _get_func(func_name)

    def _adjust_func(
        self,
        func: Callable,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
    ) -> Callable:
        if as_absolute and as_sign_change_latency:
            raise ValueError(
                "Only specify one of as_absolute and as_sign_change_latency."
            )
        if as_absolute:
            return decorators.as_absolute(func)
        if as_sign_change_latency:
            return decorators.as_sign_change_latency(func)
        return func

    @property
    def feature_names(self):
        return self._get_feature_names("individual") + self._get_feature_names("dyadic")

    def _get_feature_funcs(self, category: FeatureCategory, *, clear: bool = False):
        if category == "individual":
            feature_funcs = self._feature_funcs_individual
        elif category == "dyadic":
            feature_funcs = self._feature_funcs_dyadic
        else:
            raise ValueError(
                f"Undefined feature category {category}, specify either 'individual' and 'dyadic'."
            )
        if clear:
            feature_funcs.clear()
        return feature_funcs

    def _get_feature_names(self, category: FeatureCategory, *, clear: bool = False):
        if category == "individual":
            feature_names = self._feature_names_individual
        elif category == "dyadic":
            feature_names = self._feature_names_dyadic
        else:
            raise ValueError(
                f"Undefined feature category {category}, specify either 'individual' and 'dyadic'."
            )
        if clear:
            feature_names.clear()
        return feature_names

    def _init_features(
        self,
        feature_funcs: list[tuple[Callable, dict[str, Any]]],
        *,
        category: FeatureCategory,
    ) -> None:
        _feature_funcs = self._get_feature_funcs(category, clear=True)
        _feature_names = self._get_feature_names(category, clear=True)
        for func, kwargs in feature_funcs:
            func = self._adjust_func(
                func,
                **{
                    kwarg: kwargs[kwarg]
                    for kwarg in self.allowed_additional_kwargs
                    if kwarg in kwargs
                },
            )
            names = utils.get_feature_names(func, **kwargs)
            pruned_names = utils.prune_feature_names(
                names,
                keep=kwargs["keep"] if "keep" in kwargs else None,
                discard=kwargs["discard"] if "discard" in kwargs else None,
            )
            _feature_names.extend(pruned_names)
            _feature_funcs.append((func, kwargs))

    def load(
        self,
        features_config: dict[FeatureCategory, list[tuple[str, dict[str, Any]]]],
    ) -> Self:
        def ensure_flat(kwargs: dict[str, Any]) -> dict[str, Any]:
            kwargs["flat"] = True
            return kwargs

        for category, feature_funcs in features_config.items():
            self._init_features(
                [
                    (self._get_func(func_name), ensure_flat(kwargs))
                    for func_name, kwargs in feature_funcs
                ],
                category=category,
            )
        return self

    def _concatenate(self, *args: Any) -> Any:
        raise NotImplementedError

    def extract_features(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
        *,
        category: FeatureCategory,
    ) -> Any:
        def prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            kwargs = kwargs.copy()
            for kwarg in self.allowed_additional_kwargs:
                if kwarg not in kwargs:
                    continue
                kwargs.pop(kwarg)
            if category == "dyadic":
                kwargs["trajectory_other"] = trajectory_other
            return kwargs

        feature_funcs = self._get_feature_funcs(category)
        if len(feature_funcs) == 0:
            raise ValueError("No features specified.")
        return self._concatenate(
            *[
                func(trajectory, **prepare_kwargs(kwargs))
                for func, kwargs in feature_funcs
            ],
        )

    def extract(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
    ) -> Any:
        if trajectory_other is None and len(self._feature_funcs_dyadic) > 0:
            with warning_only():
                warnings.warn(
                    "Extracting only non-dyadic features, although dyadic features are specified."
                )
        if trajectory_other is None:
            return self.extract_features(trajectory, category="individual")
        if len(self._feature_funcs_dyadic) == 0:
            return self.extract_features(trajectory, category="individual")
        if len(self._feature_funcs_individual) == 0:
            return self.extract_features(
                trajectory, trajectory_other, category="dyadic"
            )
        return self._concatenate(
            *[
                self.extract_features(trajectory, category="individual"),
                self.extract_features(trajectory, trajectory_other, category="dyadic"),
            ],
        )


class FeatureExtractor(BaseExtractor):
    _feature_funcs_individual: list[tuple[utils.Feature, dict[str, Any]]] = []
    _feature_funcs_dyadic: list[tuple[utils.Feature, dict[str, Any]]] = []

    def _concatenate(self, *args: NDArray) -> NDArray:
        return np.concatenate(args, axis=1)

    if TYPE_CHECKING:

        def extract_features(
            self,
            trajectory: Trajectory,
            trajectory_other: Optional[Trajectory] = None,
            *,
            category: FeatureCategory,
        ) -> NDArray: ...

        def extract(
            self, trajectory: Trajectory, trajectory_other: Optional[Trajectory] = None
        ) -> NDArray: ...


class DataFrameFeatureExtractor(BaseExtractor):
    _feature_funcs_individual: list[tuple[utils.DataFrameFeature, dict[str, Any]]] = []
    _feature_funcs_dyadic: list[tuple[utils.DataFrameFeature, dict[str, Any]]] = []
    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
        "keep",
        "discard",
    )

    def _adjust_func(
        self,
        func: Callable,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
        keep: list[str] | str | None = None,
        discard: list[str] | str | None = None,
    ) -> Callable:
        return decorators.as_dataframe(
            super()._adjust_func(func, as_absolute, as_sign_change_latency),
            keep=keep,
            discard=discard,
        )

    def _concatenate(self, *args: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(args, axis=1)

    if TYPE_CHECKING:

        def extract_features(
            self,
            trajectory: Trajectory,
            trajectory_other: Optional[Trajectory] = None,
            *,
            category: FeatureCategory,
        ) -> pd.DataFrame: ...

        def extract_dyadic_features(
            self, trajectory: Trajectory, trajectory_other: Trajectory
        ) -> pd.DataFrame: ...

        def extract(
            self, trajectory: Trajectory, trajectory_other: Trajectory | None = None
        ) -> pd.DataFrame: ...

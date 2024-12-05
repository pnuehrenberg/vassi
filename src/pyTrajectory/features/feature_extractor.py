import os
import pickle
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self

import numpy as np
import pandas as pd

# import portalocker
import yaml
from numpy.typing import NDArray

from pyTrajectory.utils import warning_only

from ..data_structures import Trajectory
from ..utils import hash_dict
from . import decorators, features, temporal_features, utils

FeatureCategory = Literal["individual", "dyadic"]


def _to_cache(obj: Any, cache_file: str) -> None:
    # TODO writing should catch keyboardinterrupt to avoid corrupted files
    # with portalocker.Lock(cache_file, "wb") as cached:
    with open(cache_file, "wb") as cached:
        pickle.dump(obj, cached)


def _from_cache(cache_file: str):
    if not os.path.isfile(cache_file):
        raise FileNotFoundError
    # reading should not require locking, so maybe unneccesary
    # with portalocker.Lock(cache_file, "rb") as cached:
    # TODO or explicitly fail here and so that value gets recomputed
    with open(cache_file, "rb") as cached:
        return pickle.load(cached)


def _hash_args(extractor: "BaseExtractor", *args, **kwargs) -> str:
    def to_hash_string(arg):
        if arg is None:
            return "none"
        if isinstance(arg, str):
            return arg
        if isinstance(arg, Trajectory):
            return arg.sha1
        raise NotImplementedError("invalid argument type")

    d = {"extractor": extractor.sha1}
    for idx, arg in enumerate(args):
        d[f"arg_{idx}"] = to_hash_string(arg)
    for key, value in kwargs.items():
        d[key] = to_hash_string(value)
    return hash_dict(d)


def cache(func: Callable) -> Callable:
    def _cache(extractor: "BaseExtractor", *args, **kwargs):
        hash_value = _hash_args(extractor, *args, **kwargs)
        cache_file = os.path.join(extractor.cache_directory, hash_value)
        try:
            return _from_cache(cache_file)
        except FileNotFoundError:
            pass
        value = func(extractor, *args, **kwargs)
        _to_cache(value, cache_file)
        return value

    return _cache


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def construct_yaml_tuple(self, node):
    seq = self.construct_sequence(node)
    if seq and isinstance(seq, list):
        return tuple(seq)
    return seq


class TupleLoader(yaml.SafeLoader):
    pass


TupleLoader.add_constructor("tag:yaml.org,2002:seq", construct_yaml_tuple)


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
    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
    )

    def __init__(
        self,
        *,
        features: list[tuple[utils.Feature, dict[str, Any]]] | None = None,
        dyadic_features: list[tuple[utils.Feature, dict[str, Any]]] | None = None,
        cache_directory: str,
    ):
        self._feature_funcs_individual: list[tuple[Callable, dict[str, Any]]] = []
        self._feature_funcs_dyadic: list[tuple[Callable, dict[str, Any]]] = []
        self._feature_names_individual: list[str] = []
        self._feature_names_dyadic: list[str] = []
        if features is not None:
            self._init_features(features, category="individual")
        if dyadic_features is not None:
            self._init_features(dyadic_features, category="dyadic")
        self.cache_directory = cache_directory
        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory, exist_ok=True)

    @property
    def sha1(self):
        d = self.config
        d["type"] = str(type(self))
        return hash_dict(d)

    def __hash__(self) -> int:
        return hash(self.sha1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def _get_func(self, func_name: str) -> utils.Feature:
        return _get_func(func_name)

    def _adjust_func(
        self,
        func: utils.Feature,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
    ) -> utils.Feature:
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
            feature = decorators._inner(func)
            prefix = decorators._get_prefix(func)
            # names = utils.get_feature_names(func, **kwargs)
            names = [
                f"{prefix}{name}" for name in utils.get_feature_names(feature, **kwargs)
            ]
            pruned_names = utils.prune_feature_names(
                names,
                keep=kwargs["keep"] if "keep" in kwargs else None,
                discard=kwargs["discard"] if "discard" in kwargs else None,
            )
            _feature_names.extend(pruned_names)
            _feature_funcs.append((func, kwargs))

    @property
    def config(self):
        config = {}
        feature_categories: list[FeatureCategory] = ["individual", "dyadic"]
        for feature_category in feature_categories:
            features = self._get_feature_funcs(feature_category)
            if len(features) == 0:
                continue
            config[feature_category] = []
            for func, kwargs in features:
                config[feature_category].append(
                    (decorators._inner(func).__name__, kwargs)
                )
        return config

    def save_yaml(self, features_config_file: str) -> None:
        with open(features_config_file, "w") as yaml_file:
            yaml_file.write(
                yaml.dump(self.config, Dumper=NoAliasDumper, sort_keys=False)
            )

    def read_yaml(self, features_config_file: str) -> Self:
        with open(features_config_file, "r") as yaml_file:
            features_config = yaml.load(yaml_file.read(), Loader=TupleLoader)
        self.load(features_config)
        return self

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

    @classmethod
    def concatenate(cls, *args: Any, axis: int = 1) -> Any:
        raise NotImplementedError

    # @cache
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
        return type(self).concatenate(
            *[
                func(trajectory, **prepare_kwargs(kwargs))
                for func, kwargs in feature_funcs
            ],
        )

    @cache
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
        return type(self).concatenate(
            *[
                self.extract_features(trajectory, category="individual"),
                self.extract_features(trajectory, trajectory_other, category="dyadic"),
            ],
        )


class FeatureExtractor(BaseExtractor):
    @classmethod
    def concatenate(cls, *args: NDArray, axis: int = 1) -> NDArray:
        return np.concatenate(args, axis=axis)

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
    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
        "keep",
        "discard",
    )

    def _adjust_func(  # type: ignore
        self,
        func: utils.Feature,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
        keep: list[str] | str | None = None,
        discard: list[str] | str | None = None,
    ) -> utils.DataFrameFeature:
        return decorators.as_dataframe(
            super()._adjust_func(func, as_absolute, as_sign_change_latency),
            keep=keep,
            discard=discard,
        )

    @classmethod
    def concatenate(cls, *args: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        dataframe = pd.concat(args, axis=axis)
        assert isinstance(dataframe, pd.DataFrame)
        return dataframe

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

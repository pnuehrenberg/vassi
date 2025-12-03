from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping
from inspect import signature
from pathlib import Path
from typing import Literal, Self, override

import h5py
import numpy as np
import pandas as pd

from .._utils import get_inner
from ..data_structures import Trajectory
from ..io.yaml import from_yaml, to_yaml
from ..sliding_metrics import SlidingWindowAggregator
from ..type_guards import is_iterable_of, is_iterable_of_tuple, is_valid_features_config
from ..utils import hash_mapping
from ..warnings import warn
from .features import load_by_name as load_feature_by_name
from .modifiers import as_absolute, as_sign_change_latency, reversed_dyad
from .naming import as_dataframe, generate_feature_names, prune_feature_names
from .temporal_features import load_by_name as load_temporal_feature_by_name
from .utils import Shaped


def _load_by_name(func_name: str) -> Callable[..., np.ndarray]:
    func = load_feature_by_name(func_name)
    if func is None:
        func = load_temporal_feature_by_name(func_name)
    if func is None:
        raise ValueError(f"Unknown feature function: {func_name}")
    return func


def _params_config(config: Mapping[str, object]) -> dict[str, Hashable]:
    valid_config: dict[str, Hashable] = {}
    for k, v in config.items():
        if isinstance(v, Hashable):
            valid_config[k] = v
        elif is_iterable_of_tuple(v, int):
            valid_config[k] = tuple(v)
        elif is_iterable_of(v, int):
            valid_config[k] = tuple(v)
        elif isinstance(v, str):
            valid_config[k] = v
        else:
            raise ValueError(f"Unsopported value for {k}: {v}")
    return valid_config


def _parse_reversed_dyad(**kwargs: ...) -> bool:
    if not isinstance(reversed_dyad := kwargs.pop("reversed_dyad", False), bool):
        raise ValueError("reversed_dyad must be a boolean if specified")
    return reversed_dyad


def _parse_as_absolute(**kwargs: ...) -> bool:
    if not isinstance(as_absolute := kwargs.pop("as_absolute", False), bool):
        raise ValueError("as_absolute must be a boolean if specified")
    return as_absolute


def _parse_as_sign_change_latency(**kwargs: ...) -> bool:
    if not isinstance(
        as_sign_change_latency := kwargs.get("as_sign_change_latency", False), bool
    ):
        raise ValueError("as_sign_change_latency must be a boolean if specified")
    return as_sign_change_latency


def _parse_keep(**kwargs: ...) -> str | Iterable[str] | None:
    if (
        (keep := kwargs.pop("keep", None)) is not None
        and not isinstance(keep, str)
        and not is_iterable_of(keep, str)
    ):
        raise ValueError("keep must be a string or an iterable of strings if specified")
    return keep


def _parse_discard(**kwargs: ...) -> str | Iterable[str] | None:
    if (
        (discard := kwargs.pop("discard", None)) is not None
        and not isinstance(discard, str)
        and not is_iterable_of(discard, str)
    ):
        raise ValueError(
            "discard must be a string or an iterable of strings if specified"
        )
    return discard


class BaseExtractor[F: Shaped](ABC):
    _feature_funcs: dict[
        Literal["individual", "dyad"],
        list[tuple[Callable[..., F], Mapping[str, object], Mapping[str, object]]],
    ]
    _feature_names: dict[Literal["individual", "dyad"], list[str]]
    _supported_modify_params: set[str]
    _num_features: int

    def __init__(
        self,
        individual_features: Iterable[
            tuple[Callable[..., np.ndarray], Mapping[str, object]]
        ]
        | None = None,
        dyadic_features: Iterable[
            tuple[Callable[..., np.ndarray], Mapping[str, object]]
        ]
        | None = None,
        *,
        cache_mode: bool | Literal["required"] = False,
        cache_directory: str | Path = "./feature_cache",
        aggregator: SlidingWindowAggregator | None = None,
    ):
        self._feature_funcs = {
            "individual": [],
            "dyad": [],
        }
        self._feature_names = {
            "individual": [],
            "dyad": [],
        }
        if individual_features is None:
            individual_features = []
        if dyadic_features is None:
            dyadic_features = []
        self._init_features(individual_features, target="individual")
        self._init_features(dyadic_features, target="dyad")
        self.cache_mode: bool | Literal["required"] = cache_mode
        self.aggregator: SlidingWindowAggregator | None = aggregator
        self._num_original_features: int = len(self._feature_names["individual"]) + len(
            self._feature_names["dyad"]
        )
        self.cache_directory: Path = Path(cache_directory) / self.sha1

    @property
    def feature_names(self) -> list[str]:
        if self.aggregator is None:
            return self._feature_names["individual"] + self._feature_names["dyad"]
        return self.aggregator.get_feature_names_out(
            self._feature_names["individual"] + self._feature_names["dyad"]
        ).tolist()

    @property
    def num_features(self) -> int:
        if self.aggregator is None:
            return self._num_original_features
        return self.aggregator.get_num_features_out(self._num_original_features)

    @property
    def config(
        self,
    ) -> dict[
        Literal["individual", "dyad"], tuple[tuple[str, dict[str, Hashable]], ...]
    ]:
        return {
            "individual": tuple(
                (
                    get_inner(func).__name__,
                    _params_config({**kwargs, **modifier_kwargs}),
                )
                for func, kwargs, modifier_kwargs in self._feature_funcs["individual"]
            ),
            "dyad": tuple(
                (
                    get_inner(func).__name__,
                    _params_config({**kwargs, **modifier_kwargs}),
                )
                for func, kwargs, modifier_kwargs in self._feature_funcs["dyad"]
            ),
        }

    def save_yaml(self, features_config_file: str | Path) -> None:
        to_yaml(self.config, file_name=features_config_file)

    @classmethod
    def from_config(
        cls,
        config: Mapping[
            Literal["individual", "dyad"], Iterable[tuple[str, Mapping[str, object]]]
        ],
        *,
        cache_mode: bool | Literal["required"] = False,
        cache_directory: str | Path = "./feature_cache",
        aggregator: SlidingWindowAggregator | None = None,
    ) -> Self:
        individual_features = (
            [
                (_load_by_name(str(func)), kwargs)
                for func, kwargs in config["individual"]
            ]
            if "individual" in config
            else None
        )
        dyadic_features = (
            [(_load_by_name(str(func)), kwargs) for func, kwargs in config["dyad"]]
            if "dyad" in config
            else None
        )
        return cls(
            individual_features,
            dyadic_features,
            cache_mode=cache_mode,
            cache_directory=cache_directory,
            aggregator=aggregator,
        )

    @classmethod
    def from_yaml(
        cls,
        features_config_file: str | Path,
        *,
        cache_mode: bool | Literal["required"] = False,
        cache_directory: str | Path = "./feature_cache",
        aggregator: SlidingWindowAggregator | None = None,
    ) -> Self:
        config = from_yaml(features_config_file)

        if not is_valid_features_config(config):
            raise ValueError(f"Invalid features configuration: {features_config_file}")
        return cls.from_config(
            config,
            cache_mode=cache_mode,
            cache_directory=cache_directory,
            aggregator=aggregator,
        )

    @property
    def sha1(self) -> str:
        d: dict[object, object] = {**self.config}
        d["type"] = str(type(self))
        if self.aggregator is not None:
            d["aggregator"] = self.aggregator.sha1
        return hash_mapping(d)

    @override
    def __hash__(self) -> int:
        return hash(self.sha1)

    @override
    def __eq__(self, other: ...) -> bool:
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    @abstractmethod
    def _parse_modify_params(self, **kwargs: ...) -> tuple[object, ...]: ...

    def _separate_modify_params(
        self, kwargs: Mapping[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {k: v for k, v in kwargs.items() if k not in self._supported_modify_params},
            {k: v for k, v in kwargs.items() if k in self._supported_modify_params},
        )

    def _init_features(
        self,
        feature_config: Iterable[
            tuple[Callable[..., np.ndarray], Mapping[str, object]]
        ],
        *,
        target: Literal["individual", "dyad"],
    ) -> None:
        feature_funcs: list[
            tuple[Callable[..., F], Mapping[str, object], Mapping[str, object]]
        ] = []
        feature_names: list[str] = []
        for func, kwargs in feature_config:
            func = self._modify_feature_func(
                func,
                **kwargs,
            )
            kwargs, modifier_kwargs = self._separate_modify_params(kwargs)

            keyword_only_params = {
                param
                for param in signature(func).parameters.values()
                if param.kind == param.KEYWORD_ONLY
            }
            if unsupported := set(kwargs) - set(
                param.name for param in keyword_only_params
            ):
                warn(
                    f"{get_inner(func).__name__}: Ignoring unsupported keyword-only parameters: {unsupported}"
                )
                modifier_kwargs.update(
                    {kwarg: kwargs.pop(kwarg) for kwarg in unsupported}
                )
            if missing_required := set(
                param.name
                for param in keyword_only_params
                if param.default == param.empty
            ) - set(kwargs):
                raise ValueError(
                    f"{get_inner(func).__name__}: Missing required keyword-only parameters: {missing_required}"
                )
            kwargs["flat"] = True
            feature_funcs.append((func, kwargs, modifier_kwargs))
            if target == "dyad":
                kwargs = kwargs.copy()
                kwargs["trajectory_other"] = Trajectory()
            names = generate_feature_names(func, **kwargs)
            if (
                "keep" in self._supported_modify_params
                and "discard" in self._supported_modify_params
            ):
                keep = _parse_keep(**modifier_kwargs)
                discard = _parse_discard(**modifier_kwargs)
                names = prune_feature_names(names, keep=keep, discard=discard)
            feature_names.extend(names)
        self._feature_funcs[target] = feature_funcs
        self._feature_names[target] = feature_names

    @abstractmethod
    def _modify_feature_func(
        self,
        func: Callable[..., np.ndarray],
        **kwargs: ...,
    ) -> Callable[..., F]: ...

    @abstractmethod
    def finalize(self, x: np.ndarray) -> F: ...

    def extract(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory | None = None,
        *,
        indices: np.ndarray | None,
        out: np.ndarray | None,
    ) -> F:
        cache_file = None
        if self.cache_mode:
            if not self.cache_directory.exists():
                self.cache_directory.mkdir(parents=True)
            cache_file = (
                self.cache_directory
                / f"{trajectory.sha1}{trajectory_other.sha1 if trajectory_other is not None else ''}.h5"
            )
            if self.cache_mode == "required" and not cache_file.exists():
                raise FileNotFoundError(
                    f"Cache file not found, but cache_mode is set to '{self.cache_mode}'."
                )
        if trajectory_other is not None and len(trajectory) != len(trajectory_other):
            raise ValueError("Trajectories must have the same length.")
        if indices is None:
            num_samples = len(trajectory)
        else:
            if not indices.ndim == 1:
                raise ValueError("Indices must be a 1-dimensional array.")
            if np.isdtype(indices.dtype, "bool"):
                num_samples = indices.sum()
            elif np.issubdtype(indices.dtype, np.integer):
                num_samples = len(indices)
            else:
                raise TypeError("Indices must be a boolean or integer array.")
        if out is None:
            out = np.zeros((num_samples, self.num_features))
        elif out.shape != (num_samples, self.num_features):
            raise ValueError(
                f"Output array has incorrect shape, expected {(num_samples, self.num_features)} and not {out.shape}"
            )
        if self.cache_mode and cache_file is not None and cache_file.exists():
            with h5py.File(str(cache_file), "r") as h5_file:
                if "features" not in h5_file:
                    raise ValueError("Cache file does not contain features.")
                cached = h5_file["features"]
                if not isinstance(cached, h5py.Dataset):
                    raise TypeError("Cached features are not a dataset.")
                out[:] = cached[:] if indices is None else cached[:][indices]
            return self.finalize(out)
        cached = None
        feature_idx = 0
        if self.aggregator is None:
            if self.cache_mode:
                cached = np.zeros((len(trajectory), self.num_features))
            for target in self._feature_funcs:
                for func, kwargs, _ in self._feature_funcs[target]:
                    if target == "dyad":
                        if trajectory_other is None:
                            raise ValueError(
                                "trajectory_other must be provided for dyadic features."
                            )
                        features = func(
                            trajectory, trajectory_other=trajectory_other, **kwargs
                        )
                    else:
                        features = func(trajectory, **kwargs)
                    features = np.asarray(features)
                    num_current_features = features.shape[1]
                    if cached is not None:
                        # can write full directly to cached
                        cached[:, feature_idx : feature_idx + num_current_features] = (
                            features
                        )
                    if indices is not None:
                        features = features[indices]
                    # can write indexed directly to out
                    out[:, feature_idx : feature_idx + num_current_features] = features
                    feature_idx += num_current_features
            assert feature_idx == self.num_features
        else:
            intermediate = np.zeros((len(trajectory), self._num_original_features))
            # here, it is more efficient to write results to intermediate, which is then transformed once
            for target in self._feature_funcs:
                for func, kwargs, _ in self._feature_funcs[target]:
                    if target == "dyad":
                        if trajectory_other is None:
                            raise ValueError(
                                "trajectory_other must be provided for dyadic features."
                            )
                        features = func(
                            trajectory, trajectory_other=trajectory_other, **kwargs
                        )
                    else:
                        features = func(trajectory, **kwargs)
                    features = np.asarray(features)
                    num_current_features = features.shape[1]
                    intermediate[
                        :, feature_idx : feature_idx + num_current_features
                    ] = features
                    feature_idx += num_current_features
            assert feature_idx == self._num_original_features
            if not self.cache_mode and (
                indices is None
                or (
                    len(indices) == len(trajectory)
                    and (indices == np.arange(len(trajectory))).all()
                )
            ):
                _ = self.aggregator.transform(intermediate, indices=indices, out=out)
            else:
                # can't directly write to out, needs creating full transformed result, and then writing indexed to out
                # full transformed result will get cached if cache_mode
                transformed = self.aggregator.transform(
                    intermediate, indices=None, out=None
                )
                cached = transformed
                out[:] = transformed[indices]
        if self.cache_mode:
            assert cached is not None
            assert cache_file is not None
            with h5py.File(str(cache_file), "w") as h5_file:
                _ = h5_file.create_dataset("features", data=cached)
        return self.finalize(out)


class Extractor(BaseExtractor[np.ndarray]):
    _supported_modify_params: set[str] = {
        "reversed_dyad",
        "as_absolute",
        "as_sign_change_latency",
    }

    @override
    def _parse_modify_params(self, **kwargs: ...) -> tuple[bool, bool, bool]:
        return (
            _parse_reversed_dyad(**kwargs),
            _parse_as_absolute(**kwargs),
            _parse_as_sign_change_latency(**kwargs),
        )

    @override
    def _modify_feature_func(
        self,
        func: Callable[..., np.ndarray],
        **kwargs: ...,
    ) -> Callable[..., np.ndarray]:
        param_reversed_dyad, param_as_absolute, param_as_sign_change_latency = (
            self._parse_modify_params(**kwargs)
        )
        if param_reversed_dyad:
            func = reversed_dyad(func)
        if param_as_absolute and param_as_sign_change_latency:
            raise ValueError(
                "Only specify one of as_absolute and as_sign_change_latency."
            )
        if param_as_absolute:
            func = as_absolute(func)
        if param_as_sign_change_latency:
            func = as_sign_change_latency(func)
        return func

    @override
    def finalize(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected 2D array with {self.num_features} columns, got {x.shape}"
            )
        return x


class DataFrameExtractor(BaseExtractor[pd.DataFrame]):
    _supported_modify_params: set[str] = {
        "reversed_dyad",
        "as_absolute",
        "as_sign_change_latency",
        "keep",
        "discard",
    }

    @override
    def _parse_modify_params(
        self, **kwargs: ...
    ) -> tuple[bool, bool, bool, Iterable[str] | None, Iterable[str] | None]:
        return (
            _parse_reversed_dyad(**kwargs),
            _parse_as_absolute(**kwargs),
            _parse_as_sign_change_latency(**kwargs),
            _parse_keep(**kwargs),
            _parse_discard(**kwargs),
        )

    @override
    def _modify_feature_func(
        self,
        func: Callable[..., np.ndarray],
        **kwargs: ...,
    ) -> Callable[..., pd.DataFrame]:
        (
            param_reversed_dyad,
            param_as_absolute,
            param_as_sign_change_latency,
            keep,
            discard,
        ) = self._parse_modify_params(**kwargs)
        if param_reversed_dyad:
            func = reversed_dyad(func)
        if param_as_absolute and param_as_sign_change_latency:
            raise ValueError(
                "Only specify one of as_absolute and as_sign_change_latency."
            )
        if param_as_absolute:
            func = as_absolute(func)
        if param_as_sign_change_latency:
            func = as_sign_change_latency(func)
        return as_dataframe(func, keep=keep, discard=discard)

    @override
    def finalize(self, x: np.ndarray) -> pd.DataFrame:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected 2D array with {self.num_features} columns, got {x.shape}"
            )
        return pd.DataFrame(x, columns=self.feature_names)

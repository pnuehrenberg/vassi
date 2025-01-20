import os
import pickle
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self

import numpy as np
import pandas as pd

# import portalocker
import yaml
from numpy.typing import NDArray

from ..data_structures import Trajectory
from ..utils import hash_dict, warning_only
from . import decorators, features, temporal_features, utils

FeatureCategory = Literal["individual", "dyadic"]


def _to_cache(obj: Any, cache_file: str) -> None:
    """
    Helper function to write an object to a cache file using pickle.

    Parameters
    ----------
    obj : Any
        The object to write.
    cache_file : str
        The path to the cache file.
    """
    # TODO writing should catch keyboardinterrupt to avoid corrupted files
    with open(cache_file, "wb") as cached:
        pickle.dump(obj, cached)


def _from_cache(cache_file: str):
    """
    Helper function to read an object from a cache file using pickle.

    Parameters
    ----------
    cache_file : str
        The path to the cache file.

    Returns
    -------
    Any
        The object read from the cache file.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    if not os.path.isfile(cache_file):
        raise FileNotFoundError
    # TODO or explicitly fail here and so that value gets recomputed
    with open(cache_file, "rb") as cached:
        return pickle.load(cached)


def _hash_args(extractor: "BaseExtractor", *args, **kwargs) -> str:
    """
    Return a hash (hex digest) of the arguments of a method implemented by the BaseExtractor class.

    Parameters
    ----------
    extractor : BaseExtractor
        The extractor.
    *args : Any
        The arguments.
    **kwargs : Any
        The keyword arguments.

    Returns
    -------
    str
        The hash of the arguments.
    """

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
    """
    Decorator to cache the result of a method implemented by the BaseExtractor class.

    Parameters
    ----------
    func : Callable
        The method to cache.

    Returns
    -------
    Callable
        The decorated method.
    """

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


class _NoAliasDumper(yaml.SafeDumper):
    """
    Helper class to dump yaml without aliases.
    """

    def ignore_aliases(self, data):
        return True


def _construct_yaml_tuple(self, node):
    """
    Helper function to construct a tuple from a yaml sequence.
    """
    seq = self.construct_sequence(node)
    if seq and isinstance(seq, list):
        return tuple(seq)
    return seq


class _TupleLoader(yaml.SafeLoader):
    """
    Helper class to load all sequences in yaml as tuples.
    """

    pass


_TupleLoader.add_constructor("tag:yaml.org,2002:seq", _construct_yaml_tuple)


def load_feature_func(func_name: str) -> utils.Feature:
    """
    Helper function to get a feature function (from features module) from its name.

    Parameters
    ----------
    func_name : str
        The name of the feature function.

    Returns
    -------
    utils.Feature
        The feature function.

    Raises
    ------
    ValueError
        If the feature function is not implemented in the features module.
    """
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
    """
    The base class for feature extractors.

    Additional keyword arguments can be passed to the feature functions.
    For the BaseExtractor class, this is ("as_absolute", "as_sign_change_latency").

    Parameters
    ----------
    features : list[tuple[utils.Feature, dict[str, Any]]] | None, optional
        The features to extract, by default None.
    dyadic_features : list[tuple[utils.Feature, dict[str, Any]]] | None, optional
        The dyadic features to extract, by default None.
    cache_directory : str, optional
        The directory to use for caching, by default "cache".
    """

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
        """
        The SHA1 hash (digest) of the extractor configuration.
        """
        d = self.config
        d["type"] = str(type(self))
        return hash_dict(d)

    def __hash__(self) -> int:
        """
        Return the hash of the extractor configuration.
        """
        return hash(self.sha1)

    def __eq__(self, other: object) -> bool:
        """
        Return whether the extractor is equal to another object (by comparing the configuration).
        """
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def _get_func(self, func_name: str) -> utils.Feature:
        """
        Helper function to get a feature function (from features module) from its name.

        See also
        --------
        load_feature_func
        """
        return load_feature_func(func_name)

    def _adjust_func(
        self,
        func: utils.Feature,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
    ) -> utils.Feature:
        """
        Adjust a feature function to be either absolute or sign change latency.

        Parameters
        ----------
        func : utils.Feature
            The feature function to adjust.
        as_absolute : bool, optional
            Whether to adjust the feature function to be absolute.
        as_sign_change_latency : bool, optional
            Whether to adjust the feature function to be sign change latency.

        Returns
        -------
        utils.Feature
            The adjusted feature function.
        """
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
        """
        The names of all features (both individual and dyadic).
        """
        return self._get_feature_names("individual") + self._get_feature_names("dyadic")

    def _get_feature_funcs(self, category: FeatureCategory, *, clear: bool = False):
        """
        Get the feature functions for a given category.

        Parameters
        ----------
        category : Literal["individual", "dyadic"]
            The category of the feature functions to get.
        clear : bool, optional
            Whether to clear the feature functions before returning them.

        Returns
        -------
        list[tuple[Callable, dict[str, Any]]]
            The feature functions for the specified category.
        """
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
        """
        Get the names of the features for a given category.

        Parameters
        ----------
        category : Literal["individual", "dyadic"]
            The category of the feature names to get.
        clear : bool, optional
            Whether to clear the feature names before returning them.

        Returns
        -------
        list[str]
            The feature names for the specified category.
        """
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
        """
        Initialize the features for a given category.

        Parameters
        ----------
        feature_funcs : list[tuple[Callable, dict[str, Any]]]
            The feature functions to initialize.
        category : Literal["individual", "dyadic"]
            The category of the features to initialize.
        """
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
        """
        The configuration of the extractor.
        """
        config = {}
        feature_categories: list[FeatureCategory] = ["individual", "dyadic"]
        for feature_category in feature_categories:
            features = self._get_feature_funcs(feature_category)
            if len(features) == 0:
                continue
            config[feature_category] = []
            for func, kwargs in features:
                kwargs = kwargs.copy()
                if "flat" in kwargs:
                    kwargs.pop("flat")
                config[feature_category].append(
                    (decorators._inner(func).__name__, kwargs)
                )
        return config

    def save_yaml(self, features_config_file: str) -> None:
        """
        Save the extractor configuration to a yaml file.

        Parameters
        ----------
        features_config_file : str
            The path to the yaml file to save the configuration to.
        """
        with open(features_config_file, "w") as yaml_file:
            yaml_file.write(
                yaml.dump(self.config, Dumper=_NoAliasDumper, sort_keys=False)
            )

    def read_yaml(self, features_config_file: str) -> Self:
        """
        Load the extractor configuration from a yaml file.

        Parameters
        ----------
        features_config_file : str
            The path to the yaml file to load the configuration from.

        Returns
        -------
        Self
            The extractor with the loaded configuration.
        """
        with open(features_config_file, "r") as yaml_file:
            features_config = yaml.load(yaml_file.read(), Loader=_TupleLoader)
        self.load(features_config)
        return self

    def load(
        self,
        features_config: dict[FeatureCategory, list[tuple[str, dict[str, Any]]]],
    ) -> Self:
        """
        Load the extractor configuration from a dictionary.

        Parameters
        ----------
        features_config : dict[Literal["individual", "dyadic"], list[tuple[str, dict[str, Any]]]]
            The configuration to load.

        Returns
        -------
        Self
            The extractor with the loaded configuration.
        """

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
        """
        Each subclass must implement this method to concatenate the computed features.
        """
        raise NotImplementedError

    def extract_features(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
        *,
        category: FeatureCategory,
    ) -> Any:
        """
        Extract features of one category (individual or dyadic) from a trajectory.
        If the category is dyadic, the trajectory_other argument should be provided.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to extract features from.
        trajectory_other : Trajectory, optional
            The other trajectory in a dyad, by default None.
        category : Literal["individual", "dyadic"]
            The category of the features to extract.

        Returns
        -------
        Any
            The computed features. The type depends on the subclass.
        """

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
        """
        Extract features from a trajectory.

        If the trajectory_other argument is None and dyadic features are specified, a warning is raised.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to extract features from.
        trajectory_other : Trajectory, optional
            The other trajectory in a dyad, by default None.

        Returns
        -------
        Any
            The computed features. The type depends on the subclass.
        """
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
    """
    A subclass to extract features as numpy arrays.
    """

    @classmethod
    def concatenate(cls, *args: NDArray, axis: int = 1) -> NDArray:
        """
        Concatenate the computed features as a numpy array.

        Parameters
        ----------
        *args : NDArray
            The computed features.
        axis : int, optional
            The axis to concatenate along, by default 1.

        Returns
        -------
        NDArray
            The concatenated features.
        """
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
    """
    A subclass to extract features as pandas DataFrames.

    Extends the additional keyword arguments that can be passed to the feature functions.
    For the DataFrameFeatureExtractor class, this is ("as_absolute", "as_sign_change_latency", "keep", "discard").
    """

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
        """
        Adjust a feature function to be a dataframe feature.

        Parameters
        ----------
        func : utils.Feature
            The feature function to adjust.
        as_absolute : bool, optional
            Whether to adjust the feature function to be absolute.
        as_sign_change_latency : bool, optional
            Whether to adjust the feature function to be sign change latency.
        keep : list[str] | str | None, optional
            Substring(s) in the feature names to keep, by default None.
            This can be used to keep feautres that would otherwise be discarded
        discard : list[str] | str | None, optional
            Substring(s) in the feature names to discard, by default None.

        Returns
        -------
        utils.DataFrameFeature
            The adjusted feature function.
        """
        return decorators.as_dataframe(
            super()._adjust_func(func, as_absolute, as_sign_change_latency),
            keep=keep,
            discard=discard,
        )

    @classmethod
    def concatenate(cls, *args: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """
        Concatenate the computed features as a pandas DataFrame.

        Parameters
        ----------
        *args : pd.DataFrame
            The computed features.
        axis : int, optional
            The axis to concatenate along, by default 1.

        Returns
        -------
        pd.DataFrame
            The concatenated features.
        """
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

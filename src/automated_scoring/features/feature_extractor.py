import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    Self,
    TypeVar,
)

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ..data_structures import Trajectory
from ..logging import set_logging_level
from ..utils import hash_dict
from . import decorators, features, temporal_features, utils
from ._caching import cache

FeatureCategory = Literal["individual", "dyadic"]


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


class Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


F = TypeVar("F", bound=Shaped)


class BaseExtractor(ABC, Generic[F]):
    """
    The base class for feature extractors.

    Additional keyword arguments can be passed to the feature functions.
    For the BaseExtractor class, this is ("as_absolute", "as_sign_change_latency").

    Parameters
    ----------
    features : list[tuple[utils.Feature, Mapping[str, Any]]] | None, optional
        The features to extract, by default None.
    dyadic_features : list[tuple[utils.Feature, Mapping[str, Any]]] | None, optional
        The dyadic features to extract, by default None.
    cache_directory : str, optional
        The directory to use for caching, by default "cache".
    """

    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
        "reversed_dyad",
    )

    def __init__(
        self,
        *,
        features: list[tuple[utils.Feature, Mapping[str, Any]]] | None = None,
        dyadic_features: list[tuple[utils.Feature, Mapping[str, Any]]] | None = None,
        cache_directory: str,
        pipeline: Optional[Pipeline] = None,
        refit_pipeline: bool = False,
    ):
        self._feature_funcs_individual: list[tuple[utils.Feature, dict[str, Any]]] = []
        self._feature_funcs_dyadic: list[tuple[utils.Feature, dict[str, Any]]] = []
        self._feature_names_individual: list[str] = []
        self._feature_names_dyadic: list[str] = []
        if features is not None:
            self._init_features(features, category="individual")
        if dyadic_features is not None:
            self._init_features(dyadic_features, category="dyadic")
        self.cache_directory = cache_directory
        self.pipeline = pipeline
        self.refit_pipeline = refit_pipeline
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
        **kwargs: Any,  # Additional keyword arguments are allowed, but ignored
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
    def feature_names(self) -> list[str]:
        """
        The names of all features (both individual and dyadic).
        """
        return self._get_feature_names("individual") + self._get_feature_names("dyadic")

    def _get_feature_funcs(
        self, category: FeatureCategory, *, clear: bool = False
    ) -> list[tuple[utils.Feature, dict[str, Any]]]:
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
        list[tuple[utils.Feature, Mapping[str, Any]]]
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
        feature_funcs: list[tuple[utils.Feature, Mapping[str, Any]]],
        *,
        category: FeatureCategory,
    ) -> None:
        """
        Initialize the features for a given category.

        Parameters
        ----------
        feature_funcs : list[tuple[utils.Feature, Mapping[str, Any]]]
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
            if "reversed_dyad" in kwargs and kwargs["reversed_dyad"]:
                prefix = f"r_{prefix}"
            names = [
                f"{prefix}{name}" for name in utils.get_feature_names(feature, **kwargs)
            ]
            pruned_names = utils.prune_feature_names(
                names,
                keep=kwargs["keep"] if "keep" in kwargs else None,
                discard=kwargs["discard"] if "discard" in kwargs else None,
            )
            _feature_names.extend(pruned_names)
            _feature_funcs.append((func, {key: value for key, value in kwargs.items()}))

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
        features_config: dict[FeatureCategory, list[tuple[str, Mapping[str, Any]]]],
    ) -> Self:
        """
        Load the extractor configuration from a dictionary.

        Parameters
        ----------
        features_config : dict[Literal["individual", "dyadic"], list[tuple[str, Mapping[str, Any]]]]
            The configuration to load.

        Returns
        -------
        Self
            The extractor with the loaded configuration.
        """

        def ensure_flat(kwargs: Mapping[str, Any]) -> dict[str, Any]:
            kwargs = {key: value for key, value in kwargs.items()}
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
    @abstractmethod
    def concatenate(cls, *args: F, axis: int = 1, **kwargs: Any) -> F: ...

    @classmethod
    @abstractmethod
    def empty(cls) -> F: ...

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

        def prepare_args(
            kwargs: Mapping[str, Any],
        ) -> tuple[Trajectory, dict[str, Any]]:
            nonlocal trajectory, trajectory_other
            input_trajectory = trajectory
            input_trajectory_other = trajectory_other
            kwargs = {key: value for key, value in kwargs.items()}
            for kwarg in self.allowed_additional_kwargs:
                if kwarg not in kwargs:
                    continue
                if kwarg == "reversed_dyad" and kwargs[kwarg]:
                    if input_trajectory_other is None:
                        raise ValueError(
                            "Can only reverse feature when trajectory_other is specified."
                        )
                    input_trajectory, input_trajectory_other = (
                        input_trajectory_other,
                        input_trajectory,
                    )
                kwargs.pop(kwarg)
            if category == "dyadic":
                if input_trajectory_other is None:
                    raise ValueError(
                        "Can only calculate dyadic feature when trajectory_other is specified."
                    )
                kwargs["trajectory_other"] = input_trajectory_other
            return input_trajectory, kwargs

        feature_funcs = self._get_feature_funcs(category)
        if len(feature_funcs) == 0:
            raise ValueError("No features specified.")
        features = []
        for func, kwargs in feature_funcs:
            input_trajectory, kwargs = prepare_args(kwargs)
            features.append(func(input_trajectory, **kwargs))
        features = type(self).concatenate(*features)
        if isinstance(features, pd.DataFrame):
            features.columns = self._get_feature_names(category)
        if self.pipeline is not None:
            if isinstance(features, pd.DataFrame):
                self.pipeline.set_output(transform="pandas")
            if self.refit_pipeline:
                self.pipeline.fit(features)
            return self.pipeline.transform(features)
        return features

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

        def extract_category(category: Literal["individual", "dyadic"]) -> Any:
            nonlocal trajectory, trajectory_other
            if category == "individual":
                return self.extract_features(
                    trajectory, trajectory_other, category="individual"
                )
            if category == "dyadic":
                return self.extract_features(
                    trajectory, trajectory_other, category="dyadic"
                )
            raise ValueError(f"Invalid feature category {category}.")

        if trajectory_other is None and len(self._feature_funcs_dyadic) > 0:
            set_logging_level().warning(
                "Extracting only non-dyadic features, although dyadic features are specified."
            )
        if trajectory_other is None:
            return extract_category("individual")
        if len(self._feature_funcs_dyadic) == 0:
            return extract_category("individual")
        if len(self._feature_funcs_individual) == 0:
            return extract_category("dyadic")
        return type(self).concatenate(
            extract_category("individual"),
            extract_category("dyadic"),
        )


class FeatureExtractor(BaseExtractor[NDArray]):
    """
    A subclass to extract features as numpy arrays.
    """

    @classmethod
    def concatenate(cls, *args: NDArray, axis: int = 1, **kwargs: Any) -> NDArray:
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

        def prepare_array(array: NDArray, num_features: int | None) -> NDArray:
            if array.size == 0:
                if num_features is None:
                    raise ValueError("num_features can not be None when array is empty")
                array = array.reshape(-1, num_features)
            return array

        if "ignore_index" in kwargs:
            kwargs.pop("ignore_index")
        num_features = None
        if "num_features" in kwargs:
            num_features = kwargs.pop("num_features")
        prepared_args = [prepare_array(arg, num_features) for arg in args]
        return np.concatenate(prepared_args, axis=axis, **kwargs)

    @classmethod
    def empty(cls) -> NDArray:
        return np.array([])

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


class DataFrameFeatureExtractor(BaseExtractor[pd.DataFrame]):
    """
    A subclass to extract features as pandas DataFrames.

    Extends the additional keyword arguments that can be passed to the feature functions.
    For the DataFrameFeatureExtractor class, this is ("as_absolute", "as_sign_change_latency", "keep", "discard").
    """

    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
        "reversed_dyad",
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
        **kwargs: Any,  # Additional keyword arguments are allowed, but ignored
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
    def concatenate(
        cls, *args: pd.DataFrame, axis: int = 1, **kwargs: Any
    ) -> pd.DataFrame:
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
        if "num_features" in kwargs:
            kwargs.pop("num_features")

        dataframe = pd.concat(args, axis=axis, **kwargs)
        assert isinstance(dataframe, pd.DataFrame)
        return dataframe

    @classmethod
    def empty(cls) -> pd.DataFrame:
        return pd.DataFrame()

    if TYPE_CHECKING:

        def extract_features(
            self,
            trajectory: Trajectory,
            trajectory_other: Optional[Trajectory] = None,
            *,
            category: FeatureCategory,
        ) -> pd.DataFrame: ...

        def extract(
            self, trajectory: Trajectory, trajectory_other: Trajectory | None = None
        ) -> pd.DataFrame: ...

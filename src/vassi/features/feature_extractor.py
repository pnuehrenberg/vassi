import os
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Protocol,
    Self,
)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..data_structures import Trajectory
from ..io import from_yaml, to_yaml
from ..logging import set_logging_level
from ..utils import hash_dict
from . import decorators, features, temporal_features, utils
from ._caching import cache

FeatureCategory = Literal["individual", "dyadic"]


def load_feature_func(func_name: str) -> utils.Feature:
    """
    Helper function to get a feature function (from :mod:`~vassi.features.features`) from its name.

    Parameters:
        func_name: The name of the feature function.

    Returns:
        utils.Feature: The feature function.

    Raises:
        ValueError: If the feature function is not implemented in the features module.
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
    """The minimum requirement of extracted features. Both numpy arrays and pandas DataFrames are supported."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class BaseExtractor[F: Shaped](ABC):
    """
    The base class for feature extractors.

    Allows the following keyword arguments in feature functions:

    - as_absolute: Whether to return the absolute value of the feature.
    - as_sign_change_latency: Whether to return the latency of sign changes in the feature.
    - reversed_dyad: Whether to reverse the order of the dyad for feature computation.

    Parameters:
        features: The features to extract.
        dyadic_features: The dyadic features to extract.
        cache_mode: Whether to use caching or to allow only precomputed features.
        cache_directory: The directory to use for caching.
        pipeline: The pipeline to use for further feature transformation.
        refit_pipeline: Whether to refit the pipeline for each extraction call.
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
        cache_mode: Literal["cached"] | bool = True,
        cache_directory: Optional[str] = None,
        pipeline: Optional[Pipeline] = None,
        refit_pipeline: bool = False,
    ):
        self._feature_functions_individual: list[
            tuple[utils.Feature, dict[str, Any]]
        ] = []
        self._feature_functions_dyadic: list[tuple[utils.Feature, dict[str, Any]]] = []
        self._feature_names_individual: list[str] = []
        self._feature_names_dyadic: list[str] = []
        if features is not None:
            self._init_features(features, category="individual")
        if dyadic_features is not None:
            self._init_features(dyadic_features, category="dyadic")
        self.cache_mode = cache_mode
        if self.cache_mode:
            if cache_directory is None:
                raise ValueError(
                    f"cache_directory must be specified with cache={self.cache_mode}"
                )
            self.cache_directory = cache_directory
            if not os.path.exists(self.cache_directory):
                os.makedirs(self.cache_directory, exist_ok=True)
        else:
            self.cache_directory = None
        self.pipeline = pipeline
        self.refit_pipeline = refit_pipeline

    @property
    def sha1(self):
        """The SHA1 hash (digest) of the extractor configuration."""
        d: dict[str, Hashable] = {**self.config}
        d["type"] = str(type(self))
        return hash_dict(d)

    def __hash__(self) -> int:
        return hash(self.sha1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def adjust_function(
        self,
        function: utils.Feature,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
        **kwargs: Any,  # Additional keyword arguments are allowed, but ignored
    ) -> utils.Feature:
        """
        Adjust a feature function to be either absolute or sign change latency.

        Parameters:
            function: The feature function to adjust.
            as_absolute: Whether to adjust the feature function to be absolute.
            as_sign_change_latency: Whether to adjust the feature function to be sign change latency.

        Returns:
            The adjusted feature function.

        Raises:
            ValueError: If both :code:`as_absolute=True` and :code:`as_sign_change_latency=True`.
        """
        if as_absolute and as_sign_change_latency:
            raise ValueError(
                "Only specify one of as_absolute and as_sign_change_latency."
            )
        if as_absolute:
            return decorators.as_absolute(function)
        if as_sign_change_latency:
            return decorators.as_sign_change_latency(function)
        return function

    @property
    def feature_names(self) -> list[str]:
        """The names of all features (both individual and dyadic)."""
        return self._get_feature_names("individual") + self._get_feature_names("dyadic")

    def _get_feature_functions(
        self, category: FeatureCategory, *, clear: bool = False
    ) -> list[tuple[utils.Feature, dict[str, Any]]]:
        if category == "individual":
            feature_functions = self._feature_functions_individual
        elif category == "dyadic":
            feature_functions = self._feature_functions_dyadic
        else:
            raise ValueError(
                f"Undefined feature category {category}, specify either 'individual' and 'dyadic'."
            )
        if clear:
            feature_functions.clear()
        return feature_functions

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
        feature_functions: list[tuple[utils.Feature, Mapping[str, Any]]],
        *,
        category: FeatureCategory,
    ) -> None:
        _feature_functions = self._get_feature_functions(category, clear=True)
        _feature_names = self._get_feature_names(category, clear=True)
        for function, kwargs in feature_functions:
            function = self.adjust_function(
                function,
                **{
                    kwarg: kwargs[kwarg]
                    for kwarg in self.allowed_additional_kwargs
                    if kwarg in kwargs
                },
            )
            feature = decorators.get_inner(function)
            prefix = decorators.get_prefix(function)
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
            _feature_functions.append(
                (function, {key: value for key, value in kwargs.items()})
            )

    @property
    def config(
        self,
    ) -> dict[
        Literal["individual", "dyadic"], tuple[tuple[str, dict[str, Hashable]], ...]
    ]:
        """The configuration of the extractor, returned as a dictionary."""
        config = {}
        for feature_category in ("individual", "dyadic"):
            features = self._get_feature_functions(feature_category)
            if len(features) == 0:
                continue
            config[feature_category] = []
            for function, kwargs in features:
                kwargs = kwargs.copy()
                if "flat" in kwargs:
                    kwargs.pop("flat")
                config[feature_category].append(
                    (decorators.get_inner(function).__name__, kwargs)
                )
        return config

    def save_yaml(self, features_config_file: str) -> None:
        """
        Save the extractor configuration to a yaml file.

        Parameters:
            features_config_file: The path to the yaml file to save the configuration to.
        """
        to_yaml(self.config, file_name=features_config_file)

    def read_yaml(self, features_config_file: str) -> Self:
        """
        Load the extractor configuration from a yaml file.

        Parameters:
            features_config_file: The path to the yaml file to load the configuration from.
        """
        self.load(from_yaml(features_config_file))
        return self

    def load(
        self,
        features_config: dict[FeatureCategory, list[tuple[str, Mapping[str, Any]]]],
    ) -> Self:
        """
        Load the extractor configuration from a dictionary.

        Parameters:
            features_config: The configuration to load.

        Returns:
            The feature extractor after loading the configuration.
        """

        def ensure_flat(kwargs: Mapping[str, Any]) -> dict[str, Any]:
            kwargs = {key: value for key, value in kwargs.items()}
            kwargs["flat"] = True
            return kwargs

        for category, feature_functions in features_config.items():
            self._init_features(
                [
                    (load_feature_func(func_name), ensure_flat(kwargs))
                    for func_name, kwargs in feature_functions
                ],
                category=category,
            )
        return self

    @classmethod
    @abstractmethod
    def concatenate(cls, *args: F, axis: int = 1, **kwargs: Any) -> F: ...

    @classmethod
    @abstractmethod
    def empty(cls) -> F:
        """Returns an empty feature object, type depending on the implementation in the subclass."""
        ...

    def extract_features(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
        *,
        category: FeatureCategory,
    ) -> F:
        """
        Extract features of one category (:code:`'individual'` or :code:`'dyadic'`) from a trajectory.
        If the category is :code:`'dyadic'`, the :code:`trajectory_other` argument must be provided.

        Parameters:
            trajectory: The trajectory to extract features from.
            trajectory_other: A second trajectory, used for dyadic features.
            category: The category of the features to extract.

        Returns:
            The extracted features.
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

        feature_functions = self._get_feature_functions(category)
        if len(feature_functions) == 0:
            raise ValueError("No features specified.")
        features = []
        for function, kwargs in feature_functions:
            input_trajectory, kwargs = prepare_args(kwargs)
            features.append(function(input_trajectory, **kwargs))
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
    ) -> F:
        """
        Extract features from a trajectory.

        If the trajectory_other argument is None and dyadic features are specified, a warning is raised.

        Parameters:
            trajectory: The trajectory to extract features from.
            trajectory_other: The other trajectory in a dyad when extracting dyadic features.

        Returns
        -------
        Any
            The computed features. The type depends on the subclass.
        """

        def extract_category(category: Literal["individual", "dyadic"]) -> F:
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

        if trajectory_other is None and len(self._feature_functions_dyadic) > 0:
            set_logging_level().warning(
                "Extracting only non-dyadic features, although dyadic features are specified."
            )
        if trajectory_other is None:
            return extract_category("individual")
        if len(self._feature_functions_dyadic) == 0:
            return extract_category("individual")
        if len(self._feature_functions_individual) == 0:
            return extract_category("dyadic")
        return type(self).concatenate(
            extract_category("individual"),
            extract_category("dyadic"),
        )


class FeatureExtractor(BaseExtractor[np.ndarray]):
    """
    A subclass to extract features as numpy arrays (:class:`~numpy.ndarray`).
    """

    @classmethod
    def concatenate(cls, *args: np.ndarray, axis: int = 1, **kwargs: Any) -> np.ndarray:
        """
        Concatenate the computed features.

        Parameters:
            *args: The computed features.
            axis: The axis to concatenate along.

        Returns:
            The concatenated features.
        """

        def prepare_array(array: np.ndarray, num_features: int | None) -> np.ndarray:
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
    def empty(cls) -> np.ndarray:
        """Returns an empty :code:`~numpy.ndarray`."""
        return np.array([])

    if TYPE_CHECKING:

        def extract_features(
            self,
            trajectory: Trajectory,
            trajectory_other: Optional[Trajectory] = None,
            *,
            category: FeatureCategory,
        ) -> np.ndarray: ...

        def extract(
            self, trajectory: Trajectory, trajectory_other: Optional[Trajectory] = None
        ) -> np.ndarray: ...


class DataFrameFeatureExtractor(BaseExtractor[pd.DataFrame]):
    """
    A subclass to extract features as pandas DataFrames (:class:`~pandas.DataFrame`).

    Extends the additional keyword arguments that can be passed to the feature functions
    to :code:`("as_absolute", "as_sign_change_latency", "keep", "discard")`.
    """

    allowed_additional_kwargs: tuple[str, ...] = (
        "as_absolute",
        "as_sign_change_latency",
        "reversed_dyad",
        "keep",
        "discard",
    )

    def adjust_function(  # type: ignore
        self,
        function: utils.Feature,
        as_absolute: bool = False,
        as_sign_change_latency: bool = False,
        keep: list[str] | str | None = None,
        discard: list[str] | str | None = None,
        **kwargs: Any,  # Additional keyword arguments are allowed, but ignored
    ) -> utils.DataFrameFeature:
        """
        Adjust a feature function to be a dataframe feature.

        Parameters:
            function: The feature function to adjust.
            as_absolute: Whether to adjust the feature function to be absolute.
            as_sign_change_latency: Whether to adjust the feature function to be sign change latency.
            keep: Patterns in the feature names to keep. This can be used to keep features that would otherwise be discarded.
            discard: Patterns in the feature names to discard.

        Returns:
            The adjusted feature function.
        """
        return decorators.as_dataframe(
            super().adjust_function(function, as_absolute, as_sign_change_latency),
            keep=keep,
            discard=discard,
        )

    @classmethod
    def concatenate(
        cls, *args: pd.DataFrame, axis: int = 1, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Concatenate the computed features as a pandas DataFrame.

        Parameters:
            *args: The computed features.
            axis: The axis to concatenate along.

        Returns:
            The concatenated features.
        """
        ignore_index = False
        if "ignore_index" in kwargs:
            ignore_index = kwargs.pop("ignore_index")
            if TYPE_CHECKING:
                assert isinstance(ignore_index, bool)
        dataframe = pd.concat(
            args, axis="index" if axis == 0 else "columns", ignore_index=ignore_index
        )
        assert isinstance(dataframe, pd.DataFrame)
        return dataframe

    @classmethod
    def empty(cls) -> pd.DataFrame:
        """Returns an empty :class:`~pandas.DataFrame`."""
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

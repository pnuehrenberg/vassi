from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from functools import partial
from typing import Concatenate, Literal, TypedDict, TypeGuard, final

import numpy as np
import optuna

from ...sliding_metrics import get_window_slices
from .._results import AnnotatedDatasetClassification


def without_postprocessing(
    dataset_classification: AnnotatedDatasetClassification, **_: ...
) -> AnnotatedDatasetClassification:
    return dataset_classification.discretize()


def _scorer(
    result: AnnotatedDatasetClassification,
    *,
    on: Literal["timestamp", "prediction", "annotation"] | None,
    foreground_only: bool,
) -> float:
    if on is None:
        scores = result.score()
    else:
        scores = result.f1_score(on=on)
    if foreground_only:
        scores = scores[list(result.foreground_categories)]
    return float(np.mean(scores))


def _make_scorer(
    on: Literal["timestamp", "prediction", "annotation"] | None = None,
    foreground_only: bool = False,
) -> Callable[[AnnotatedDatasetClassification], float]:
    return partial(_scorer, on=on, foreground_only=foreground_only)


macro_f1_all_levels = _make_scorer()
macro_f1_foreground_all_levels = _make_scorer(foreground_only=True)
macro_f1_timestamp = _make_scorer(on="timestamp")
macro_f1_foreground_timestamp = _make_scorer(on="timestamp", foreground_only=True)


class AggregatorKwargs(TypedDict):
    sliding_metric_functions: Iterable[
        Callable[Concatenate[np.ndarray, int, ...], np.ndarray]
    ]
    windows: Iterable[int]
    num_slices_per_window: int | None
    keep_original_features: bool


def is_aggregator_kwargs(kwargs: dict[str, object]) -> TypeGuard[AggregatorKwargs]:
    required_keys = {
        "sliding_metric_functions",
        "windows",
        "num_slices_per_window",
        "keep_original_features",
    }
    if not required_keys.issubset(kwargs.keys()):
        return False
    if not isinstance(kwargs["keep_original_features"], bool):
        return False
    if kwargs["num_slices_per_window"] is not None and not isinstance(
        kwargs["num_slices_per_window"], int
    ):
        return False
    windows = kwargs["windows"]
    if not isinstance(windows, Iterable):
        return False
    if isinstance(windows, (list, tuple)):
        if not all(isinstance(w, int) for w in windows):
            return False
    funcs = kwargs["sliding_metric_functions"]
    if not isinstance(funcs, Iterable):
        return False
    if not isinstance(funcs, Generator):
        # this is unlikely, but we should not consume values if its a generator
        if not all(callable(f) for f in funcs):
            return False
    return True


class _AggregatorKwargs(TypedDict):
    metric_funcs: list[Callable[Concatenate[np.ndarray, int, ...], np.ndarray]]
    window_size: int
    window_slices: Iterable[slice] | None
    keep_original: bool


def get_validated_aggregator_kwargs(
    aggregator_kwargs: AggregatorKwargs | None,
):
    if aggregator_kwargs is None:
        return False
    sliding_metric_functions = list(aggregator_kwargs["sliding_metric_functions"])
    if len(sliding_metric_functions) == 0:
        raise ValueError("sliding_metric_functions must be at least of length 1")
    windows = list(aggregator_kwargs["windows"])
    num_slices = aggregator_kwargs["num_slices_per_window"]
    if len(windows) == 0:
        raise ValueError(
            "windows must be provided if use_sliding_window_features is True"
        )
    if num_slices is not None and num_slices <= 0:
        raise ValueError("num_slices_per_window must be >= 1 if specified")
    if len(windows) > 1 and num_slices is None:
        raise ValueError("specify exactly one window if num_slices_per_window is None")
    slices = None
    if num_slices is not None:
        windows, slices = get_window_slices(num_slices, windows=windows)
    window_size = max(windows)
    return _AggregatorKwargs(
        metric_funcs=sliding_metric_functions,
        window_size=window_size,
        window_slices=slices,
        keep_original=aggregator_kwargs["keep_original_features"],
    )


class Parameters(TypedDict):
    balance_sample_weights: bool
    classifier_kwargs: Mapping[str, object]
    sampling_function_kwargs: Mapping[str, object]
    postprocessing_function_kwargs: Mapping[str, object]
    use_sliding_window_features: bool
    aggregator_kwargs: AggregatorKwargs | None


@final
class IntParameter:
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        self.name = name
        self.low = low
        self.high = high
        self.step = step

    def __call__(self, trial: optuna.trial.Trial | optuna.trial.FrozenTrial) -> int:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step)


@final
class FloatParameter:
    def __init__(self, name: str, low: float, high: float, step: float | None = None):
        self.name = name
        self.low = low
        self.high = high
        self.step = step

    def __call__(self, trial: optuna.trial.Trial | optuna.trial.FrozenTrial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, step=self.step)


@final
class CategoricalParameter:
    def __init__[T: (int, float, bool, str)](self, name: str, choices: Sequence[T]):
        self.name = name
        self.choices = choices

    def __call__(self, trial: optuna.trial.Trial | optuna.trial.FrozenTrial) -> ...:
        return trial.suggest_categorical(self.name, self.choices)


@final
class ParameterSpace:
    def __init__(
        self,
        balance_sample_weights: Callable[[optuna.trial.Trial], bool],
        use_sliding_window_features: Callable[[optuna.trial.Trial], bool],
        classifier_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        sampling_function_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        postprocessing_function_kwargs: Mapping[
            str, Callable[[optuna.trial.Trial], object]
        ]
        | None = None,
        aggregator_kwargs: Callable[
            [optuna.trial.Trial | optuna.trial.FrozenTrial], AggregatorKwargs
        ]
        | None = None,
    ):
        self.balance_sample_weights = balance_sample_weights
        self.use_sliding_window_features = use_sliding_window_features
        self.classifier_kwargs = classifier_kwargs or {}
        self.sampling_function_kwargs = sampling_function_kwargs or {}
        self.postprocessing_function_kwargs = postprocessing_function_kwargs or {}
        self.aggregator_kwargs = aggregator_kwargs

    def suggest(self, trial: optuna.trial.Trial) -> Parameters:
        balance_sample_weights = self.balance_sample_weights(trial)
        use_sliding_window_features = self.use_sliding_window_features(trial)
        aggregator_kwargs = None
        if self.aggregator_kwargs:
            aggregator_kwargs = self.aggregator_kwargs(trial)
        classifier_kwargs = {
            key: parameter(trial) for key, parameter in self.classifier_kwargs.items()
        }
        sampling_function_kwargs = {
            key: parameter(trial)
            for key, parameter in self.sampling_function_kwargs.items()
        }
        postprocessing_function_kwargs = {
            key: parameter(trial)
            for key, parameter in self.postprocessing_function_kwargs.items()
        }
        return Parameters(
            balance_sample_weights=balance_sample_weights,
            classifier_kwargs=classifier_kwargs,
            sampling_function_kwargs=sampling_function_kwargs,
            postprocessing_function_kwargs=postprocessing_function_kwargs,
            use_sliding_window_features=use_sliding_window_features,
            aggregator_kwargs=aggregator_kwargs,
        )

    def parse(
        self,
        parameters: Mapping[str, object],
        *,
        balance_sample_weights: bool | None = None,
        classifier_kwargs: Mapping[str, object] | None = None,
        sampling_function_kwargs: Mapping[str, object] | None = None,
        postprocessing_function_kwargs: Mapping[str, object] | None = None,
        use_sliding_window_features: bool | None = None,
        aggregator_kwargs: Mapping[str, object] | None = None,
    ) -> Parameters:
        classifier_kwargs = dict(classifier_kwargs) if classifier_kwargs else {}
        sampling_function_kwargs = (
            dict(sampling_function_kwargs) if sampling_function_kwargs else {}
        )
        postprocessing_function_kwargs = (
            dict(postprocessing_function_kwargs)
            if postprocessing_function_kwargs
            else {}
        )
        aggregator_kwargs = dict(aggregator_kwargs) if aggregator_kwargs else {}
        available_aggregator_kwargs: list[object] = []
        if self.aggregator_kwargs is not None:
            mock_trial = optuna.trial.create_trial()
            available_aggregator_kwargs = list(
                self.aggregator_kwargs(mock_trial).keys()
            )
        for param, value in parameters.items():
            if not param.startswith("param_"):
                continue
            param = param.replace("param_", "", 1)
            if param == "balance_sample_weights":
                balance_sample_weights = bool(value)
            elif param == "use_sliding_window_features":
                use_sliding_window_features = bool(value)
            elif param in self.classifier_kwargs:
                classifier_kwargs[param] = value
            elif param in self.sampling_function_kwargs:
                sampling_function_kwargs[param] = value
            elif param in self.postprocessing_function_kwargs:
                postprocessing_function_kwargs[param] = value
            elif param in available_aggregator_kwargs:
                aggregator_kwargs[param] = value
            else:
                raise ValueError(f"undefined parameter {param}")
        if balance_sample_weights is None:
            raise ValueError("expected value for balance_sample_weights")
        if use_sliding_window_features is None:
            raise ValueError("expected value for use_sliding_window_features")
        for param in (
            set(self.classifier_kwargs)
            | set(self.sampling_function_kwargs)
            | set(self.postprocessing_function_kwargs)
            | set(available_aggregator_kwargs)
        ) - (
            set(classifier_kwargs)
            | set(sampling_function_kwargs)
            | set(postprocessing_function_kwargs)
            | set(aggregator_kwargs)
        ):
            raise ValueError(f"expected value for {param}")
        _aggregator_kwargs = (
            AggregatorKwargs(**aggregator_kwargs)
            if (valid := is_aggregator_kwargs(aggregator_kwargs))
            else None
        )
        if not valid and aggregator_kwargs:
            raise ValueError("invalid aggregator_kwargs")
        return Parameters(
            balance_sample_weights=balance_sample_weights,
            classifier_kwargs=classifier_kwargs,
            sampling_function_kwargs=sampling_function_kwargs,
            postprocessing_function_kwargs=postprocessing_function_kwargs,
            use_sliding_window_features=use_sliding_window_features,
            aggregator_kwargs=_aggregator_kwargs,
        )

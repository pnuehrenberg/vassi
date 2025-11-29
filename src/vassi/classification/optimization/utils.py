from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Literal, TypedDict, final

import numpy as np
import optuna

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


class Parameters(TypedDict):
    balance_sample_weights: bool
    classifier_kwargs: Mapping[str, object]
    sampling_function_kwargs: Mapping[str, object]
    postprocessing_function_kwargs: Mapping[str, object]


@final
class IntParameter:
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        self.name = name
        self.low = low
        self.high = high
        self.step = step

    def __call__(self, trial: optuna.trial.Trial) -> int:
        return trial.suggest_int(self.name, self.low, self.high, step=self.step)


@final
class FloatParameter:
    def __init__(self, name: str, low: float, high: float, step: float | None = None):
        self.name = name
        self.low = low
        self.high = high
        self.step = step

    def __call__(self, trial: optuna.trial.Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, step=self.step)


@final
class CategoricalParameter:
    def __init__[T: (int, float, bool, str)](self, name: str, choices: Sequence[T]):
        self.name = name
        self.choices = choices

    def __call__(self, trial: optuna.trial.Trial) -> ...:
        return trial.suggest_categorical(self.name, self.choices)


@final
class ParameterSpace:
    def __init__(
        self,
        balance_sample_weights: Callable[[optuna.trial.Trial], bool],
        classifier_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        sampling_function_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        postprocessing_function_kwargs: Mapping[
            str, Callable[[optuna.trial.Trial], object]
        ]
        | None = None,
    ):
        self.balance_sample_weights = balance_sample_weights
        self.classifier_kwargs = classifier_kwargs or {}
        self.sampling_function_kwargs = sampling_function_kwargs or {}
        self.postprocessing_function_kwargs = postprocessing_function_kwargs or {}

    def suggest(self, trial: optuna.trial.Trial) -> Parameters:
        balance_sample_weights = self.balance_sample_weights(trial)
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
        )

    def parse(
        self,
        parameters: Mapping[str, object],
        *,
        balance_sample_weights: bool | None = None,
        classifier_kwargs: Mapping[str, object] | None = None,
        sampling_function_kwargs: Mapping[str, object] | None = None,
        postprocessing_function_kwargs: Mapping[str, object] | None = None,
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
        for param, value in parameters.items():
            if not param.startswith("param_"):
                continue
            param = param.replace("param_", "", 1)
            if param == "balance_sample_weights":
                balance_sample_weights = bool(value)
            elif param in self.classifier_kwargs:
                classifier_kwargs[param] = value
            elif param in self.sampling_function_kwargs:
                sampling_function_kwargs[param] = value
            elif param in self.postprocessing_function_kwargs:
                postprocessing_function_kwargs[param] = value
            else:
                raise ValueError(f"undefined parameter {param}")
        if balance_sample_weights is None:
            raise ValueError("expected value for balance_sample_weights")
        for param in (
            set(self.classifier_kwargs)
            | set(self.sampling_function_kwargs)
            | set(self.postprocessing_function_kwargs)
        ) - (
            set(classifier_kwargs)
            | set(sampling_function_kwargs)
            | set(postprocessing_function_kwargs)
        ):
            raise ValueError(f"expected value for {param}")
        return Parameters(
            balance_sample_weights=balance_sample_weights,
            classifier_kwargs=classifier_kwargs,
            sampling_function_kwargs=sampling_function_kwargs,
            postprocessing_function_kwargs=postprocessing_function_kwargs,
        )

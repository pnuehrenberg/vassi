from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, TypedDict, TypeVar

import numpy as np
import optuna
import pandas as pd

from ..classification.results import ClassificationResult, _NestedResult
from ..dataset.types import AnnotatedDataset, SamplingFunction
from ..features import BaseExtractor, F
from ..logging import log_time, set_logging_level, with_loop
from ..utils import Experiment, ensure_generator, to_int_seed
from .predict import k_fold_predict

if TYPE_CHECKING:
    from loguru import Logger


class OverlappingPredictionsKwargs(TypedDict):
    priority_func: Callable[[pd.DataFrame], Iterable[float]]
    prefilter_recipient_bouts: bool
    max_bout_gap: float
    max_allowed_bout_overlap: float


T = TypeVar("T", bound=ClassificationResult | _NestedResult)


class PostprocessingFunction(Protocol):
    def __call__(
        self,
        result: T,
        *args: ...,
        postprocessing_parameters: dict[str, Any],
        decision_thresholds: Iterable[float],
        **kwargs: ...,
    ) -> T: ...


class PostprocessingParameters(TypedDict):
    postprocessing_parameters: dict[str, Any]
    decision_thresholds: Iterable[float]


class ParameterSuggestionFunction(Protocol):
    def __call__(
        self,
        trial: optuna.trial.Trial,
        *args: ...,
        categories: Iterable[str],
        **kwargs: ...,
    ) -> PostprocessingParameters: ...


@log_time(
    level_start="debug",
    level_finish="info",
    description="optuna optimization trial",
)
def optuna_score_postprocessing_trial(
    classification_result: ClassificationResult | _NestedResult,
    trial: optuna.trial.Trial,
    *,
    postprocessing_function: PostprocessingFunction,
    postprocessing_parameters: PostprocessingParameters | ParameterSuggestionFunction,
    log: Optional[Logger],
) -> float:
    if callable(postprocessing_parameters):
        postprocessing_parameters = postprocessing_parameters(
            trial, categories=classification_result.categories
        )
    classification_result_processed = postprocessing_function(
        classification_result,
        **postprocessing_parameters,
        default_decision="none",
    )
    return float(np.nanmean(classification_result_processed.score()))


@log_time(
    level_start="info",
    level_finish="success",
    description="optuna optimization study",
)
def optuna_parameter_optimization(
    classification_result: ClassificationResult | _NestedResult,
    *,
    num_trials: int,
    random_state: Optional[int | np.random.Generator],
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    log: Optional[Logger],
) -> optuna.study.Study:
    random_state = ensure_generator(random_state)
    objective = partial(
        optuna_score_postprocessing_trial,
        classification_result,
        postprocessing_function=postprocessing_function,
        postprocessing_parameters=suggest_postprocessing_parameters_function,
        log=log,
    )
    optuna.logging.disable_default_handler()
    sampler = optuna.samplers.TPESampler(seed=to_int_seed(random_state))
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    return study


def postprocessing_optimization_run(
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    num_trials: int,
    k: int,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool,
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    random_state: Optional[int | np.random.Generator],
    log: Optional[Logger],
) -> optuna.study.Study:
    k_fold_result = k_fold_predict(
        dataset,
        extractor,
        classifier,
        k=k,
        random_state=random_state,
        sampling_function=sampling_function,
        balance_sample_weights=balance_sample_weights,
        log=log,
    )
    return optuna_parameter_optimization(
        k_fold_result,
        num_trials=num_trials,
        random_state=random_state,
        postprocessing_function=postprocessing_function,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters_function,
        log=log,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="postprocessing parameter optimization",
)
def optimize_postprocessing_parameters(
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    *,
    num_runs: int,
    num_trials: int,
    k: int,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs] = None,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    experiment: Optional[Experiment] = None,
    log: Logger | None = None,
) -> list[optuna.study.Study] | None:
    if experiment is None:
        experiment = Experiment()
    if log is None:
        log = set_logging_level()
    for run in range(num_runs):
        if not experiment.performs_run(run):
            continue
        log = with_loop(log, name="run", step=run)[0]
        experiment.add(
            run,
            postprocessing_optimization_run(
                dataset,
                extractor,
                classifier,
                k=k,
                random_state=experiment.get_random_state(run, num_runs=num_runs),
                sampling_function=sampling_function,
                balance_sample_weights=balance_sample_weights,
                postprocessing_function=postprocessing_function,
                suggest_postprocessing_parameters_function=suggest_postprocessing_parameters_function,
                num_trials=num_trials,
                log=log,
            ),
        )
    if not experiment.is_root:
        return
    studies = []
    for run, study in sorted(experiment.collect(num_runs=num_runs).items()):
        studies.append(study)
    return studies

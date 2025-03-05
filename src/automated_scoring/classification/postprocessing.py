from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, TypedDict, TypeVar

import numpy as np
import optuna
import pandas as pd
from scipy.stats import gaussian_kde

from ..dataset.types import AnnotatedDataset, SamplingFunction
from ..features import BaseExtractor, Shaped
from ..io import to_yaml
from ..logging import increment_loop, log_time, set_logging_level, with_loop
from ..utils import Experiment, to_int_seed
from .predict import k_fold_predict
from .results import ClassificationResult, _NestedResult

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


# @log_time(
#     level_start="debug",
#     level_finish="info",
#     description="optuna optimization trial",
# )
def optuna_score_postprocessing_trial(
    classification_result: ClassificationResult | _NestedResult,
    trial: optuna.trial.Trial,
    *,
    postprocessing_function: PostprocessingFunction,
    postprocessing_parameters: PostprocessingParameters | ParameterSuggestionFunction,
    loop_log: tuple[Logger, str],
) -> float:
    log, loop_name = loop_log
    if callable(postprocessing_parameters):
        postprocessing_parameters = postprocessing_parameters(
            trial, categories=classification_result.categories
        )
    classification_result_processed = postprocessing_function(
        classification_result,
        **postprocessing_parameters,
        default_decision="none",
    )
    score = float(np.nanmean(classification_result_processed.score()))
    increment_loop(log, name=loop_name).info(f"score: {score:.3f}")
    return score


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
    log: Logger,
) -> optuna.study.Study:
    random_state = np.random.default_rng(random_state)
    objective = partial(
        optuna_score_postprocessing_trial,
        classification_result,
        postprocessing_function=postprocessing_function,
        postprocessing_parameters=suggest_postprocessing_parameters_function,
        loop_log=with_loop(log, name="optuna trial", step=0, total=num_trials),
    )
    optuna.logging.disable_default_handler()
    sampler = optuna.samplers.TPESampler(seed=to_int_seed(random_state))
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    return study


def postprocessing_optimization_run[F: Shaped](
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
    log: Logger,
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
def optimize_postprocessing_parameters[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    *,
    num_trials: int,
    k: int,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    experiment: Experiment,
    log: Optional[Logger] = None,
) -> list[optuna.study.Study] | None:
    if log is None:
        log = set_logging_level()
    for run in experiment:
        log = with_loop(log, name="run", step=run)[0]
        experiment.add(
            postprocessing_optimization_run(
                dataset,
                extractor,
                classifier,
                k=k,
                random_state=experiment.random_state,
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
    return list(experiment.collect().values())


def summarize_experiment(
    studies: list[optuna.study.Study],
    *,
    results_file: str = "optimization-results.yaml",
    summary_file: str = "optimization-summary.yaml",
    log: Optional[Logger] = None,
):
    if log is None:
        log = set_logging_level()
    results = [
        {
            "best": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "results": study.trials_dataframe(("number", "params", "value")).to_dict(
                orient="list"
            ),
        }
        for study in studies
    ]
    parameters = list(results[0]["best_params"].keys())
    best_parameters = {}
    for parameter in parameters:
        tested_values = [
            trial[f"params_{parameter}"]
            for result in results
            for trial in pd.DataFrame(result["results"]).to_dict(orient="records")
        ]
        values = np.unique(tested_values)
        best_values = [result["best_params"][parameter] for result in results]
        density = gaussian_kde(best_values)(np.unique(tested_values))
        best = np.argmax(density)
        best_value = values[best]
        if "window" in parameter:
            best_value = int(best_value + 1)
        else:
            best_value = float(best_value)
        best_parameters[parameter] = best_value
    to_yaml(results, file_name=results_file)
    to_yaml(best_parameters, file_name=summary_file)
    space = max([len(parameter) for parameter in best_parameters]) + 10
    decimals = 3
    summary = "\n".join(
        [
            f"{key:<{space}}{value:{f'.{decimals}f' if isinstance(value, float) else ''}}"
            for key, value in best_parameters.items()
        ]
    )
    log.success("Best parameters:\n" + summary)
    return best_parameters

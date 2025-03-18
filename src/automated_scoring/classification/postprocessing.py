from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from functools import partial
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypedDict, cast

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import gaussian_kde

from ..dataset.types import AnnotatedDataset, SamplingFunction
from ..features import BaseExtractor, Shaped
from ..features.caching import from_cache, remove_cache, to_cache
from ..io import to_yaml
from ..logging import (
    _create_log_in_subprocess,
    increment_loop,
    log_time,
    set_logging_level,
    with_loop,
)
from ..utils import Experiment, to_int_seed
from .predict import k_fold_predict
from .results import ClassificationResult, DatasetClassificationResult, _NestedResult

if TYPE_CHECKING:
    from loguru import Logger


class PostprocessingParameters(TypedDict):
    postprocessing_parameters: dict[str, Any]
    decision_thresholds: Iterable[float]


class PostprocessingFunction[T: ClassificationResult | _NestedResult](Protocol):
    def __call__(
        self,
        result: T,
        *args: ...,
        postprocessing_parameters: dict[str, Any],
        decision_thresholds: Iterable[float],
        **kwargs: ...,
    ) -> T: ...


class ParameterSuggestionFunction(Protocol):
    def __call__(
        self,
        trial: optuna.trial.Trial,
        *args: ...,
        categories: Iterable[str],
        **kwargs: ...,
    ) -> PostprocessingParameters: ...


_log: Logger | None = None


def _result_from_cache[T: ClassificationResult | _NestedResult](result: str | T) -> T:
    if isinstance(result, str):
        cached = from_cache(result)
        if TYPE_CHECKING:
            cached = cast(T, cached)
        return cached
    return result


def optuna_score_postprocessing_trial[T: ClassificationResult | _NestedResult](
    classification_result: T | str | list[T | str],
    trial: optuna.trial.Trial,
    *,
    postprocessing_function: PostprocessingFunction[T],
    postprocessing_parameters: PostprocessingParameters | ParameterSuggestionFunction,
    loop_log: tuple[Logger, str] | tuple[tuple[dict[str, Any], int], str],
) -> float:
    global _log
    if not isinstance(classification_result, list):
        classification_result = [_result_from_cache(classification_result)]
    result = _result_from_cache(classification_result[0])
    if callable(postprocessing_parameters):
        postprocessing_parameters = postprocessing_parameters(
            trial,
            categories=result.categories,
        )
    scores = []
    if "scores" in postprocessing_parameters["postprocessing_parameters"]:
        score_selection: list[str] | slice = postprocessing_parameters[
            "postprocessing_parameters"
        ]["scores"]
        score_levels = ["timestamp", "annotation", "prediction"]
        if not isinstance(score_selection, list) or not all(
            [level in score_levels for level in score_selection]
        ):
            raise ValueError(f"scores must be a list constrained to {score_levels}")
    else:
        score_selection = slice(None)
    for result in [result] + classification_result[1:]:
        scores.append(
            np.nanmean(
                postprocessing_function(
                    _result_from_cache(result),
                    **postprocessing_parameters,
                    default_decision="none",
                )
                .score()
                .loc[slice(None), score_selection]
            )
        )
    score = float(np.mean(scores))
    log, loop_name = loop_log
    if isinstance(log, tuple) and _log is not None:
        log = _log
    elif isinstance(log, tuple) and _log is None:
        _log = _create_log_in_subprocess(*log)
        log = _log
    if TYPE_CHECKING:
        assert isinstance(log, Logger)
    increment_loop(log, name=loop_name).info(f"score: {score:.3f}")
    return score


def _optimize_study(
    study_name: str,
    storage: optuna.storages.BaseStorage,
    objective: Callable,
    num_trials: int,
) -> None:
    global _log
    optuna.logging.disable_default_handler()
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True
    )
    study.optimize(objective, n_trials=num_trials)
    _log = None


def _execute_parallel_study[T: ClassificationResult | _NestedResult](
    objective: Callable,
    *,
    study_name: str,
    storage: optuna.storages.BaseStorage,
    num_trials: int,
    log: Logger,
) -> None:
    num_cpus = cpu_count()
    num_inner_threads = num_cpus // 4
    num_jobs = num_cpus // num_inner_threads
    with parallel_config(backend="loky", inner_max_num_threads=num_inner_threads):
        Parallel(n_jobs=num_jobs)(
            delayed(_optimize_study)(
                study_name,
                storage,
                partial(
                    objective,
                    loop_log=with_loop(
                        log,
                        name=f"job: {job} | optuna trial",
                        step=0,
                        total=num_trials // num_jobs,
                        prepare_for_subprocess=True,
                    ),
                ),
                num_trials=num_trials // num_jobs,
            )
            for job in range(num_jobs)
        )


@log_time(
    level_start="info",
    level_finish="success",
    description="optuna optimization study",
)
def optuna_parameter_optimization[T: ClassificationResult | _NestedResult](
    classification_result: T | str | list[T | str],
    *,
    num_trials: int,
    random_state: Optional[int | np.random.Generator],
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    parallel_optimization: bool,
    experiment: Optional[Experiment],
    log: Logger,
) -> optuna.study.Study:
    random_state = np.random.default_rng(random_state)
    objective = partial(
        optuna_score_postprocessing_trial,
        classification_result,
        postprocessing_function=postprocessing_function,
        postprocessing_parameters=suggest_postprocessing_parameters_function,
    )
    optuna.logging.disable_default_handler()
    if not parallel_optimization:
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=to_int_seed(random_state)),
            direction="maximize",
        )
        study.optimize(
            partial(
                objective,
                loop_log=with_loop(log, name="optuna trial", step=0, total=num_trials),
            ),
            n_trials=num_trials,
        )
        return study
    if experiment is not None and experiment.is_distributed:
        num_trials = num_trials // experiment.num_runs
    storage = None
    if experiment is None or experiment.is_root:
        os.makedirs("optuna_logs", exist_ok=True)
        _, storage_file = tempfile.mkstemp(suffix=".optuna-log", dir="optuna_logs")
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage_file),  # type: ignore
        )
        optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=to_int_seed(random_state)),
            storage=storage,
            direction="maximize",
            study_name="optuna_study",
        )
    if experiment is not None:
        storage = experiment.broadcast(storage)
    if TYPE_CHECKING:
        assert storage is not None
    execute_parallel_study = partial(
        _execute_parallel_study,
        objective,
        study_name="optuna_study",
        storage=storage,
        num_trials=num_trials,
    )
    if experiment is not None:
        for run in experiment:
            execute_parallel_study(log=with_loop(log, name="MPI process", step=run)[0])
        experiment.barrier()
    else:
        execute_parallel_study(log=log)
    return optuna.load_study(storage=storage, study_name="optuna_study")


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
    optimize_across_runs: bool = False,
    parallel_optimization: bool = False,
    log: Optional[Logger] = None,
) -> list[optuna.study.Study] | optuna.study.Study:
    if log is None:
        log = set_logging_level()
    for run in experiment:
        _log = with_loop(log, name="run", step=run)[0]
        k_fold_result = k_fold_predict(
            dataset,
            extractor,
            classifier,
            k=k,
            random_state=experiment.random_state,
            sampling_function=sampling_function,
            balance_sample_weights=balance_sample_weights,
            log=_log,
        )
        k_fold_result = (
            k_fold_result if not parallel_optimization else to_cache(k_fold_result)
        )
        if not optimize_across_runs:
            study = optuna_parameter_optimization(
                k_fold_result,
                num_trials=num_trials,
                random_state=experiment.random_state,
                postprocessing_function=postprocessing_function,
                suggest_postprocessing_parameters_function=suggest_postprocessing_parameters_function,
                parallel_optimization=parallel_optimization,
                experiment=None,
                log=_log,
            )
            experiment.add(study)
            continue
        experiment.add(k_fold_result)
    if not optimize_across_runs:
        return list(experiment.collect().values())
    k_fold_results: list[str | DatasetClassificationResult] = list(
        experiment.collect().values()
    )
    study = optuna_parameter_optimization(
        k_fold_results,
        num_trials=num_trials,
        random_state=experiment.get_random_state(
            experiment.num_runs, num_runs=experiment.num_runs + 1
        ),
        postprocessing_function=postprocessing_function,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters_function,
        parallel_optimization=parallel_optimization,
        experiment=experiment,
        log=log,
    )
    experiment.barrier()
    for result in k_fold_results:
        if not isinstance(result, str):
            continue
        remove_cache(result)
    return study


def summarize_experiment(
    studies: list[optuna.study.Study] | optuna.study.Study,
    *,
    results_file: str = "optimization-results.yaml",
    summary_file: str = "optimization-summary.yaml",
    trials_file: str = "optimization-trials.csv",
    log: Optional[Logger] = None,
):
    if log is None:
        log = set_logging_level()
    all_trials = []
    if not isinstance(studies, list):
        studies = [studies]
    for idx, study in enumerate(studies):
        trials = study.trials_dataframe(("number", "params", "value"))
        columns = trials.columns
        trials["study"] = idx
        all_trials.append(trials[["study", *columns]])
    all_trials = pd.concat(all_trials, axis=0, ignore_index=True)
    results = [
        {
            "best": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": {**study.best_params},
        }
        for study in studies
    ]
    parameters = list(results[0]["best_params"].keys())
    best_parameters = {}
    for parameter in parameters:
        tested_values = all_trials[f"params_{parameter}"]
        values = np.unique(tested_values)
        best_values = np.array([result["best_params"][parameter] for result in results])
        if len(best_values) == 1:
            best_value = best_values.tolist()[0]
        elif not (
            np.issubdtype(values.dtype, np.floating)
            or np.issubdtype(values.dtype, np.integer)
        ):
            unique_best_values, counts = np.unique(best_values, return_counts=True)
            # tolist also converts np dtype to native
            best_value = unique_best_values.tolist()[np.argmax(counts)]
        else:
            density = gaussian_kde(best_values)(values)
            best = np.argmax(density)
            best_value = values[best]
        if "window" in parameter:
            best_value = int(best_value + 1)
        else:
            best_value = float(best_value)
        best_parameters[parameter] = best_value
    all_trials.to_csv(trials_file, index=False)
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

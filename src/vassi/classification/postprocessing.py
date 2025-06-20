from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    cast,
    overload,
)

import loguru
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import gaussian_kde

from ..dataset.types import AnnotatedDataset, SamplingFunction
from ..features import BaseExtractor, Shaped
from ..io import from_cache, remove_cache, to_cache, to_yaml
from ..logging import (
    _create_log_in_subprocess,
    increment_loop,
    log_time,
    set_logging_level,
    with_loop,
)
from ..utils import Experiment, available_resources, to_int_seed, to_scalars
from .predict import k_fold_predict
from .results import (
    ClassificationResult,
    DatasetClassificationResult,
    GroupClassificationResult,
)

Result = ClassificationResult | GroupClassificationResult | DatasetClassificationResult


class PostprocessingParameters(TypedDict):
    """Typed dictionary that holds parameters for postprocessing."""

    postprocessing_parameters: dict[str, Any]
    decision_thresholds: Iterable[float]


class PostprocessingFunction[T: Result](Protocol):
    """
    Protocol for postprocessing functions.

    Parameters:
        result (:class:`~vassi.classification.results.ClassificationResult` | :class:`~vassi.classification.results.GroupClassificationResult` | :class:`~vassi.classification.results.DatasetClassificationResult`): The result to be postprocessed.
        *args: Additional arguments.
        postprocessing_parameters (dict[str, Any]): Parameters for postprocessing.
        decision_thresholds (Iterable[float]): Category-specific decision thresholds.
        **kwargs: Additional keyword arguments.

    Returns:
        (:class:`~vassi.classification.results.ClassificationResult` | :class:`~vassi.classification.results.GroupClassificationResult` | :class:`~vassi.classification.results.DatasetClassificationResult`): The result after postprocessing.

    See also:
        :class:`PostprocessingParameters` to specify the two required keyword arguments with :code:`**postprocessing_parameters`.
    """

    def __call__(
        self,
        result: T,
        *args: ...,
        postprocessing_parameters: dict[str, Any],
        decision_thresholds: Iterable[float],
        **kwargs: ...,
    ) -> T: ...


class ParameterSuggestionFunction(Protocol):
    """
    Protocol for parameter suggestion functions.

    Parameters:
        trial (:class:`~optuna.trial.Trial`): The trial object.
        *args: Additional arguments.
        categories (Iterable[str]): Category names.
        **kwargs: Additional keyword arguments.

    Returns:
        :class:`PostprocessingParameters`: Parameters for postprocessing.
    """

    def __call__(
        self,
        trial: optuna.trial.Trial,
        *args: ...,
        categories: Iterable[str],
        **kwargs: ...,
    ) -> PostprocessingParameters: ...


_log: loguru.Logger | None = None


def _result_from_cache[T: Result](result: str | T) -> T:
    if isinstance(result, str):
        cached = from_cache(result)
        if TYPE_CHECKING:
            cached = cast(T, cached)
        return cached
    return result


def optuna_score_postprocessing_trial[T: Result](
    classification_result: T | str | list[T] | list[str],
    trial: optuna.trial.Trial,
    *,
    postprocessing_function: PostprocessingFunction[T],
    postprocessing_parameters: PostprocessingParameters | ParameterSuggestionFunction,
    loop_log: tuple[loguru.Logger, str] | tuple[tuple[dict[str, Any], int], str],
) -> float:
    """
    Run a single :class:`~optuna.trial.Trial` to evaluate postprocessing parameters for a specified postprocessing function.

    Parameters:
        classification_result: The classification result(s) to postprocess, can also be passed as a string or a list of strings to read from cache files.
        trial: One optuna trial of the current optuna study.
        postprocessing_function: The postprocessing function to use.
        postprocessing_parameters: The postprocessing parameters to evaluate, can also be a callable that returns a dictionary of parameters given the optuna trial.
        loop_log: The logger and log name to use for logging. In a multiprocessing environment, the logger should be passed as a tuple of logger parameters (dict) and log level (int).

    Returns:
        The score of the postprocessed classification result, calculated as the average of 'timestamp', 'annotation' and 'prediction' macro F1 scores.
    """
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
                .loc[score_selection, slice(None)]
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
        assert isinstance(log, loguru.Logger)
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


def _execute_parallel_study[T: Result](
    objective: Callable,
    *,
    study_name: str,
    storage: optuna.storages.BaseStorage,
    num_trials: int,
    log: loguru.Logger,
) -> None:
    num_jobs, num_inner_threads = available_resources()
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
def optuna_parameter_optimization[T: Result](
    classification_result: T | str | list[T] | list[str],
    *,
    num_trials: int,
    random_state: Optional[int | np.random.Generator],
    postprocessing_function: PostprocessingFunction,
    suggest_postprocessing_parameters_function: ParameterSuggestionFunction,
    parallel_optimization: bool,
    experiment: Optional[Experiment],
    log: loguru.Logger,
) -> optuna.study.Study:
    """
    Perform a parameter optimization study using Optuna.

    Parameters:
        classification_result: The classification result(s) to postprocess, can also be passed as a string or a list of strings to read from cache files.
        num_trials: The number of Optuna trials to run.
        random_state: The random state to use for the optimization.
        postprocessing_function: The postprocessing function to use.
        suggest_postprocessing_parameters_function: A callable that suggests postprocessing parameters for the specified postprocessing function.
        parallel_optimization: Whether to run the Optuna study in parallel.
        experiment: Should be specified in a distributed setting.
        log: The Loguru logger to use for logging.

    Returns:
        The Optuna study.
    """
    random_state = np.random.default_rng(random_state)
    objective = partial(
        optuna_score_postprocessing_trial,
        classification_result,
        postprocessing_function=postprocessing_function,
        postprocessing_parameters=suggest_postprocessing_parameters_function,
    )
    optuna.logging.disable_default_handler()
    if not parallel_optimization:
        study = None
        if experiment is None or experiment.is_root:
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=to_int_seed(random_state)),
                direction="maximize",
            )
            study.optimize(
                partial(
                    objective,
                    loop_log=with_loop(
                        log, name="optuna trial", step=0, total=num_trials
                    ),
                ),
                n_trials=num_trials,
            )
            if experiment is None:
                return study
        study = experiment.broadcast(study)
        if TYPE_CHECKING:
            assert study is not None
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


@overload
def run_k_fold_experiment[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    k: int,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    experiment: Experiment,
    log: Optional[loguru.Logger] = None,
    cache: Literal[True],
) -> list[str]: ...


@overload
def run_k_fold_experiment[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    k: int,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    experiment: Experiment,
    log: Optional[loguru.Logger] = None,
    cache: Literal[False] = False,
) -> list[DatasetClassificationResult]: ...


@log_time(
    level_start="info",
    level_finish="success",
    description="k-fold experiment",
)
def run_k_fold_experiment[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    k: int,
    sampling_function: SamplingFunction,
    balance_sample_weights: bool = True,
    experiment: Experiment,
    log: Optional[loguru.Logger] = None,
    cache: bool = True,
) -> list[DatasetClassificationResult] | list[str]:
    """
    Perform a k-fold prediction experiment on the given dataset.

    Parameters:
        dataset: The dataset to perform the experiment on.
        extractor: The extractor to use for feature extraction.
        classifier: The classifier to use for classification.
        k: The number of folds to use.
        sampling_function: The sampling function to use.
        balance_sample_weights: Whether to balance sample weights for model fitting.
        experiment: The experiment to run, can also be a :class:`~vassi.distributed.DistributedExperiment`.
        log: The logger to use.
        cache: Whether to cache the results.

    Returns:
        The results of the experiment (as list of cache files if :code:`cache=True`)
    """
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
        if cache:
            k_fold_result = to_cache(k_fold_result)
        experiment.add(k_fold_result)
    k_fold_results = list(experiment.collect().values())
    if TYPE_CHECKING and cache:
        k_fold_results = [
            k_fold_result
            for k_fold_result in k_fold_results
            if isinstance(k_fold_result, str)
        ]
    elif TYPE_CHECKING:
        k_fold_results = [
            k_fold_result
            for k_fold_result in k_fold_results
            if isinstance(k_fold_result, DatasetClassificationResult)
        ]
    return k_fold_results


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
    log: Optional[loguru.Logger] = None,
) -> list[optuna.study.Study] | optuna.study.Study:
    """
    Sequentially perform a k-fold prediction experiment and use the results to optimize postprocessing parameters.

    See also:
        - :func:`run_k_fold_experiment` to perform a k-fold prediction experiment.
        - :func:`optuna_parameter_optimization` to optimize postprocessing parameters on existing classification results.

    Parameters:
        dataset: The dataset to use.
        extractor: The extractor to use for feature extraction.
        classifier: The classifier to use for classification.
        postprocessing_function: The postprocessing function to use.
        suggest_postprocessing_parameters_function: A callable that suggests postprocessing parameters.
        num_trials: The number of trials to perform in the Optuna optimization study.
        k: The number of folds to use for the k-fold experiment.
        sampling_function: The sampling function to use during k-fold prediction.
        balance_sample_weights: Whether to balance the sample weights during model fitting.
        experiment: The experiment to use for the experiment (specifies number of runs and random state).
        optimize_across_runs: Whether to optimize postprocessing parameters across experiment runs, or for each run individually.
        parallel_optimization: Whether to perform the Optuna optimization study/studies in parallel.
        log: The Loguru logger to use for the experiment.

    Returns:
        The list of Optuna studies (:code:`optimize_across_runs=False`) or alternatively, a single study.
    """
    if log is None:
        log = set_logging_level()
    k_fold_results = run_k_fold_experiment(
        dataset,
        extractor,
        classifier,
        k=k,
        sampling_function=sampling_function,
        balance_sample_weights=balance_sample_weights,
        experiment=experiment,
        log=log,
        cache=optimize_across_runs | parallel_optimization,
    )
    if not optimize_across_runs:
        for run in experiment:
            _log = with_loop(log, name="run", step=run)[0]
            study = optuna_parameter_optimization(
                k_fold_results[run],
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
        return list(experiment.collect().values())
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
    log: Optional[loguru.Logger] = None,
):
    """
    Summarize one or more Optuna studies resulting from an optimization experiment.

    See also:
        :func:`optimize_postprocessing_parameters`

    Parameters:
        studies: One or more Optuna studies to summarize.
        results_file: Path to the file where the results will be saved.
        summary_file: Path to the file where the summary will be saved.
        trials_file: Path to the file where the trials will be saved.
        log: Loguru logger to use for logging.

    Returns:
        None
    """
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
            best_value = to_scalars(best_values)[0]
        elif not (
            np.issubdtype(values.dtype, np.floating)
            or np.issubdtype(values.dtype, np.integer)
        ):
            unique_best_values, counts = np.unique(best_values, return_counts=True)
            best_value = to_scalars(unique_best_values)[np.argmax(counts)]
        else:
            density = gaussian_kde(best_values)(values)
            best = np.argmax(density)
            best_value = values[best]
        if "window" in parameter:
            best_value = int(best_value + 1)
        elif isinstance(best_value, bool | np.bool_):
            best_value = bool(best_value)
        else:
            try:
                best_value = float(best_value)
            except ValueError:
                best_value = str(best_value)
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

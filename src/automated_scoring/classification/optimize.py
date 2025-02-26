from __future__ import annotations

import os
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..dataset import AnnotatedDataset
from ..features import BaseExtractor, F
from ..logging import log_loop, log_time, set_logging_level, with_loop
from ..utils import IterationManager, SmoothingFunction
from . import optimization_utils as utils
from .predict import k_fold_predict
from .results import DatasetClassificationResult
from .utils import EncodingFunction, SamplingFunction

if TYPE_CHECKING:
    from loguru import Logger


@log_time(
    level_start="info",
    level_finish="success",
    description="scoring smoothing parameters",
)
def _score_smoothed_results(
    smoothing_func: SmoothingFunction,
    parameter_combinations: list[list[dict[str, Any]]],
    classification_result: DatasetClassificationResult,
    encode_func: EncodingFunction,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs],
    iteration: int,
    log: Logger,
) -> dict[str, list[dict[str, Any]]]:
    num_categories = len(parameter_combinations)
    categories = classification_result.categories
    if not num_categories == len(categories):
        raise ValueError(
            "number of decision thresholds does not match number of categories"
        )
    results = {category: [] for category in categories}
    for log, (category_idx, category) in log_loop(
        enumerate(categories),
        level="info",
        message="scored smoothing parameters",
        name="category",
        total=num_categories,
        log=log,
    ):
        for log, parameters in log_loop(
            parameter_combinations[category_idx],
            level="info",
            message="scored parameters",
            log=log,
        ):
            smoothing_funcs = [utils.passthrough for _ in range(num_categories)]
            smoothing_funcs[category_idx] = partial(smoothing_func, parameters)
            classification_result = classification_result.smooth(smoothing_funcs)
            if remove_overlapping_predictions:
                if overlapping_predictions_kwargs is None:
                    raise ValueError(
                        "overlapping_predictions_kwargs must be provided when remove_overlapping_predictions is True"
                    )
                classification_result = (
                    classification_result.remove_overlapping_predictions(
                        **overlapping_predictions_kwargs
                    )
                )
            results[category].append(
                {
                    "iteration": iteration,
                    **parameters,
                    **classification_result.score(encode_func=encode_func, macro=True),
                }
            )
    return results


def score_smoothing(
    smoothing_func: SmoothingFunction,
    parameter_combinations: list[list[dict[str, Any]]],
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs],
    iteration: int,
    k: int,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    log: Logger,
) -> dict[str, list[dict[str, Any]]]:
    classification_result = k_fold_predict(
        dataset,
        extractor,
        classifier,
        k=k,
        random_state=random_state,
        sampling_func=sampling_func,
        balance_sample_weights=balance_sample_weights,
        encode_func=dataset.encode,
        log=log,
    )
    return _score_smoothed_results(
        smoothing_func,
        parameter_combinations,
        classification_result,
        dataset.encode,
        remove_overlapping_predictions=remove_overlapping_predictions,
        overlapping_predictions_kwargs=overlapping_predictions_kwargs,
        iteration=iteration,
        log=log,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="scoring decision thresholds",
)
def _score_thresholds(
    decision_thresholds: list[NDArray],
    classification_result: DatasetClassificationResult,
    encode_func: EncodingFunction,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs],
    default_decision: int | str,
    iteration: int,
    log: Logger,
) -> dict[str, list[dict[str, Any]]]:
    num_categories = len(decision_thresholds)
    if not num_categories == len(classification_result.categories):
        raise ValueError(
            "number of decision thresholds does not match number of categories"
        )
    categories = classification_result.categories
    results = {category: [] for category in categories}
    for log, (category_idx, category) in log_loop(
        enumerate(categories),
        level="info",
        message="scored thresholds",
        name="category",
        total=num_categories,
        log=log,
    ):
        for log, threshold in log_loop(
            decision_thresholds[category_idx],
            level="info",
            message="scored thresholds",
            name="threshold",
            log=log,
        ):
            thresholds = np.zeros(num_categories)
            thresholds[category_idx] = threshold
            classification_result = classification_result.threshold(
                thresholds,
                default_decision=default_decision,
            )
            if remove_overlapping_predictions:
                if overlapping_predictions_kwargs is None:
                    raise ValueError(
                        "overlapping_predictions_kwargs must be provided if remove_overlapping_predictions is True"
                    )
                classification_result = (
                    classification_result.remove_overlapping_predictions(
                        **overlapping_predictions_kwargs
                    )
                )
            results[category].append(
                {
                    "iteration": iteration,
                    "threshold": threshold,
                    **classification_result.score(encode_func=encode_func, macro=True),
                }
            )
    return results


def score_thresholds(
    decision_thresholds: list[NDArray],
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs],
    default_decision: int | str,
    iteration: int,
    k: int,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    smoothing_funcs: Iterable[SmoothingFunction] | None,
    log: Logger,
) -> dict[str, list[dict[str, Any]]]:
    classification_result = k_fold_predict(
        dataset,
        extractor,
        classifier,
        k=k,
        random_state=random_state,
        sampling_func=sampling_func,
        balance_sample_weights=balance_sample_weights,
        encode_func=dataset.encode,
        log=log,
    )
    if smoothing_funcs is not None:
        classification_result = classification_result.smooth(
            smoothing_funcs, threshold=False
        )
    return _score_thresholds(
        decision_thresholds,
        classification_result,
        dataset.encode,
        remove_overlapping_predictions=remove_overlapping_predictions,
        overlapping_predictions_kwargs=overlapping_predictions_kwargs,
        default_decision=default_decision,
        iteration=iteration,
        log=log,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="smoothing optimization",
)
def optimize_smoothing(
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    smoothing_func: SmoothingFunction,
    smoothing_parameters_grid: dict[str, Iterable] | list[dict[str, Iterable]],
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs] = None,
    num_iterations: int,
    plot_results: bool = True,
    k: int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    results_path: Optional[str] = None,
    log: Logger | None = None,
    iteration_manager: IterationManager | None = None,
) -> dict[str, dict[str, Any]] | None:
    if iteration_manager is None:
        iteration_manager = IterationManager()
    if log is None:
        log = set_logging_level()
    num_categories = len(dataset.categories)
    if isinstance(smoothing_parameters_grid, dict):
        smoothing_parameters_grid = [smoothing_parameters_grid] * num_categories
    parameter_combinations: list[list[dict[str, Any]]] = []
    for idx in range(len(smoothing_parameters_grid)):
        parameter_combinations.append(
            utils.parameter_grid_to_combinations(smoothing_parameters_grid[idx])
        )
    for iteration in range(num_iterations):
        if not iteration_manager.do_iteration(iteration):
            continue
        log = with_loop(log, name="iteration", step=iteration)[0]
        iteration_manager.add(
            iteration,
            score_smoothing(
                smoothing_func,
                parameter_combinations,
                dataset,
                extractor,
                classifier,
                remove_overlapping_predictions=remove_overlapping_predictions,
                overlapping_predictions_kwargs=overlapping_predictions_kwargs,
                iteration=iteration,
                k=k,
                random_state=iteration_manager.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                log=log,
            ),
        )
    if not iteration_manager.is_root:
        return
    results = {category: [] for category in dataset.categories}
    for iteration_results in iteration_manager.collect(
        num_iterations=num_iterations
    ).values():
        for category, category_results in iteration_results.items():
            results[category].extend(category_results)
    results = {
        category: pd.DataFrame(category_results)
        for category, category_results in results.items()
    }
    if results_path is not None:
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        for category, category_results in results.items():
            category_results.to_csv(
                os.path.join(results_path, f"results_smoothing-{category}.csv"),
                index=False,
            )
    parameter_names = list(smoothing_parameters_grid[0].keys())
    return {
        category: utils.evaluate_results(
            category_results,
            parameter_names=parameter_names,
            plot_results=plot_results,
        )
        for category, category_results in results.items()
    }


@log_time(
    level_start="info",
    level_finish="success",
    description="decision threshold optimization",
)
def optimize_decision_thresholds(
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[utils.OverlappingPredictionsKwargs] = None,
    num_iterations: int,
    decision_threshold_range: tuple[float, float] | Iterable[tuple[float, float]] = (
        0.0,
        1.0,
    ),
    decision_threshold_step: float | Iterable[float] = 0.01,
    default_decision: int | str = "none",
    smoothing_funcs: Optional[Iterable[SmoothingFunction]] = None,
    plot_results: bool = True,
    k: int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    results_path: Optional[str] = None,
    log: Logger | None = None,
    iteration_manager: IterationManager | None = None,
) -> dict[str, dict[str, Any]] | None:
    if iteration_manager is None:
        iteration_manager = IterationManager()
    if log is None:
        log = set_logging_level()
    num_categories = len(dataset.categories)
    decision_thresholds = utils.prepare_thresholds(
        decision_threshold_range, decision_threshold_step, num_categories
    )
    if encode_func is None:
        try:
            encode_func = dataset.encode
        except ValueError:
            raise ValueError("specify encode_func for non-annotated datasets")
    for iteration in range(num_iterations):
        if not iteration_manager.do_iteration(iteration):
            continue
        log = with_loop(log, name="iteration", step=iteration)[0]
        iteration_manager.add(
            iteration,
            score_thresholds(
                decision_thresholds,
                dataset,
                extractor,
                classifier,
                remove_overlapping_predictions=remove_overlapping_predictions,
                overlapping_predictions_kwargs=overlapping_predictions_kwargs,
                default_decision=default_decision,
                iteration=iteration,
                k=k,
                random_state=iteration_manager.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                smoothing_funcs=smoothing_funcs,
                log=log,
            ),
        )
    if not iteration_manager.is_root:
        return
    results = {category: [] for category in dataset.categories}
    for iteration_results in iteration_manager.collect(
        num_iterations=num_iterations
    ).values():
        for category, category_results in iteration_results.items():
            results[category].extend(category_results)
    results = {
        category: pd.DataFrame(category_results)
        for category, category_results in results.items()
    }
    if results_path is not None:
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        for category, category_results in results.items():
            category_results.to_csv(
                os.path.join(results_path, f"results_thresholding-{category}.csv"),
                index=False,
            )
    return {
        category: utils.evaluate_results(
            category_results,
            parameter_names=["threshold"],
            plot_results=plot_results,
        )
        for category, category_results in results.items()
    }

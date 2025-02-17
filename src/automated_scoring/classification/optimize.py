from __future__ import annotations

import os
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..dataset import AnnotatedDataset
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..logging import log_loop, log_time, set_logging_level, with_loop
from ..utils import MPIContext, SmoothingFunction
from . import _optimization_utils as utils
from .predict import k_fold_predict
from .results import DatasetClassificationResult
from .utils import EncodingFunction, SamplingFunction

if TYPE_CHECKING:
    from loguru import Logger


class OverlappingPredictionsKwargs(TypedDict):
    priority_func: Callable[[pd.DataFrame], Iterable[float]]
    prefilter_recipient_bouts: bool
    max_bout_gap: float
    max_allowed_bout_overlap: float


@log_time(
    level_start="info",
    level_finish="success",
    description="scoring smoothing parameters",
)
def _score_smoothed_results(
    smoothing_func: SmoothingFunction,
    parameter_combinations: list[dict[str, Any]],
    classification_result: DatasetClassificationResult,
    encode_func: EncodingFunction,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs],
    iteration: int,
    log: Logger,
) -> list[dict[str, Any]]:
    results = []
    for log, parameters in log_loop(
        parameter_combinations,
        level="info",
        message="scored parameters",
        log=log,
    ):
        classification_result = classification_result.smooth(
            [partial(smoothing_func, parameters)],
        )
        if remove_overlapping_predictions:
            if overlapping_predictions_kwargs is None:
                raise ValueError("overlapping_predictions_kwargs must be provided when remove_overlapping_predictions is True")
            classification_result = classification_result.remove_overlapping_predictions(
                **overlapping_predictions_kwargs
            )
        results.append(
            {
                "iteration": iteration,
                **parameters,
                **classification_result.score(encode_func=encode_func, macro=True),
            }
        )
    return results


def score_smoothing(
    smoothing_func: SmoothingFunction,
    parameter_combinations: list[dict[str, Any]],
    dataset: AnnotatedDataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs],
    iteration: int,
    k: int,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    log: Logger,
) -> list[dict[str, Any]]:
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
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs],
    default_decision: int | str,
    iteration: int,
    log: Logger,
) -> list[list[dict[str, Any]]]:
    num_categories = len(decision_thresholds)
    if not num_categories == len(classification_result.categories):
        raise ValueError(
            "number of decision thresholds does not match number of categories"
        )
    categories = classification_result.categories
    results = [[] for _ in range(num_categories)]
    for log, category_idx in log_loop(
        range(num_categories),
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
                    raise ValueError("overlapping_predictions_kwargs must be provided if remove_overlapping_predictions is True")
                classification_result = classification_result.remove_overlapping_predictions(
                    **overlapping_predictions_kwargs
                )
            results[category_idx].append(
                {
                    "iteration": iteration,
                    f"threshold_{categories[category_idx]}": threshold,
                    **classification_result.score(encode_func=encode_func, macro=True),
                }
            )
    return results


def score_thresholds(
    decision_thresholds: list[NDArray],
    dataset: AnnotatedDataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs],
    default_decision: int | str,
    iteration: int,
    k: int,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    smoothing_func: SmoothingFunction | None,
    log: Logger,
) -> list[list[dict[str, Any]]]:
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
    if smoothing_func is not None:
        classification_result = classification_result.smooth(
            [smoothing_func], threshold=False
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
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    smoothing_func: SmoothingFunction,
    smoothing_parameters_grid: dict[str, Iterable],
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs] = None,
    num_iterations: int,
    tolerance: float = 0.01,
    plot_results: bool = True,
    k: int,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    results_path: Optional[str] = None,
    log: Logger | None = None,
) -> dict[str, Any] | None:
    mpi_context = MPIContext(random_state)
    if log is None:
        log = set_logging_level()
    parameter_combinations = utils.parameter_grid_to_combinations(
        smoothing_parameters_grid
    )
    for iteration in range(num_iterations):
        if not mpi_context.do_iteration(iteration):
            continue
        log = with_loop(log, name="iteration", step=iteration)[0]
        mpi_context.add(
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
                random_state=mpi_context.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                log=log,
            ),
        )
    if not mpi_context.is_root:
        return
    results = []
    for iteration_results in mpi_context.collect(
        num_iterations=num_iterations
    ).values():
        results.extend(iteration_results)
    results = pd.DataFrame(results)
    if results_path is not None:
        results.to_csv(os.path.join(results_path, "results_smoothing.csv"))
    return utils.evaluate_results(
        results,
        parameter_names=parameter_combinations[0].keys(),
        tolerance=tolerance,
        plot_results=plot_results,
    )


@log_time(
    level_start="info",
    level_finish="success",
    description="decision threshold optimization",
)
def optimize_decision_thresholds(
    dataset: AnnotatedDataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    overlapping_predictions_kwargs: Optional[OverlappingPredictionsKwargs] = None,
    num_iterations: int,
    decision_threshold_range: tuple[float, float] | Iterable[tuple[float, float]] = (
        0.0,
        1.0,
    ),
    decision_threshold_step: float | Iterable[float] = 0.01,
    default_decision: int | str = "none",
    smoothing_func: Optional[SmoothingFunction] = None,
    tolerance: float = 0.01,
    plot_results: bool = True,
    k: int,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    results_path: Optional[str] = None,
    log: Logger | None = None,
):
    mpi_context = MPIContext(random_state)
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
        if not mpi_context.do_iteration(iteration):
            continue
        log = with_loop(log, name="iteration", step=iteration)[0]
        mpi_context.add(
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
                random_state=mpi_context.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                smoothing_func=smoothing_func,
                log=log,
            ),
        )
    if not mpi_context.is_root:
        return
    results = [[] for _ in range(num_categories)]
    for iteration_results in mpi_context.collect(
        num_iterations=num_iterations
    ).values():
        for category_idx, category_results in enumerate(iteration_results):
            results[category_idx].extend(category_results)
    results = [pd.DataFrame(category_results) for category_results in results]
    if results_path is not None:
        for category_results, category in zip(results, dataset.categories):
            category_results.to_csv(
                os.path.join(results_path, f"results_thresholding-{category}.csv")
            )
    return tuple(
        utils.evaluate_results(
            category_results,
            parameter_names=[f"threshold_{category}"],
            tolerance=tolerance,
            plot_results=plot_results,
        )
        for category_results, category in zip(results, dataset.categories)
    )

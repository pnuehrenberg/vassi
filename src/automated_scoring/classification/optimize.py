from __future__ import annotations

import os
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from ..dataset import Dataset, Identifier
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import MPIContext, SmoothingFunction
from . import _optimization_utils as utils
from .predict import k_fold_predict
from .utils import EncodingFunction, SamplingFunction

if TYPE_CHECKING:
    from loguru import Logger


def score_smoothing(
    smoothing_func: SmoothingFunction,
    parameter_combinations: list[dict[str, Any]],
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    iteration: int,
    k: int,
    exclude: Iterable[Identifier] | None,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    encode_func: EncodingFunction,
    log: Logger | None,
) -> list[dict[str, Any]]:
    if log is None:
        log = logger.bind(iteration=iteration)
    classification_result = k_fold_predict(
        dataset,
        extractor,
        classifier,
        k=k,
        exclude=exclude,
        random_state=random_state,
        sampling_func=sampling_func,
        balance_sample_weights=balance_sample_weights,
        encode_func=encode_func,
        log=log,
    )
    results = []
    log.info("scoring smoothing parameters")
    for idx, parameters in enumerate(parameter_combinations):
        classification_result = classification_result.smooth(
            [partial(smoothing_func, parameters)],
            remove_overlapping_predictions=remove_overlapping_predictions,
        )
        results.append(
            {
                "iteration": iteration,
                **parameters,
                **classification_result.score(encode_func=encode_func, macro=True),
            }
        )
        log.bind(level={"name": "", "step": idx + 1, "total": len(parameter_combinations)}).trace(
            "scored parameters"
        )
    log.success("finished scoring smoothing parameters")
    return results


def score_thresholds(
    decision_thresholds: list[NDArray],
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
    default_decision: int | str,
    iteration: int,
    k: int,
    exclude: Iterable[Identifier] | None,
    random_state: np.random.Generator | int,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool,
    encode_func: EncodingFunction,
    smoothing_func: SmoothingFunction | None,
    log: Logger | None,
) -> list[list[dict[str, Any]]]:
    if log is None:
        log = logger.bind(iteration=iteration)
    num_categories = len(decision_thresholds)
    assert num_categories == len(dataset.categories)
    classification_result = k_fold_predict(
        dataset,
        extractor,
        classifier,
        k=k,
        exclude=exclude,
        random_state=random_state,
        sampling_func=sampling_func,
        balance_sample_weights=balance_sample_weights,
        encode_func=encode_func,
        log=log,
    )
    results = [[] for _ in range(num_categories)]
    log.info("scoring decision thresholds")
    if smoothing_func is not None:
        classification_result = classification_result.smooth(
            [smoothing_func], threshold=False
        )
    for category_idx in range(num_categories):
        _log = log.bind(
            level={
                "name": "category",
                "step": category_idx + 1,
                "total": num_categories,
            }
        )
        for idx, threshold in enumerate(decision_thresholds[category_idx]):
            thresholds = np.zeros(num_categories)
            thresholds[category_idx] = threshold
            results[category_idx].append(
                {
                    "iteration": iteration,
                    f"threshold_{dataset.categories[category_idx]}": threshold,
                    **classification_result.threshold(
                        thresholds,
                        default_decision=default_decision,
                        remove_overlapping_predictions=remove_overlapping_predictions,
                    ).score(encode_func=encode_func, macro=True),
                }
            )
            _log.bind(
                sublevel={
                    "step": idx + 1,
                    "total": len(decision_thresholds[category_idx]),
                }
            ).trace("scored thresholds")
        _log.debug(
            f"finished scoring thresholds for category {dataset.categories[category_idx]}"
        )
    log.success("finished scoring thresholds")
    return results


def optimize_smoothing(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    smoothing_func: SmoothingFunction,
    smoothing_parameters_grid: dict[str, Iterable],
    *,
    remove_overlapping_predictions: bool,
    num_iterations: int,
    tolerance: float = 0.01,
    plot_results: bool = True,
    k: int,
    exclude: Optional[Iterable[Identifier]] = None,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    results_path: Optional[str] = None,
    log: Logger | None = None,
) -> dict[str, Any] | None:
    mpi_context = MPIContext(random_state)
    if log is None:
        log = logger.bind(mpi=mpi_context.info)
    parameter_combinations = utils.parameter_grid_to_combinations(
        smoothing_parameters_grid
    )
    if encode_func is None:
        try:
            encode_func = dataset.encode
        except ValueError:
            raise ValueError("specify encode_func for non-annotated datasets")
    for iteration in range(num_iterations):
        if not mpi_context.do_iteration(iteration):
            continue
        log = log.bind(iteration=iteration)
        log.info("evaluating smoothing")
        mpi_context.add(
            iteration,
            score_smoothing(
                smoothing_func,
                parameter_combinations,
                dataset,
                extractor,
                classifier,
                remove_overlapping_predictions=remove_overlapping_predictions,
                iteration=iteration,
                k=k,
                exclude=exclude,
                random_state=mpi_context.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                encode_func=encode_func,
                log=log,
            ),
        )
        log.success("finished smoothing evaluation")
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


def optimize_decision_thresholds(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    *,
    remove_overlapping_predictions: bool,
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
    exclude: Optional[Iterable[Identifier]] = None,
    random_state: Optional[np.random.Generator | int] = None,
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    encode_func: Optional[EncodingFunction] = None,
    results_path: Optional[str] = None,
    log: Logger | None = None,
):
    mpi_context = MPIContext(random_state)
    if log is None:
        log = logger.bind(mpi=mpi_context.info)
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
        log = log.bind(iteration=iteration)
        log.info("eveluating decision thresholds")
        mpi_context.add(
            iteration,
            score_thresholds(
                decision_thresholds,
                dataset,
                extractor,
                classifier,
                remove_overlapping_predictions=remove_overlapping_predictions,
                default_decision=default_decision,
                iteration=iteration,
                k=k,
                exclude=exclude,
                random_state=mpi_context.get_random_state(
                    iteration, num_iterations=num_iterations
                ),
                sampling_func=sampling_func,
                balance_sample_weights=balance_sample_weights,
                encode_func=encode_func,
                smoothing_func=smoothing_func,
                log=log,
            ),
        )
        log.success("finished decision threshold evaluation")
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

from collections.abc import Iterable
from itertools import product
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..dataset import Dataset, DyadIdentity, Identity
from ..features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import ensure_generator, formatted_tqdm
from .predict import k_fold_predict
from .utils import EncodingFunction, SamplingFunction, SmoothingFunction
from .visualization import Array


def _parameter_grid_to_combinations(
    paramter_grid: dict[str, Iterable],
) -> list[dict[str, Any]]:
    """Convert a parameter grid specified as a dictionary of iterables to a list of combinations."""
    return [
        {key: value for key, value in zip(paramter_grid.keys(), combination)}
        for combination in product(*paramter_grid.values())
    ]


def _prepare_thresholds(
    decision_threshold_range: tuple[float, float] | Iterable[tuple[float, float]],
    decision_threshold_step: float | Iterable[float],
    num_categories: int,
) -> list[NDArray]:
    if isinstance(decision_threshold_range, tuple) and isinstance(
        decision_threshold_range[0], float
    ):
        decision_threshold_ranges: list[tuple[float, float]] = [
            decision_threshold_range
        ] * num_categories  # type: ignore  # see check above
    elif (
        len(decision_threshold_ranges := list(decision_threshold_range))  # type: ignore  # see check above
        != num_categories
    ):
        raise ValueError(
            f"decision_threshold_range must be a list of {num_categories} ranges (number of categories), but is {len(decision_threshold_ranges)}"
        )
    if isinstance(decision_threshold_step, float | int):
        decision_threshold_steps = [decision_threshold_step] * num_categories
    elif (
        len(decision_threshold_steps := list(decision_threshold_step)) != num_categories
    ):
        raise ValueError(
            f"decision_threshold_step must be a list of {num_categories} steps (number of categories), but is {len(decision_threshold_steps)}"
        )
    return [
        np.arange(*threshold_range, step)
        for threshold_range, step in zip(
            decision_threshold_ranges, decision_threshold_steps
        )
    ]


def _evaluate_results(
    results: pd.DataFrame,
    *,
    parameter_names: Iterable[str],
    parameter_weight: Iterable[float] | float = 1.0,
    tolerance: float,
    plot_results: bool,
    figsize: Optional[tuple[float, float]] = None,
    dpi: float = 100,
    axes: Optional[Array[Axes]] = None,
    score_names: Iterable[str] = (
        "category_count_score",
        "f1_per_timestamp",
        "f1_per_annotation",
        "f1_per_prediction",
    ),
) -> dict[str, Any]:
    """
    Find the best parameter combination from a dataframe of results.

    Parameters
    ----------
    results : pd.DataFrame
        The results dataframe.
    parameter_names : Iterable[str]
        The names of the parameters to evaluate.
    parameter_weight : Iterable[float] | float, optional
        The cost of each parameter when choosing the best combination within the tolerance interval. Defaults to 1.0.
    tolerance : float
        The tolerance for the best parameter combination.
    plot_results : bool
        Whether to plot the results.
    figsize : tuple[float, float], optional
        The size of the figure if axes is not specified. Defaults to None.
    dpi : float, optional
        The DPI of the figure if axes is not specified. Defaults to 100.
    axes : Array[Axes], optional
        The axes to plot the results on. Defaults to None, which creates a new figure.
    score_names : Iterable[str], optional
        The names of the scores to evaluate, by default
        ("category_count_score", "f1_per_timestamp", "f1_per_annotation", "f1_per_prediction")

    Raises
    ------
    ValueError
        If any of the parameter names or score names are not in the results dataframe.

    Returns
    -------
    dict[str, Any]
        The best parameter combination.
    """
    parameter_names = list(parameter_names)
    score_names = list(score_names)
    if not np.isin(parameter_names, results.columns).all():
        raise ValueError(
            f"all parameter names must be in columns of results {results.columns}, got {parameter_names}"
        )
    if not np.isin(score_names, results.columns).all():
        raise ValueError(
            f"all score names must be in columns of results {results.columns}, got {score_names}"
        )
    if "iteration" not in results.columns:
        raise ValueError("results must contain an iteration column")
    if not isinstance(parameter_weight, float | int):
        parameter_weight = list(parameter_weight)
        if len(parameter_weight) != len(parameter_names):
            raise ValueError(
                f"parameter_weight must be a single value or an iterable of {len(parameter_names)} values, but is {len(parameter_weight)}"
            )
    else:
        parameter_weight = [parameter_weight] * len(parameter_names)
    results["average_score"] = results[score_names].mean(axis=1)
    # average across iterations
    average_results = (
        results.groupby(parameter_names)
        .aggregate({"average_score": "mean"})
        .reset_index()
    )
    max_score = average_results["average_score"].max()
    if TYPE_CHECKING:
        assert isinstance(max_score, float)
    within_tolerance = np.argwhere(
        average_results["average_score"] >= max_score - tolerance
    ).ravel()
    # normalize and weight parameters and find lowest combination within the tolerance interval
    # by default, lower parameters are preferred, otherwise, negative weights can be used
    parameters = np.asarray(average_results[parameter_names]).astype(float)
    parameters -= parameters.min(axis=0)
    parameters /= parameters.max(axis=0)
    parameter_costs = parameters * np.asarray(parameter_weight)
    best = parameter_costs[within_tolerance].sum(axis=1).argmin()
    best_parameters = average_results.iloc[within_tolerance[best]][
        parameter_names
    ].to_dict()
    if not plot_results:
        return best_parameters

    show_on_return = False
    if axes is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axes = fig.subplots(len(parameter_names), 1, sharey=True, squeeze=False)  # type: ignore  # for some reason, squeeze changes return type
        show_on_return = True
    if TYPE_CHECKING:
        assert axes is not None
    axes = axes.ravel()
    for idx, parameter_name in enumerate(parameter_names):
        ax = axes[idx]
        for iteration, results_iteration in results.groupby("iteration"):
            x = np.asarray(results_iteration[parameter_name])
            y = np.asarray(results_iteration["average_score"])
            ax.plot(
                x[np.argsort(x)],
                y[np.argsort(x)],
                lw=1,
                alpha=0.5,
                color="grey",
                zorder=1,
            )
        x = average_results[parameter_name]
        y = average_results["average_score"]
        ax.plot(x[np.argsort(x)], y[np.argsort(x)], lw=1, color="k", zorder=2)
        best_value = float(best_parameters[parameter_name])
        if best_value == round(best_value):
            best_value = int(best_value)
        ax.annotate(
            str(best_value),
            (best_parameters[parameter_name], max_score),
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(width=0.5, headwidth=5, headlength=5, shrink=0.1, fc="k"),
            ha="center",
            va="center",
        )
        if tolerance <= 0:
            continue
        ax.axhspan(
            max_score - tolerance, max_score, color="r", lw=0, alpha=0.2, zorder=0
        )
        ax.text(
            1.01,
            max_score - tolerance / 2,
            "tolerance",
            transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="left",
            va="center",
        )
        ax.set_xlabel(parameter_name.capitalize().replace("_", " "))
        ax.set_ylabel("Score")
        ax.spines[["right", "top"]].set_visible(False)
    if show_on_return:
        plt.show()

    return best_parameters


def optimize_smoothing(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    smoothing_func: SmoothingFunction,
    smoothing_parameters_grid: dict[str, Iterable],
    *,
    remove_overlapping_predictions: bool,
    num_iterations: int,
    show_progress: bool = False,
    tolerance: float = 0.01,
    plot_results: bool = True,
    # k fold paramters
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    # random_state is also used for sampling
    random_state: Optional[np.random.Generator | int] = None,
    # sampling parameters
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    # encode_func required for k-fold prediction of datasets with non-annotated groups
    encode_func: Optional[EncodingFunction] = None,
    show_k_fold_progress: bool = False,
):
    """
    Find the best parameter combination for output smoothing of a classifier on a given dataset.
    K-fold prediction is used to evaluate on the entire dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to evaluate.
    extractor : FeatureExtractor | DataFrameFeatureExtractor
        The feature extractor to use.
    classifier : Any
        The classifier to evaluate. Should be compatible with the scikit-learn API.
    smoothing_func : SmoothingFunction
        The smoothing function to use.
    smoothing_parameters_grid : dict[str, Iterable]
        The grid of smoothing parameters to evaluate.
    remove_overlapping_predictions : bool
        Whether to remove overlapping predictions in groups in which one individual has predictions for multiple recipients (multiple dyads per individual).
    num_iterations : int
        How often to run k-fold prediction and evaluation.
    show_progress : bool, optional
        Whether to show a progress bar. Defaults to False.
    tolerance : float, optional
        The tolerance for the best parameter combination. Defaults to 0.01.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.
    k : int, optional
        The number of folds to use for k-fold cross-validation. Defaults to 5.
    exclude : Iterable[Identity | DyadIdentity], optional
        The individuals, dyads, or groups to exclude from the dataset. Defaults to None.
    random_state : int | np.random.Generator, optional
        The random state to use for and sampling and k-fold prediction. Defaults to None.
    sampling_func : SamplingFunction
        The sampling function to use.
    balance_sample_weights : bool, optional
        Whether to use balanced sample weights during classifier training. Defaults to True.
    encode_func : EncodingFunction, optional
        The encoding function to use to assign numerical predictions to categories. Required if an unannoated dataset is provided.
    show_k_fold_progress : bool, optional
        Whether to show nested progress bars for k-fold progress. Defaults to False.

    Returns
    -------
    dict[str, Any]
        The best parameter combination.
    """
    random_state = ensure_generator(random_state)
    results = []
    parameter_combinations = _parameter_grid_to_combinations(smoothing_parameters_grid)
    if encode_func is None:
        try:
            encode_func = dataset.encode
        except ValueError:
            raise ValueError("specify encode_func for non-annotated datasets")
    if TYPE_CHECKING:
        assert encode_func is not None
    for iteration in formatted_tqdm(
        range(num_iterations), desc="iterations", disable=not show_progress
    ):
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
            show_progress=show_k_fold_progress,
        )
        for parameters in formatted_tqdm(
            parameter_combinations,
            desc="scoring combinations",
            disable=not show_progress,
        ):
            if TYPE_CHECKING:
                assert isinstance(parameters, dict)
            classification_result = classification_result.smooth(
                [lambda array: smoothing_func(array, parameters)],
                remove_overlapping_predictions=remove_overlapping_predictions,
            )
            results.append(
                {
                    "iteration": iteration,
                    **parameters,
                    **classification_result.score(encode_func=encode_func, macro=True),
                }
            )
    results = pd.DataFrame(results)
    return _evaluate_results(
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
    show_progress: bool = False,
    tolerance: float = 0.01,
    plot_results: bool = True,
    # k fold paramters
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    # random_state is also used for sampling
    random_state: Optional[np.random.Generator | int] = None,
    # sampling parameters
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    # encode_func required for k-fold prediction of datasets with non-annotated groups
    encode_func: Optional[EncodingFunction] = None,
    show_k_fold_progress: bool = False,
):
    random_state = ensure_generator(random_state)
    num_categories = len(dataset.categories)
    decision_thresholds = _prepare_thresholds(
        decision_threshold_range, decision_threshold_step, num_categories
    )
    if encode_func is None:
        try:
            encode_func = dataset.encode
        except ValueError:
            raise ValueError("specify encode_func for non-annotated datasets")
    if TYPE_CHECKING:
        assert encode_func is not None
    results = [[] for _ in range(num_categories)]
    for iteration in formatted_tqdm(
        range(num_iterations), desc="iterations", disable=not show_progress
    ):
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
            show_progress=show_k_fold_progress,
        )
        if smoothing_func is not None:
            classification_result = classification_result.smooth(
                [smoothing_func], threshold=False
            )
            for category_idx in formatted_tqdm(
                range(num_categories),
                desc="thresholding categories",
                disable=not show_progress,
            ):
                for threshold in formatted_tqdm(
                    decision_thresholds[category_idx],
                    desc="scoring thresholds",
                    disable=not show_progress,
                ):
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
    results = [pd.DataFrame(category_results) for category_results in results]
    return tuple(
        _evaluate_results(
            category_results,
            parameter_names=[f"threshold_{category}"],
            tolerance=tolerance,
            plot_results=plot_results,
        )
        for category_results, category in zip(results, dataset.categories)
    )

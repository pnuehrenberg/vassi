from collections.abc import Iterable
from itertools import product
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import ensure_generator, formatted_tqdm
from .. import Dataset
from ..types.utils import DyadIdentity, Identity
from .classify import k_fold_predict
from .utils import EncodingFunction, SamplingFunction, SmoothingFunction


def parameter_grid_to_combinations(
    paramter_grid: dict[str, Iterable],
) -> list[dict[str, Any]]:
    """Convert a parameter grid specified as a dictionary of iterables to a list of combinations."""
    return [
        {key: value for key, value in zip(paramter_grid.keys(), combination)}
        for combination in product(*paramter_grid.values())
    ]


def _evaluate_results(
    results: pd.DataFrame,
    *,
    parameter_names: Iterable[str],
    tolerance: float,
    plot_results: bool,
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
    tolerance : float
        The tolerance for the best parameter combination.
    plot_results : bool
        Whether to plot the results.
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
    best_parameters = average_results.iloc[within_tolerance[0]][
        parameter_names
    ].to_dict()
    if not plot_results:
        return best_parameters

    fig, axes = plt.subplots(1, len(parameter_names), sharey=True)
    for idx, parameter_name in enumerate(parameter_names):
        ax = axes[idx]
        for iteration, results_iteration in results.groupby("iteration"):
            x = results_iteration[:, parameter_name]
            y = results_iteration[:, "average_score"]
            ax.plot(
                x[np.argsort(x)], y[np.argsort(x)], lw=1, alpha=0.2, color="k", zorder=1
            )
        x = average_results[:, parameter_name]
        y = average_results[:, "average_score"]
        ax.plot(
            x[np.argsort(x)], y[np.argsort(x)], lw=1, alpha=0.5, color="k", zorder=2
        )
        ax.annotate(
            str(best_parameters[parameter_name]),
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
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel(parameter_name)
        ax.set_ylabel("score")
    plt.show()
    return best_parameters


def optimize_smoothing(
    dataset: Dataset,
    extractor: FeatureExtractor | DataFrameFeatureExtractor,
    classifier: Any,
    smoothing_func: SmoothingFunction,
    smoothing_parameters_grid: dict[str, Iterable],
    *,
    num_iterations: int,
    show_progress: bool = False,
    tolerance: float = 0.01,
    plot_results: bool = False,
    # k fold paramters
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    # random_state is also used for sampling
    random_state: Optional[np.random.Generator | int] = None,
    # sampling parameters
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    # pipeline parameters are also used for sampling in classification
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
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
    num_iterations : int
        How often to run k-fold prediction and evaluation.
    show_progress : bool, optional
        Whether to show a progress bar. Defaults to False.
    tolerance : float, optional
        The tolerance for the best parameter combination. Defaults to 0.01.
    plot_results : bool, optional
        Whether to plot the results. Defaults to False.
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
    pipeline : Pipeline, optional
        The pipeline to use for data sampling and classifier training. Defaults to None.
    fit_pipeline : bool, optional
        Whether to fit the specified pipeline on each fold. Defaults to True.
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
    parameter_combinations = parameter_grid_to_combinations(smoothing_parameters_grid)
    for iteration in (
        formatted_tqdm(range(num_iterations), desc="iterations")
        if show_progress
        else range(num_iterations)
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
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            encode_func=encode_func,
            show_progress=show_k_fold_progress,
        )
        for parameters in (
            formatted_tqdm(parameter_combinations, desc="scoring combinations")
            if show_progress
            else parameter_combinations
        ):
            classification_result = classification_result.smooth(
                [lambda array: smoothing_func(array, parameters)]
            )
            if encode_func is None:
                try:
                    encode_func = dataset.encode
                except ValueError:
                    raise ValueError("specify encode_func for non-annotated datasets")
            if TYPE_CHECKING:
                assert encode_func is not None
            results.append(
                {
                    "iteration": iteration,
                    **parameters,
                    **{
                        key: value
                        for key, value in zip(
                            [
                                "category_count_score",
                                "f1_per_timestamp",
                                "f1_per_annotation",
                                "f1_per_prediction",
                            ],
                            classification_result.score(encode_func=encode_func).mean(
                                axis=-1
                            ),
                        )
                    },
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
    smoothing_func: SmoothingFunction,
    num_iterations: int,
    show_progress: bool = False,
    tolerance: float = 0.01,
    plot_results: bool = False,
    # k fold paramters
    k: int,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    # random_state is also used for sampling
    random_state: Optional[np.random.Generator | int] = None,
    # sampling parameters
    sampling_func: SamplingFunction,
    balance_sample_weights: bool = True,
    # pipeline parameters are also used for sampling in classification
    pipeline: Optional[Pipeline] = None,
    fit_pipeline: bool = True,
    # encode_func required for k-fold prediction of datasets with non-annotated groups
    encode_func: Optional[EncodingFunction] = None,
    show_k_fold_progress: bool = False,
):
    pass

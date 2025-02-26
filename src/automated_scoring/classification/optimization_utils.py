from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .visualization import Array


def parameter_grid_to_combinations(
    paramter_grid: dict[str, Iterable],
) -> list[dict[str, Any]]:
    """Convert a parameter grid specified as a dictionary of iterables to a list of combinations."""
    return [
        {key: value for key, value in zip(paramter_grid.keys(), combination)}
        for combination in product(*paramter_grid.values())
    ]


def prepare_thresholds(
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


def evaluate_results(
    results: pd.DataFrame,
    *,
    parameter_names: Iterable[str],
    plot_results: bool,
    figsize: Optional[tuple[float, float]] = None,
    dpi: float = 100,
    axes: Optional[Array[Axes]] = None,
    score_names: Iterable[str] = (
        "f1_per_timestamp",
        "f1_per_annotation",
        "f1_per_prediction",
    ),
) -> dict[str, Any]:
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
        .reset_index(inplace=False)
    )
    if TYPE_CHECKING:
        # reset_index with inplace=False not correctly detected by pyright
        assert average_results is not None
    average_scores = np.array(average_results["average_score"])
    max_score = average_scores.max()
    best = average_scores.argmax()
    if TYPE_CHECKING:
        assert isinstance(max_score, float)
    best_parameters = {
        str(parameter_name): value
        for parameter_name, value in (
            average_results.iloc[best][parameter_names].to_dict().items()
        )
    }
    if not plot_results:
        return best_parameters
    show_on_return = False
    if axes is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axes = cast(  # cast NDArray to specific type
            Array[Axes],
            fig.subplots(len(parameter_names), 1, sharey=True, squeeze=False),
        )
        show_on_return = True
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
        x = np.asarray(average_results[parameter_name])
        y = np.asarray(average_results["average_score"])
        ax.plot(x[np.argsort(x)], y[np.argsort(x)], lw=1, color="k", zorder=2)
        best_value = float(best_parameters[parameter_name])
        if best_value == round(best_value):
            best_value = int(best_value)
        ax.annotate(
            str(best_value),
            (best_value, max_score),
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(width=0.5, headwidth=5, headlength=5, shrink=0.1, fc="k"),
            ha="center",
            va="center",
        )
        ax.set_xlabel(parameter_name.capitalize().replace("_", " "))
        ax.set_ylabel("Score")
        ax.spines[["right", "top"]].set_visible(False)
    if show_on_return:
        plt.show()
    return best_parameters


class OverlappingPredictionsKwargs(TypedDict):
    priority_func: Callable[[pd.DataFrame], Iterable[float]]
    prefilter_recipient_bouts: bool
    max_bout_gap: float
    max_allowed_bout_overlap: float


def passthrough(*, array: NDArray) -> NDArray:
    return array

from __future__ import annotations

import functools
from collections.abc import Iterable
from contextlib import contextmanager
from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..logging import set_logging_level
from .visualization import Array

if TYPE_CHECKING:
    from loguru import Logger


@contextmanager
def catch_time() -> Generator[Callable[[], float], None, None]:
    # https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    start = perf_counter()
    yield lambda: perf_counter() - start


def _log_time[T, **P](
    *args,
    func: Callable[P, T],
    level: Literal["trace", "debug", "info", "success", "warning", "error"],
    description: str,
    **kwargs,
) -> T:
    if "log" not in kwargs:
        raise ValueError("missing keyword-only argument log: loguru.Logger | None")
    log = kwargs.pop("log")
    if log is None:
        log = set_logging_level()
    if TYPE_CHECKING:
        assert isinstance(log, Logger)
    kwargs["log"] = log
    getattr(log, level)(f"started {description}")
    with catch_time() as get_time:
        result = func(*args, **kwargs)
    getattr(log, level)(f"finished {description} in {get_time():.2f} seconds")
    return result


def log_time[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    result_func = functools.partial(_log_time, func=func)
    decorated = functools.wraps(func)(result_func)
    return decorated


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
        ax.set_xlabel(parameter_name.capitalize().replace("_", " "))
        ax.set_ylabel("Score")
        ax.spines[["right", "top"]].set_visible(False)
        if tolerance <= 0:
            continue
        ax.axhspan(
            max_score - tolerance, max_score, color="r", lw=0, alpha=0.2, zorder=0
        )
        # ax.text(
        #     1,
        #     max_score,
        #     "tolerance",
        #     transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
        #     ha="right",
        #     va="bottom",
        #     color="r",
        #     alpha=0.2,
        # )
    if show_on_return:
        plt.show()
    return best_parameters

from collections.abc import Iterable
from typing import Optional

from .classification.results import BaseResult

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def summarize_scores(result: BaseResult, *, foreground_categories: Iterable[str], run, postprocessing_step: str):
    # this is a helper function to aggregate the f1 scores for one postprocessing in one run
    scores = result.score()
    summary = scores.stack().reset_index()
    summary = pd.DataFrame(
        np.array(summary[0]),
        index=summary["level_0"] + "_f1" + "-" + summary["level_1"],
    ).T
    columns = summary.columns
    summary["run"] = run
    summary["postprocessing_step"] = postprocessing_step
    summary = summary[["run", "postprocessing_step", *columns]]
    for level in scores.index:
        summary[f"{level}_f1-macro-foreground"] = scores.loc[level, list(foreground_categories)].mean()
        summary[f"{level}_f1-macro-all"] = scores.loc[level].mean()
    summary.columns = pd.MultiIndex.from_tuples(
        [
            tuple(map(str, (column.split("-", 1) if "-" in column else (column, ""))))
            for column in summary.columns
        ]
    )
    return summary


def aggregate_scores(summary: pd.DataFrame, score_level: str, *, categories: Iterable[str]):
    return (
        summary.loc[:, ["postprocessing_step", score_level]]
        .sort_index(axis=1)  # avoid unsorted index warning
        .groupby("postprocessing_step")
        .aggregate(["mean", "std"])
        .loc[:, score_level]
        .loc[:, ["macro-foreground", "macro-all", *categories]]
    )


def plot_errorbars(
    ax: Axes,
    means: Iterable[float],
    stds: Iterable[float],
    *,
    x: Optional[Iterable[float]]=None,
    padding: float = 0.5,
    ls="none",
    marker="_",
    ms: float = 10,
    lw: float = 6,
    markeredgecolor="k",
    color="k",
    xticklabels: Iterable[str] = ("model", "smooth", "thresh"),
    ylabel: str,
):
    means = np.array(means)
    stds = np.array(stds)
    if x is None:
        x = np.arange(means.size)
    else:
        x = np.array(x)
    ax.errorbar(x, means, stds, ls=ls, marker=marker, ms=ms, lw=lw, markeredgecolor=markeredgecolor, color=color)
    ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=75)
    ax.set_ylabel(ylabel)

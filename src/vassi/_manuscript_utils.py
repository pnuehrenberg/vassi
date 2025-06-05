from collections.abc import Iterable, Sequence, Callable
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Optional,
    Self,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle, ConnectionStyle, FancyArrowPatch

from .dataset.types import AnnotatedGroup
from .dataset.types.dyad import AnnotatedDyad

from .classification.results import BaseResult
from .classification.visualization import _Array


def summarize_scores(
    result: BaseResult,
    *,
    foreground_categories: Iterable[str],
    run,
    postprocessing_step: str,
):
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
        summary[f"{level}_f1-macro-foreground"] = scores.loc[
            level, list(foreground_categories)
        ].mean()
        summary[f"{level}_f1-macro-all"] = scores.loc[level].mean()
    summary.columns = pd.MultiIndex.from_tuples(
        [
            tuple(map(str, (column.split("-", 1) if "-" in column else (column, ""))))
            for column in summary.columns
        ]
    )
    return summary


def aggregate_scores(
    summary: pd.DataFrame, score_level: str, *, categories: Iterable[str]
):
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
    x: Optional[Iterable[float]] = None,
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
    ax.errorbar(
        x,
        means,
        stds,
        ls=ls,
        marker=marker,
        ms=ms,
        lw=lw,
        markeredgecolor=markeredgecolor,
        color=color,
    )
    ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=75)
    ax.set_ylabel(ylabel)


def adjust_node_positions_repulsion_vectorized(
    positions, min_distance, step=0.1, max_iterations=1000, convergence_threshold=1e-5
):
    adjusted_positions = positions.copy()
    for iteration in range(max_iterations):
        max_displacement = 0.0
        # Calculate pairwise distances
        diffs = (
            adjusted_positions[:, np.newaxis, :] - adjusted_positions[np.newaxis, :, :]
        )  # (N, N, 2)
        dists = np.linalg.norm(diffs, axis=2)  # (N, N)
        # Create a mask for distances less than min_distance (excluding self-distances)
        overlap_mask = dists < min_distance  # (N, N)
        overlap_mask[np.diag_indices_from(overlap_mask)] = False
        # Calculate repulsive forces
        forces = np.zeros_like(adjusted_positions)  # (N, 2)
        # Avoid division by zero by setting zero distances to a small value
        dists_safe = np.where(dists == 0, 1e-10, dists)
        # Calculate normalized vectors and apply force
        normalized_diffs = diffs / dists_safe[:, :, np.newaxis]
        repulsion_forces = (min_distance - dists[:, :, np.newaxis]) * normalized_diffs
        repulsion_forces = np.where(overlap_mask[:, :, np.newaxis], repulsion_forces, 0)
        forces = np.sum(repulsion_forces, axis=1)
        adjusted_positions += forces * step  # Adjust the step size as needed.
        max_displacement = np.linalg.norm(forces, axis=1).max()
        if max_displacement < convergence_threshold:
            break
    return adjusted_positions


def draw_network(
    ax,
    connectivity_matrix,
    locations,
    cmap,
    norm,
    edge_weight_threshold=0,
    fc="lightgray",
):
    ax.scatter(
        *locations.T, ec="k", fc=fc, s=10, lw=0.5, zorder=connectivity_matrix.max() + 1
    )
    assert connectivity_matrix.shape[0] == connectivity_matrix.shape[1]
    num_individuals = connectivity_matrix.shape[0]
    for actor_idx in range(num_individuals):
        for recipient_idx in range(num_individuals):
            if actor_idx == recipient_idx:
                continue
            edge_weight = connectivity_matrix[actor_idx, recipient_idx]
            if edge_weight <= edge_weight_threshold:
                continue
            edge = FancyArrowPatch(
                locations[actor_idx],
                locations[recipient_idx],
                arrowstyle=ArrowStyle("CurveB", head_length=1, head_width=1),
                connectionstyle=ConnectionStyle("Arc3", rad=0.5),
                shrinkA=3,
                shrinkB=3,
                joinstyle="miter",
                capstyle="round",
                color=cmap(norm(np.log(edge_weight))),
                zorder=edge_weight,
                clip_on=False,
            )
            ax.add_patch(edge)


def dyadic_interactions(group: AnnotatedGroup, *, kind: Literal["count", "duration"]):
    interaction_matrices = {
        category: np.zeros((len(group.individuals), len(group.individuals)))
        for category in group.foreground_categories
    }
    for identifier, sampleable in group:
        if not isinstance(identifier, tuple):
            raise ValueError("group target should be 'dyad'")
        if TYPE_CHECKING:
            assert isinstance(sampleable, AnnotatedDyad)
        actor, recipient = identifier
        actor_idx = group.individuals.index(actor)
        recipient_idx = group.individuals.index(recipient)
        observations = sampleable.observations
        for category in group.foreground_categories:
            try:
                observations_category = observations.set_index("category").loc[
                    [category]
                ]
            except KeyError:
                continue
            match kind:
                case "count":
                    interaction_matrices[category][actor_idx, recipient_idx] = len(
                        observations_category
                    )
                case "duration":
                    interaction_matrices[category][actor_idx, recipient_idx] = (
                        observations_category["duration"].sum()
                    )
                case _:
                    raise ValueError(
                        f"invalid value for 'kind', specify either 'count' or 'duration' (got '{kind}')"
                    )
    return interaction_matrices


def plot_classification_timeline_multiple(
    predictions: pd.DataFrame,
    categories: Iterable[str],
    *,
    annotations: Optional[pd.DataFrame] = None,
    timestamps: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    y_proba_smoothed: Optional[np.ndarray] = None,
    axes: Optional[_Array[Axes]] = None,
    figsize: tuple[float, float] = (10, 3),
    dpi: float = 100,
    category_labels: Optional[Iterable[str]] = None,
    interval: Optional[tuple[float, float]] = None,
    limit_interval: bool = True,
    x_tick_step: Optional[float] = None,
    x_tick_conversion: Optional[Callable[[Sequence[float]], Sequence[str]]] = None,
    x_label: Optional[str] = None,
    y_offset=0,
    x_offset=0,
    zorder=1,
):
    zorder *= 3

    def _plot_timeline(
        ax: Axes,
        observations: pd.DataFrame,
        categories: list[str],
        y_range: tuple[float, float],
        color,
    ):
        try:
            intervals = (
                observations.set_index("category")
                .loc[[categories[idx]], ["start", "duration"]]
                .to_numpy()
            )
        except KeyError:
            return
        ax.broken_barh(
            [(float(start) + x_offset, float(stop)) for start, stop in intervals],
            yrange=y_range,
            lw=0,
            color=color,
            zorder=zorder - 1,
        )

    if interval is None or limit_interval:
        interval = (-np.inf, np.inf)
        interval = (
            max(interval[0], predictions["start"].min()),
            min(interval[1], predictions["stop"].max()),
        )
    categories = list(categories)
    category_labels = categories if category_labels is None else list(category_labels)
    show_on_return = False
    if axes is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axes = fig.subplots(len(categories), 1, sharey=True)
        show_on_return = True
    if TYPE_CHECKING:
        assert axes is not None
    predictions_y_range = (
        (0.5 if annotations is not None else 0) + y_offset,
        0.5 if annotations is not None else 1,
    )
    for idx in range(len(categories)):
        _plot_timeline(
            axes[idx], predictions, categories, (predictions_y_range), "#ef8a62"
        )
        if annotations is not None:
            _plot_timeline(
                axes[idx], annotations, categories, (0 + y_offset, 0.5), "#67a9cf"
            )
        if y_proba_smoothed is not None:
            assert timestamps is not None, (
                "specify timestamps when plotting probabilities"
            )
            # tolist is a hack for type checking: Type "NDArray[numpy.bool[builtins.bool]]" is not assignable to type "Sequence[bool] | None"
            axes[idx].fill_between(
                timestamps + x_offset,
                y_proba_smoothed[:, idx] + y_offset,
                where=(y_proba_smoothed[:, idx] > 0.01).tolist(),
                lw=0,
                color="#f7f7f7",
                zorder=zorder - 2,
            )
            axes[idx].plot(
                timestamps + x_offset,
                y_proba_smoothed[:, idx] + y_offset,
                lw=1,
                c="k",
                zorder=zorder,
            )
        axes[idx].set_facecolor("#f7f7f7")
        axes[idx].spines[["right", "top", "bottom"]].set_visible(False)
        if y_proba is None and y_proba_smoothed is None:
            axes[idx].set_yticks([])
            axes[idx].spines[["left"]].set_visible(False)
        axes[idx].set_xticks([])
        axes[idx].set_xlim(interval[0], interval[1] + x_offset)
        axes[idx].set_ylim(-0.1, 1.1 + y_offset)
        axes[idx].set_ylabel(category_labels[idx], ha="right", va="center", rotation=0)
    if x_tick_step is not None:
        axes[-1].set_xticks(np.arange(*interval, x_tick_step))
    else:
        axes[-1].set_xticks([])
    if x_tick_conversion is not None:
        axes[-1].set_xticklabels(x_tick_conversion(list(axes[-1].get_xticks())))
    if x_label is not None:
        axes[-1].set_xlabel(x_label)
    if show_on_return:
        plt.show()

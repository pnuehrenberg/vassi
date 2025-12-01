from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.typing as mpt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import ArrowStyle, ConnectionStyle, FancyArrowPatch

from .classification.visualization import AxesArray
from .dataset.types import AnnotatedDyad, AnnotatedGroup
from .type_guards import is_tuple_of


def aggregate_scores(
    summary: pd.DataFrame,
    *,
    categories: Iterable[str],
    level: Literal["timestamp", "annotation", "prediction"],
    pipeline_step: Literal["raw", "smoothed", "thresholded"],
):
    relevant_columns = summary[
        [
            column
            for column in summary.columns
            if column.endswith(pipeline_step)
            and column.startswith(level)
            and any(category in column for category in categories)
        ]
    ]
    scores = relevant_columns.mean(axis=1)
    return tuple(scores.aggregate(["mean", "std"], axis=0))  # pyright: ignore[reportUnknownMemberType]


def plot_errorbars(
    ax: Axes,
    means: Iterable[float] | float,
    stds: Iterable[float] | float,
    *,
    x: Iterable[float] | None = None,
    xticklabels: Iterable[str] = ("model", "smooth", "thresh"),
    ylabel: str,
    padding: float = 0.5,
    ms: float = 10,
    lw: float = 6,
    ls: mpt.LineStyleType = "none",
    marker: mpt.MarkerType = "_",
    markeredgecolor: mpt.ColorType = "k",
    color: mpt.ColorType = "k",
):
    if isinstance(means, float):
        means = [means]
    if isinstance(stds, float):
        stds = [stds]
    means = np.array(means)
    stds = np.array(stds)
    if x is None:
        x = np.arange(means.size)
    else:
        x = np.array(x)
    _ = ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
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
    _ = ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
    _ = ax.set_xticks(x)  # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_xticklabels(xticklabels, rotation=75)  # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_ylabel(ylabel)  # pyright: ignore[reportUnknownMemberType]


def adjust_node_positions_repulsion_vectorized(
    positions: np.ndarray,
    min_distance: float,
    step: float = 0.1,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-5,
):
    adjusted_positions = positions.copy()
    for _ in range(max_iterations):
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
    ax: Axes,
    connectivity_matrix: np.ndarray,
    locations: np.ndarray,
    cmap: Colormap,
    norm: Normalize,
    edge_weight_threshold: float = 0,
    fc: mpt.ColorType = "lightgray",
):
    _ = ax.scatter(  # pyright: ignore[reportUnknownMemberType]
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
                arrowstyle=ArrowStyle("->", head_length=1, head_width=1),
                connectionstyle=ConnectionStyle("arc3", rad=0.5),
                shrinkA=3,
                shrinkB=3,
                joinstyle="miter",
                capstyle="round",
                color=cmap(norm(np.log(edge_weight))),  # pyright: ignore[reportUnknownArgumentType]
                zorder=edge_weight,
                clip_on=False,
            )
            _ = ax.add_patch(edge)


def dyadic_interactions(group: AnnotatedGroup, *, kind: Literal["count", "duration"]):
    individuals = list(group.actors())
    interaction_matrices = {
        category: np.zeros((len(individuals), len(individuals)))
        for category in group.foreground_categories
    }
    for identifier, sampleable in group:
        if not is_tuple_of(identifier, Hashable):
            raise ValueError("Expected group of dyads")
        if not isinstance(sampleable, AnnotatedDyad):
            raise ValueError("sampleable should be AnnotatedDyad")
        actor, recipient = identifier
        actor_idx = individuals.index(actor)
        recipient_idx = individuals.index(recipient)
        observations = sampleable.observations
        for category in group.foreground_categories:
            try:
                observations_category = observations.set_index("category").loc[
                    [category]
                ]
            except KeyError:
                continue
                if kind == "count":
                    interaction_matrices[category][actor_idx, recipient_idx] = len(
                        observations_category
                    )
                elif kind == "duration":
                    interaction_matrices[category][actor_idx, recipient_idx] = (
                        observations_category["duration"].sum()
                    )
                else:
                    raise ValueError(
                        f"invalid value for 'kind', specify either 'count' or 'duration' (got '{kind}')"
                    )
    return interaction_matrices


def plot_classification_timeline_multiple(
    predictions: pd.DataFrame,
    categories: Iterable[str],
    *,
    annotations: pd.DataFrame | None = None,
    timestamps: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    y_proba_smoothed: np.ndarray | None = None,
    axes: AxesArray | None = None,
    figsize: tuple[float, float] = (10, 3),
    dpi: float = 100,
    category_labels: Iterable[str] | None = None,
    interval: tuple[float, float] | None = None,
    limit_interval: bool = True,
    x_tick_step: float | None = None,
    x_tick_conversion: Callable[[Sequence[float]], Sequence[str]] | None = None,
    x_label: str | None = None,
    y_offset: int = 0,
    x_offset: int = 0,
    zorder: int = 1,
):
    zorder *= 3

    def _plot_timeline(
        ax: Axes,
        observations: pd.DataFrame,
        categories: list[str],
        y_range: tuple[float, float],
        color: mpt.ColorType,
    ):
        try:
            intervals = (
                observations.set_index("category")
                .loc[[categories[idx]], ["start", "duration"]]
                .to_numpy()
            )
        except KeyError:
            return
        _ = ax.broken_barh(  # pyright: ignore[reportUnknownMemberType]
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
        fig = plt.figure(figsize=figsize, dpi=dpi)  # pyright: ignore[reportUnknownMemberType]
        axes = fig.subplots(len(categories), 1, sharey=True)
        show_on_return = True
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
            _ = axes[idx].fill_between(  # pyright: ignore[reportUnknownMemberType]
                timestamps + x_offset,
                y_proba_smoothed[:, idx] + y_offset,
                where=(y_proba_smoothed[:, idx] > 0.01).tolist(),
                lw=0,
                color="#f7f7f7",
                zorder=zorder - 2,
            )
            _ = axes[idx].plot(  # pyright: ignore[reportUnknownMemberType]
                timestamps + x_offset,
                y_proba_smoothed[:, idx] + y_offset,
                lw=1,
                c="k",
                zorder=zorder,
            )
        axes[idx].set_facecolor("#f7f7f7")
        axes[idx].spines[["right", "top", "bottom"]].set_visible(False)
        if y_proba is None and y_proba_smoothed is None:
            _ = axes[idx].set_yticks([])  # pyright: ignore[reportUnknownMemberType]
            axes[idx].spines[["left"]].set_visible(False)
        _ = axes[idx].set_xticks([])  # pyright: ignore[reportUnknownMemberType]
        _ = axes[idx].set_xlim(interval[0], interval[1] + x_offset)
        _ = axes[idx].set_ylim(-0.1, 1.1 + y_offset)
        _ = axes[idx].set_ylabel(  # pyright: ignore[reportUnknownMemberType]
            category_labels[idx], ha="right", va="center", rotation=0
        )
    if x_tick_step is not None:
        _ = axes[-1].set_xticks(np.arange(*interval, x_tick_step))  # pyright: ignore[reportUnknownMemberType]
    else:
        _ = axes[-1].set_xticks([])  # pyright: ignore[reportUnknownMemberType]
    if x_tick_conversion is not None:
        _ = axes[-1].set_xticklabels(x_tick_conversion(list(axes[-1].get_xticks())))  # pyright: ignore[reportUnknownMemberType]
    if x_label is not None:
        _ = axes[-1].set_xlabel(x_label)  # pyright: ignore[reportUnknownMemberType]
    if show_on_return:
        plt.show()  # pyright: ignore[reportUnknownMemberType]

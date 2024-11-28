from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

# Little helper class, which is only used as a type.
# https://stackoverflow.com/questions/72649220/precise-type-annotating-array-numpy-ndarray-of-matplotlib-axes-from-plt-subplo
DType = TypeVar("DType")


class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:  # type: ignore
        return super().__getitem__(key)  # type: ignore


def plot_confusion_matrix(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (3, 3),
    dpi: float = 100,
    category_labels: Optional[Sequence[str]] = None,
    show_colorbar: bool = True,
):
    cm = confusion_matrix(
        y_true, y_pred, labels=range(max(max(y_true), max(y_pred)) + 1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_prob = (cm.T / cm.sum(axis=1)).T
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
    ax_colorbar = None
    if show_colorbar:
        ax_colorbar = ax.inset_axes((1.05, 0, 0.05, 1))
    mappable = ax.matshow(cm_prob, vmin=0, vmax=1)
    grid = np.indices(cm_prob.shape)
    for row_idx, col_idx in zip(grid[0].ravel(), grid[1].ravel()):
        ax.text(
            col_idx,
            row_idx,
            f"{cm_prob[row_idx, col_idx]:.2f}\n({cm[row_idx, col_idx]})",
            ha="center",
            va="center",
            c="k" if cm_prob[row_idx, col_idx] > 0.5 else "w",
            fontsize=8,
        )
    if show_colorbar and ax_colorbar is not None:
        plt.colorbar(mappable, cax=ax_colorbar)
        ax_colorbar.yaxis.set_ticks_position("right")
    if category_labels is not None:
        ax.set_yticks(range(len(category_labels)))
        ax.set_yticklabels(category_labels, rotation=90, va="center")
        ax.set_xticks(range(len(category_labels)))
        ax.set_xticklabels(category_labels)
    ax.set_ylabel("Annotated")
    ax.set_xlabel("Predicted")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)


def plot_classification_timeline(
    predictions: pd.DataFrame,
    categories: Iterable[str],
    *,
    annotations: Optional[pd.DataFrame] = None,
    y_proba: Optional[NDArray] = None,
    y_proba_smoothed: Optional[NDArray] = None,
    axes: Optional[Array[Axes]] = None,
    figsize: tuple[float, float] = (10, 3),
    dpi: float = 100,
    category_labels: Optional[Iterable[str]] = None,
    interval: tuple[float, float] = (0, 500),
    x_tick_step: float = 30,
    x_tick_conversion: Optional[Callable[[Sequence[float]], Sequence]] = None,
    x_label: Optional[str] = None,
):
    interval = (
        max(interval[0], predictions["start"].min()),
        min(interval[1], predictions["stop"].max()),
    )  # type: ignore
    categories = list(categories)
    if category_labels is None:
        category_labels = categories
    else:
        category_labels = list(category_labels)
    if axes is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        axes = fig.subplots(len(categories), 1, sharey=True)
        if TYPE_CHECKING:
            assert axes is not None
    for idx in range(len(categories)):
        try:
            axes[idx].broken_barh(
                predictions.set_index("category")
                .loc[[categories[idx]], ["start", "duration"]]
                .to_numpy(),
                yrange=(
                    0.5 if annotations is not None else 0,
                    0.5 if annotations is not None else 1,
                ),
                lw=0,
                color="#ef8a62",
            )
        except KeyError:
            pass
        if annotations is not None:
            try:
                axes[idx].broken_barh(
                    annotations.set_index("category")
                    .loc[[categories[idx]], ["start", "duration"]]
                    .to_numpy(),
                    yrange=(0, 0.5),
                    lw=0,
                    color="#67a9cf",
                )
            except KeyError:
                pass
        if y_proba is not None:
            axes[idx].plot(
                y_proba[:, idx],
                lw=1,
                c="k",
                alpha=0.5 if y_proba_smoothed is not None else 1,
            )
        if y_proba_smoothed is not None:
            axes[idx].plot(y_proba_smoothed[:, idx], lw=1, c="k")
        axes[idx].set_facecolor("#f7f7f7")
        axes[idx].spines[["right", "top", "bottom"]].set_visible(False)
        if y_proba is None and y_proba_smoothed is None:
            axes[idx].set_yticks([])
            axes[idx].spines[["left"]].set_visible(False)
        axes[idx].set_xticks([])
        axes[idx].set_xlim(interval[0], interval[1])
        axes[idx].set_ylim(-0.1, 1.1)
        axes[idx].set_ylabel(category_labels[idx], ha="right", va="center", rotation=0)
    axes[-1].set_xticks(np.arange(*interval, x_tick_step))
    if x_tick_conversion is not None:
        axes[-1].set_xticklabels(x_tick_conversion(list(axes[-1].get_xticks())))
    if x_label is not None:
        axes[-1].set_xlabel(x_label)
    plt.show()

from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Optional,
    Self,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

# Little helper class, only used for array type annotations.
# https://stackoverflow.com/questions/72649220/precise-type-annotating-array-numpy-ndarray-of-matplotlib-axes-from-plt-subplo
DType = TypeVar("DType")


class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:  # type: ignore
        return super().__getitem__(key)  # type: ignore

    def ravel(self, *args, **kwargs) -> Self: ...


def plot_confusion_matrix(
    y_true: NDArray[np.integer] | Iterable[NDArray[np.integer]],
    y_pred: NDArray[np.integer] | Iterable[NDArray[np.integer]],
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (3, 3),
    dpi: float = 100,
    category_labels: Optional[Sequence[str]] = None,
    show_colorbar: bool = True,
):
    def format_count(count):
        if count > 10000:
            return f"{count / 1000:.1f}k"
        return count

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not y_true.shape == y_pred.shape:
        raise ValueError("y_true and y_pred must be of same shape")
    if y_true.ndim == 1:
        cm = confusion_matrix(
            y_true, y_pred, labels=range(max(max(y_true), max(y_pred)) + 1)
        )
    else:
        cm = (
            np.asarray(
                [
                    confusion_matrix(
                        y_true, y_pred, labels=range(max(max(y_true), max(y_pred)) + 1)
                    )
                    for y_true, y_pred in zip(y_true, y_pred)
                ]
            )
            .mean(axis=0)
            .round()
            .astype(int)
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_prob = (cm.T / cm.sum(axis=1)).T
    show_on_return = False
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        show_on_return = True
    ax_colorbar = None
    if show_colorbar:
        ax_colorbar = ax.inset_axes((1.05, 0, 0.05, 1))
    mappable = ax.matshow(cm_prob, vmin=0, vmax=1)
    grid = np.indices(cm_prob.shape)
    for row_idx, col_idx in zip(grid[0].ravel(), grid[1].ravel()):
        ax.text(
            col_idx,
            row_idx,
            f"{cm_prob[row_idx, col_idx]:.2f}\n({format_count(cm[row_idx, col_idx])})",
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
    if show_on_return:
        plt.show()


def plot_classification_timeline(
    predictions: pd.DataFrame,
    categories: Iterable[str],
    *,
    annotations: Optional[pd.DataFrame] = None,
    timestamps: Optional[NDArray] = None,
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
    limit_interval: bool = True,
):
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
            intervals,
            yrange=y_range,
            lw=0,
            color=color,
        )

    if limit_interval:
        interval = (
            max(interval[0], predictions["start"].min()),
            min(interval[1], predictions["stop"].max()),
        )  # type: ignore
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
        0.5 if annotations is not None else 0,
        0.5 if annotations is not None else 1,
    )
    for idx in range(len(categories)):
        _plot_timeline(
            axes[idx], predictions, categories, predictions_y_range, "#ef8a62"
        )
        if annotations is not None:
            _plot_timeline(axes[idx], annotations, categories, (0, 0.5), "#67a9cf")
        if y_proba is not None:
            assert (
                timestamps is not None
            ), "specify timestamps when plotting probabilities"
            axes[idx].plot(
                timestamps,
                y_proba[:, idx],
                lw=1,
                c="k",
                alpha=0.5 if y_proba_smoothed is not None else 1,
            )
        if y_proba_smoothed is not None:
            assert (
                timestamps is not None
            ), "specify timestamps when plotting probabilities"
            axes[idx].plot(timestamps, y_proba_smoothed[:, idx], lw=1, c="k")
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
    if show_on_return:
        plt.show()

from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Self,
    TypeVar,
    Union,
)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from sklearn.metrics import confusion_matrix

# Little helper class, only used for array type annotations.
# https://stackoverflow.com/questions/72649220/precise-type-annotating-array-numpy-ndarray-of-matplotlib-axes-from-plt-subplo
DType = TypeVar("DType")


class _Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:  # type: ignore
        return super().__getitem__(key)  # type: ignore

    def ravel(self, *args, **kwargs) -> Self: ...


def _matshow_patches(
    ax: Axes,
    X: np.ndarray,
    cmap: Optional[Union[str, Colormap]] = None,
    norm: Optional[Normalize] = None,
    aspect: Union[Literal["equal"], float] = "equal",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    line_width: float = 0.5,
    **kwargs: Any,
) -> plt.cm.ScalarMappable:
    """
    Plot a matrix as an image using matplotlib.patches.Rectangle,
    mimicking plt.matshow for confusion matrices with explicit pixel rendering.

    Parameters:
        X: The matrix to be plotted.
        cmap: A Colormap instance or registered colormap name. If :code:`None`, defaults
            to Matplotlib's default.
        norm: A Normalize instance is used to map the data values to the 0-1 range
            before mapping to colors using the colormap.
        aspect: The aspect ratio of the axes. 'equal' makes cells square.
        vmin: Minimum and maximum values to normalize the colormap. If not provided,
            they are inferred from the data.
        vmax: Minimum and maximum values to normalize the colormap. If not provided,
            they are inferred from the data.
        line_width: The width of the edge line for each patch. Setting this to a non-zero
            value (e.g., 0.5) with :code:`edgecolor` matching :code:`facecolor` can help
            prevent tiny white lines/gaps in SVG exports due to anti-aliasing.
        kwargs: Additional keyword arguments are not directly used by this function's
            core drawing logic but are accepted for API compatibility.

    Returns:
        The ScalarMappable object that can be passed to :code:`plt.colorbar()`
        for external colorbar creation.
    """

    X = np.asarray(X)
    rows, cols = X.shape

    if cmap is None:
        cmap = plt.rcParams["image.cmap"]
    colormap = plt.get_cmap(cmap)

    if norm is None:
        norm = Normalize(
            vmin=vmin if vmin is not None else np.min(X),
            vmax=vmax if vmax is not None else np.max(X),
        )

    for i in range(rows):
        for j in range(cols):
            color = colormap(norm(X[i, j]))
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5),  # Position of the patch
                1,
                1,  # Size of the patch
                facecolor=color,
                edgecolor=color,  # Set edge color to match face color
                linewidth=line_width,  # Use the specified line width
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # Invert y-axis to match matshow

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)

    ax.set_aspect(aspect)

    # Create a dummy mappable for potential external colorbar creation
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Important: set an empty array for the scalar mappable

    # Return the ScalarMappable for compatibility with colorbar creation
    return sm


def plot_confusion_matrix(
    y_true: np.ndarray | Iterable[np.ndarray],
    y_pred: np.ndarray | Iterable[np.ndarray],
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (3, 3),
    dpi: float = 100,
    category_labels: Optional[Sequence[str]] = None,
    show_colorbar: bool = True,
):
    """
    Plot a confusion matrix of one or more sets of predictions.

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        ax: Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        figsize: Figure size in inches.
        dpi: Dots per inch.
        category_labels: List of category labels to use for the confusion matrix.
        show_colorbar: Whether to show the colorbar.
    """

    def format_count(count) -> str:
        if count > 100000:
            return f"{count:.1e}".replace("e+0", "e").replace("e+", "e")
        return str(count)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not y_true.shape == y_pred.shape:
        raise ValueError("y_true and y_pred must be of same shape")
    if y_true.ndim == 1 and y_true.dtype != "O":
        cm = confusion_matrix(
            y_true, y_pred, labels=range(max(max(y_true), max(y_pred)) + 1)
        )
    else:
        cm = (
            np.asarray(
                [
                    confusion_matrix(
                        y_true,
                        y_pred,
                        labels=range(
                            max(
                                max(y_true),
                                max(y_pred),
                            )
                            + 1
                        ),
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
    mappable = _matshow_patches(
        ax, cm_prob, vmin=0, vmax=1, interpolation="none", rasterized=True
    )
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
):
    """
    Plot a timeline of predictions and annotations.

    Parameters:
        predictions: Prediction data to visualize.
        categories: Category names.
        annotations: Annotation data to visualize.
        timestamps: Corresponding timestamps for :code:`y_proba` and :code:`y_proba_smoothed`.
        y_proba: Predicted probabilities.
        y_proba_smoothed: Smoothed predicted probabilities.
        axes: Matplotlib Axes objects to plot on, should be of length :code:`len(categories)`.
        figsize: Figure size in inches.
        dpi: Dots per inch.
        category_labels: Category labels to use for the timeline.
        interval: Start and end times for the timeline.
        limit_interval: Whether to limit the interval to the data range.
        x_tick_step: Step size for x-axis ticks.
        x_tick_conversion: Function to convert x-axis ticks.
        x_label: Label for the x-axis.
    """

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
            [(float(start), float(stop)) for start, stop in intervals],
            yrange=y_range,
            lw=0,
            color=color,
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
            assert timestamps is not None, (
                "specify timestamps when plotting probabilities"
            )
            axes[idx].plot(
                timestamps,
                y_proba[:, idx],
                lw=1,
                c="k",
                alpha=0.5 if y_proba_smoothed is not None else 1,
            )
        if y_proba_smoothed is not None:
            assert timestamps is not None, (
                "specify timestamps when plotting probabilities"
            )
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

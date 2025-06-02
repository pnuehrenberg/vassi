from collections.abc import Sequence
from typing import Any, Literal, Optional, Self

import numpy as np
from matplotlib import transforms
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .data_structures import Instance, Trajectory


def get_instance_range(
    instance: Instance, padding: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    cfg = instance.cfg
    if cfg.key_box is not None:
        box = instance[cfg.key_box]
        return (
            (box[0] - padding, box[2] + padding),
            (box[1] - padding, box[3] + padding),
        )
    if cfg.key_keypoints is not None:
        keypoints = instance[cfg.key_keypoints][:, :2]
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    raise NotImplementedError(
        "instance range not implemented for instances without boxes or keypoints"
    )


def get_trajectory_range(
    trajectory: Trajectory, padding: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    cfg = trajectory.cfg
    if cfg.key_box is not None:
        boxes = trajectory[cfg.key_box]
        x_min, y_min = boxes[:, :2].min(axis=0)
        x_max, y_max = boxes[:, 2:].max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    if cfg.key_keypoints is not None:
        keypoints = trajectory[cfg.key_keypoints][..., :2].reshape(-1, 2)
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    raise NotImplementedError(
        "trajectories range not implemented for trajectories without boxes or keypoints"
    )


class Panel:
    def __init__(self, width, height, *, extent: tuple[float, float, float, float]):
        self._width = width
        self._height = height
        self._extent = extent

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self._extent

    def dimension(self, orientation: Literal["vertical", "horizontal"]) -> float:
        match orientation:
            case "vertical":
                return self._height
            case "horizontal":
                return self._width
        raise ValueError("Invalid orientation")

    def start(self, orientation: Literal["vertical", "horizontal"]) -> float:
        match orientation:
            case "vertical":
                return self._extent[1]
            case "horizontal":
                return self._extent[0]
        raise ValueError("Invalid orientation")

    def get_ax(
        self,
        fig: Figure,
        label: Optional[str] = None,
        label_offset: tuple[float, float] = (-0.75, 0.25),
        spines: tuple[bool, bool, bool, bool] = (True, True, False, False),
    ):
        x, y, w, h = (*self.extent[:2], self.width, self.height)
        width = self.extent[2]
        height = self.extent[3]
        ax = fig.add_axes((x / width, y / height, w / width, h / height))
        if label is not None:
            transform = ax.transAxes + transforms.ScaledTranslation(
                *label_offset, fig.dpi_scale_trans
            )
            ax.text(
                0,
                1,
                label,
                transform=transform,
                fontsize=12,
                va="top",
                weight="semibold",
            )
        visible = [
            spine
            for spine, show in zip(["left", "bottom", "top", "right"], spines)
            if show
        ]
        hidden = [
            spine
            for spine, show in zip(["left", "bottom", "top", "right"], spines)
            if not show
        ]
        if len(visible) > 0:
            ax.spines[visible].set_visible(True)
        if len(hidden) > 0:
            ax.spines[hidden].set_visible(False)
        return ax

    def divide(
        self,
        *,
        sizes: Optional[Sequence[float]] = None,
        sizes_absolute: Optional[Sequence[float]] = None,  # use 0 for expanding panels
        spacing: Optional[float] = None,
        spacing_absolute: Optional[float] = None,
        orientation: Literal["vertical", "horizontal"],
    ) -> list[Self]:
        dimension = self.dimension(orientation)
        if spacing_absolute is not None:
            if spacing_absolute < 0 or spacing_absolute > dimension:
                raise ValueError(
                    "Spacing absolute must be between 0 and panel dimension"
                )
            spacing = spacing_absolute / dimension
        elif spacing is None:
            raise ValueError("Either spacing or spacing_absolute must be provided")
        if sizes_absolute is not None:
            sizes = [size_absolute / dimension for size_absolute in sizes_absolute]
            whitespace = spacing * (len(sizes) - 1)
            available = 1 - (sum(sizes) + whitespace)
            num_expanding = sum(size == 0 for size in sizes)
            sizes = [
                (size if size > 0 else available / num_expanding) for size in sizes
            ]
        elif sizes is not None:
            sizes_relative = np.array(sizes) / np.sum(sizes)
            whitespace = spacing * (len(sizes_relative) - 1)
            available = 1 - whitespace
            sizes = available * sizes_relative
        if sizes is None:
            raise ValueError("Either sizes or sizes_absolute must be provided")
        interval = (self.start(orientation) / self.dimension(orientation), 1)
        adjusted = []
        for size in sizes:
            if len(adjusted) > 0:
                adjusted.append(spacing * interval[1])
            adjusted.append(size * interval[1])
        starts = np.cumsum(
            [interval[0] if orientation == "horizontal" else 0] + adjusted
        )[::2]
        dimensions = np.array(adjusted)[::2]
        if orientation == "vertical":
            starts = 1 - starts - dimensions + interval[0]
        subpanels = []
        for start, dimension in zip(starts, dimensions):
            if orientation == "vertical":
                subpanels.append(
                    self.__class__(
                        self.width,
                        dimension * self.height,
                        extent=(self.extent[0], start * self.height, *self.extent[2:]),
                    )
                )
            else:
                subpanels.append(
                    self.__class__(
                        dimension * self.width,
                        self.height,
                        extent=(start * self.width, self.extent[1], *self.extent[2:]),
                    )
                )
        return subpanels


def get_box_height(height_in_inches, ax):
    fig_height = ax.get_figure().get_size_inches()[1]
    bbox = ax.get_position()
    return (height_in_inches / fig_height) / bbox.height


def get_box_width(width_in_inches, ax):
    fig_width = ax.get_figure().get_size_inches()[0]
    bbox = ax.get_position()
    return (width_in_inches / fig_width) / bbox.width


def add_xtick_box(
    center,
    width,
    ax,
    y="bottom",
    offset_in_inches=0.03,
    height_in_inches=0.2,
    text=None,
    color=None,
    **kwargs,
):
    y = 0 if y == "bottom" else 1
    text_kwargs: dict[str, Any] = {"ha": "center", "va": "center"}
    if color is not None:
        kwargs["fc"] = color
        kwargs["ec"] = adjust_lightness(color, 0.5)
        text_kwargs["color"] = kwargs["ec"]
    offset = get_box_height(offset_in_inches, ax)
    height = get_box_height(height_in_inches, ax)
    if y == 0:
        offset *= -1
        height *= -1
    pos = (center - width / 2, y + offset)
    transform = transform = ax.get_xaxis_transform()
    box = Rectangle(pos, width, height, clip_on=False, transform=transform, **kwargs)
    ax.add_patch(box)
    if text:
        ax.text(
            center, y + offset + (height) / 2, text, transform=transform, **text_kwargs
        )


def add_ytick_box(
    center,
    height,
    ax,
    x="left",
    offset_in_inches=0.03,
    width_in_inches=0.2,
    text=None,
    color=None,
    text_rotation=90,
    **kwargs,
):
    x = 0 if x == "left" else 1
    text_kwargs = {"ha": "center", "va": "center", "rotation": text_rotation}
    if color is not None:
        kwargs["fc"] = color
        kwargs["ec"] = adjust_lightness(color, 0.5)
        text_kwargs["color"] = kwargs["ec"]
    offset = get_box_width(offset_in_inches, ax)
    width = get_box_width(width_in_inches, ax)
    if x == 0:
        offset *= -1
        width *= -1
    pos = (x + offset, center - height / 2)
    transform = transform = ax.get_yaxis_transform()
    box = Rectangle(pos, width, height, clip_on=False, transform=transform, **kwargs)
    ax.add_patch(box)
    if text:
        ax.text(
            x + offset + (width) / 2, center, text, transform=transform, **text_kwargs
        )


def add_xtick_boxes(
    ticks, width, ax, labels=None, colors=None, disable_ticks=True, y="bottom", **kwargs
):
    num_ticks = len(ticks)
    if labels is None:
        labels = [None] * num_ticks
    if colors is None:
        colors = [None] * num_ticks
    for tick, label, color in zip(ticks, labels, colors):
        add_xtick_box(tick, width, ax, y=y, text=label, color=color, **kwargs)
    if disable_ticks:
        ax.set_xticks([])


def add_ytick_boxes(
    ticks, height, ax, labels=None, colors=None, disable_ticks=True, x="left", **kwargs
):
    num_ticks = len(ticks)
    if labels is None:
        labels = [None] * num_ticks
    if colors is None:
        colors = [None] * num_ticks
    for tick, label, color in zip(ticks, labels, colors):
        add_ytick_box(tick, height, ax, x=x, text=label, color=color, **kwargs)
    if disable_ticks:
        ax.set_yticks([])


def adjust_lightness(color, amount):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

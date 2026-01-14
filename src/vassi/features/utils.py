from typing import Protocol


class Shaped(Protocol):
    """The minimum requirement of extracted features. Both numpy arrays and polars DataFrames are supported."""

    @property
    def shape(self) -> tuple[int, ...]: ...

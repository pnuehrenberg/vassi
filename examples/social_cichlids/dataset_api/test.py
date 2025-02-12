from typing import TYPE_CHECKING

import pandas as pd

from automated_scoring.data_structures import Trajectory

from .types import (
    AnnotatedDataset,
    AnnotatedDyad,
    AnnotatedGroup,
    AnnotatedIndividual,
    Dataset,
    Dyad,
    Group,
    Individual,
)

if TYPE_CHECKING:
    Individual.REQUIRED_COLUMNS()
    Dyad.REQUIRED_COLUMNS()
    Group.REQUIRED_COLUMNS("individual")
    Group.REQUIRED_COLUMNS("dyad")

    Individual(Trajectory())
    AnnotatedIndividual(
        Trajectory(),
        observations=pd.DataFrame(),
        categories=("a", "b"),
        background_category="c",
    )
    Dyad(Trajectory(), Trajectory())
    AnnotatedDyad(
        Trajectory(),
        Trajectory(),
        observations=pd.DataFrame(),
        categories=("a", "b"),
        background_category="c",
    )
    Group([], target="dyad", exclude=None)
    AnnotatedGroup(
        [],
        target="dyad",
        exclude=None,
        observations=pd.DataFrame(),
        categories=("a", "b"),
        background_category="c",
    )
    Dataset({})
    AnnotatedDataset(
        {}, observations=pd.DataFrame(), categories=("a", "b"), background_category="c"
    )

from typing import Iterable, Optional, Sequence

import pandas as pd

from ..types._sampleable import AnnotatedSampleable
from ..types.dyad import AnnotatedDyad
from ..types.group import AnnotatedGroup
from ..types.individual import AnnotatedIndividual
from ..types.utils import DyadIdentity, Identity


def concatenate_observations(
    sampleables: (
        dict[Identity, AnnotatedGroup]
        | Sequence[AnnotatedGroup]
        | dict[Identity | DyadIdentity, AnnotatedSampleable]
        | Sequence[AnnotatedSampleable]
    ),
    *,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
):
    concatenated_observations = []
    if isinstance(sampleables, dict):
        if exclude is None:
            exclude = []
        sampleables_dict = {
            key: sampleable
            for key, sampleable in sampleables.items()
            if key not in exclude
        }
    else:
        sampleables_dict = {
            idx: sampleable for idx, sampleable in enumerate(sampleables)
        }
    for key, sampleable in sampleables_dict.items():
        if isinstance(sampleable, AnnotatedIndividual):
            observations = sampleable.observations
            if not isinstance(key, (int, str)):
                raise ValueError(f"invalid identity {key} for annotated individual")
            observations["actor"] = key
            observations = observations[["actor", *observations.columns[:-1]]]
            concatenated_observations.append(observations)
            continue
        elif isinstance(sampleable, AnnotatedDyad):
            observations = sampleable.observations
            if not isinstance(key, tuple):
                raise ValueError(f"invalid identity {key} for annotated dyad")
            actor, recipient = key
            observations["actor"] = actor
            observations["recipient"] = recipient
            observations = observations[
                ["actor", "recipient", *observations.columns[:-2]]
            ]
            concatenated_observations.append(observations)
            continue
        if not isinstance(sampleable, AnnotatedGroup):
            raise ValueError(
                f"Pass either annotated groups, dyads or individuals, but not {type(sampleable)}"
            )
        observations = concatenate_observations(
            sampleable._sampleables, exclude=exclude
        )
        observations["group"] = key
        observations = observations[["group", *observations.columns[:-1]]]
        concatenated_observations.append(observations)
    observations = pd.concat(concatenated_observations).reset_index(drop=True)
    for categorical_column in ["group", "actor", "recipient", "category"]:
        if categorical_column not in observations.columns:
            continue
        observations[categorical_column] = observations[categorical_column].astype(
            pd.CategoricalDtype()
        )
    return observations

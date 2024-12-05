from typing import Iterable, Optional, Sequence

import pandas as pd

from ..types._sampleable import AnnotatedSampleable
from ..types.dyad import AnnotatedDyad
from ..types.group import AnnotatedGroup
from ..types.individual import AnnotatedIndividual
from ..types.utils import DyadIdentity, Identity


def concatenate_annotations(
    sampleables: (
        dict[Identity, AnnotatedGroup]
        | Sequence[AnnotatedGroup]
        | dict[Identity | DyadIdentity, AnnotatedSampleable]
        | Sequence[AnnotatedSampleable]
    ),
    *,
    exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
):
    concatenated_annotations = []
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
            annotations = sampleable.annotations
            if not isinstance(key, (int, str)):
                raise ValueError(f"invalid identity {key} for annotated individual")
            annotations["actor"] = key
            annotations = annotations[["actor", *annotations.columns[:-1]]]
            concatenated_annotations.append(annotations)
            continue
        elif isinstance(sampleable, AnnotatedDyad):
            annotations = sampleable.annotations
            if not isinstance(key, tuple):
                raise ValueError(f"invalid identity {key} for annotated dyad")
            actor, recipient = key
            annotations["actor"] = actor
            annotations["recipient"] = recipient
            annotations = annotations[["actor", "recipient", *annotations.columns[:-2]]]
            concatenated_annotations.append(annotations)
            continue
        if not isinstance(sampleable, AnnotatedGroup):
            raise ValueError(
                f"Pass either annotated groups, dyads or individuals, but not {type(sampleable)}"
            )
        annotations = concatenate_annotations(sampleable._sampleables, exclude=exclude)
        annotations["group"] = key
        annotations = annotations[["group", *annotations.columns[:-1]]]
        concatenated_annotations.append(annotations)
    annotations = pd.concat(concatenated_annotations).reset_index(drop=True)
    for categorical_column in ["group", "actor", "recipient", "category"]:
        if categorical_column not in annotations.columns:
            continue
        annotations[categorical_column] = annotations[categorical_column].astype(
            pd.CategoricalDtype()
        )
    return annotations

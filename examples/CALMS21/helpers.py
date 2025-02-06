import numpy as np
import pandas as pd


def subsample_train(
    dataset,
    extractor,
    *,
    random_state=None,
    exclude=None,
):
    X_subsample_even, y_subsample_even = dataset.subsample(
        extractor,
        0.1,
        categories=("none", "investigation"),
        random_state=random_state,
        exclude=exclude,
    )
    X_subsample_all, y_subsample_all = dataset.subsample(
        extractor,
        1.0,
        try_even_subsampling=False,
        categories=("attack", "mount"),
        random_state=random_state,
        exclude=exclude,
    )
    return (
        pd.concat([X_subsample_even, X_subsample_all]),
        np.concatenate([y_subsample_even, y_subsample_all]),
    )

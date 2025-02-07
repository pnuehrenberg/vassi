import numpy as np
import pandas as pd

from automated_scoring.dataset.sampling.permutation import permute_recipients


def subsample_train(
    dataset,
    extractor,
    *,
    random_state=None,
    exclude=None,
    log,
):
    X_train_none, y_train_none = dataset.subsample(
        extractor,
        0.001,
        categories=("none",),
        try_even_subsampling=False,
        random_state=random_state,
        log=log,
    )

    X_frontal, y_frontal = dataset.subsample(
        extractor,
        0.25,
        categories=("frontal_display",),
        try_even_subsampling=False,
        random_state=random_state,
        log=log,
    )

    X_minorities, y_minorities = dataset.subsample(
        extractor,
        1.0,
        categories=(
            "approach",
            "chase",
            "dart_bite",
            "lateral_display",
            "quiver",
        ),
        try_even_subsampling=False,
        random_state=random_state,
        log=log,
    )

    # sample close neighbors more frequently
    sampling_frequency = {0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05}
    X_additional = pd.concat(
        [
            permute_recipients(dataset, neighbor_rank=neighbor_rank).subsample(
                extractor,
                sampling_frequency[neighbor_rank],
                categories=(
                    "approach",
                    "frontal_display",
                    "chase",
                    "dart_bite",
                    "lateral_display",
                    "quiver",
                ),
                try_even_subsampling=False,
                random_state=random_state,
                log=log,
            )[0]  # only keep samples (X) but not labels (y)
            for neighbor_rank in range(5)
        ]
    )
    y_additional = np.repeat(
        "none", len(X_additional)
    )  # all labels are "none" because of switched recipients
    X = pd.concat([X_train_none, X_frontal, X_minorities, X_additional])
    y = np.concat([y_train_none, y_frontal, y_minorities, y_additional])
    return X, y

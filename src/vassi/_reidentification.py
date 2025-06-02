from collections.abc import Iterable, Callable
from typing import Optional

import networkx as nx
import numpy as np
from numpy.dtypes import StringDType
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from . import config, math
from .data_structures import TimestampedInstanceCollection
from .data_structures.utils import OutOfInterval
from .features.utils import Feature
from .utils import perform_operation

FeatureDistance = Callable[[np.ndarray, np.ndarray], np.ndarray]


def assign_identities(
    instances: TimestampedInstanceCollection,
    *,
    position_func: Feature,
    max_lag: int,
    max_distance: float,
    similarity_features: Optional[Iterable[tuple[Feature, FeatureDistance]]] = None,
    dissimilarity_weights: float | Iterable[float] = 1,
    cfg: Optional[config.Config] = None,
    show_progress: bool = True,
):
    """
    Automatically assigns instances into trajectories (formatted as TimestampedInstanceCollection) with unique identities.

    Assignments are optimized using the hungarian algorithm on similarity matrices, with thresholds
    (max_lag, max_distance) to avoid assigning instances to trajectories with too much lag or over too far distances.

    Similarity matrices are calculated as the weighted average (dissimilarity_weights) of similarity features computed
    on pairs (previous instances and current instances) of instances.

    Parameters
    ----------
    instances: TimestampedInstanceCollection
        The instances to assign into trajectories.
    position_func: Feature
        Function to compute the position of an instance.
    max_lag: int
        Maximum lag to allow for an instance to be assigned to a trajectory.
    max_distance: float
        Maximum distance to allow for an instance to be assigned to a trajectory.
    similarity_features: Iterable[tuple[Feature, FeatureDistance]], optional
        Tuples of feature functions and distance functions (between calculated features) to calculate similarity features.
        If None (default), the the euclidean distances between instances (calculated using the position_func) are used.
    dissimilarity_weights: float | Iterable[float], optional
        Weights to use for the similarity features. If a single float is provided, the same weight is used for all
        similarity features. If an iterable is provided, it must have the same length as the number of similarity
        features.
    cfg: config.Config, optional
        Configuration object. If None, a the configuration of the instance collection is used.
    show_progress: bool, optional
        Whether to show a progress bar.

    Returns
    -------
    trajectories: TimestampedInstanceCollection
        The instances with assigned identities.

    Raises
    ------
    ValueError
        If the number of similarity features is not equal to the number of dissimilarity weights.
    ValueError
        If the identity key of the configuration is not set.
    ValueError
        If the dtype of the identity key of the configuration is not int or numpy.dtypes.StringDType.
    """

    def next_identity() -> str | int:
        if len(used_decimal_identities) == 0:
            identity = 0
        else:
            identity = max(used_decimal_identities) + 1
        used_decimal_identities.add(identity)
        active.add(identity)
        return identity

    cfg = instances.cfg
    if (key_identity := cfg.key_identity) is None:
        raise ValueError("undefined identity key")
    dtype = instances[key_identity].dtype
    if isinstance(dtype, StringDType):
        unassigned = "undefined"
    elif np.issubdtype(dtype, int):
        unassigned = -1
    else:
        raise ValueError(
            f"value for {key_identity} should be a numpy array of type int or numpy.dtypes.StringDType"
        )
    with instances.validate(False):
        instances[key_identity] = unassigned
    if similarity_features is None:
        similarity_features = []
    else:
        similarity_features = list(similarity_features)
    if not isinstance(dissimilarity_weights, Iterable):
        dissimilarity_weights = [dissimilarity_weights] * len(similarity_features)
    else:
        dissimilarity_weights = list(dissimilarity_weights)
    if (num_features := len(similarity_features)) != (
        num_weights := len(dissimilarity_weights)
    ):
        raise ValueError(
            f"provided {num_features} similarity features, but {num_weights} dissimilarity weights."
        )
    max_dissimilarity = [1] * num_features
    used_decimal_identities: set[int] = set()
    active: set[int | str] = set()
    archived: set[int | str] = set()
    if not instances.is_sorted:
        instances = instances.sort()
    timestamps = np.arange(
        instances.timestamps[0],
        instances.timestamps[-1] + 1,
    )
    start = timestamps[0]
    if show_progress:
        timestamps = tqdm(timestamps)
    for timestamp in timestamps:
        instances_window = instances.slice_window(
            max(start, timestamp - max_lag - 1), timestamp
        )
        # archive active trajectories with too high lag
        for identity in list(active):  # may change size
            last_timestamp = instances_window.select(identity=identity)[
                instances.key_timestamp
            ][-1]
            if timestamp - last_timestamp <= max_lag:
                continue
            active.discard(identity)
            archived.add(identity)
        if len(instances_window) == 0:
            # should probably move up
            continue
        try:
            current_instances = instances_window.slice_window(timestamp, timestamp)
        except OutOfInterval:
            # the if below is probably wrong!
            continue
        if (num_current := len(current_instances)) == 0:
            # nothing to do
            continue
        if (num_active := len(active)) == 0 and len(archived) == 0:
            # first
            with current_instances.validate(False):
                current_instances[key_identity] = np.asarray(
                    [next_identity() for _ in range(num_current)], dtype=dtype
                )
            continue
        elif num_active == 0:
            # only archived, assign new identities to all
            with current_instances.validate(False):
                current_instances[key_identity] = np.asarray(
                    [next_identity() for _ in range(num_current)], dtype=dtype
                )
            continue
        # find optimal assignments from active to current
        active_instances = TimestampedInstanceCollection.concatenate(
            *[instances_window.select(identity=identity)[-1:] for identity in active],
            validate=False,
        )
        instances_to_match = TimestampedInstanceCollection.concatenate(
            active_instances, current_instances, validate=False
        )
        positions_to_match = position_func(instances_to_match)
        distance_matrix = perform_operation(
            math.euclidean_distance,
            positions_to_match.reshape(1, num_active + num_current, -1),
            positions_to_match.reshape(1, num_active + num_current, -1),
            element_wise=False,
            flat=False,
        )[0]
        matched_category = np.ones(
            (len(instances_to_match), len(instances_to_match)), dtype=bool
        )
        if cfg.key_category is not None and cfg.key_category in cfg.trajectory_keys:
            categories_to_match = instances_to_match[cfg.key_category]
            matched_category = np.equal(
                categories_to_match[:, np.newaxis], categories_to_match[np.newaxis]
            )
        distance_matrix[~matched_category] = np.inf
        # create a graph and determine the connected components within max_distance
        distance_matrix[:num_active, :num_active] = np.inf
        distance_matrix[num_active:, num_active:] = np.inf
        G = nx.from_numpy_array(distance_matrix <= max_distance)
        components = []
        for g in nx.connected_components(G):
            components.append(np.array(sorted(g)))
        # solve assignment for each component
        for component in components:
            # indices relating to active_instances and current_instances
            from_idx = component[component < num_active]
            to_idx = component[component >= num_active] - num_active
            if to_idx.size == 0:
                # no new instance to assign to
                continue
            if from_idx.size == 0:
                # no active to assign from, create new
                for idx in to_idx:
                    with current_instances.validate(False):
                        current_instances[idx, key_identity] = np.asarray(
                            next_identity(), dtype=dtype
                        )
                continue
            # n-to-n assignment
            from_features = [
                feature_func(active_instances.select_index(from_idx))
                for feature_func, _ in similarity_features
            ]
            to_features = [
                feature_func(current_instances.select_index(to_idx))
                for feature_func, _ in similarity_features
            ]
            # dissimilarity for each given feature
            dissimilarity_matrices = []
            for feature_idx, (_, distance_func) in enumerate(similarity_features):
                dissimilarity_matrix = perform_operation(
                    distance_func,
                    from_features[feature_idx].reshape(1, len(from_idx), -1),
                    to_features[feature_idx].reshape(1, len(to_idx), -1),
                    element_wise=False,
                    flat=False,
                )[0]
                assert dissimilarity_matrix.shape == (len(from_idx), len(to_idx))
                if (max_diss := dissimilarity_matrix.max()) > max_dissimilarity[
                    feature_idx
                ]:
                    # update max dissimilarity
                    max_dissimilarity[feature_idx] = max_diss
                dissimilarity_matrix /= max_dissimilarity[feature_idx]
                dissimilarity_matrices.append(dissimilarity_matrix)
            if len(dissimilarity_matrices) > 0:
                cost_matrix = np.average(
                    dissimilarity_matrices, weights=dissimilarity_weights
                )
            else:
                # fall back to distance matrix
                cost_matrix = distance_matrix[from_idx][:, to_idx + num_active]
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            assigned_row = []
            assigned_col = []
            for row, col in zip(row_idx, col_idx):
                # append to active
                assert (row not in assigned_row) or (col not in assigned_col), (
                    f"double assignment {row_idx} {col_idx}"
                )
                assigned_row.append(row)
                assigned_col.append(col)
                with current_instances.validate(False):
                    current_instances[to_idx[col], key_identity] = active_instances[
                        from_idx[row], key_identity
                    ]
        # create new from unassigned
        unassigned_idx = np.argwhere(
            current_instances[key_identity] == unassigned
        ).ravel()
        for idx in unassigned_idx:
            with current_instances.validate(False):
                current_instances[int(idx), key_identity] = np.asarray(
                    next_identity(), dtype=dtype
                )
        assert (current_instances[key_identity] != unassigned).all()
    return instances

from typing import Optional, TypedDict, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline

from ...utils import ensure_generator, sklearn_seed


class SampleKMeansKwargs(TypedDict, total=False):
    num_samples: Optional[int]
    sampling_frequency: Optional[float]
    num_clusters: Optional[int]
    scale_p_with_distance: bool
    relative_to_cluster_size: bool
    preparation_pipeline: Optional[Pipeline]
    fit_preparation_pipeline: bool
    pca_reduce_dims: bool
    pca_num_components: Optional[int]
    pca_ev_ratio_cutoff: Optional[float]
    random_state: Optional[np.random.Generator | int]


_default_sample_k_means_kwargs: SampleKMeansKwargs = {
    "num_samples": None,
    "sampling_frequency": None,
    "num_clusters": None,
    "scale_p_with_distance": True,
    "relative_to_cluster_size": False,
    "preparation_pipeline": None,
    "fit_preparation_pipeline": True,
    "pca_reduce_dims": True,
    "pca_num_components": None,
    "pca_ev_ratio_cutoff": None,
    "random_state": None,
}


@overload
def reduce_dims_pca(X: NDArray, *args, **kwargs) -> NDArray: ...


@overload
def reduce_dims_pca(X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame: ...


def reduce_dims_pca(
    X: NDArray | pd.DataFrame,
    *,
    pca: Optional[PCA | IncrementalPCA] = None,
    num_components: Optional[int] = None,
    ev_ratio_cutoff: Optional[float] = None,
    random_state: Optional[np.random.Generator | int] = None,
) -> NDArray | pd.DataFrame:
    random_state = ensure_generator(random_state)
    if pca is None:
        pca = PCA(
            n_components=num_components,
            random_state=sklearn_seed(random_state),
        )
        if isinstance(X, pd.DataFrame):
            pca.set_output(transform="pandas")
        pca.fit(X)
    X = pca.transform(X)
    if ev_ratio_cutoff is None:
        return X
    ev_ratios: NDArray = pca.explained_variance_ratio_  # type: ignore
    cumulative_ev_ratios = np.cumsum(ev_ratios)
    if cumulative_ev_ratios.max() <= ev_ratio_cutoff:
        return X
    cutoff = np.argmax(cumulative_ev_ratios > ev_ratio_cutoff)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(X[:, :cutoff])
    return X[:, :cutoff]


@overload
def sample_k_means(
    X: NDArray,
    kwargs: Optional[SampleKMeansKwargs],
) -> NDArray: ...


@overload
def sample_k_means(
    X: pd.DataFrame,
    kwargs: Optional[SampleKMeansKwargs],
) -> pd.DataFrame: ...


def sample_k_means(
    X: NDArray | pd.DataFrame,
    kwargs: Optional[SampleKMeansKwargs],
) -> NDArray | pd.DataFrame:
    global _default_sample_k_means_kwargs

    def kwarg_or_default(kwarg):
        if kwargs is not None and kwarg in kwargs:
            return kwargs[kwarg]
        return _default_sample_k_means_kwargs[kwarg]

    num_samples = kwarg_or_default("num_samples")
    sampling_frequency = kwarg_or_default("sampling_frequency")
    num_clusters = kwarg_or_default("num_clusters")
    scale_p_with_distance = kwarg_or_default("scale_p_with_distance")
    relative_to_cluster_size = kwarg_or_default("relative_to_cluster_size")
    preparation_pipeline = kwarg_or_default("preparation_pipeline")
    fit_preparation_pipeline = kwarg_or_default("fit_preparation_pipeline")
    pca_reduce_dims = kwarg_or_default("pca_reduce_dims")
    pca_num_components = kwarg_or_default("pca_num_components")
    pca_ev_ratio_cutoff = kwarg_or_default("pca_ev_ratio_cutoff")
    random_state = kwarg_or_default("random_state")

    random_state = ensure_generator(random_state)
    if num_clusters is None:
        num_clusters = int(round(np.sqrt(X.shape[0])))
    if num_samples is not None:
        num_samples = min(max(0, num_samples), X.shape[0])
        sampling_frequency = num_samples / X.shape[0]
    elif sampling_frequency is None:
        raise ValueError("Specify either sampling_frequency or num_samples.")
    else:
        sampling_frequency = min(max(0.0, sampling_frequency), 1.0)
    X_prepared = X
    if preparation_pipeline is not None:
        if isinstance(X, pd.DataFrame):
            preparation_pipeline.set_output(transform="pandas")
        if fit_preparation_pipeline:
            X_prepared = preparation_pipeline.fit_transform(X)
        else:
            X_prepared = preparation_pipeline.transform(X)
    X_reduced = X_prepared
    if pca_reduce_dims:
        X_reduced = reduce_dims_pca(
            X_prepared,
            num_components=pca_num_components,
            ev_ratio_cutoff=pca_ev_ratio_cutoff,
            random_state=random_state,
        )
    k_means = KMeans(
        n_clusters=num_clusters, random_state=sklearn_seed(random_state)
    ).fit(X_reduced)
    distances = k_means.transform(X_reduced)
    labels = k_means.predict(X_reduced)
    min_cluster_distances = distances[np.arange(distances.shape[0]), labels]
    num_cluster_samples = None
    if not relative_to_cluster_size:
        num_cluster_samples = int(round(sampling_frequency * X.shape[0] / num_clusters))
    samples_idx = []
    for label in np.unique(labels):
        cluster_samples_idx = np.argwhere(labels == label).ravel()
        cluster_size = cluster_samples_idx.shape[0]
        cluster_distances = min_cluster_distances[cluster_samples_idx]
        if cluster_distances.sum() == 0:
            cluster_distances += 1
        samples_idx.append(
            random_state.choice(
                cluster_samples_idx,
                size=(
                    min(cluster_size, num_cluster_samples)
                    if num_cluster_samples
                    else int(round(sampling_frequency * cluster_size))
                ),
                p=(
                    cluster_distances / cluster_distances.sum()
                    if scale_p_with_distance
                    else None
                ),
                replace=False,
            )
        )
    samples_idx = np.sort(np.concatenate(samples_idx))
    if isinstance(X, pd.DataFrame):
        return X.iloc[samples_idx]
    return X[samples_idx]

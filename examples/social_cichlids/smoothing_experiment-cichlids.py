import numpy as np
from helpers import overlapping_predictions_kwargs, smooth, subsample_train
from numba import config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from automated_scoring.classification import (
    optimize_smoothing,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset
from automated_scoring.logging import set_logging_level
from automated_scoring.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = (
    "pose",
    "time_stamp",
)


if __name__ == "__main__":
    # set the threading layer before any parallel target compilation
    config.THREADING_LAYER = "safe"  # type: ignore

    from automated_scoring.mpi_utils import MPIContext

    dataset_full = load_dataset(
        "cichlids",
        directory="../../datasets/social_cichlids",
        target="dyad",
        background_category="none",
    )

    dataset_train, dataset_test = dataset_full.split(
        0.8,
        random_state=1,
    )

    time_scales, slices = get_window_slices(3, time_scales=(91,))

    aggregator = ColumnTransformer(
        [
            (
                "aggregate",
                SlidingWindowAggregator(
                    [metrics.median, metrics.q10, metrics.q90], max(time_scales), slices
                ),
                make_column_selector(),
            ),
            ("original", "passthrough", make_column_selector()),
        ],
    )

    pipeline = Pipeline(
        [("impute", KNNImputer()), ("aggregate", aggregator)]
    ).set_output(transform="pandas")

    extractor = DataFrameFeatureExtractor(
        cache_directory="cichlids_cache",
        pipeline=pipeline,
        refit_pipeline=True,
    ).read_yaml("config_file-cichlids.yaml")

    best_parameters = optimize_smoothing(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        smooth,
        smoothing_parameters_grid={"median_filter_window": np.arange(3, 91, 2)},
        remove_overlapping_predictions=True,
        overlapping_predictions_kwargs=overlapping_predictions_kwargs,
        num_iterations=20,
        k=5,
        sampling_func=subsample_train,
        tolerance=0.005,
        plot_results=False,
        results_path=".",
        log=set_logging_level("info"),
        iteration_manager=MPIContext(random_state=1),
    )

    if best_parameters is not None:
        print(best_parameters)

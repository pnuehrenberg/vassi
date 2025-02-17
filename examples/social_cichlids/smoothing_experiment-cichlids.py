import numpy as np
from helpers import subsample_train
from scipy.signal import medfilt
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
from automated_scoring.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)

if __name__ == "__main__":
    cfg.key_keypoints = "pose"
    cfg.key_timestamp = "time_stamp"

    cfg.trajectory_keys = (
        "pose",
        "time_stamp",
    )

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

    observations = dataset_train.observations
    observations = observations[observations["category"] != "none"]
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

    def smooth(parameters, *, array):
        return medfilt(array, parameters["median_filter_window"])

    best_parameters = optimize_smoothing(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        smooth,
        smoothing_parameters_grid={"median_filter_window": np.arange(3, 91, 2)},
        remove_overlapping_predictions=False,
        num_iterations=4,
        k=5,
        sampling_func=subsample_train,
        tolerance=0.005,
        random_state=1,
        plot_results=False,
        results_path=".",
    )

    if best_parameters is not None:
        print(best_parameters)

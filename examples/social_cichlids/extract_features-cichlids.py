from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

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
        "cichlids", directory="../../datasets/social_cichlids", target="dyads"
    )

    dataset_train, dataset_test = dataset_full.split(0.8, random_state=1)

    observations = dataset_train.get_observations()
    observations = observations[observations["category"] != "none"]
    time_scales, slices = get_window_slices(
        3, time_scales=(91,), durations=observations["duration"]
    )

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

    for sampleable in dataset_full.sampling_targets:
        sampleable.extract_features(extractor)

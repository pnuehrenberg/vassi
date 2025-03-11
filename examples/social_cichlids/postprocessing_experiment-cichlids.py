from helpers import postprocessing, subsample_train, suggest_postprocessing_parameters
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from automated_scoring.classification.postprocessing import (
    optimize_postprocessing_parameters,
    summarize_experiment,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset
from automated_scoring.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)
from automated_scoring.logging import set_logging_level

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = ("pose", "time_stamp")


if __name__ == "__main__":
    from automated_scoring.distributed import DistributedExperiment

    dataset_full = load_dataset(
        "cichlids",
        directory="../../datasets/social_cichlids",
        target="dyad",
        background_category="none",
    )

    dataset_train, _ = dataset_full.split(
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

    log = set_logging_level("info")

    studies = optimize_postprocessing_parameters(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        postprocessing_function=postprocessing,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters,
        num_trials=2000,
        k=5,
        sampling_function=subsample_train,
        balance_sample_weights=True,
        experiment=DistributedExperiment(20, random_state=1),
        log=log,
    )

    if studies is not None:
        summarize_experiment(studies, log=log)

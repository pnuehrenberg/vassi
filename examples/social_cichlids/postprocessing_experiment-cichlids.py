from helpers import postprocessing, subsample_train, suggest_postprocessing_parameters
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from vassi.classification.postprocessing import (
    optimize_postprocessing_parameters,
    summarize_experiment,
)
from vassi.config import cfg
from vassi.features import DataFrameFeatureExtractor
from vassi.io import load_dataset
from vassi.logging import set_logging_level
from vassi.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = ("pose", "time_stamp")


if __name__ == "__main__":
    from vassi.distributed import DistributedExperiment

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
        cache_mode="cached",
    ).read_yaml("config_file-cichlids.yaml")

    log = set_logging_level("info")

    experiment = DistributedExperiment(20, random_state=1)

    studies = optimize_postprocessing_parameters(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        postprocessing_function=postprocessing,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters,
        num_trials=4000,
        k=5,
        sampling_function=subsample_train,
        balance_sample_weights=True,
        experiment=experiment,
        optimize_across_runs=True,
        parallel_optimization=True,
        log=log,
    )

    if experiment.is_root:
        summarize_experiment(studies, log=log)

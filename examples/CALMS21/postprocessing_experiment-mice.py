from helpers import postprocessing, subsample_train, suggest_postprocessing_parameters
from xgboost import XGBClassifier

from automated_scoring.classification.postprocessing import (
    optimize_postprocessing_parameters,
    summarize_experiment,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset
from automated_scoring.logging import set_logging_level

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"
cfg.trajectory_keys = ("keypoints", "timestamps")

if __name__ == "__main__":
    from automated_scoring.distributed import DistributedExperiment

    dataset_train = load_dataset(
        "mice_train",
        directory="../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    )
    dataset_train = dataset_train.exclude_individuals(["intruder"])

    extractor = DataFrameFeatureExtractor(
        cache_directory="feature_cache_mice",
        cache_mode="cached",
    ).read_yaml("config_file.yaml")

    log = set_logging_level("info")

    experiment = DistributedExperiment(20, random_state=1)

    studies = optimize_postprocessing_parameters(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        postprocessing_function=postprocessing,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters,
        num_trials=800,
        k=5,
        sampling_function=subsample_train,
        balance_sample_weights=True,
        experiment=experiment,
        optimize_across_runs=True,
        parallel_optimization=True,
        log=log,
    )

    experiment.barrier()

    if experiment.is_root:
        summarize_experiment(studies, log=log)

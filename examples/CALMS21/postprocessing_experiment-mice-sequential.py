from helpers import postprocessing, subsample_train, suggest_postprocessing_parameters
from xgboost import XGBClassifier

from vassi.classification.postprocessing import (
    optuna_parameter_optimization,
    run_k_fold_experiment,
    summarize_experiment,
)
from vassi.config import cfg
from vassi.features import DataFrameFeatureExtractor
from vassi.io import from_cache, from_yaml, load_dataset, to_yaml
from vassi.logging import set_logging_level

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"
cfg.trajectory_keys = ("keypoints", "timestamps")


def step_1():
    from vassi.distributed import DistributedExperiment

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

    experiment = DistributedExperiment(20, random_state=1)

    k_fold_results = run_k_fold_experiment(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=1000),
        k=5,
        sampling_function=subsample_train,
        balance_sample_weights=True,
        experiment=experiment,
        log=set_logging_level("info"),
        cache=True,
    )

    if experiment.is_root:
        to_yaml(k_fold_results, file_name="k_fold_results.yaml")


def step_2():
    log = set_logging_level("info")
    k_fold_results = [
        from_cache(k_fold_result) for k_fold_result in from_yaml("k_fold_results.yaml")
    ]
    studies = optuna_parameter_optimization(
        k_fold_results,
        postprocessing_function=postprocessing,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters,
        num_trials=800,
        random_state=1,
        parallel_optimization=False,
        experiment=None,
        log=log,
    )
    summarize_experiment(studies, log=log)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    match parse_args().step:
        case 1:
            step_1()
        case 2:
            step_2()

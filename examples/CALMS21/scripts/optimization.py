from xgboost import XGBClassifier

from vassi.classification.optimization import (
    Environment,
    macro_f1_all_levels,
    run_optuna_hyperparameter_search,
    summarize_study,
)
from vassi.config import cfg
from vassi.features import DataFrameExtractor
from vassi.io import load_dataset

from .helpers import parameter_space, postprocessing_function, sampling_function

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"
cfg.trajectory_keys = ("keypoints", "timestamps")

if __name__ == "__main__":
    dataset_train = load_dataset(
        "mice_train",
        directory="../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    )[0].exclude({"intruder"})

    dataset_test = load_dataset(
        "mice_test",
        directory="../../datasets/CALMS21/test",
        target="dyad",
        background_category="none",
    )[0].exclude({"intruder"})

    extractor = DataFrameExtractor.from_yaml(
        "features-mice.yaml",
        cache_mode=True,
    )

    study = run_optuna_hyperparameter_search(
        dataset_train,
        extractor,
        XGBClassifier,
        parameter_space,
        num_trials=1000,
        k=5,
        sampling_function=sampling_function,
        postprocessing_function=postprocessing_function,
        scoring_function=macro_f1_all_levels,
        n_jobs=4,
    )

    if Environment().is_root:
        _ = summarize_study(study)

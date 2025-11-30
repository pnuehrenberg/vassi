from xgboost import XGBClassifier

from vassi.classification.optimization import (
    macro_f1_all_levels,
    run_optuna_hyperparameter_search,
    summarize_study,
)
from vassi.config import cfg
from vassi.distributed import Environment
from vassi.features import DataFrameExtractor
from vassi.io import load_dataset

from .helpers import parameter_space, postprocessing_function, sampling_function

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"
cfg.trajectory_keys = ("keypoints", "timestamps")

if __name__ == "__main__":
    env = Environment()
    dataset_train = load_dataset(
        "mice_train",
        directory="../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    )[0].exclude({"intruder"})

    extractor = DataFrameExtractor.from_yaml(
        "features-mice.yaml",
        cache_mode="required",
    )

    study = run_optuna_hyperparameter_search(
        dataset_train,
        extractor,
        XGBClassifier,
        parameter_space,
        num_trials=2000,
        k=5,
        sampling_function=sampling_function,
        postprocessing_function=postprocessing_function,
        scoring_function=macro_f1_all_levels,
        n_jobs=4,
    )

    if env.is_root:
        _ = summarize_study(study)

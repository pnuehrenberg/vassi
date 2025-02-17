from helpers import subsample_train
from scipy.signal import medfilt
from xgboost import XGBClassifier

from automated_scoring.classification import (
    optimize_decision_thresholds,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset

if __name__ == "__main__":
    cfg.key_keypoints = "keypoints"
    cfg.key_timestamp = "timestamps"

    cfg.trajectory_keys = (
        "keypoints",
        "timestamps",
    )

    dataset_train = load_dataset(
        "mice_train",
        directory="../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    )
    dataset_test = load_dataset(
        "mice_test",
        directory="../../datasets/CALMS21/test",
        target="dyad",
        background_category="none",
    )

    extractor = DataFrameFeatureExtractor(
        cache_directory="feature_cache_mice"
    ).read_yaml("config_file.yaml")

    def smooth(*, array):
        return medfilt(array, 47)  # results from smoothing_experiment-mice.py

    best_parameters = optimize_decision_thresholds(
        dataset_train.exclude_individuals(["intruder"]),
        extractor,
        XGBClassifier(n_estimators=1000),
        remove_overlapping_predictions=False,
        smoothing_func=smooth,
        num_iterations=20,
        k=5,
        sampling_func=subsample_train,
        decision_threshold_range=(0.0, 1.0),
        decision_threshold_step=0.01,
        tolerance=0.005,
        random_state=1,
        plot_results=False,
        results_path=".",
    )

    if best_parameters is not None:
        print(best_parameters)

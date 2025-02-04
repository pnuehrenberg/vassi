import numpy as np
from helpers import subsample_train
from scipy.signal import medfilt
from xgboost import XGBClassifier

from automated_scoring.classification import (
    optimize_smoothing,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset
from automated_scoring.utils import ensure_generator

if __name__ == "__main__":
    cfg.key_keypoints = "keypoints"
    cfg.key_timestamp = "timestamps"

    cfg.trajectory_keys = (
        "keypoints",
        "timestamps",
    )

    dataset_train = load_dataset(
        "mice_train", directory="datasets/CALMS21/train", target="dyads"
    )
    dataset_test = load_dataset(
        "mice_test", directory="datasets/CALMS21/test", target="dyads"
    )

    extractor = DataFrameFeatureExtractor(
        cache_directory="feature_cache_mice"
    ).read_yaml("config_file.yaml")

    def smooth(parameters, *, array):
        return medfilt(array, parameters["median_filter_window"])

    best_parameters = optimize_smoothing(
        dataset_train,
        extractor,
        XGBClassifier(n_estimators=100),
        smooth,
        smoothing_parameters_grid={"median_filter_window": np.arange(3, 91, 10)},
        remove_overlapping_predictions=False,
        num_iterations=20,
        show_progress=True,
        k=2,
        exclude=[("intruder", "resident")],
        sampling_func=subsample_train,
        show_k_fold_progress=True,
        tolerance=0.005,
        random_state=ensure_generator(1),
        plot_results=False,
    )

    if best_parameters is not None:
        print(best_parameters)  # 47

# import numpy as np
# from helpers import smooth, subsample_train
# from xgboost import XGBClassifier

# from automated_scoring.classification import (
#     optimize_smoothing,
# )
# from automated_scoring.config import cfg
# from automated_scoring.features import DataFrameFeatureExtractor
# from automated_scoring.io import load_dataset
# from automated_scoring.logging import set_logging_level

# cfg.key_keypoints = "keypoints"
# cfg.key_timestamp = "timestamps"

# cfg.trajectory_keys = (
#     "keypoints",
#     "timestamps",
# )

# if __name__ == "__main__":
#     from automated_scoring.mpi_utils import MPIContext

#     dataset_train = load_dataset(
#         "mice_train",
#         directory="../../datasets/CALMS21/train",
#         target="dyad",
#         background_category="none",
#     )
#     dataset_test = load_dataset(
#         "mice_test",
#         directory="../../datasets/CALMS21/test",
#         target="dyad",
#         background_category="none",
#     )

#     extractor = DataFrameFeatureExtractor(
#         cache_directory="feature_cache_mice"
#     ).read_yaml("config_file.yaml")

#     best_parameters = optimize_smoothing(
#         dataset_train.exclude_individuals(["intruder"]),
#         extractor,
#         XGBClassifier(n_estimators=1000),
#         smooth,
#         smoothing_parameters_grid={"median_filter_window": np.arange(3, 91, 2)},
#         remove_overlapping_predictions=False,
#         num_iterations=20,
#         k=5,
#         sampling_func=subsample_train,
#         plot_results=False,
#         results_path="optimization_results/smoothing",
#         log=set_logging_level("info"),
#         iteration_manager=MPIContext(random_state=1),
#     )

#     if best_parameters is not None:
#         print(best_parameters)

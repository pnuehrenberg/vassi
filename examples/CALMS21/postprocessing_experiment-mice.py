from helpers import postprocessing, subsample_train, suggest_postprocessing_parameters
from xgboost import XGBClassifier

from automated_scoring.classification.postprocessing import (
    optimize_postprocessing_parameters,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import load_dataset, to_yaml
from automated_scoring.logging import set_logging_level

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"

cfg.trajectory_keys = (
    "keypoints",
    "timestamps",
)

if __name__ == "__main__":
    from automated_scoring.distributed import DistributedExperiment

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

    studies = optimize_postprocessing_parameters(
        dataset_train.exclude_individuals(["intruder"]),
        extractor,
        XGBClassifier(n_estimators=100),
        postprocessing_function=postprocessing,
        suggest_postprocessing_parameters_function=suggest_postprocessing_parameters,
        remove_overlapping_predictions=False,
        overlapping_predictions_kwargs=None,
        num_runs=2,
        k=5,
        sampling_function=subsample_train,
        balance_sample_weights=True,
        results_path=".",
        num_trials=100,
        experiment=DistributedExperiment(random_state=1),
        log=set_logging_level("info"),
    )

    if studies is not None:
        to_yaml(
            [
                {
                    "best": study.best_trial.number,
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "results": study.trials_dataframe(
                        ("number", "params", "value")
                    ).to_dict(orient="records"),
                }
                for study in studies
            ],
            file_name="optimization_results.yaml",
        )

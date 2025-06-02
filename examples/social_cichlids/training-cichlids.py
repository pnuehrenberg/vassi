import os

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from vassi.config import cfg
from vassi.io import from_cache, to_cache

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = ("pose", "time_stamp")


if __name__ == "__main__":
    from vassi.distributed import DistributedExperiment

    experiment = DistributedExperiment(20, random_state=1)
    cache_directory = "samples_cache"

    for run in experiment:
        X, y = from_cache(os.path.join(cache_directory, f"samples_{run:02d}.cache"))

        classifier = XGBClassifier(
            n_estimators=1000, random_state=experiment.random_state
        ).fit(X.to_numpy(), y, sample_weight=compute_sample_weight("balanced", y))
        to_cache(
            classifier, cache_file=f"clf_{run:02d}.cache", directory=cache_directory
        )

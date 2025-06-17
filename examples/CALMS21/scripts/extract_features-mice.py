from vassi.config import cfg
from vassi.features import DataFrameFeatureExtractor
from vassi.io import load_dataset

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"

cfg.trajectory_keys = (
    "keypoints",
    "timestamps",
)

if __name__ == "__main__":
    dataset_train = load_dataset(
        "mice_train",
        directory="../../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    )
    dataset_test = load_dataset(
        "mice_test",
        directory="../../../datasets/CALMS21/test",
        target="dyad",
        background_category="none",
    )

    extractor = DataFrameFeatureExtractor(
        cache_directory="../feature_cache_mice"
    ).read_yaml("../features-mice.yaml")

    for _, group in dataset_train.exclude_individuals(["intruder"]):
        for _, sampleable in group:
            sampleable.sample_X(extractor)

    for _, group in dataset_test.exclude_individuals(["intruder"]):
        for _, sampleable in group:
            sampleable.sample_X(extractor)

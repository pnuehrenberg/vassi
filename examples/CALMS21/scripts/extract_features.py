from vassi.config import cfg
from vassi.dataset import AnnotatedDataset
from vassi.features import DataFrameExtractor

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"

cfg.trajectory_keys = (
    "keypoints",
    "timestamps",
)

if __name__ == "__main__":
    dataset_train = AnnotatedDataset.load_legacy(
        "../../datasets/CALMS21/train/mice_train_trajectories.h5",
        observation_file="../../datasets/CALMS21/train/mice_train_annotations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    dataset_test = AnnotatedDataset.load_legacy(
        "../../datasets/CALMS21/test/mice_train_trajectories.h5",
        observation_file="../../datasets/CALMS21/test/mice_test_observations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    extractor = DataFrameExtractor.from_yaml(
        "features-mice.yaml",
        cache_mode=True,
    )

    for element in [element for _, group in dataset_train for _, element in group] + [
        element for _, group in dataset_test for _, element in group
    ]:
        _ = element.sample_X(extractor, indices=None, out=None)

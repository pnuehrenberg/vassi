from vassi.config import cfg
from vassi.features import DataFrameExtractor
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

    for element in [element for _, group in dataset_train for _, element in group] + [
        element for _, group in dataset_test for _, element in group
    ]:
        _ = element.sample_X(extractor, indices=None, out=None)

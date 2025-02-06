import json

import numpy as np

from automated_scoring.data_structures import Trajectory
from automated_scoring.dataset import AnnotatedGroup, Dataset
from automated_scoring.dataset.observations import to_observations
from automated_scoring.io import save_dataset


def load_calms21_sequences(calms21_json_file):
    sequences = []
    with open(calms21_json_file) as json_file:
        # load entire dataset file
        json_data = json.load(json_file)
        for data in json_data.values():
            # each value is one annotator (task one only has one annotator)
            for pair in [key for key in data.keys()]:
                # iterate all video sequences
                pair_data = data.pop(pair)
                annotations = to_observations(
                    np.asarray(pair_data["annotations"]),
                    category_names=["attack", "investigation", "mount", "other"],
                    drop=["other"],
                )
                annotations["actor"] = "resident"
                annotations["recipient"] = "intruder"
                sequences.append(
                    (
                        {
                            "resident": Trajectory(
                                data={
                                    "keypoints": np.asarray(pair_data["keypoints"])[
                                        :, 0
                                    ].transpose(0, 2, 1),
                                    "timestamps": np.arange(
                                        len(pair_data["keypoints"])
                                    ),
                                }
                            ),
                            "intruder": Trajectory(
                                data={
                                    "keypoints": np.asarray(pair_data["keypoints"])[
                                        :, 1
                                    ].transpose(0, 2, 1),
                                    "timestamps": np.arange(
                                        len(pair_data["keypoints"])
                                    ),
                                }
                            ),
                        },
                        annotations,
                    )
                )
            del data
    return sequences


if __name__ == "__main__":
    dataset_train = Dataset(
        {
            idx: AnnotatedGroup(
                trajectories,
                target="dyads",
                observations=annotations,
                categories=("attack", "investigation", "mount"),
            )
            for idx, (trajectories, annotations) in enumerate(
                load_calms21_sequences(
                    "/home/paul/Downloads/task1_classic_classification/calms21_task1_train.json"
                )
            )
        }
    )

    dataset_test = Dataset(
        {
            idx: AnnotatedGroup(
                trajectories,
                target="dyads",
                observations=annotations,
                categories=("attack", "investigation", "mount"),
            )
            for idx, (trajectories, annotations) in enumerate(
                load_calms21_sequences(
                    "/home/paul/Downloads/task1_classic_classification/calms21_task1_test.json"
                )
            )
        }
    )

    save_dataset(dataset_test, directory="datasets", dataset_name="mice_test")
    save_dataset(dataset_train, directory="datasets", dataset_name="mice_train")

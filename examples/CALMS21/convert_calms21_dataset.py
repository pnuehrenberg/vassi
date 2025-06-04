import argparse
import json
import os

import numpy as np
import pandas as pd

from vassi.config import cfg
from vassi.data_structures import Trajectory
from vassi.dataset import (
    AnnotatedDataset,
    AnnotatedGroup,
    IndividualIdentifier,
)
from vassi.dataset.observations import to_observations
from vassi.io import save_dataset


def load_calms21_sequences(
    calms21_json_file: str,
) -> list[tuple[dict[IndividualIdentifier, Trajectory], pd.DataFrame]]:
    sequences = []
    with open(calms21_json_file) as json_file:
        # load entire dataset file
        json_data = json.load(json_file)
        for data in json_data.values():
            # each value is one annotator (task one only has one annotator)
            for sequence in [key for key in data.keys()]:
                # iterate all video sequences
                sequence_data = data.pop(sequence)
                annotations = to_observations(
                    np.asarray(sequence_data["annotations"]),
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
                                    "keypoints": np.asarray(sequence_data["keypoints"])[
                                        :, 0
                                    ].transpose(0, 2, 1),
                                    "timestamps": np.arange(
                                        len(sequence_data["keypoints"])
                                    ),
                                }
                            ),
                            "intruder": Trajectory(
                                data={
                                    "keypoints": np.asarray(sequence_data["keypoints"])[
                                        :, 1
                                    ].transpose(0, 2, 1),
                                    "timestamps": np.arange(
                                        len(sequence_data["keypoints"])
                                    ),
                                }
                            ),
                        },
                        annotations,
                    )
                )
            del data
    return sequences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sequences", type=str, required=True)
    parser.add_argument("--test_sequences", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    return parser.parse_args()


def convert_calms21_sequences(
    train_sequences: str,
    test_sequences: str,
    output_directory: str,
):
    global cfg

    # this is how the data will be named in all data structures
    cfg.key_keypoints = "keypoints"
    cfg.key_timestamp = "timestamps"
    cfg.trajectory_keys = ("keypoints", "timestamps")

    groups_train: dict[IndividualIdentifier, AnnotatedGroup] = {
        idx: AnnotatedGroup(
            trajectories,
            target="dyad",
            observations=annotations,
            categories=("attack", "investigation", "mount"),
            background_category="none",
        )
        for idx, (trajectories, annotations) in enumerate(
            load_calms21_sequences(train_sequences)
        )
    }

    groups_test: dict[IndividualIdentifier, AnnotatedGroup] = {
        idx: AnnotatedGroup(
            trajectories,
            target="dyad",
            observations=annotations,
            categories=("attack", "investigation", "mount"),
            background_category="none",
        )
        for idx, (trajectories, annotations) in enumerate(
            load_calms21_sequences(test_sequences)
        )
    }

    save_dataset(
        AnnotatedDataset.from_groups(groups_train),
        directory=os.path.join(output_directory, "train"),
        dataset_name="mice_train",
    )
    save_dataset(
        AnnotatedDataset.from_groups(groups_test),
        directory=os.path.join(output_directory, "test"),
        dataset_name="mice_test",
    )


if __name__ == "__main__":
    args = parse_args()
    convert_calms21_sequences(
        args.train_sequences,
        args.test_sequences,
        args.output_directory,
    )

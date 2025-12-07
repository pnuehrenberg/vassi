from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import (
    compute_sample_weight,  # pyright: ignore[reportUnknownVariableType]
)
from xgboost import XGBClassifier

from vassi.classification import predict
from vassi.classification.optimization.utils import get_validated_aggregator_kwargs
from vassi.config import cfg
from vassi.dataset import AnnotatedDataset
from vassi.distributed import Environment
from vassi.features import DataFrameExtractor
from vassi.io.h5 import write_h5_data
from vassi.io.yaml import from_yaml
from vassi.sliding_metrics import SlidingWindowAggregator
from vassi.type_guards import is_mapping_of

from .helpers import parameter_space, sampling_function, smooth_model_outputs

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"
cfg.trajectory_keys = ("keypoints", "timestamps")


def _flat(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return pd.DataFrame(
        np.array(df).flatten().reshape(1, -1),
        columns=pd.Index(
            map(
                lambda pair: "-".join(map(str, pair)) + f"-{suffix}",
                product(df.index, df.columns),
            )
        ),
    )


if __name__ == "__main__":
    env = Environment()
    dataset_train = AnnotatedDataset.load(
        "../../datasets/CALMS21/train/mice_train_trajectories.h5",
        observation_file="../../datasets/CALMS21/train/mice_train_observations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    dataset_test = AnnotatedDataset.load(
        "../../datasets/CALMS21/test/mice_train_trajectories.h5",
        observation_file="../../datasets/CALMS21/test/mice_test_observations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    best_parameters = from_yaml("optimization/session_2/optimization-summary.yaml")
    if not is_mapping_of(best_parameters, str, object):
        raise ValueError("Expected parameters to be a mapping of strings to objects")

    parameters = parameter_space.parse(best_parameters)
    min_samples_per_stratum = parameters["sampling_function_kwargs"][
        "min_samples_per_stratum"
    ]
    if not isinstance(min_samples_per_stratum, int):
        raise ValueError("Expected min_samples_per_stratum to be an integer")

    n_estimators = parameters["classifier_kwargs"]["n_estimators"]
    if not isinstance(n_estimators, int):
        raise ValueError("Expected n_estimators to be an integer")

    n_estimators = parameters["classifier_kwargs"]["n_estimators"]
    if not isinstance(n_estimators, int):
        raise ValueError("Expected n_estimators to be an integer")

    aggregator = None
    if parameters["use_sliding_window_features"] and (
        kwargs := get_validated_aggregator_kwargs(parameters["aggregator_kwargs"])
    ):
        aggregator = SlidingWindowAggregator(**kwargs)

    extractor = DataFrameExtractor.from_yaml(
        "features-mice.yaml",
        cache_mode=False,  # not safe when using sliding window features with unknown aggregation and MPI
        aggregator=aggregator,
    )

    summary: dict[
        int, tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]
    ] = {}

    result_for_visualization = None

    for run in range(20):
        if run % env.size != env.rank:
            continue
        x, y = sampling_function(
            dataset_train,
            extractor,
            min_samples_per_stratum=min_samples_per_stratum,
            random_state=run,
        )
        classifier = XGBClassifier(n_estimators=n_estimators, random_state=run)

        sample_weights = None
        if parameters["balance_sample_weights"]:
            sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        classifier = classifier.fit(x, y, sample_weight=sample_weights)

        result = predict(dataset_test, classifier, extractor)

        # # instead of running the postprocessing function, we run smoothing and thresholding separately
        # result_postprocessed = postprocessing_function(
        #     result, **parameters["postprocessing_function_kwargs"]
        # )

        # see helpers.py postprocessing_function
        smoothing_function_kwargs = dict(parameters["postprocessing_function_kwargs"])
        decision_thresholds = {
            threshold.replace("decision_threshold-", "", 1): value
            for threshold in parameters["postprocessing_function_kwargs"]
            if threshold.startswith("decision_threshold-")
            and (value := smoothing_function_kwargs.pop(threshold))
            and isinstance(value, (float, int))
        }
        result_smoothed = result.smooth(
            partial(
                smooth_model_outputs, result.categories, **smoothing_function_kwargs
            ),
        )

        result_thresholded = result_smoothed.discretize(decision_thresholds)

        if run == 0:
            result_for_visualization = result_thresholded

        summary[run] = (
            {
                "raw": _flat(result.score(), "raw").assign(run=run),
                "smoothed": _flat(result_smoothed.score(), "smoothed").assign(run=run),
                "thresholded": _flat(result_thresholded.score(), "thresholded").assign(
                    run=run
                ),
            },
            {
                "true": {
                    "timestamp": result_thresholded.y_gt,
                    "annotation": result_thresholded.encode(
                        np.array(result_thresholded.annotations["category"])
                    ),
                    "prediction": result_thresholded.encode(
                        np.array(result_thresholded.predictions["true_category"])
                    ),
                },
                "pred": {
                    "timestamp": result_thresholded.y,
                    "annotation": result_thresholded.encode(
                        np.array(result_thresholded.annotations["predicted_category"])
                    ),
                    "prediction": result_thresholded.encode(
                        np.array(result_thresholded.predictions["category"])
                    ),
                },
            },
        )

    gathered_summaries = env.gather(summary)

    if not env.is_root:
        exit()

    summary = {}
    for _summary in gathered_summaries:
        for key, value in _summary.items():
            if key in summary:
                raise ValueError(f"Duplicate run: {key}")
            summary[key] = value

    scores_raw = pd.concat(
        [scores["raw"] for scores, _ in summary.values()], ignore_index=True
    )

    scores_smoothed = pd.concat(
        [scores["smoothed"] for scores, _ in summary.values()], ignore_index=True
    )

    scores_thresholded = pd.concat(
        [scores["thresholded"] for scores, _ in summary.values()], ignore_index=True
    )

    scores = pd.concat(
        [
            scores_raw.drop(columns=["run"]),
            scores_smoothed.drop(columns=["run"]),
            scores_thresholded,
        ],
        axis=1,
    )

    for run, (_, y_data) in summary.items():
        for name, y in y_data.items():
            write_h5_data("results.h5", data=y, data_path=f"y/{run}/{name}", key=name)
    scores.to_hdf("results.h5", key="scores", index=False)

    if result_for_visualization is None:
        raise ValueError("expected result, evaulation requires runs > 0")
    result_for_visualization.to_h5(
        "results.h5", data_path="result_thresholded", key="result_thresholded"
    )

    print(scores.drop(columns=["run"]).mean(axis=0))

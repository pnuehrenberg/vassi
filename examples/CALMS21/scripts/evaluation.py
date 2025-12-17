from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import (
    compute_sample_weight,  # pyright: ignore[reportUnknownVariableType]
)
from xgboost import XGBClassifier

from vassi.classification import predict
from vassi.config import cfg
from vassi.dataset import AnnotatedDataset
from vassi.distributed import Environment
from vassi.features import DataFrameExtractor
from vassi.io.h5 import write_h5_data
from vassi.io.yaml import from_yaml
from vassi.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    sliding_mean,
    sliding_median,
)
from vassi.type_guards import is_mapping_of

from .helpers import (
    CALMS21PipelineParams,
    ParamsClassifier,
    sampling_function,
    smooth_model_outputs,
)

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
        observation_file="../../datasets/CALMS21/train/mice_train_annotations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    dataset_test = AnnotatedDataset.load(
        "../../datasets/CALMS21/test/mice_test_trajectories.h5",
        observation_file="../../datasets/CALMS21/test/mice_test_annotations.csv",
        target="dyad",
        background_category="none",
    ).exclude({"intruder"})

    best_parameters = from_yaml("optimization/session_6/optimization-summary.yaml")
    if not is_mapping_of(best_parameters, str, object):
        raise ValueError("Expected parameters to be a mapping of strings to objects")

    params = CALMS21PipelineParams.init_from(dict(best_parameters))
    n_estimators = (
        params.params_classifier
        if isinstance(params.params_classifier, ParamsClassifier)
        else None
    )

    aggregator = None
    if params.use_sliding_window_features and (
        aggregator_params := params.params_feature_aggregator
    ):
        # see classification.optimization.search, could be moved to helper function (init_aggregator)
        if aggregator_params.sliding_metric == "mean":
            metric_funcs = [sliding_mean]
        elif aggregator_params.sliding_metric == "median":
            metric_funcs = [sliding_median]
        else:
            raise ValueError("expected one of 'mean', 'median'")
        windows, window_slices = get_window_slices(
            aggregator_params.num_slices_per_window,
            windows=[aggregator_params.window_size],
        )
        aggregator = SlidingWindowAggregator(
            metric_funcs,
            window_size=max(windows),
            window_slices=window_slices,
            keep_original=aggregator_params.keep_original_features,
        )

    # see helpers.py postprocessing_function, this could also moved to a helper function (parse_decision_thresholds)
    decision_thresholds = (
        {
            category: threshold
            for category, category_params in params.params_category_output.items()
            if category_params.use_thresholding
            and (threshold := category_params.threshold)
        }
        if params.params_category_output
        else None
    )

    extractor = DataFrameExtractor.from_yaml(
        "features-mice.yaml",
        cache_mode=False,  # not safe when using sliding window features (may consume a lot of disk space)
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
            params=params,
            random_state=run,
        )
        classifier = XGBClassifier(n_estimators=n_estimators, random_state=run)

        sample_weights = None
        if params.balance_sample_weights:
            sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        classifier = classifier.fit(x, y, sample_weight=sample_weights)

        result = predict(dataset_test, classifier, extractor)

        # # instead of running the postprocessing function, we run smoothing and thresholding separately
        # result_postprocessed = postprocessing_function(
        #     result, params=params
        # )

        result_smoothed = result.smooth(
            partial(smooth_model_outputs, result.categories, params=params),
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

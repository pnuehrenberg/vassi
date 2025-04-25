import os
from functools import partial

import numpy as np
import pandas as pd
from helpers import score_priority, smooth_model_outputs
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from automated_scoring.classification import (
    predict,
)
from automated_scoring.config import cfg
from automated_scoring.features import DataFrameFeatureExtractor
from automated_scoring.io import from_cache, from_yaml, load_dataset, to_cache
from automated_scoring.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = ("pose", "time_stamp")


def summarize_scores(result, *, foreground_categories, run, postprocessing_step):
    scores = test_result.score()
    summary = scores.stack().reset_index()
    summary = pd.DataFrame(
        np.array(summary[0]),
        index=summary["level_0"] + "_f1" + "-" + summary["level_1"],
    ).T
    columns = summary.columns
    summary["run"] = run
    summary["postprocessing_step"] = postprocessing_step
    summary = summary[["run", "postprocessing_step", *columns]]
    for level in scores.index:
        summary[f"{level}_f1-macro-foreground"] = scores.loc[
            level, list(foreground_categories)
        ].mean()
        summary[f"{level}_f1-macro-all"] = scores.loc[level].mean()
    summary.columns = pd.MultiIndex.from_tuples(
        [
            (column.split("-", 1) if "-" in column else (column, ""))
            for column in summary.columns
        ]
    )
    return summary


if __name__ == "__main__":
    from automated_scoring.distributed import DistributedExperiment

    dataset_full = load_dataset(
        "cichlids",
        directory="../../datasets/social_cichlids",
        target="dyad",
        background_category="none",
    )

    _, dataset_test = dataset_full.split(
        0.8,
        random_state=1,
    )

    best_parameters = from_yaml("optimization-summary.yaml")

    priority_function = partial(
        score_priority,
        weight_max_probability=best_parameters["weight_max_probability"],
        weight_mean_probability=1 - best_parameters["weight_max_probability"],
    )
    best_thresholds = [
        best_parameters[f"threshold-{category}"] for category in dataset_test.categories
    ]

    time_scales, slices = get_window_slices(3, time_scales=(91,))

    aggregator = ColumnTransformer(
        [
            (
                "aggregate",
                SlidingWindowAggregator(
                    [metrics.median, metrics.q10, metrics.q90], max(time_scales), slices
                ),
                make_column_selector(),
            ),
            ("original", "passthrough", make_column_selector()),
        ],
    )

    pipeline = Pipeline(
        [("impute", KNNImputer()), ("aggregate", aggregator)]
    ).set_output(transform="pandas")

    extractor = DataFrameFeatureExtractor(
        cache_directory="cichlids_cache",
        pipeline=pipeline,
        refit_pipeline=True,
        cache_mode="cached",
    ).read_yaml("config_file-cichlids.yaml")

    experiment = DistributedExperiment(20, random_state=1)
    cache_directory = "samples_cache"

    for run in experiment:
        classifier = from_cache(os.path.join(cache_directory, f"clf_{run:02d}.cache"))

        summary = []
        y = {"true": {}, "pred": {}}

        test_result = predict(
            dataset_test, classifier, extractor, log=None
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="model_outputs",
            )
        )

        test_result = test_result.smooth(
            partial(smooth_model_outputs, best_parameters)
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="smoothed",
            )
        )

        test_result = test_result.threshold(
            best_thresholds, default_decision="none"
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="thresholded",
            )
        )

        summary = pd.concat(summary, ignore_index=True)

        y["true"]["timestamp"] = test_result.y_true_numeric
        y["pred"]["timestamp"] = test_result.y_pred_numeric
        y["true"]["annotation"] = dataset_test.encode(
            test_result.annotations["category"].to_numpy()
        )
        y["pred"]["annotation"] = dataset_test.encode(
            test_result.annotations["predicted_category"].to_numpy()
        )
        y["true"]["prediction"] = dataset_test.encode(
            test_result.predictions["true_category"].to_numpy()
        )
        y["pred"]["prediction"] = dataset_test.encode(
            test_result.predictions["category"].to_numpy()
        )

        experiment.add((summary, y))

    summary = pd.concat(
        [summary for summary, _ in experiment.collect().values()], ignore_index=True
    )
    confusion = [y for _, y in experiment.collect().values()]

    if experiment.is_root:
        to_cache([summary, confusion], cache_file="results.cache")
        to_cache(test_result, cache_file="predictions.cache")

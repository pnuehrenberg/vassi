import os
from functools import partial

import numpy as np
import pandas as pd
from helpers import score_priority, smooth_model_outputs
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

import vassi._manuscript_utils as manuscript_utils
from vassi.classification import (
    predict,
)
from vassi.config import cfg
from vassi.features import DataFrameFeatureExtractor
from vassi.io import from_cache, from_yaml, load_dataset, save_data
from vassi.logging import set_logging_level
from vassi.sliding_metrics import (
    SlidingWindowAggregator,
    get_window_slices,
    metrics,
)

cfg.key_keypoints = "pose"
cfg.key_timestamp = "time_stamp"

cfg.trajectory_keys = ("pose", "time_stamp")


if __name__ == "__main__":
    from vassi.distributed import DistributedExperiment

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

    log = set_logging_level("info")

    experiment = DistributedExperiment(20, random_state=1)
    cache_directory = "samples_cache"
    test_result = None  # dummy variable if no predictions are made

    for run in experiment:
        classifier = from_cache(os.path.join(cache_directory, f"clf_{run:02d}.cache"))

        log.info("classifier loaded")

        summary = []
        y = {"true": {}, "pred": {}}

        test_result = predict(
            dataset_test, classifier, extractor, log=log
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="model_outputs",
            )
        )

        log.info("finished scoring model outputs")

        test_result = test_result.smooth(
            partial(smooth_model_outputs, best_parameters)
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="smoothed",
            )
        )

        log.info("finished scoring smoothed results")

        test_result = test_result.threshold(
            best_thresholds, default_decision="none"
        ).remove_overlapping_predictions(
            priority_function=priority_function,
            prefilter_recipient_bouts=best_parameters["prefilter_recipient_bouts"],
            max_bout_gap=best_parameters["max_bout_gap"],
            max_allowed_bout_overlap=best_parameters["max_allowed_bout_overlap"],
        )
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="thresholded",
            )
        )

        log.info("finished scoring thresholded results")

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

        log.info("finished run")

    results = experiment.collect()

    summary = pd.concat([summary for summary, _ in results.values()], ignore_index=True)
    confusion = [y for _, y in results.values()]

    log.info("collected results")

    if not experiment.is_root:
        exit()

    for run, confusion_data in enumerate(confusion):
        save_data(
            "results.h5",
            confusion_data["true"],
            os.path.join(f"run_{run:02d}", "true"),
        )
        save_data(
            "results.h5",
            confusion_data["pred"],
            os.path.join(f"run_{run:02d}", "pred"),
        )
    save_data(
        "results.h5",
        {"runs": np.array([f"run_{run:02d}" for run in range(len(confusion))])},
    )
    summary.to_hdf("results.h5", key="summary")

    if test_result is None:
        log.error("no results to save")
    else:
        test_result.to_h5("results.h5", dataset_name="test_dataset")
        log.info("saved results")

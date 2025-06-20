import sys

sys.path.append("..")

import os
from functools import partial

import numpy as np
import pandas as pd
from helpers import smooth_model_outputs, subsample_train
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import vassi._manuscript_utils as manuscript_utils
from vassi.classification import (
    predict,
)
from vassi.config import cfg
from vassi.features import DataFrameFeatureExtractor
from vassi.io import from_yaml, load_dataset, save_data
from vassi.logging import set_logging_level, with_loop  # log_loop

cfg.key_keypoints = "keypoints"
cfg.key_timestamp = "timestamps"

cfg.trajectory_keys = (
    "keypoints",
    "timestamps",
)

if __name__ == "__main__":
    from vassi.distributed import DistributedExperiment

    dataset_train = load_dataset(
        "mice_train",
        directory="../../../datasets/CALMS21/train",
        target="dyad",
        background_category="none",
    ).exclude_individuals(["intruder"])

    dataset_test = load_dataset(
        "mice_test",
        directory="../../../datasets/CALMS21/test",
        target="dyad",
        background_category="none",
    ).exclude_individuals(["intruder"])

    extractor = DataFrameFeatureExtractor(
        cache_directory="../feature_cache_mice", cache_mode="cached"
    ).read_yaml("../features-mice.yaml")

    best_parameters = from_yaml("optimization-summary.yaml")
    best_thresholds = [
        best_parameters[f"threshold-{category}"] for category in dataset_test.categories
    ]

    log = set_logging_level("info")

    experiment = DistributedExperiment(20, random_state=1)
    test_result = None  # dummy variable if no predictions are made

    # for _log, run in log_loop(
    #     experiment,
    #     level="info",
    #     name="run",
    #     total=experiment.num_runs,
    #     message="evaluation",
    # ):
    for run in experiment:
        _log, _ = with_loop(log, name="run", step=run)

        X, y = subsample_train(
            dataset_train,
            extractor,
            random_state=experiment.random_state,
            log=_log,
        )
        y = dataset_train.encode(y)

        classifier = XGBClassifier(
            n_estimators=1000, random_state=experiment.random_state
        ).fit(X.to_numpy(), y, sample_weight=compute_sample_weight("balanced", y))

        summary = []
        y = {"true": {}, "pred": {}}

        test_result = predict(dataset_test, classifier, extractor, log=_log)
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="model_outputs",
            )
        )

        _log.info("finished scoring model outputs")

        test_result = test_result.smooth(partial(smooth_model_outputs, best_parameters))
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="smoothed",
            )
        )

        _log.info("finished scoring smoothed results")

        test_result = test_result.threshold(best_thresholds, default_decision="none")
        summary.append(
            manuscript_utils.summarize_scores(
                test_result,
                foreground_categories=dataset_test.foreground_categories,
                run=run,
                postprocessing_step="thresholded",
            )
        )

        _log.info("finished scoring thresholded results")

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

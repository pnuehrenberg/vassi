Basic usage
===========

The *automated-scoring* package provides high-level functions and classes for a behavioral classification pipeline. In the following example, we break down a minimal example of running the entire pipeline on the *CALMS21* dataset. You can find the full notebook on one of the next pages.

First, we load the *CALMS21* training dataset:

.. code-block:: python

    from automated_scoring.config import cfg
    from automated_scoring.io import load_dataset

    # set configuration keys (see conversion.py for details)
    cfg.key_keypoints = "keypoints"
    cfg.key_timestamp = "timestamps"
    cfg.trajectory_keys = ("keypoints", "timestamps")

    dataset_train = load_dataset(
        "mice_train",
        directory="../../datasets/CALMS21/train",  # set directory
        target="dyad",
        background_category="none",
    )

    # remove all dyads in which the intruder mouse is "actor"
    # (i.e., first individual in the dyad)
    dataset_train = dataset_train.exclude_individuals(["intruder"])

Then, we can create a feature extractor to sample the dataset.

.. note::
    Feature extraction can be cached by specifying the :code:`cache_mode` argument to either :code:`True` or :code:`'cached'`. For the latter, feature extraction requires already cached feature files. You need to specify :code:`cache_directory` in both cases.
    Caching can help you to avoid recalculation of the same individuals or dyads and is also aware of the feature extractor configuration.

.. code-block:: python

    from automated_scoring.features import DataFrameFeatureExtractor

    extractor = DataFrameFeatureExtractor(cache_mode=False)
    extractor.read_yaml("config_file.yaml")

With this extractor, we can compute the defined features for all dyads in the training dataset. For this quick example, we subsample the dataset so that each behavioral category is represented with 1000 samples each. This produces a dataframe for the features, and a numpy array containing corresponding labels.

.. note::
    Have a look at the documentation for more details on subsampling. Here, we use the default arguments for stratified subsampling that ensures that each dyad is proportionally represented in the resulting samples. You can fix the random state to ensure reproducibility.

.. code-block:: python

    X, y = dataset_train.subsample(
        extractor,
        size={category: 1000 for category in dataset_train.categories},
        random_state=1,
    )
    y = dataset_train.encode(y)  # Encode the labels as integers

Now, we can train a model on the subsampled dataset. For this basic example, we choose a :code:`RandomForestClassifier` from the :code:`sklearn.ensemble` `module <https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles>`_.

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(random_state=1)
    classifier.fit(X, y)

Until now, the example only used the training dataset. Let's load the test dataset for evaluation and use the fitted classifier to predict on all dyads.

.. code-block:: python

    from automated_scoring.classification.predict import predict

    dataset_test = load_dataset(
        "mice_test",
        directory="../../datasets/CALMS21/test",  # set directory
        target="dyad",
        background_category="none",
    )
    dataset_test = dataset_test.exclude_individuals(["intruder"])

    result_test = predict(dataset_test, classifier, extractor)

The resulting object :code:`result_test` holds the true and predicted labels for each dyad, for all timestamps (video frames), but also aggregated as intervals for :code:`predictions` and :code:`annotations` (both as properties that return a :code:`DataFrame`). Since we predicted on the entire test dataset, the result is a nested object that contains predictions for each group (video sequences of the *CALMS21* dataset) and each dyad (only one dyad per group: :code:`('resident', 'intruder')`).

These result objects provide easy access to evaluation metrics, such as F1 scores and confusion matrices. We can also visualize predictions as behavioral timelines.

.. code-block:: python

    from automated_scoring.classification.visualization import (
        plot_confusion_matrix,
        plot_classification_timeline,
    )

    plot_confusion_matrix(
        result_test.y_true_numeric,
        result_test.y_pred_numeric,
        category_labels=result_test.categories,
    )

    result_group = result_test.classification_results[10]
    result_dyad = result_group.classification_results[("resident", "intruder")]

    plot_classification_timeline(
        result_dyad.predictions,
        annotations=result_dyad.annotations,
        categories=result_dyad.categories,
        y_proba=result_dyad.y_proba,
        timestamps=result_dyad.timestamps,
    )

.. image:: 3_getting_started_confusion.svg
    :width: 350
    :align: center
    :alt: Confusion matrix for all frames of the test dataset.

.. image:: 3_getting_started_timeline.svg
    :alt: Behavioral timeline for test sequence 11.

Although we only trained a simple model on a subset of 4000 samples, the model already seems to classify the majority of the frames correctly.
You can fit any classification model that implements the :code:`sklearn` predictor `API <https://scikit-learn.org/stable/developers/develop.html#estimators>`_ to improve classification results, for example also :code:`XGBoost` classifiers. The *automated-scoring* package further provides two postprocessing steps to improve classification results, *smoothing* and *thresholding*. Have a look at the example notebooks to reproduce the results as presented in the paper.

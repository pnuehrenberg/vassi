.. vassi documentation master file, created by
   sphinx-quickstart on Tue Dec 17 17:23:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. centered::
   Verifiable, automated scoring of social interactions in animal groups using trajectory data

.. image:: source/vassi_logo.svg
    :width: 400
    :align: center
    :alt: vassi logo.

.. admonition:: *vassi* can help you to

    - organize :doc:`trajectory and posture data <source/4_import_data>` in datasets with groups of multiple individuals
    - extract :doc:`individual and dyadic spatiotemporal features <source/5_feature_extraction>` to describe movement and posture
    - sample behavioral datasets to :doc:`train machine-learning algorithms <source/3_basic_usage>`
    - post-process behavioral classification results for down-stream analyses
    - interactively :doc:`visualize and validate behavioral sequences <source/interactive_validation>`

You can use *vassi* to implement a full behavioral scoring pipeline in Python, train a machine-learning model, and use it to predict behavioral sequences.

.. code-block:: python

    # load training dataset
    dataset_train = load_dataset("train", ...)

    # configure feature extractor
    extractor = FeatureExtractor().read_yaml("feature_config.yaml")

    # extract samples from dataset
    X, y = dataset_train.subsample(extractor, size=0.1)

    # train classifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier.fit(X, dataset_train.encode(y))

    # load test dataset and predict
    dataset_test = load_dataset("test",  ...)
    classification_result = predict(dataset_test, classifier, extractor)

    # postprocessing
    processed_result = classification_result.smooth(
        lambda result: sliding_mean(result, window_size=5)
    ).threshold(
        [0.1, 0.8]  # assuming two categories
    )

    # save for downstream behavioral analyses
    processed_result.predictions.to_csv("predictions.csv")

Refer to the :doc:`basic usage <source/3_basic_usage>` page if you want to test *vassi* on an existing dataset.

The following video gives an overview of the interactive validation tool that complements the classification pipeline.

.. video:: source/SI6_interactive_validation.mp4
    :width: 100%

You can also have a look at the :doc:`interactive validation notebook <source/interactive_validation>` that we used to record this video.

.. toctree::
    :caption: Getting Started
    :hidden:

    source/1_installation
    source/2_example_datasets
    source/3_basic_usage
    source/4_import_data
    source/5_feature_extraction
    source/interactive_validation

.. toctree::
    :caption: Case studies
    :hidden:

    source/case_studies

.. toctree::
    :caption: API Reference
    :hidden:

    source/vassi.classification
    source/vassi.data_structures
    source/vassi.dataset
    source/vassi.features
    source/vassi.sliding_metrics
    source/vassi.config
    source/vassi.distributed
    source/vassi.io
    source/vassi.logging
    source/vassi.math
    source/vassi.series_operations
    source/vassi.utils
    source/vassi.visualization

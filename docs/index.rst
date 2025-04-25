.. automated-scoring documentation master file, created by
   sphinx-quickstart on Tue Dec 17 17:23:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

automated-scoring
=================

Welcome to *automated-scoring*, a Python package for automated scoring of animal behavior using trajectory data.

*automated-scoring* provides classes and methods to

- handle and manipulate trajectory and posture data
- extract individual and dyadic spatiotemporal features to describe movement and posture
- handle datasets containing multiple groups of individuals
- sample datasets to train machine-learning algorithms
- post-process behavioral classification results for down-stream analyses
- interactive visualization of behavioral sequences

.. toctree::
    :caption: Getting Started
    :hidden:

    source/1_installation
    source/2_example_datasets
    source/3_basic_usage
    source/4_import_data
    source/5_feature_extraction

.. toctree::
    :caption: CALMS21 dataset
    :hidden:

    source/minimal_example
    source/results_and_figures
    source/postprocessing_parameters

.. toctree::
    :caption: API Reference
    :hidden:

    source/automated_scoring.classification
    source/automated_scoring.data_structures
    source/automated_scoring.dataset
    source/automated_scoring.features
    source/automated_scoring.sliding_metrics
    source/automated_scoring.config
    source/automated_scoring.distributed
    source/automated_scoring.io
    source/automated_scoring.logging
    source/automated_scoring.math
    source/automated_scoring.reidentification
    source/automated_scoring.series_operations
    source/automated_scoring.utils
    source/automated_scoring.visualization

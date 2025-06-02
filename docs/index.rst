.. vassi documentation master file, created by
   sphinx-quickstart on Tue Dec 17 17:23:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

vassi
=================

Welcome to *vassi*, a Python package for verifiable, automated scoring of social interactions in animal groups using trajectory data.

.. admonition:: *vassi* provides classes and methods to

    - handle and manipulate trajectory and posture data in datasets with groups of multiple individuals (see :doc:`source/4_import_data`)
    - extract individual and dyadic spatiotemporal features to describe movement and posture (see :doc:`source/5_feature_extraction`)
    - sample datasets to train machine-learning algorithms (see :doc:`source/3_basic_usage`)
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

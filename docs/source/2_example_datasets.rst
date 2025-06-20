Example datasets
================

1. *CALMS21* dataset
--------------------

One of the two datasets that we use to benchmark and test our package is the *CALMS21* dataset of mouse resident-intruder interactions [CALMS21]_.
You can download the original *CALMS21* dataset `here <https://data.caltech.edu/records/s0vdx-0k302>`_.

We provide a `conversion script <https://github.com/pnuehrenberg/vassi/blob/main/src/vassi/case_studies/calms21/convert.py>`_ to format this dataset for the use with *vassi*.
You can run the :code:`convert.py` script from the command line. If you have previously created a virtual environment, activate it before running the script to make sure that the package can be imported.
Adjust the path arguments to match the location of the downloaded dataset files and the desired output directory.

.. code-block:: bash

    conda activate vassi  # activate the environment
    python -m vassi.case_studies.calms21.convert \
        --train_sequences path/to/calms21_task1_train.json \
        --test_sequences path/to/calms21_task1_test.json \
        --output_directory path/to/output/CALMS21


.. note::
    You may notice that all frames labeled as 'other' are labeled 'none' as the behavioral background category after conversion. This is done for consistency in our examples.


You will find the converted dataset in the output directory with :code:`train` and :code:`test` subdirectories, containing the converted trajectories and annotations for the train and test data, respectively.

For convenience, we also provide a `script to download <https://github.com/pnuehrenberg/vassi/blob/main/src/vassi/case_studies/calms21/download.py>`_ (and optionally convert) this dataset directly:

.. code-block:: bash

    python -m vassi.case_studies.calms21.download \
        # if not specified, this will default to "datasets/CALMS21"
        --output_directory path/to/datasets/CALMS21

If you want to keep the original dataset as well, or only want to download it, you can specify the following options:

.. code-block:: bash

    python -m vassi.case_studies.calms21.download --download-only

Or (only relevant when converting):

.. code-block:: bash

    python -m vassi.case_studies.calms21.download --keep-original

You can also download the videos along with the dataset, note that this requires additional storage space (~60 GB for download and extraction, ~30 GB afterwards):

.. code-block:: bash

    python -m vassi.case_studies.calms21.download --download-videos

2. *Social cichlids* dataset
----------------------------

We provide a second dataset (*social cichlids*), which is a dataset of social interactions in groups of cichlid fish (*N. multifiasciatus*).
The dataset is comprised of 9 video recordings of groups with 15 fish. Each fish is individually tracked with three keypoints (head, center and tail).
We provide matching behavioral annotations, i.e., start-stop intervals with a behavioral category of the following categories: 'approach', 'frontal display', 'lateral display', 'dart/bite', 'chase' and 'quiver'.
Unlabeled intervals are considered as the background category 'none'.


.. raw:: html

    <video controls="controls" id="video_social_cichlids" width="100%">
        <source src="../_static/social_cichlids.mp4" type="video/mp4" />
    </video>

You can find this dataset in our `data repository <https://doi.org/10.17617/3.3R0QYI>`_.

Again, for convenience, we provide a script to download this dataset directly:

.. code-block:: bash

    python -m vassi.case_studies.social_cichlids.download \
        # if not specified, this will default to "datasets/social_cichlids"
        --output_directory path/to/datasets/social_cichlids

Or, if you want to additionally download the videos alongside the dataset, you can use the following command (this requires ~5 GB of additional disk space):

.. code-block:: bash

    python -m vassi.case_studies.social_cichlids.download --download-videos

|

.. [CALMS21] Sun JJ, Karigo T, Chakraborty D, Mohanty SP, Wild B, Sun Q, Chen C, Anderson DJ, Perona P, Yue Y, Kennedy A. The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions. Adv Neural Inf Process Syst. 2021 Dec;2021(DB1):1-15. PMID: 38706835; PMCID: PMC11067713.

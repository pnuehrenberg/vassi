Example datasets
================

1. *CALMS21* dataset
--------------------

One of the two datasets that we use to benchmark and test our package is the *CALMS21* dataset of mouse resident-intruder interactions [CALMS21]_.
You can download the original *CALMS21* dataset `here <https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1>`_.

We provide a `conversion script <https://github.com/pnuehrenberg/automated-scoring/blob/main/examples/CALMS21/conversion.py>`_ to load and reformat this dataset.
You can run the :code:`conversion.py` script from the command line. If you have previously created a virtual environment, activate it before running the script to make sure that the package can be imported.
Adjust the path arguments to match the location of the downloaded dataset files and the desired output directory.

.. code-block:: bash

    conda activate automated-scoring  # activate the environment
    cd path/to/examples/CALMS21  # go to the directory with conversion.py
    python -m conversion \
        --train_sequences path/to/calms21_task1_train.json \
        --test_sequences path/to/calms21_task1_test.json \
        --output_directory path/to/output/CALMS21


.. note::
    You may notice that all frames labeled as 'other' are labeled 'none' as the behavioral background category after conversion. This is done for consistency in our examples.


You will find the converted dataset in the output directory with :code:`train` and :code:`test` subdirectories, containing the converted trajectories and annotations for the train and test data, respectively.

Alternatively, you can download the converted dataset in our `data repository <https://doi.org/10.17617/3.3R0QYI>`_.

2. *Social cichlids* dataset
----------------------------

We provide a second dataset (*social cichlids*), which is a dataset of social interactions in groups of cichlid fish (*N. multifiasciatus*).
The dataset is comprised of 9 video recordings of groups with 15 fish. Each fish is individually tracked with three keypoints (head, center and tail).
We provide matching behavioral annotations, i.e., start-stop intervals with a behavioral category of the following categories: 'approach', 'frontal display', 'lateral display', 'dart/bite', 'chase' and 'quiver'.
Unlabeled intervals are considered as the background category 'none'.

You can find the dataset in our `data repository <https://doi.org/10.17617/3.3R0QYI>`_.

|

.. [CALMS21] Sun JJ, Karigo T, Chakraborty D, Mohanty SP, Wild B, Sun Q, Chen C, Anderson DJ, Perona P, Yue Y, Kennedy A. The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions. Adv Neural Inf Process Syst. 2021 Dec;2021(DB1):1-15. PMID: 38706835; PMCID: PMC11067713.

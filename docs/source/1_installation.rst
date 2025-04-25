Installation
============

The automated-scoring package requires Python 3.12 or higher. It will install all dependencies automatically.
To avoid conflicts with other packages, we recommend to install the package in a virtual environment.

We recommend setting up a virtual environment via conda, for example obtained via a `miniforge <https://github.com/conda-forge/miniforge>`_ installation:

.. code-block:: bash

    conda create -n automated-scoring python=3.12
    conda activate automated-scoring

Install from PyPI
-----------------

.. warning::
    This is not yet available.

You can install the latest version from PyPI using pip:

.. code-block:: bash

    pip install automated-scoring

Install from GitHub
-------------------

.. warning::
    The repository is currently private, ask paul.nuehrenberg@uni-konstanz.de for access.

You can install the latest stable or development version from GitHub using pip:

.. code-block:: bash

    pip install git+https://github.com/pnuehrenberg/automated-scoring.git

Install from local directory
----------------------------

If you want to explore our code examples and datasets, you may obtain the package from GitHub.
You can then install it locally using pip:

.. code-block:: bash

    git clone https://github.com/pnuehrenberg/automated-scoring.git
    cd automated-scoring
    pip install .

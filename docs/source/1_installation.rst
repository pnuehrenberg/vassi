Installation
============

The *vassi* package requires Python 3.12 or higher. It will install all dependencies automatically.
To avoid conflicts with other packages, we recommend to install the package in a virtual environment.

This can be achieved with conda (or your favorite package manager). A minimal version of conda can be obtained from `miniforge <https://github.com/conda-forge/miniforge>`_.
Then, create a virtual environment:

.. code-block:: bash

    conda create -n vassi python=3.12
    conda activate vassi

Install from GitHub
-------------------

.. warning::
    The repository is currently private, ask paul.nuehrenberg@uni-konstanz.de for access.

You can install the latest stable or development version from GitHub using pip:

.. code-block:: bash

    pip install git+https://github.com/pnuehrenberg/vassi.git

Install from local directory
----------------------------

You can also first clone the package to have a local copy and then install it if you want to explore the code examples.

.. code-block:: bash

    git clone https://github.com/pnuehrenberg/vassi.git
    pip install vassi/

Installation and Usage
==============================

*Only MacOS and Linux supported at this time. It may or may not work on Windows but is not yet tested so YMMV*

1. Clone the repo

2. Install using below instructions or your own way

3. Run using the run command below

**Python virtual environment**

.. code-block:: bash

   python --version                # hopefully this says >= 3.9
   python -m venv venv             # make an environment in this folder here
   source venv/bin/activate        # activate the new environment
   pip install -r requirements.txt # install dependencies

**Conda**

.. code-block:: bash

   mamba env create -f env.yml
   mamba activate ts-cpu


**Run command**

There are two run "modes".

One performs a fitting procedure

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

And the other just performs a forward pass and gives you the spectra given some input parameters

.. code-block:: bash

   python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode forward
